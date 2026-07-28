# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni talker runner with FIFO text/feedback decode handoff."""

from __future__ import annotations

import logging
from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.sglang_execution import attn_forward_context
from sglang_omni.models.qwen3_omni.components.feedback_slots import feedback_slot_rows
from sglang_omni.scheduling.messages import OutgoingMessage

logger = logging.getLogger(__name__)


class QwenTalkerModelRunner(ModelRunner):

    def __init__(
        self,
        tp_worker: Any,
        output_processor: Any,
        outbox: Any,
        *,
        code2wav_target: str = "code2wav",
        feedback_enabled: bool = True,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self._outbox = outbox
        self._code2wav_target = code2wav_target
        self._feedback_enabled = bool(feedback_enabled)
        if self._feedback_enabled:
            self._check_feedback_slots_cover_pool()

    def _check_feedback_slots_cover_pool(self) -> None:
        """Cross-check the req_pool_idx-keyed feedback table against the real pool.

        The model allocates ``_feedback_slots`` in its own ``__init__``, before
        ``req_to_token_pool`` exists, so it has to size from server args. This is the
        first point where both are visible; checking once here turns any future
        divergence into a startup error instead of an async device-side index assert
        that only sustained slot churn reaches.
        """
        slots = getattr(self.model, "_feedback_slots", None)
        pool = getattr(self.tp_worker.model_runner, "req_to_token_pool", None)
        # Note (wenyao): a skip must be audible. If either attribute is ever renamed
        # upstream the index sites keep working and only this check goes quiet, which
        # puts us back on an async device assert with no startup signal.
        if slots is None or pool is None:
            missing = "model._feedback_slots" if slots is None else ""
            if pool is None:
                missing = f"{missing} and " if missing else ""
                missing += "model_runner.req_to_token_pool"
            logger.warning(
                "Talker feedback slot bound check skipped: %s not found. A slot table "
                "too small for the request pool would now fail as a device-side index "
                "assert under load instead of at startup.",
                missing,
            )
            return
        # Note (wenyao): prefer the pool's own row count so this stays an independent
        # check rather than a restatement of the model's sizing formula.
        required = getattr(pool, "_alloc_size", None)
        if required is None:
            required = feedback_slot_rows(pool.size)
            logger.info(
                "Talker feedback slot bound check fell back to req_to_token_pool.size "
                "+ 1: %s has no _alloc_size. Still correct, but no longer independent "
                "of the model's own sizing formula.",
                type(pool).__name__,
            )
        if slots.shape[0] < required:
            raise RuntimeError(
                "Talker feedback slots are too small for the request pool: "
                f"_feedback_slots has {slots.shape[0]} rows but req_to_token_pool "
                f"of size {pool.size} allocates req_pool_idx in [1, {pool.size}], "
                f"needing {required} rows"
            )

    def execute(self, scheduler_output: Any):
        return super().execute(scheduler_output)

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        return self._run_projected_prefill_forward(
            forward_batch, schedule_batch, requests
        )

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del is_lookahead
        del schedule_batch
        if not self._feedback_enabled:
            return

        if not self._requests_ready_for_decode(requests):
            raise RuntimeError(
                "Talker decode reached model runner without ready feedback/text input"
            )

        self.model.prepare_decode_buffers(requests)
        self._write_feedback_buffers(
            requests, self._batch_pool_indices(forward_batch, len(requests))
        )

    def post_prefill(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        # Note (Xuesong): Do not clear data.prefill_input_embeds: decode retract may requeue
        # the Req for another prefill pass and Req.input_embeds is None.
        if not self._feedback_enabled:
            return

        if result.next_token_ids is None:
            return
        layer0_codes = result.next_token_ids
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        talker_hidden = result.logits_output.hidden_states
        if isinstance(talker_hidden, torch.Tensor) and talker_hidden.ndim == 2:
            talker_hidden = talker_hidden.unsqueeze(1)
        self.model.code_predictor_forward(layer0_codes, talker_hidden)
        self._stage_token_ids(result, result.next_token_ids)
        self._emit_code_chunks_and_feedback(
            requests=requests,
            pool_indices=self._batch_pool_indices(forward_batch, len(requests)),
        )

    def post_decode(
        self,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        if not self._feedback_enabled:
            return

        result.next_token_ids = self._collect_sampled_token_ids(requests)
        self._stage_token_ids(result, result.next_token_ids)
        self._emit_code_chunks_and_feedback(
            requests=requests,
            pool_indices=self._batch_pool_indices(forward_batch, len(requests)),
        )

    def post_decode_launch(
        self,
        result: Any,
        forward_batch: Any,
        requests: list,
    ) -> Any:
        """Async-decode GPU half of ``post_decode``: publish the in-forward
        sampled ids and emit this step's codec frame + feedback row, with no
        host sync. The emit MUST stay here: it snapshots ``_output_codes`` and
        scatters ``_output_embeds`` into the slot table, both fixed buffers the
        next step's forward overwrites, and running it right after this step's
        forward on the same stream is what orders those reads before that write.

        Returns the sampled ids as the resolve payload; ``_finalize`` reads them
        from the staged pinned copy, which the caller's event covers.
        """
        if not self._feedback_enabled or not requests:
            return None

        result.next_token_ids = self._collect_sampled_token_ids(requests)
        self._stage_token_ids(result, result.next_token_ids)
        self._emit_code_chunks_and_feedback(
            requests=requests,
            pool_indices=self._batch_pool_indices(forward_batch, len(requests)),
        )
        return result.next_token_ids

    def post_decode_resolve(
        self,
        launch_buf: Any,
        result: Any,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> None:
        """Async-decode host half: restore the launch-time ids so the shared
        ``_finalize`` tail reads this step's tokens. Every other host-side step
        (pinned token-id sync, finish bookkeeping, ``generation_steps``) already
        lives in that tail, so there is nothing else to collect here.
        """
        del forward_batch, schedule_batch, requests
        if launch_buf is not None:
            result.next_token_ids = launch_buf

    def lookahead_eligible(self, batch: Any) -> bool:
        """The feedback talker is always lookahead-eligible.

        The base gate exists because its launch samples one step before resolve
        appends the token to ``req.output_ids``, so any history-scored sampling
        term would read a stale view. Talker decode never samples on that path:
        it samples inside the forward against a device-side repetition mask that
        is advanced from ``_sampled_token_ids`` (the previous forward's output),
        and it ignores frequency/presence penalties and ``min_new_tokens``
        entirely. ``req.output_ids`` is read only when ``prepare_decode_buffers``
        falls off its fast path — a batch composition change — and there the
        rebuilt mask can miss the newest token, one step per rebuild.
        """
        if not self._feedback_enabled:
            return super().lookahead_eligible(batch)
        return True

    def _collect_sampled_token_ids(self, requests: list) -> torch.Tensor:
        # Note (wenyao): clone, not a view: the next forward writes
        # _sampled_token_ids in place, and under lookahead that write lands
        # before this step's resolve reads the ids.
        return self.model._sampled_token_ids[: len(requests)].clone()

    @staticmethod
    def _batch_pool_indices(forward_batch: Any, bs: int) -> torch.Tensor:
        """This batch's ``req_pool_idx`` per row, already on the device.

        Row i belongs to ``requests[i]``: upstream fills this tensor from the same
        allocation that sets ``req.req_pool_idx`` and reslices it alongside
        ``batch.reqs`` on every filter and merge, and any CUDA-graph padding is
        appended past ``bs``.

        Building the index from a Python list instead would be a pageable
        host-to-device copy on every frame, which ends in a stream synchronize and
        serializes that step against its own forward.
        """
        rows = forward_batch.req_pool_indices
        if int(rows.shape[0]) < bs:
            raise RuntimeError(
                "Talker forward batch carries fewer pool indices than requests: "
                f"{int(rows.shape[0])} rows for {bs} requests"
            )
        return rows[:bs]

    def _emit_code_chunks_and_feedback(
        self,
        *,
        requests: list,
        pool_indices: torch.Tensor,
    ) -> None:
        bs = len(requests)
        # Note (wenyao): one batched clone, not one per row: the snapshot must be a
        # fresh allocation so its rows survive the next in-graph write to the
        # fixed-address _output_codes.
        codes_snap = self.model._output_codes[:bs].detach().clone()
        reqs = [sched_req.data.req for sched_req in requests]
        slot_ids = [req.req_pool_idx for req in reqs]
        pool_ids = torch.tensor(
            slot_ids,
            dtype=torch.long,
            device=self.model._output_embeds.device,
        )
        # Note (wenyao): slots must be written before the next forward's in-graph
        # write to _output_embeds; emit runs post-forward on the same stream, so the
        # ordering holds without a sync.
        self.model._feedback_slots[pool_indices] = self.model._output_embeds[:bs]
        for idx, sched_req in enumerate(requests):
            req = reqs[idx]
            # Note (wenyao): a row that finished or retracted in an earlier step is
            # still carried by this batch; shipping its frame would append audio past
            # the end of the request. The slot write and the counter below stay
            # unconditional so the retract snapshot still sees a consistent row.
            if not self._req_is_done(req):
                code_chunk = codes_snap[idx]
                # Tell code2wav whether to forward audio chunks to the Coordinator.
                stage_payload = sched_req.data.stage_payload
                is_streaming = bool(
                    stage_payload is not None
                    and (stage_payload.request.params or {}).get("stream", False)
                )
                self._outbox.put(
                    OutgoingMessage(
                        request_id=req.rid,
                        type="stream",
                        data=code_chunk,
                        target=self._code2wav_target,
                        metadata={"stream": is_streaming},
                    )
                )
            sched_req.data.pending_feedback_count += 1
            # Note (wenyao): retract frees req_pool_idx, so the slot this frame was
            # written into is only recoverable from the request's own record. Read
            # off the request rather than pool_indices: that is a device tensor and
            # touching it on the host would sync.
            sched_req.data.feedback_slot_idx = req.req_pool_idx

    def _req_is_done(self, req: Any) -> bool:
        """Whether this row was finished or retracted by an earlier step.

        Same predicate the base resolve uses to build its skip set, so the codec
        stream and the token stream drop the same rows.
        """
        try:
            finished = bool(req.finished())
        except AttributeError:
            finished = False
        return finished or self._req_is_retracted(req)

    def snapshot_feedback_for_retract(self, req: Any) -> None:
        """Copy an unconsumed feedback row out of its slot before the slot is reused.

        Called while the retracted request is being requeued: its pool index is
        already freed, but no other request has emitted into the row yet. That makes
        this the last legal read of the recorded slot, so the index is retired here
        on every retract - a second retract before a new emit would otherwise read a
        row the pool has since handed to another request.

        Replay depth is one row: the slot table keeps only the newest emit per
        request, so a request retracted with more than one unconsumed frame cannot be
        fully replayed. ``_generated_prefill_slice`` raises if replay ever needs a
        second row.
        """
        if not self._feedback_enabled:
            return
        data = getattr(req, "_omni_data", None)
        if data is None:
            return
        slot_idx = getattr(data, "feedback_slot_idx", None)
        data.feedback_slot_idx = None
        pending = getattr(data, "pending_feedback_count", 0)
        if pending <= 0:
            return
        if getattr(data, "retracted_feedback_embed", None) is not None:
            # Note (wenyao): retiring the index here discards the newer row the
            # re-prefill emitted into the slot, which is safe because that prefill
            # regenerates the same frame; the older held snapshot is the one still
            # owed to the model.
            return
        if slot_idx is None:
            raise RuntimeError(
                "Talker request has pending feedback but no recorded slot to "
                f"snapshot on retract (pending_feedback_count={pending}): the "
                "recorded slot is retired by every retract, so a second retract "
                "before a new emit has no row left to read and the freed pool "
                "index may already belong to another request"
            )
        data.retracted_feedback_embed = self.model._feedback_slots[slot_idx].clone()

    def sample_before_post_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return False

    def is_decode_batch_ready(self, schedule_batch: Any) -> bool:
        if not self._feedback_enabled or not schedule_batch.forward_mode.is_decode():
            return True
        return all(
            self._data_has_next_decode_input(getattr(req, "_omni_data", None))
            for req in schedule_batch.reqs
        )

    def _run_projected_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        has_projected = forward_batch.input_embeds is not None or any(
            bool(req.data.input_embeds_are_projected) for req in requests
        )
        if not has_projected:
            return None

        projected_flags = [
            bool(req.data.input_embeds_are_projected) for req in requests
        ]
        has_projected_requests = any(projected_flags)
        if has_projected_requests and not all(projected_flags):
            raise RuntimeError(
                "Talker projected and unprojected prefill requests cannot be "
                "batched together"
            )

        input_embeds_are_projected = has_projected_requests
        input_embeds = forward_batch.input_embeds
        if has_projected_requests:
            parts: list[torch.Tensor] = []
            for sched_req in requests:
                req = sched_req.data.req
                prefix_len = len(req.prefix_indices)
                extend_len = int(req.extend_range.length)
                part = self._projected_prefill_slice(
                    sched_req=sched_req,
                    prefix_len=prefix_len,
                    extend_len=extend_len,
                    device=forward_batch.input_ids.device,
                )
                if part is not None and part.shape[0] > 0:
                    parts.append(part)
            if not parts:
                return None
            input_embeds = torch.cat(parts, dim=0)
        elif input_embeds is None:
            return None

        expected_rows = int(forward_batch.input_ids.shape[0])
        if input_embeds.shape[0] != expected_rows:
            raise RuntimeError(
                "Talker projected prefill embeds must align with forward input_ids: "
                f"got {input_embeds.shape[0]} rows for {expected_rows} input ids"
            )

        result = self._forward_with_input_embeds(
            forward_batch,
            input_embeds=input_embeds,
            input_embeds_are_projected=input_embeds_are_projected,
        )
        return result

    @staticmethod
    def _projected_prefill_slice(
        *,
        sched_req: Any,
        prefix_len: int,
        extend_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if extend_len <= 0:
            return None

        data = sched_req.data
        req = data.req
        end = prefix_len + extend_len
        tensor = data.prefill_input_embeds
        if tensor is not None:
            prompt_len = int(tensor.shape[0])
            dtype = tensor.dtype
            embed_device = tensor.device
            parts = QwenTalkerModelRunner._prefill_prompt_parts_from_tensor(
                tensor=tensor,
                prefix_len=prefix_len,
                end=end,
            )
        else:
            embeds = req.input_embeds
            if not embeds:
                return None
            prompt_len = len(embeds)
            dtype = torch.float32
            embed_device = device
            parts = QwenTalkerModelRunner._prefill_prompt_parts_from_list(
                embeds=embeds,
                prefix_len=prefix_len,
                end=end,
                device=device,
            )

        if end > prompt_len:
            generated = QwenTalkerModelRunner._generated_prefill_slice(
                sched_req=sched_req,
                gen_start=max(prefix_len, prompt_len) - prompt_len,
                gen_end=end - prompt_len,
                device=embed_device,
                dtype=dtype,
            )
            if generated is not None:
                parts.append(generated)

        if not parts:
            return None
        return torch.cat(parts, dim=0)

    @staticmethod
    def _prefill_prompt_parts_from_tensor(
        *,
        tensor: torch.Tensor,
        prefix_len: int,
        end: int,
    ) -> list[torch.Tensor]:
        prompt_len = int(tensor.shape[0])
        start = min(prefix_len, prompt_len)
        stop = min(end, prompt_len)
        return [tensor[start:stop]] if stop > start else []

    @staticmethod
    def _prefill_prompt_parts_from_list(
        *,
        embeds: list,
        prefix_len: int,
        end: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        prompt_len = len(embeds)
        start = min(prefix_len, prompt_len)
        stop = min(end, prompt_len)
        if stop <= start:
            return []
        return [
            torch.as_tensor(
                embeds[start:stop],
                device=device,
                dtype=torch.float32,
            )
        ]

    @staticmethod
    def _generated_prefill_slice(
        *,
        sched_req: Any,
        gen_start: int,
        gen_end: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if gen_end <= gen_start:
            return None

        data = sched_req.data
        history = QwenTalkerModelRunner._decode_input_history(data)
        while len(history) < gen_end:
            combined = QwenTalkerModelRunner._take_next_decode_input_embed(
                sched_req=sched_req,
                device=device,
                dtype=dtype,
            )
            if combined is None:
                raise RuntimeError(
                    "Cannot replay retracted talker decode tokens: missing "
                    "feedback/text input embeds for generated-token prefill "
                    "(pending_feedback_count="
                    f"{getattr(data, 'pending_feedback_count', 0)}). A retract "
                    "recovers at most one feedback row, so a request retracted with "
                    "more than one unconsumed frame cannot be fully replayed"
                )
            QwenTalkerModelRunner._append_decode_input_history(data, combined)

        rows = [
            QwenTalkerModelRunner._decode_row(row, device=device, dtype=dtype)
            for row in history[gen_start:gen_end]
        ]
        if not rows:
            return None
        return torch.stack(rows, dim=0)

    def _write_feedback_buffers(
        self, requests: list, pool_indices: torch.Tensor | None = None
    ) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return

        feedback_buffer = self.model._feedback_buffer
        feedback_mask = self.model._feedback_mask
        device = feedback_buffer.device
        dtype = feedback_buffer.dtype
        feedback_mask[:batch_size] = False

        rows: list[int] = []
        datas: list[Any] = []
        pool_ids: list[int] = []
        overrides: list[torch.Tensor | None] = []
        text_rows: list[torch.Tensor] = []
        any_missing_pool_idx = False
        for row_idx, sched_req in enumerate(requests):
            data = sched_req.data
            if getattr(data, "pending_feedback_count", 0) <= 0:
                continue
            override = getattr(data, "retracted_feedback_embed", None)
            pool_idx = getattr(getattr(data, "req", None), "req_pool_idx", None)
            if override is None and pool_idx is None:
                raise RuntimeError(
                    "Talker request has pending feedback but no pool slot to read it "
                    "from: req_pool_idx is None and no retracted feedback snapshot "
                    "was taken"
                )
            next_text = self._peek_next_text_row(data)
            if next_text is None:
                continue
            rows.append(row_idx)
            datas.append(data)
            any_missing_pool_idx = any_missing_pool_idx or pool_idx is None
            pool_ids.append(0 if pool_idx is None else int(pool_idx))
            overrides.append(override)
            text_rows.append(self._decode_row(next_text, device=device, dtype=dtype))
        if not rows:
            return

        if (
            pool_indices is not None
            and len(rows) == batch_size
            and not any_missing_pool_idx
        ):
            # Note (wenyao): the readiness gate only runs a decode batch when every
            # row has feedback and text, so this is the steady state and it must stay
            # free of host-built index tensors. The branch below covers rows the gate
            # cannot see (a retract snapshot with no pool slot) and pays one
            # host-to-device copy, which serializes that step against the forward.
            pool_ids_t = pool_indices
        else:
            pool_ids_t = torch.tensor(pool_ids, dtype=torch.long, device=device)
        feedback_rows = self.model._feedback_slots[pool_ids_t]
        for i, override in enumerate(overrides):
            if override is not None:
                feedback_rows[i] = self._decode_row(
                    override, device=device, dtype=dtype
                )
        combined = feedback_rows + torch.stack(text_rows, dim=0)

        for i, data in enumerate(datas):
            self._append_decode_input_history(data, combined[i])
            self._consume_feedback_and_text(data)

        if len(rows) == batch_size:
            # Note (wenyao): a slice-assign here keeps the steady state free of a
            # host-built row-index tensor.
            feedback_buffer[:batch_size] = combined
            feedback_mask[:batch_size] = True
            return
        rows_t = torch.tensor(rows, dtype=torch.long, device=device)
        feedback_buffer[rows_t] = combined
        feedback_mask[rows_t] = True

    @staticmethod
    def _data_has_next_decode_input(data: Any) -> bool:
        if data is None:
            return False
        if getattr(data, "pending_feedback_count", 0) <= 0:
            return False
        pending_text_queue = getattr(data, "pending_text_queue", None)
        if pending_text_queue:
            return True
        return bool(
            data.thinker_chunks_done
            and getattr(data, "tts_pad_embed", None) is not None
        )

    def _requests_ready_for_decode(self, requests: list) -> bool:
        return all(
            self._data_has_next_decode_input(sched_req.data) for sched_req in requests
        )

    @staticmethod
    def _pop_left(queue: Any) -> torch.Tensor | None:
        if not queue:
            return None
        if hasattr(queue, "popleft"):
            return queue.popleft()
        if isinstance(queue, list):
            return queue.pop(0)
        return None

    @staticmethod
    def _peek_left(queue: Any) -> torch.Tensor | None:
        if not queue:
            return None
        if isinstance(queue, list):
            return queue[0]
        if hasattr(queue, "__getitem__"):
            return queue[0]
        return None

    @staticmethod
    def _decode_input_history(data: Any) -> list[torch.Tensor]:
        history = getattr(data, "decode_input_embeds", None)
        if history is None:
            history = []
            data.decode_input_embeds = history
        return history

    @staticmethod
    def _append_decode_input_history(data: Any, row: torch.Tensor) -> None:
        QwenTalkerModelRunner._decode_input_history(data).append(row.detach())

    @staticmethod
    def _decode_row(
        row: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        row = row.reshape(-1)
        if row.device != device or row.dtype != dtype:
            raise RuntimeError(
                "Talker decode rows must already match the feedback buffer "
                f"device/dtype, got {row.device}/{row.dtype}, "
                f"expected {device}/{dtype}"
            )
        return row

    @staticmethod
    def _peek_next_text_row(data: Any) -> torch.Tensor | None:
        next_text = QwenTalkerModelRunner._peek_left(
            getattr(data, "pending_text_queue", None)
        )
        if next_text is not None:
            return next_text
        if not getattr(data, "thinker_chunks_done", False):
            return None
        return getattr(data, "tts_pad_embed", None)

    @staticmethod
    def _consume_feedback_and_text(data: Any) -> None:
        data.pending_feedback_count -= 1
        data.retracted_feedback_embed = None
        if getattr(data, "pending_text_queue", None):
            QwenTalkerModelRunner._pop_left(data.pending_text_queue)

    @staticmethod
    def _combine_feedback_with_next_text(
        *,
        data: Any,
        feedback: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if feedback is None:
            return None
        next_text = QwenTalkerModelRunner._peek_next_text_row(data)
        if next_text is None:
            return None
        return QwenTalkerModelRunner._decode_row(
            feedback,
            device=device,
            dtype=dtype,
        ) + QwenTalkerModelRunner._decode_row(
            next_text,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _take_next_decode_input_embed(
        *,
        sched_req: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        # Note (wenyao): retract replay must not read _feedback_slots — the pool idx
        # is already freed and may belong to another request.
        data = sched_req.data
        if getattr(data, "pending_feedback_count", 0) <= 0:
            return None
        combined = QwenTalkerModelRunner._combine_feedback_with_next_text(
            data=data,
            feedback=getattr(data, "retracted_feedback_embed", None),
            device=device,
            dtype=dtype,
        )
        if combined is None:
            return None

        QwenTalkerModelRunner._consume_feedback_and_text(data)
        return combined

    def _forward_with_input_embeds(
        self,
        forward_batch: Any,
        *,
        input_embeds: torch.Tensor,
        input_deepstack_embeds: torch.Tensor | None = None,
        input_deepstack_mask: torch.Tensor | None = None,
        input_embeds_are_projected: bool = False,
    ) -> GenerationBatchResult:
        model_runner = self.tp_worker.model_runner
        model_dtype = self.model.activation_dtype

        model_runner.attn_backend.init_forward_metadata(forward_batch)

        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions

        input_embeds = input_embeds.to(
            device=forward_batch.input_ids.device,
            dtype=model_dtype,
        )
        if input_deepstack_embeds is not None:
            input_deepstack_embeds = input_deepstack_embeds.to(
                device=forward_batch.input_ids.device,
                dtype=model_dtype,
            )

        with attn_forward_context(model_runner.attn_backend):
            logits_output = self.model(
                input_ids=forward_batch.input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
                input_deepstack_embeds=input_deepstack_embeds,
                input_deepstack_mask=input_deepstack_mask,
                input_embeds_are_projected=input_embeds_are_projected,
            )
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )
