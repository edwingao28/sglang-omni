# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni talker scheduler policy on top of the generic OmniScheduler."""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Any

import torch

from sglang.srt.managers import scheduler as _upstream_scheduler
from sglang.srt.managers.scheduler import Scheduler as _Upstream

from sglang_omni.models.qwen3_omni.config import MIN_PARTIAL_START_CHUNKS
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

logger = logging.getLogger(__name__)


def talker_overlap_requested() -> bool:
    """Whether the talker stage was asked to overlap decode steps.

    Single source of truth for the env gate: it both keeps
    ``disable_overlap_schedule`` unforced here and turns on the scheduler's
    one-step-lookahead async decode loop in ``create_talker_scheduler``.
    """
    return os.environ.get("SGLANG_OMNI_TALKER_OVERLAP", "0") == "1"


def configure_talker_server_args(
    server_args: Any,
    *,
    feedback_enabled: bool = True,
) -> bool:
    """Apply talker-specific scheduler/runtime defaults.

    Returns whether CUDA graphs were originally requested so the caller can
    re-enable graph capture after the model worker is constructed.
    """

    want_cuda_graph = not bool(server_args.disable_cuda_graph)
    overlap_requested = talker_overlap_requested()
    if feedback_enabled:
        if not overlap_requested:
            server_args.disable_overlap_schedule = True
        if want_cuda_graph:
            server_args.disable_cuda_graph = True
    server_args.disable_radix_cache = True
    server_args.chunked_prefill_size = 0
    return want_cuda_graph


class QwenTalkerScheduler(OmniScheduler):
    """Talker scheduler with Qwen-specific request and decode readiness."""

    def __init__(
        self,
        *args: Any,
        enable_partial_start: bool = False,
        partial_start_min_chunks: int = MIN_PARTIAL_START_CHUNKS,
        im_end_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if partial_start_min_chunks < MIN_PARTIAL_START_CHUNKS:
            raise ValueError(
                f"partial_start_min_chunks must be >= {MIN_PARTIAL_START_CHUNKS}, "
                f"got {partial_start_min_chunks}"
            )
        self._enable_partial_start = bool(enable_partial_start)
        self._partial_start_min_chunks = int(partial_start_min_chunks)
        self._im_end_token_id = im_end_token_id

    def _count_usable_prefetched_chunks(self, prefetched: list[Any]) -> int:
        im_end = self._im_end_token_id
        if im_end is None or not prefetched:
            return len(prefetched)
        metadata = getattr(prefetched[-1], "metadata", None) or {}
        token_id = metadata.get("token_id")
        if token_id is not None and int(token_id) == int(im_end):
            return len(prefetched) - 1
        return len(prefetched)

    def _is_request_build_ready(
        self,
        payload: Any,
        *,
        pending_stream_done: bool,
    ) -> bool:
        if pending_stream_done:
            return True
        if not self._enable_partial_start:
            return False
        prefetched = getattr(payload, "prefetched_chunks", None) or []
        return (
            self._count_usable_prefetched_chunks(prefetched)
            >= self._partial_start_min_chunks
        )

    def _initialize_request_stream_state(self, req_data: Any, payload: Any) -> None:
        del req_data, payload
        return None

    def _should_recheck_deferred_request_on_stream_chunk(
        self, request_id: str, chunk: Any
    ) -> bool:
        del request_id, chunk
        return self._enable_partial_start

    def _is_batch_ready_to_run(self, batch: Any) -> bool:
        if (
            batch is not None
            and batch.forward_mode.is_decode()
            and self._model_runner is not None
            and hasattr(self._model_runner, "is_decode_batch_ready")
            and not self._model_runner.is_decode_batch_ready(batch)
        ):
            logger.debug(
                "Deferring decode batch until talker feedback/text input is ready"
            )
            return False
        return True

    def get_next_batch_to_run(self) -> Any | None:
        batch = _Upstream.get_next_batch_to_run(self)
        if batch is not None and not self._is_batch_ready_to_run(batch):
            self._rollback_decode_prep_after_skip(batch)
            return None
        return batch

    def update_running_batch(self, batch: Any) -> Any:
        # Note (wenyao): retract_decode frees the KV and sets is_retracted before it
        # hands the request back, and the async resolve then skips retracted rows —
        # so a step that was launched but not yet resolved loses the token it already
        # emitted a codec frame for, and the replay desyncs against the text queue.
        # Land the in-flight step while the request still counts as running.
        if self._async_pending_batch() is not None and self._retract_is_imminent(batch):
            self._resolve_pending_async()
        return _Upstream.update_running_batch(self, batch)

    def _retract_is_imminent(self, batch: Any) -> bool:
        """Upstream's own retract trigger, read before it mutates the batch.

        ``check_decode_mem`` is evaluated here on the unfiltered batch, so this can
        only be true more often than the upstream check that follows it — draining a
        step early costs one round of overlap, missing one corrupts the replay.
        """
        if batch is None or not batch.reqs:
            return False
        interval = _upstream_scheduler.TEST_RETRACT_INTERVAL
        if (
            _upstream_scheduler.TEST_RETRACT
            and getattr(self, "forward_ct", 0) % interval == 0
        ):
            return True
        return not batch.check_decode_mem()

    def _add_request_to_queue(self, req: Any, is_retracted: bool = False) -> None:
        # Note (wenyao): retract has already freed req_pool_idx but nothing has
        # emitted into that feedback slot yet, so this is the last point where an
        # unconsumed feedback row can still be read back for replay.
        if is_retracted or bool(getattr(req, "is_retracted", False)):
            runner = getattr(self, "_model_runner", None)
            if runner is not None:
                try:
                    runner.snapshot_feedback_for_retract(req)
                except Exception as exc:
                    # Note (wenyao): retract runs inside get_next_batch_to_run, which
                    # the event loop calls outside any try, so an error escaping here
                    # kills the scheduler thread and every co-resident request. Fail
                    # this request the way a batch failure does and keep the stage up.
                    logger.exception(
                        "Talker retract feedback snapshot failed for request=%s; "
                        "aborting that request alone",
                        req.rid,
                    )
                    self._emit_request_error(req.rid, exc)
                    self.abort(req.rid, defer_running_cleanup=False)
                    return
        return _Upstream._add_request_to_queue(self, req, is_retracted=is_retracted)

    def _rollback_decode_prep_after_skip(self, batch: Any) -> None:
        # Note(Chenchen Hong, Xuesong): This is talker-only. It does not fully
        # invert prepare_for_decode; talker disables overlap/spec/Mamba/hisparse,
        # and its SamplingParams defaults keep the upstream penalizer branch
        # inactive. Also zero the req_to_token_pool cell that alloc_for_decode
        # wrote at (req_pool_indices, pre-increment seq_lens).
        if not batch.forward_mode.is_decode():
            return
        if not isinstance(batch.seq_lens_sum, int):
            raise TypeError(
                f"seq_lens_sum is {type(batch.seq_lens_sum).__name__}, expected int; "
                "sglang upstream prepare_for_decode changed; update rollback."
            )
        if batch.out_cache_loc is not None:
            self.token_to_kv_pool_allocator.free(batch.out_cache_loc)
            batch.out_cache_loc = None
        if batch.output_ids is None:
            batch.output_ids = batch.input_ids
        for req in batch.reqs:
            req.decode_batch_idx -= 1
            req.kv_committed_len -= 1
            req.kv_allocated_len -= 1
        batch.seq_lens.sub_(1)
        batch.seq_lens_cpu.sub_(1)
        batch.orig_seq_lens.sub_(1)
        batch.seq_lens_sum -= len(batch.reqs)
        req_to_token = batch.req_to_token_pool.req_to_token
        zero = getattr(self, "_rollback_zero", None)
        if (
            zero is None
            or zero.dtype != req_to_token.dtype
            or zero.device != req_to_token.device
        ):
            # Note (wenyao): a Python-scalar store wraps host-side and blocks on the
            # in-flight overlapped step; a cached device scalar keeps it async.
            zero = torch.zeros((), dtype=req_to_token.dtype, device=req_to_token.device)
            self._rollback_zero = zero
        req_to_token.index_put_((batch.req_pool_indices, batch.seq_lens), zero)

    def self_check_during_idle(self) -> None:
        if self.running_batch is not None and not self.running_batch.is_empty():
            return
        if self.waiting_queue:
            return
        super().self_check_during_idle()

    @staticmethod
    def _append_stream_chunk_default(req_data: Any, chunk: Any) -> None:
        pending_text_queue = getattr(req_data, "pending_text_queue", None)
        if pending_text_queue is None:
            pending_text_queue = deque()
            req_data.pending_text_queue = pending_text_queue
        pending_text_queue.append(getattr(chunk, "data", chunk))

    def _mark_stream_done(self, req_data: Any) -> None:
        if self._stream_done_handler is None:
            req_data.thinker_chunks_done = True
            return
        self._stream_done_handler(req_data)
