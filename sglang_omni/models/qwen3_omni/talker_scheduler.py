# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni talker scheduler policy on top of the generic OmniScheduler."""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from sglang.srt.managers.scheduler import Scheduler as _Upstream

from sglang_omni.models.qwen3_omni.config import MIN_PARTIAL_START_CHUNKS
from sglang_omni.models.qwen3_omni.request_builders import (
    PROMPT_SEGMENT_FUTURE_ATTR,
)
from sglang_omni.models.qwen3_omni.two_phase_kv import (
    ASSISTANT_TAIL_ROWS,
    adopt_parked_prompt_kv,
    is_tail_pending,
    snapshot_prompt_kv,
    uninstall_parked_prefix_shadow,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler, _detach_request_data
from sglang_omni.vendor.sglang.server_args import override_server_args

logger = logging.getLogger(__name__)

KVSTAT_ENABLED = os.environ.get("SGLANG_OMNI_TALKER_KVSTAT", "0") == "1"


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
    overrides = {
        "disable_radix_cache": True,
        "chunked_prefill_size": 0,
    }
    if feedback_enabled:
        overrides["disable_overlap_schedule"] = True
        if want_cuda_graph:
            overrides["disable_cuda_graph"] = True
    override_server_args(server_args, "qwen3_omni.talker", **overrides)
    return want_cuda_graph


class QwenTalkerScheduler(OmniScheduler):
    """Talker scheduler with Qwen-specific request and decode readiness."""

    # Note (wenyao): class scope, because schedulers get built with
    # object.__new__ in tests and the two-phase hooks run before any __init__.
    _two_phase_prefill: bool = False
    _prompt_segment_prebuilder: Callable[[Any], Any] | None = None
    _prompt_segment_executor: ThreadPoolExecutor | None = None
    _prompt_segment_futures: dict[str, Future] | None = None
    _two_phase_kv: bool = False
    _two_phase_max_parked: int = 0
    _phase_one_builder: Callable[[Any, Any], Any] | None = None
    _phase_one_queue: deque | None = None
    _phase_one_data: dict[str, Any] | None = None
    _phase_one_denied: set[str] | None = None
    _parked_reqs: dict[str, Any] | None = None
    _parked_total: int = 0
    _two_phase_slice_rows: int = 0
    _two_phase_slice_every: int = 1
    _phase_one_slice_ct: int = -1
    # Note (wenyao): the prefill/decode interleave is a wave2-only scheduler
    # feature that this tree does not carry, so the seam phase 1 buys steps
    # back from is permanently closed here. Declared rather than deleted
    # because the slice logic above is written against it and porting the
    # interleave must not require re-deriving that logic.
    prefill_decode_interleave: bool = False
    _interleave_defer_prefill: bool = False

    def __init__(
        self,
        *args: Any,
        enable_partial_start: bool = False,
        partial_start_min_chunks: int = MIN_PARTIAL_START_CHUNKS,
        im_end_token_id: int | None = None,
        talker_two_phase_prefill: bool = False,
        talker_two_phase_kv: bool = True,
        talker_two_phase_max_parked: int = 24,
        talker_two_phase_min_batch: int = 4,
        talker_two_phase_coalesce_above: int = 8,
        talker_two_phase_pool_reserve: int = 8,
        talker_two_phase_slice_rows: int = 0,
        talker_two_phase_slice_every: int = 4,
        prompt_segment_prebuilder: Callable[[Any], Any] | None = None,
        phase_one_builder: Callable[[Any, Any], Any] | None = None,
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
        self._prompt_segment_prebuilder = prompt_segment_prebuilder
        self._two_phase_prefill = bool(talker_two_phase_prefill) and (
            prompt_segment_prebuilder is not None
        )
        self._prompt_segment_futures: dict[str, Future] = {}
        # Note (wenyao): its own worker, not the request-build pool — that pool
        # is sized for admission ordering and is disabled at tp_size > 1, while
        # this work only reads the payload the talker already holds.
        self._prompt_segment_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="omni-talker-prompt-segment",
            )
            if self._two_phase_prefill
            else None
        )
        self._phase_one_builder = phase_one_builder
        self._two_phase_max_parked = max(0, int(talker_two_phase_max_parked))
        self._two_phase_min_batch = max(1, int(talker_two_phase_min_batch))
        self._two_phase_coalesce_above = max(0, int(talker_two_phase_coalesce_above))
        self._two_phase_pool_reserve = max(1, int(talker_two_phase_pool_reserve))
        self._two_phase_slice_rows = max(0, int(talker_two_phase_slice_rows))
        self._two_phase_slice_every = max(1, int(talker_two_phase_slice_every))
        self._phase_one_slice_ct = -self._two_phase_slice_every
        self._phase_one_queue = deque()
        self._phase_one_data = {}
        self._phase_one_denied = set()
        self._parked_reqs = {}
        self._parked_total = 0
        # Note (wenyao): a chunked phase-1 extend would claim the scheduler's
        # single ``chunked_req`` slot and serialize the stage to one request, so
        # the early KV write only runs where chunking is off.
        chunked_prefill_size = int(getattr(self, "chunked_prefill_size", 0) or 0)
        self._two_phase_kv = (
            self._two_phase_prefill
            and bool(talker_two_phase_kv)
            and phase_one_builder is not None
            and self._two_phase_max_parked > 0
            and chunked_prefill_size <= 0
        )
        logger.info(
            "talker two-phase prefill: requested=%s active=%s kv=%s max_parked=%s "
            "min_batch=%s coalesce_above=%s pool_reserve=%s slice_rows=%s "
            "slice_every=%s",
            bool(talker_two_phase_prefill),
            self._two_phase_prefill,
            self._two_phase_kv,
            self._two_phase_max_parked,
            self._two_phase_min_batch,
            self._two_phase_coalesce_above,
            self._two_phase_pool_reserve,
            self._two_phase_slice_rows,
            self._two_phase_slice_every,
        )

    def _prebuild_deferred_payload(self, payload: Any) -> None:
        """Project the prompt rows while the K-gate is still closed.

        The prompt segment reads only the prompt ids and the merged multimodal
        features, both of which the talker payload already carries on arrival,
        so none of it has to wait for thinker chunks.
        """
        executor = self._prompt_segment_executor
        if executor is None:
            return
        request_id = payload.request_id
        futures = self._prompt_segment_futures
        if futures is None:
            futures = self._prompt_segment_futures = {}
        if request_id in futures:
            return
        try:
            future = executor.submit(self._prompt_segment_prebuilder, payload)
        except RuntimeError:
            return
        futures[request_id] = future
        setattr(payload, PROMPT_SEGMENT_FUTURE_ATTR, future)

    def _release_prebuilt_payload(self, request_id: str) -> None:
        futures = self._prompt_segment_futures
        if futures:
            future = futures.pop(request_id, None)
            if future is not None:
                future.cancel()
        if not self._two_phase_kv:
            return
        self._phase_one_data.pop(request_id, None)
        self._phase_one_denied.discard(request_id)
        self._drop_queued_phase_one(request_id)
        parked = self._parked_reqs.pop(request_id, None)
        if parked is not None:
            self._reclaim_parked(parked)

    def _reclaim_parked(self, req: Any) -> None:
        """Give a parked request's prompt KV and pool row back.

        Abort between the two phases arrives here through
        ``_release_prebuilt_payload``: a parked request sits in no batch and no
        waiting queue, so the scheduler's batch-scanning reclaim never sees it.
        """
        uninstall_parked_prefix_shadow(req)
        req.cache_protected_len = 0
        self._release_request_kv_cache(req)
        _detach_request_data(req)

    def _drop_queued_phase_one(self, request_id: str) -> None:
        queue = self._phase_one_queue
        if not queue:
            return
        for req in list(queue):
            if req.rid == request_id:
                queue.remove(req)
                _detach_request_data(req)

    def _shutdown_resources(self) -> None:
        executor = self._prompt_segment_executor
        self._prompt_segment_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        parked = self._parked_reqs
        while parked:
            self._reclaim_parked(parked.popitem()[1])
        super()._shutdown_resources()

    def process_input_requests(self, recv_reqs: list[Any]) -> None:
        super().process_input_requests(recv_reqs)
        if self._two_phase_kv:
            self._admit_phase_one_requests()

    def _admit_phase_one_requests(self) -> None:
        """Queue a prompt-only request for every prebuild that has landed.

        The prompt segment consumes nothing from the thinker stream, so its
        rows can take their KV slots while the readiness gate is still closed;
        only the 9-row assistant tail is left on the gate's critical path.
        """
        futures = self._prompt_segment_futures or {}
        for request_id, future in list(futures.items()):
            # Note (wenyao): every parked request still holds its _phase_one_data
            # entry, so adding the two dicts charged those requests twice and
            # halved the budget exactly when concurrency needed all of it.
            if len(self._phase_one_data) >= self._two_phase_max_parked:
                return
            if (
                request_id in self._phase_one_data
                or request_id in self._parked_reqs
                or request_id in self._phase_one_denied
            ):
                continue
            payload = self._deferred_request_payloads.get(request_id)
            if payload is None or not future.done():
                continue
            # Note (wenyao): parked rows hold a pool slot outside running_batch,
            # where max_running_requests does not see them, and
            # get_num_allocatable_reqs floors on the same pool — so prepaying
            # down to the last row would stall ordinary admission instead.
            if (
                self.req_to_token_pool.available_size()
                <= self._two_phase_pool_reserve
            ):
                return
            try:
                req_data = self._phase_one_builder(payload, future.result())
            except Exception:
                # Note (wenyao): deny before logging — this payload is retried on
                # every scheduler pass until its gate opens, so a build that
                # always fails would otherwise raise once per iteration.
                self._phase_one_denied.add(request_id)
                logger.exception(
                    "talker phase-1 build failed for %s; leaving it whole",
                    request_id,
                )
                continue
            req = req_data.req
            self._normalize_req_token_arrays(req)
            req._coalesce_enqueue_t = time.perf_counter()
            req._omni_terminal_claimed = False
            req._omni_data = req_data
            self._phase_one_data[request_id] = req_data
            self._phase_one_queue.append(req)
            if KVSTAT_ENABLED:
                self._kvstat_log("p1queue", rid=request_id)

    def _kvstat_state(self, running_batch: Any) -> dict[str, Any]:
        return {
            "run_bs": 0 if running_batch is None else len(running_batch.reqs),
            "full": int(
                bool(running_batch is not None and running_batch.batch_is_full)
            ),
            "wait": len(self.waiting_queue),
            "pool": self.req_to_token_pool.available_size(),
            "kvtok": self.token_to_kv_pool_allocator.available_size(),
            "parked": len(self._parked_reqs or ()),
            "p1q": len(self._phase_one_queue or ()),
            "p1d": len(self._phase_one_data or ()),
        }

    def _kvstat_log(self, tag: str, **fields: Any) -> None:
        logger.info(
            "KVSTAT %s t=%.6f %s",
            tag,
            time.perf_counter(),
            " ".join(f"{k}={v}" for k, v in fields.items()),
        )

    def _phase_one_ready(self, running_batch: Any) -> bool:
        """Whether a phase-1 extend is worth the forward pass it costs.

        A pass carrying one or two rows is free while the stage is quiet, but
        on a colocated GPU it competes with thinker prefill once the talker is
        loaded — which is what erased the win at high concurrency. So coalesce
        only above that load; below it the prepay stays opportunistic and the
        low-concurrency win is untouched.
        """
        queue = self._phase_one_queue
        if not queue:
            return False
        if len(queue) >= self._two_phase_min_batch:
            return True
        running_bs = 0 if running_batch is None else len(running_batch.reqs)
        return running_bs < self._two_phase_coalesce_above

    def _phase_one_slice_claim(self) -> bool:
        """Whether phase 1 may take a step the decode interleave just claimed.

        Purely opportunistic prepay starves itself at concurrency: the ordinary
        pass and the interleave between them own nearly every step once the
        stage is loaded, so the requests that would gain most from a parked
        prompt are the ones that never get it. The slice buys those steps back
        from the interleave, and one claim per ``slice_every`` steps bounds
        what decode gives up.

        Only steps the interleave already deferred are on offer, so a queued
        ordinary request is not overtaken: the deferral denied it this step
        whatever phase 1 does, and the next undeferred step still admits it
        ahead of any prepay.
        """
        if self._two_phase_slice_rows <= 0:
            return False
        return self.forward_ct - self._phase_one_slice_ct >= self._two_phase_slice_every

    def get_new_batch_prefill(self, running_batch: Any) -> Any:
        deferring = bool(
            self.prefill_decode_interleave and self._interleave_defer_prefill
        )
        before = self._kvstat_state(running_batch) if KVSTAT_ENABLED else None
        plan = super().get_new_batch_prefill(running_batch)
        slicing = deferring and self._phase_one_slice_claim()
        skip_phase_one = (
            plan.batch_to_run is not None
            or (deferring and not slicing)
            or not self._two_phase_kv
            or not self._phase_one_ready(plan.running_batch)
        )
        p1_plan = (
            None
            if skip_phase_one
            else self._phase_one_prefill_plan(plan.running_batch)
        )
        if slicing and p1_plan is not None and p1_plan.batch_to_run is not None:
            self._phase_one_slice_ct = self.forward_ct
        if KVSTAT_ENABLED and (
            before["wait"] or before["p1q"] or before["parked"] or plan.batch_to_run
        ):
            ord_batch = plan.batch_to_run
            p1_batch = None if p1_plan is None else p1_plan.batch_to_run
            after_ord = self._kvstat_state(plan.running_batch)
            final = (
                after_ord
                if p1_plan is None
                else self._kvstat_state(p1_plan.running_batch)
            )
            self._kvstat_log(
                "step",
                ct=self.forward_ct,
                defer=int(deferring),
                full_in=before["full"],
                wait_in=before["wait"],
                run_bs=before["run_bs"],
                pool=before["pool"],
                kvtok=before["kvtok"],
                parked=before["parked"],
                p1q=before["p1q"],
                p1d=before["p1d"],
                ordn=-1 if ord_batch is None else len(ord_batch.reqs),
                full_ord=after_ord["full"],
                p1try=int(not skip_phase_one),
                slice=int(slicing),
                p1n=-1 if p1_batch is None else len(p1_batch.reqs),
                full_p1=final["full"],
            )
            if ord_batch is not None:
                self._kvstat_log(
                    "admit",
                    ct=self.forward_ct,
                    rids=",".join(req.rid for req in ord_batch.reqs),
                )
        return plan if p1_plan is None else p1_plan

    def _phase_one_prefill_plan(self, running_batch: Any) -> Any:
        """Admit prompt-only requests, and only those, into one extend batch.

        Their rows sample a token the runner must not ship, and the suppression
        is per batch rather than per row, so phase 1 gets its own queue and its
        own admission pass. It never runs on a step the ordinary pass claimed,
        so the waiting queue always outranks the prepay; only the decode
        interleave yields to it, and only for a budgeted slice.
        """
        queued = list(self._phase_one_queue)
        rows = self._two_phase_slice_rows
        candidates = queued[:rows] if rows > 0 else queued
        # Note (wenyao): rows held back never enter self.waiting_queue, so they
        # are absent from the leftover set the queue is rebuilt from and would
        # be dropped — with their pool row and prompt KV still allocated.
        held_back = {id(req) for req in queued[len(candidates) :]}
        saved_queue = self.waiting_queue
        # Note (wenyao): batch_is_full latches the ordinary pass's own "no room"
        # verdict and upstream clears it only when the running batch shrinks.
        # Upstream never evaluates it while the ordinary queue is empty, so a
        # phase-1 pass tripping it there would block ordinary admission for a
        # whole decode window on a verdict reached about a different queue.
        saved_full = running_batch.batch_is_full
        self.waiting_queue = candidates
        try:
            plan = _Upstream.get_new_batch_prefill(self, running_batch)
        finally:
            leftover = {id(req) for req in self.waiting_queue} | held_back
            self.waiting_queue = saved_queue
            running_batch.batch_is_full = saved_full
        plan.running_batch.batch_is_full = saved_full
        if plan.batch_to_run is None:
            return plan
        self._phase_one_queue = deque(
            req for req in self._phase_one_queue if id(req) in leftover
        )
        if self.prefill_decode_interleave:
            self._interleave_defer_prefill = True
        return plan

    def process_batch_result(self, batch: Any, result: Any) -> None:
        if self._two_phase_kv and batch is not None and batch.forward_mode.is_extend():
            if self._is_phase_one_batch(batch):
                self._park_phase_one_batch(batch)
                return
            for req in batch.reqs:
                uninstall_parked_prefix_shadow(req)
        _Upstream.process_batch_result(self, batch, result)

    @staticmethod
    def _is_phase_one_batch(batch: Any) -> bool:
        return bool(batch.reqs) and all(
            is_tail_pending(req._omni_data) for req in batch.reqs
        )

    def _park_phase_one_batch(self, batch: Any) -> None:
        """Keep the prompt KV and take these requests back off the scheduler.

        Emptying the batch is what stops the next ``get_next_batch_to_run`` from
        merging these rows into the running decode batch: with no assistant tail
        they have nothing to decode from yet.
        """
        for req in batch.reqs:
            prefix_indices = snapshot_prompt_kv(req, self.req_to_token_pool)
            req.prefix_indices = prefix_indices
            req.cache_protected_len = 0
            self._parked_reqs[req.rid] = req
        # Note (wenyao): the KV half degrades silently to the compute half when
        # the prompt segment is unusable, so its first success has to be
        # visible in the log or a full fallback run looks like a working one.
        if self._parked_total == 0:
            logger.info(
                "talker two-phase KV: first prompt extend parked (batch size %d)",
                len(batch.reqs),
            )
        self._parked_total += len(batch.reqs)
        if KVSTAT_ENABLED:
            self._kvstat_log(
                "p1park",
                ct=self.forward_ct,
                rids=",".join(req.rid for req in batch.reqs),
            )
        batch.reqs = []

    def _adopt_built_request(self, payload: Any, req_data: Any) -> None:
        if not self._two_phase_kv:
            return
        request_id = payload.request_id
        phase_one = self._phase_one_data.pop(request_id, None)
        self._phase_one_denied.discard(request_id)
        self._drop_queued_phase_one(request_id)
        parked = self._parked_reqs.pop(request_id, None)
        if KVSTAT_ENABLED:
            self._kvstat_log(
                "build", rid=request_id, engaged=int(parked is not None)
            )
        if parked is None:
            return
        fresh = req_data.req
        prompt_rows = len(parked.origin_input_ids)
        if (
            not getattr(req_data, "two_phase_composed", False)
            or len(fresh.origin_input_ids) != prompt_rows + ASSISTANT_TAIL_ROWS
        ):
            logger.warning(
                "talker two-phase KV: request %s built whole; reclaiming its "
                "parked prompt KV",
                request_id,
            )
            self._reclaim_parked(parked)
            return
        adopt_parked_prompt_kv(fresh, parked)
        self._replay_tail_pending_stream(phase_one, req_data)

    def _replay_tail_pending_stream(self, phase_one: Any, req_data: Any) -> None:
        if phase_one is None:
            return
        for chunk in getattr(phase_one, "tail_pending_chunks", None) or ():
            self._append_stream_chunk(req_data, chunk)
        if getattr(phase_one, "tail_pending_stream_done", False):
            self._mark_stream_done(req_data)

    def _stream_skip_rids(self, sched_output: Any) -> tuple[str, ...]:
        if not self._two_phase_kv:
            return ()
        return tuple(
            sched_req.request_id
            for sched_req in sched_output.requests
            if is_tail_pending(sched_req.data)
        )

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
        # Via OmniScheduler, which supplies running_batch/last_batch and unpacks
        # the 0.5.16 NextBatchPlan.
        batch = super().get_next_batch_to_run()
        if batch is not None and not self._is_batch_ready_to_run(batch):
            self._rollback_decode_prep_after_skip(batch)
            return None
        return batch

    def _rollback_decode_prep_after_skip(self, batch: Any) -> None:
        # Note(Chenchen Hong, Xuesong): This is talker-only. It does not fully
        # invert prepare_for_decode; talker disables overlap/spec/Mamba/hisparse,
        # and the penalizer's cumulate scatter_ is idempotent under the talker's
        # own SamplingBatchInfo. Zero the req_to_token_pool cell that
        # alloc_for_decode wrote at (req_pool_indices, pre-increment seq_lens);
        # seq_lens_sum stays untouched (always None after prepare_for_decode,
        # recomputed at the next forward).
        if not batch.forward_mode.is_decode():
            return
        if batch.out_cache_loc is not None:
            self.token_to_kv_pool_allocator.free(batch.out_cache_loc)
            batch.out_cache_loc = None
        for req in batch.reqs:
            req.decode_batch_idx -= 1
            req.kv_committed_len -= 1
            req.kv.kv_allocated_len -= 1
        batch.seq_lens.sub_(1)
        batch.seq_lens_cpu.sub_(1)
        batch.orig_seq_lens.sub_(1)
        batch.req_to_token_pool.req_to_token[batch.req_pool_indices, batch.seq_lens] = 0

    def self_check_during_idle(self) -> None:
        if self.running_batch is not None and not self.running_batch.is_empty():
            return
        if self.waiting_queue:
            return
        if self._parked_reqs or self._phase_one_queue:
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
