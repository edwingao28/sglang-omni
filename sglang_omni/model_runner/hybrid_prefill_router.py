# SPDX-License-Identifier: Apache-2.0
"""Shape-scoped hybrid prefill CUDA graphs: BREAKABLE small, FULL large."""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)


class HybridPrefillGraphRouter:
    """Route prefill graph replay by batch shape across two runners.

    The production BREAKABLE runner keeps every bucket it always had; a
    second FULL-backend runner covers the large token buckets that would
    otherwise run eager (cold-burst wave batches). Routing defers to each
    runner's own ``can_run_graph``, breakable first, so the ladders decide
    and no threshold is duplicated here.
    """

    def __init__(self, breakable_runner: Any, full_runner: Any) -> None:
        self.breakable_runner = breakable_runner
        self.full_runner = full_runner
        # ModelWorker's replay accounting bisects this ladder.
        self.capture_num_tokens = sorted(
            [*breakable_runner.capture_num_tokens, *full_runner.capture_num_tokens]
        )

    def _select(self, forward_batch: Any) -> Any:
        if self.breakable_runner.can_run_graph(forward_batch):
            return self.breakable_runner
        if self.full_runner.can_run_graph(forward_batch):
            return self.full_runner
        return None

    def can_run_graph(self, forward_batch: Any) -> bool:
        return self._select(forward_batch) is not None

    def execute(self, forward_batch: Any, **kwargs: Any) -> Any:
        runner = self._select(forward_batch)
        assert runner is not None, "execute() without a passing can_run_graph()"
        return runner.execute(forward_batch, **kwargs)


def install_hybrid_full_prefill(
    model_worker: Any, full_bs: Sequence[int], server_args: Any
) -> None:
    """Capture the FULL big-bucket runner and install the hybrid router.

    Runs after ``attest_prefill_cuda_graphs`` validated the production
    BREAKABLE runner. PrefillCudaGraphRunner snapshots its whole policy from
    ``server_args.cuda_graph_config.prefill`` at ``__init__``, so the FULL
    runner is constructed under a temporary view of that config (backend,
    buckets, request slots, multimodal flag) and every field is restored:
    the rest of the process keeps seeing the production BREAKABLE config.
    """
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )

    model_runner = model_worker.model_runner
    breakable_runner = model_runner.prefill_cuda_graph_runner
    if not isinstance(breakable_runner, PrefillCudaGraphRunner):
        raise RuntimeError(
            "hybrid prefill requires the BREAKABLE runner to have captured; "
            f"got {type(breakable_runner).__name__}"
        )
    full_bs = sorted(int(b) for b in full_bs)
    if full_bs[0] <= breakable_runner.max_num_tokens:
        raise ValueError(
            "hybrid FULL buckets must exceed the breakable ladder: "
            f"{full_bs[0]} <= {breakable_runner.max_num_tokens}"
        )

    prefill_cfg = server_args.cuda_graph_config.prefill
    model_config = model_runner.model_config
    saved = (prefill_cfg.backend, prefill_cfg.bs, prefill_cfg.full_prefill_max_req)
    saved_is_multimodal = model_config.is_multimodal

    max_req = prefill_cfg.full_prefill_max_req
    if max_req is None:
        chunked = server_args.chunked_prefill_size
        max_req = max(chunked // 512, 1) if chunked and chunked > 0 else 1
    max_req = min(int(max_req), model_runner.req_to_token_pool.size)

    prefill_cfg.backend = Backend.FULL
    prefill_cfg.bs = full_bs
    prefill_cfg.full_prefill_max_req = max_req
    if model_worker.enable_prefill_input_embeds:
        model_config.is_multimodal = True
    try:
        full_runner = PrefillCudaGraphRunner(model_runner)
    finally:
        model_config.is_multimodal = saved_is_multimodal
        (
            prefill_cfg.backend,
            prefill_cfg.bs,
            prefill_cfg.full_prefill_max_req,
        ) = saved

    if list(full_runner.capture_num_tokens) != full_bs:
        raise RuntimeError(
            "hybrid FULL capture shapes differ from the declared ladder: "
            f"declared={full_bs}, captured={list(full_runner.capture_num_tokens)}"
        )
    model_runner.prefill_cuda_graph_runner = HybridPrefillGraphRouter(
        breakable_runner, full_runner
    )
    logger.info(
        "hybrid prefill CUDA graphs installed: breakable buckets=%s, "
        "full buckets=%s (req slots=%d)",
        list(breakable_runner.capture_num_tokens),
        list(full_runner.capture_num_tokens),
        max_req,
    )
