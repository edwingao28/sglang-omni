# SPDX-License-Identifier: Apache-2.0
"""Parking a talker request between its prompt extend and its assistant tail."""

from __future__ import annotations

from typing import Any

import torch

# Note (wenyao): 9 is HF's assistant layout, not a tunable — 3 chat-header rows,
# 4 tts pads, tts bos, and the first spoken token (see build_assistant_part).
ASSISTANT_TAIL_ROWS = 9

_SHADOW_ATTR = "init_next_round_input"


def is_tail_pending(req_data: Any) -> bool:
    return bool(getattr(req_data, "tail_pending", False))


def snapshot_prompt_kv(req: Any, req_to_token_pool: Any) -> torch.Tensor:
    """The KV slots the prompt extend just wrote, as a prefix for the tail extend.

    Same slice ``ChunkCache.cache_unfinished_req`` takes. It is taken here
    rather than through the chunked-prefill path because upstream holds exactly
    one ``chunked_req`` per scheduler, which would serialize the stage to a
    single two-phase request.
    """
    row = req_to_token_pool.req_to_token[req.req_pool_idx, : req.extend_range.end]
    return row.to(dtype=torch.int64, copy=True)


def install_parked_prefix_shadow(req: Any, prefix_indices: torch.Tensor) -> None:
    """Per-instance override of ``Req.init_next_round_input`` for a parked req.

    The scheduler re-derives the prefix through ``tree_cache.match_prefix`` for
    every request it takes off the waiting queue, and the talker's ChunkCache
    always answers "no match" — which would strand the parked prompt KV and
    allocate the whole sequence a second time. Reinstate the parked range
    instead, with ``cache_protected_len`` at zero so ``cache_finished_req``
    still frees that range on finish or abort.
    """

    def init_next_round_input(tree_cache: Any = None, cow_mamba: Any = None) -> None:
        del tree_cache, cow_mamba
        req._refresh_fill_ids()
        req.prefix_indices = prefix_indices
        req.cache_protected_len = 0
        req.last_node = None
        req.last_host_node = None
        req.best_match_node = None
        req.host_hit_length = 0
        req.swa_host_hit_length = 0

    req.__dict__[_SHADOW_ATTR] = init_next_round_input


def uninstall_parked_prefix_shadow(req: Any) -> None:
    req.__dict__.pop(_SHADOW_ATTR, None)


def has_parked_prefix_shadow(req: Any) -> bool:
    return _SHADOW_ATTR in req.__dict__


def adopt_parked_prompt_kv(fresh: Any, parked: Any) -> None:
    """Move the parked prompt KV onto the request the whole build produced.

    Moving ownership this direction — rather than grafting the tail onto the
    parked ``Req`` — leaves every sampling, multimodal and token field exactly
    what the single-phase build produces; only the pool row and the prefix
    travel, so the tail extend is the whole build's own last 9 rows.
    """
    fresh.req_pool_idx = parked.req_pool_idx
    fresh.kv = parked.kv
    fresh.kv_committed_len = parked.kv_committed_len
    fresh.extend_batch_idx = parked.extend_batch_idx
    fresh.already_computed = parked.kv_committed_len
    install_parked_prefix_shadow(fresh, parked.prefix_indices)
    parked.req_pool_idx = None
    parked.kv = None
