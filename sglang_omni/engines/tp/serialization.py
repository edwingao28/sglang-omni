# SPDX-License-Identifier: Apache-2.0
"""Serialization helpers for TP follower batch broadcast.

Rank 0 builds a ModelWorkerBatch and must pickle it over NCCL (via
broadcast_pyobj) to follower ranks.  Followers only need tensor data for
model.forward() — they never sample, manage requests, or run penalizers.

make_follower_batch() produces a shallow copy of the batch with all
non-picklable fields nulled out so the original (which rank 0 still needs)
is left untouched.

On the first call, the result is pickle-verified to catch any new
non-picklable fields that may have been added to ModelWorkerBatch.
"""
from __future__ import annotations

import copy
import logging
import pickle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch

logger = logging.getLogger(__name__)

# Fields that are known to be non-picklable and not needed by followers.
# launch_done is a threading.Event used by overlap scheduling — not needed
# by followers and not picklable.
_FIELDS_TO_STRIP = ("sampling_info", "reqs", "launch_done")

_pickle_verified = False


def make_follower_batch(model_worker_batch: "ModelWorkerBatch") -> "ModelWorkerBatch":
    """Return a pickle-safe shallow copy of *model_worker_batch* for TP followers.

    Fields stripped (set to None on the copy):
    - ``sampling_info``  — contains penalizer_orchestrator with weakrefs,
      sampling_info_done Event, and custom processor state.
    - ``reqs``           — contains request objects with callbacks/threads.
    - ``launch_done``    — threading.Event from overlap scheduling.

    All tensor fields (input_ids, out_cache_loc, seq_lens, …) are preserved
    via the shallow copy so followers can call ForwardBatch.init_new() and
    model.forward() without extra allocations.

    On the first call, the result is pickle-verified.  If a future sglang
    version adds a new non-picklable field to ModelWorkerBatch, this will
    fail fast with a clear error instead of silently breaking at broadcast.
    """
    global _pickle_verified

    follower = copy.copy(model_worker_batch)
    for field in _FIELDS_TO_STRIP:
        setattr(follower, field, None)

    if not _pickle_verified:
        _verify_pickle_safe(follower)
        _pickle_verified = True

    return follower


def _verify_pickle_safe(batch: "ModelWorkerBatch") -> None:
    """Try to pickle *batch*; raise RuntimeError with field-level diagnosis on failure."""
    try:
        pickle.dumps(batch)
    except Exception as exc:
        # Find which field is the culprit
        bad_fields = []
        for attr, val in vars(batch).items():
            if val is None:
                continue
            try:
                pickle.dumps(val)
            except Exception:
                bad_fields.append(attr)
        raise RuntimeError(
            f"TP follower batch is not pickle-safe after stripping "
            f"{_FIELDS_TO_STRIP}. Unpicklable fields: {bad_fields}. "
            f"Add them to _FIELDS_TO_STRIP in "
            f"sglang_omni/engines/tp/serialization.py. "
            f"Original error: {exc}"
        ) from exc
    logger.debug("Follower batch pickle verification passed")
