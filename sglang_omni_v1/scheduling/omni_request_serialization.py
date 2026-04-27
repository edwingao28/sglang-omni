# SPDX-License-Identifier: Apache-2.0
"""Sanitize request payloads for cross-rank broadcast.

This module is adapted from the legacy
``sglang_omni/engines/tp/serialization.py`` batch serialization helper. V1
broadcasts request-level payloads before scheduler batch construction, so this
keeps the recursive CPU-copy and pickle-safety check while avoiding
ModelWorkerBatch-specific stripping or page-table snapshot APIs.
"""
from __future__ import annotations

import copy
import logging
import pickle
from enum import Enum
from typing import Any

import torch

logger = logging.getLogger(__name__)

_pickle_verified = False


def _to_cpu_copy(obj: Any, memo: dict[int, Any]) -> Any:
    """Return a non-mutating CPU-safe copy of request payload data."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu() if obj.device.type != "cpu" else obj
    if isinstance(obj, Enum) or isinstance(obj, type):
        return obj
    if obj is None or isinstance(obj, (int, float, bool, str, bytes)):
        return obj

    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]

    if isinstance(obj, list):
        new_list: list[Any] = [None] * len(obj)
        memo[obj_id] = new_list
        for i, value in enumerate(obj):
            new_list[i] = _to_cpu_copy(value, memo)
        return new_list

    if isinstance(obj, tuple):
        placeholder: list[Any] = []
        memo[obj_id] = placeholder
        for value in obj:
            placeholder.append(_to_cpu_copy(value, memo))
        new_tuple = tuple(placeholder)
        memo[obj_id] = new_tuple
        return new_tuple

    if isinstance(obj, dict):
        new_dict: dict[Any, Any] = {}
        memo[obj_id] = new_dict
        for key, value in obj.items():
            new_key = _to_cpu_copy(key, memo)
            new_dict[new_key] = _to_cpu_copy(value, memo)
        return new_dict

    if hasattr(obj, "__dict__"):
        cloned = copy.copy(obj)
        memo[obj_id] = cloned
        for attr, value in vars(obj).items():
            setattr(cloned, attr, _to_cpu_copy(value, memo))
        return cloned

    return obj


def _verify_pickle_safe(payload: Any) -> None:
    """Raise a diagnostic error if *payload* cannot be pickled."""
    try:
        pickle.dumps(payload)
    except Exception as exc:
        bad_fields: list[str] = []
        try:
            items = vars(payload).items()
        except TypeError:
            items = ()

        for attr, value in items:
            try:
                pickle.dumps(value)
            except Exception:
                bad_fields.append(attr)

        raise RuntimeError(
            "Request payload is not pickle-safe after sanitization in "
            "sglang_omni_v1/scheduling/omni_request_serialization.py. "
            f"Unpicklable top-level fields: {bad_fields}. "
            f"Original error: {exc}"
        ) from exc

    logger.debug("Request payload pickle verification passed")


def sanitize_request_payload(payload: Any) -> Any:
    """Return a pickle-safe, CPU-backed copy of a request broadcast payload."""
    global _pickle_verified

    sanitized = _to_cpu_copy(payload, memo={})
    if not _pickle_verified:
        _verify_pickle_safe(sanitized)
        _pickle_verified = True
    return sanitized
