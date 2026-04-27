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


class _TuplePlaceholder:
    """Mutable stand-in used while recursively copying tuple cycles."""

    __slots__ = ("items", "value", "resolving")

    def __init__(self) -> None:
        self.items: list[Any] = []
        self.value: tuple[Any, ...] | None = None
        self.resolving = False


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
        placeholder = _TuplePlaceholder()
        memo[obj_id] = placeholder
        for value in obj:
            placeholder.items.append(_to_cpu_copy(value, memo))
        return placeholder

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


def _resolve_tuple_placeholders(obj: Any, seen: set[int] | None = None) -> Any:
    """Finalize copied tuple placeholders and update mutable containers."""
    if seen is None:
        seen = set()

    if isinstance(obj, _TuplePlaceholder):
        if obj.value is not None:
            return obj.value
        if obj.resolving:
            return obj

        obj.resolving = True
        try:
            obj.value = tuple(
                _resolve_tuple_placeholders(item, seen) for item in obj.items
            )
        finally:
            obj.resolving = False

        for item in obj.value:
            _replace_tuple_placeholder_refs(item, obj, obj.value, seen=set())
        return obj.value

    obj_id = id(obj)
    if obj_id in seen:
        return obj

    if isinstance(obj, list):
        seen.add(obj_id)
        for i, value in enumerate(obj):
            obj[i] = _resolve_tuple_placeholders(value, seen)
        return obj

    if isinstance(obj, dict):
        seen.add(obj_id)
        for key, value in list(obj.items()):
            new_key = _resolve_tuple_placeholders(key, seen)
            new_value = _resolve_tuple_placeholders(value, seen)
            if new_key is key:
                obj[key] = new_value
            else:
                del obj[key]
                obj[new_key] = new_value
        return obj

    if isinstance(obj, tuple):
        seen.add(obj_id)
        for value in obj:
            _resolve_tuple_placeholders(value, seen)
        return obj

    if hasattr(obj, "__dict__"):
        seen.add(obj_id)
        for attr, value in vars(obj).items():
            setattr(obj, attr, _resolve_tuple_placeholders(value, seen))
        return obj

    return obj


def _replace_tuple_placeholder_refs(
    obj: Any,
    target: _TuplePlaceholder,
    replacement: tuple[Any, ...],
    seen: set[int],
) -> None:
    """Replace in-progress tuple backrefs inside mutable descendants."""
    if obj is target:
        return

    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if isinstance(obj, list):
        for i, value in enumerate(obj):
            if value is target:
                obj[i] = replacement
            else:
                _replace_tuple_placeholder_refs(value, target, replacement, seen)
        return

    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            current_key = key
            if key is target:
                del obj[key]
                obj[replacement] = value
                current_key = replacement
            else:
                _replace_tuple_placeholder_refs(key, target, replacement, seen)

            current_value = obj[current_key]
            if current_value is target:
                obj[current_key] = replacement
            else:
                _replace_tuple_placeholder_refs(
                    current_value, target, replacement, seen
                )
        return

    if isinstance(obj, tuple):
        for value in obj:
            _replace_tuple_placeholder_refs(value, target, replacement, seen)
        return

    if hasattr(obj, "__dict__"):
        for attr, value in vars(obj).items():
            if value is target:
                setattr(obj, attr, replacement)
            else:
                _replace_tuple_placeholder_refs(value, target, replacement, seen)


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

    sanitized = _resolve_tuple_placeholders(_to_cpu_copy(payload, memo={}))
    if not _pickle_verified:
        _verify_pickle_safe(sanitized)
        _pickle_verified = True
    return sanitized
