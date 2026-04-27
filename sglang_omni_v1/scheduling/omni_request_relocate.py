# SPDX-License-Identifier: Apache-2.0
"""Relocate request-payload tensors to a target device.

Adapted from legacy `sglang_omni/engines/tp/follower.py:relocate_batch_tensors`.
Receiver-side counterpart to omni_request_serialization.
"""
from __future__ import annotations

from typing import Any

import torch


def _move_request_tensors(
    obj: Any,
    device: torch.device,
    memo: dict[int, Any],
) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True) if obj.device != device else obj

    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj

    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]

    if isinstance(obj, dict):
        memo[obj_id] = obj
        for key, value in list(obj.items()):
            obj[key] = _move_request_tensors(value, device, memo)
        return obj

    if isinstance(obj, list):
        memo[obj_id] = obj
        for i, value in enumerate(obj):
            obj[i] = _move_request_tensors(value, device, memo)
        return obj

    if isinstance(obj, tuple):
        memo[obj_id] = obj
        new_tuple = tuple(_move_request_tensors(value, device, memo) for value in obj)
        memo[obj_id] = new_tuple
        for value in new_tuple:
            _replace_object_refs(value, obj, new_tuple, seen=set())
        return new_tuple

    if hasattr(obj, "__dict__"):
        memo[obj_id] = obj
        for attr, value in vars(obj).items():
            moved = _move_request_tensors(value, device, memo)
            if moved is not value:
                setattr(obj, attr, moved)
        return obj

    return obj


def _replace_object_refs(
    obj: Any,
    target: Any,
    replacement: Any,
    seen: set[int],
) -> None:
    """Replace object identity backrefs inside mutable descendants."""
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
                _replace_object_refs(value, target, replacement, seen)
        return

    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            current_key = key
            if key is target:
                del obj[key]
                obj[replacement] = value
                current_key = replacement
            else:
                _replace_object_refs(key, target, replacement, seen)

            current_value = obj[current_key]
            if current_value is target:
                obj[current_key] = replacement
            else:
                _replace_object_refs(current_value, target, replacement, seen)
        return

    if isinstance(obj, tuple):
        for value in obj:
            _replace_object_refs(value, target, replacement, seen)
        return

    if hasattr(obj, "__dict__"):
        for attr, value in vars(obj).items():
            if value is target:
                setattr(obj, attr, replacement)
            else:
                _replace_object_refs(value, target, replacement, seen)


def relocate_request_tensors(
    obj: Any,
    device: torch.device,
    _seen: set[int] | None = None,
) -> None:
    """Walk *obj* in-place, moving each tensor to *device*. Cycle-safe."""
    memo: dict[int, Any] = {}
    _move_request_tensors(obj, device, memo)
    if _seen is not None:
        _seen.update(memo)
