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
    seen: set[int],
) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True) if obj.device != device else obj

    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj

    if isinstance(obj, tuple):
        return tuple(_move_request_tensors(value, device, seen) for value in obj)

    obj_id = id(obj)
    if obj_id in seen:
        return obj
    seen.add(obj_id)

    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _move_request_tensors(value, device, seen)
        return obj

    if isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = _move_request_tensors(value, device, seen)
        return obj

    if hasattr(obj, "__dict__"):
        for attr, value in vars(obj).items():
            moved = _move_request_tensors(value, device, seen)
            if moved is not value:
                setattr(obj, attr, moved)
        return obj

    return obj


def relocate_request_tensors(
    obj: Any,
    device: torch.device,
    _seen: set[int] | None = None,
) -> None:
    """Walk *obj* in-place, moving each tensor to *device*. Cycle-safe."""
    _move_request_tensors(obj, device, _seen if _seen is not None else set())
