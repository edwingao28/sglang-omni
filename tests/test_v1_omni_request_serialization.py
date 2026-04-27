# SPDX-License-Identifier: Apache-2.0
"""Request-payload sanitization: pickle-safe, tensor-preserving, non-mutating."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from types import SimpleNamespace

import torch


@dataclass
class PicklePayload:
    image_embeds: torch.Tensor
    metadata: dict


@dataclass
class TensorPayload:
    embeds: torch.Tensor


@dataclass
class InnerPayload:
    t: torch.Tensor


@dataclass
class OuterPayload:
    inner: InnerPayload
    ids: list[int]


@dataclass
class TupleCyclePayload:
    items: tuple


def _assert_no_tuple_placeholder(obj, seen=None):
    if seen is None:
        seen = set()

    assert type(obj).__name__ != "_TuplePlaceholder"

    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if isinstance(obj, dict):
        for key, value in obj.items():
            _assert_no_tuple_placeholder(key, seen)
            _assert_no_tuple_placeholder(value, seen)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _assert_no_tuple_placeholder(value, seen)
    elif hasattr(obj, "__dict__"):
        for value in vars(obj).values():
            _assert_no_tuple_placeholder(value, seen)


def test_sanitize_payload_is_pickle_safe():
    from sglang_omni_v1.scheduling.omni_request_serialization import (
        sanitize_request_payload,
    )

    p = PicklePayload(
        image_embeds=torch.randn(4, 16, device="cpu"),
        metadata={"foo": [1, 2, 3]},
    )
    sanitized = sanitize_request_payload(p)
    pickle.dumps(sanitized)


def test_sanitize_payload_preserves_tensor_values():
    from sglang_omni_v1.scheduling.omni_request_serialization import (
        sanitize_request_payload,
    )

    src = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = sanitize_request_payload(TensorPayload(embeds=src))
    assert torch.equal(out.embeds, src)


def test_sanitize_payload_does_not_mutate_original():
    from sglang_omni_v1.scheduling.omni_request_serialization import (
        sanitize_request_payload,
    )

    src = torch.randn(8, device="cpu")
    p = TensorPayload(embeds=src)
    _ = sanitize_request_payload(p)
    assert p.embeds is src


def test_sanitize_payload_handles_cycle_via_memo():
    from sglang_omni_v1.scheduling.omni_request_serialization import (
        sanitize_request_payload,
    )

    p: dict = {}
    p["self"] = p
    out = sanitize_request_payload(p)
    assert out["self"] is out


def test_sanitize_payload_handles_tuple_cycle_via_memo_placeholder():
    from sglang_omni_v1.scheduling.omni_request_serialization import (
        sanitize_request_payload,
    )

    holder = []
    payload = TupleCyclePayload(items=(holder,))
    holder.append(payload.items)

    out = sanitize_request_payload(payload)

    assert isinstance(out.items, tuple)
    assert out.items is not payload.items
    assert out.items[0] is not holder
    assert isinstance(out.items[0], list)
    assert out.items[0][0] is out.items


def test_sanitize_payload_resolves_nested_tuple_cycle_placeholders():
    from sglang_omni_v1.scheduling.omni_request_serialization import (
        sanitize_request_payload,
    )

    holder = []
    a = (holder,)
    b = (a,)
    holder.append(b)

    out = sanitize_request_payload(b)

    _assert_no_tuple_placeholder(out)
    assert isinstance(out, tuple)
    assert isinstance(out[0], tuple)
    assert isinstance(out[0][0], list)
    assert out[0][0][0] is out


def test_sanitize_payload_traverses_nested_dataclass_attrs():
    from sglang_omni_v1.scheduling.omni_request_serialization import (
        sanitize_request_payload,
    )

    inner = InnerPayload(t=torch.randn(3))
    outer = OuterPayload(inner=inner, ids=[1, 2, 3])
    out = sanitize_request_payload(outer)
    assert torch.equal(out.inner.t, inner.t)
    assert out.inner is not inner


def test_sanitize_preserves_media_cache_keys():
    from sglang_omni_v1.scheduling.omni_request_serialization import (
        sanitize_request_payload,
    )

    payload = SimpleNamespace(
        request_id="r0",
        thinker_inputs={
            "media_cache_keys": {"image": "abc123", "audio": "def456"},
            "input_ids": [1, 2, 3],
        },
    )
    sanitized = sanitize_request_payload(payload)
    assert sanitized.thinker_inputs["media_cache_keys"] == {
        "image": "abc123",
        "audio": "def456",
    }
    assert payload.thinker_inputs["media_cache_keys"] == {
        "image": "abc123",
        "audio": "def456",
    }
