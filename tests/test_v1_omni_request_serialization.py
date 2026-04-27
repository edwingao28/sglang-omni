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
