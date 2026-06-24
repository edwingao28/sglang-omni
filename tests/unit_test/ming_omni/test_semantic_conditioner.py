# SPDX-License-Identifier: Apache-2.0
"""SemanticConditioner.project() unit tests.

These tests construct a real :class:`SemanticConditioner` and inject CPU
fakes onto its private attributes, bypassing ``load()`` entirely so no Ming
model files, GPU, or network are required. The fake connector is an identity
module (returns its ``inputs_embeds`` as the last hidden state) so the
projection chain is exercised end-to-end on CPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from sglang_omni.models.ming_omni.diffusion.semantic_conditioner import (  # noqa: E402
    SemanticConditioner,
)


class _IdentityConnector:
    """Stands in for the Qwen2 connector; returns inputs_embeds unchanged."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, inputs_embeds, attention_mask, output_hidden_states):
        self.calls.append(
            {
                "n": inputs_embeds.shape[1],
                "attention_mask_shape": tuple(attention_mask.shape),
                "output_hidden_states": output_hidden_states,
            }
        )
        return SimpleNamespace(hidden_states=[inputs_embeds])


def _make_conditioner(*, scales, scale_indices, proj_mid=1536):
    """Build a SemanticConditioner with CPU/float32 fakes (no load())."""
    cond = SemanticConditioner()
    cond._device = "cpu"
    cond._dtype = torch.float32
    cond._proj_in = nn.Linear(4096, proj_mid)
    cond._proj_out = nn.Linear(proj_mid, 2560)
    connector = _IdentityConnector()
    cond._connector = connector
    cond._img_gen_scales = scales
    cond._scale_indices = scale_indices
    return cond, connector


def test_project_output_shape_single_scale() -> None:
    # scales=[16] -> scale_indices=[256]; last scale slices h[:, 0:256].
    cond, _ = _make_conditioner(scales=[16], scale_indices=[256])
    hidden = torch.randn(2, 256, 4096)

    out = cond.project(hidden)

    assert out.shape == (2, 256, 2560)


def test_project_output_is_l2_normalized() -> None:
    cond, _ = _make_conditioner(scales=[16], scale_indices=[256])
    hidden = torch.randn(2, 256, 4096)

    out = cond.project(hidden)

    norms = torch.linalg.norm(out, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)


def test_project_multi_scale_slices_last_scale() -> None:
    # scales=[16, 32] -> scale_indices=[256, 1280]; last scale is rows
    # [256:1280], i.e. 1024 query tokens.
    cond, connector = _make_conditioner(scales=[16, 32], scale_indices=[256, 1280])
    hidden = torch.randn(3, 1280, 4096)

    out = cond.project(hidden)

    # Output row count equals the LAST scale's length (1024), not 256 or 1280.
    assert out.shape == (3, 1024, 2560)
    # The connector received exactly the last scale's slice.
    assert connector.calls[0]["n"] == 1024


def test_project_last_scale_slice_matches_manual_slice() -> None:
    # Lock the exact h[:, start:end] selection for the final scale.
    cond, _ = _make_conditioner(scales=[16, 32], scale_indices=[256, 1280])
    hidden = torch.randn(1, 1280, 4096)

    out = cond.project(hidden)

    # Recompute the expected projection chain on the sliced rows only.
    sliced = hidden[:, 256:1280, :]
    expected = cond._proj_in(sliced)
    expected = cond._proj_out(expected)
    expected = torch.nn.functional.normalize(expected, dim=-1)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_project_non_3d_input_raises_value_error() -> None:
    cond, _ = _make_conditioner(scales=[16], scale_indices=[256])
    hidden = torch.randn(256, 4096)  # 2D, missing batch dim

    with pytest.raises(ValueError):
        cond.project(hidden)


def test_project_not_loaded_raises_runtime_error() -> None:
    cond = SemanticConditioner()  # _connector is None
    assert cond._connector is None
    hidden = torch.randn(1, 256, 4096)

    with pytest.raises(RuntimeError):
        cond.project(hidden)
