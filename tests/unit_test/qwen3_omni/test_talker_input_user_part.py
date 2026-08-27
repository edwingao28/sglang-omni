# SPDX-License-Identifier: Apache-2.0
"""Exact-equivalence tests for build_user_part's sync-free form.

The bar is EXACT equality, not tolerance: neither change touches the math.
Dropping the `mask.any()` guards only removes a device sync, and reading
out_features off the projection only skips a shape probe. If any case here
comes back close-but-unequal, the change is wrong -- not the assertion.
"""
from __future__ import annotations

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.talker_input import build_user_part


def _reference(thinker_embed, thinker_hidden, multimodal_mask, text_projection,
               hidden_projection):
    """The pre-change implementation, verbatim."""
    out_size = text_projection(thinker_embed[:1]).shape[-1]
    result = torch.empty(
        (thinker_embed.shape[0], out_size),
        device=thinker_embed.device,
        dtype=thinker_embed.dtype,
    )
    if multimodal_mask.any():
        result[multimodal_mask] = hidden_projection(thinker_hidden[multimodal_mask])
    text_mask = ~multimodal_mask
    if text_mask.any():
        result[text_mask] = text_projection(thinker_embed[text_mask])
    return result


def _case(n_rows, mask_kind, in_dim=16, hid_dim=24, out_dim=12, seed=0):
    torch.manual_seed(seed)
    embed = torch.randn(n_rows, in_dim)
    hidden = torch.randn(n_rows, hid_dim)
    if mask_kind == "none":
        mask = torch.zeros(n_rows, dtype=torch.bool)
    elif mask_kind == "all":
        mask = torch.ones(n_rows, dtype=torch.bool)
    else:
        mask = torch.arange(n_rows) % 3 == 0
    text_proj = torch.nn.Linear(in_dim, out_dim)
    hid_proj = torch.nn.Linear(hid_dim, out_dim)
    return embed, hidden, mask, text_proj, hid_proj


@pytest.mark.parametrize("n_rows", [1, 2, 7, 64, 257])
@pytest.mark.parametrize("mask_kind", ["none", "all", "mixed"])
def test_matches_the_previous_implementation_exactly(n_rows, mask_kind):
    embed, hidden, mask, tp, hp = _case(n_rows, mask_kind, seed=n_rows)
    want = _reference(embed, hidden, mask, tp, hp)
    got = build_user_part(
        thinker_embed=embed, thinker_hidden=hidden, multimodal_mask=mask,
        text_projection=tp, hidden_projection=hp,
    )
    assert got.shape == want.shape
    assert got.dtype == want.dtype
    # Rows the mask never selects are uninitialised in BOTH implementations, so
    # compare only the rows either branch actually writes -- which, together,
    # is every row.
    assert torch.equal(got, want)


def test_all_false_mask_still_writes_every_row():
    """The empty-multimodal branch is the one whose guard was load-bearing."""
    embed, hidden, mask, tp, hp = _case(9, "none", seed=1)
    got = build_user_part(
        thinker_embed=embed, thinker_hidden=hidden, multimodal_mask=mask,
        text_projection=tp, hidden_projection=hp,
    )
    assert torch.equal(got, tp(embed))


def test_all_true_mask_still_writes_every_row():
    embed, hidden, mask, tp, hp = _case(9, "all", seed=2)
    got = build_user_part(
        thinker_embed=embed, thinker_hidden=hidden, multimodal_mask=mask,
        text_projection=tp, hidden_projection=hp,
    )
    assert torch.equal(got, hp(hidden))


def test_zero_rows_does_not_raise():
    embed, hidden, mask, tp, hp = _case(0, "none", seed=3)
    got = build_user_part(
        thinker_embed=embed, thinker_hidden=hidden, multimodal_mask=mask,
        text_projection=tp, hidden_projection=hp,
    )
    assert got.shape == (0, tp.out_features)


def test_projection_without_out_features_falls_back_to_the_probe():
    """A non-Linear projection must still work, via the retained probe."""
    embed, hidden, mask, tp, hp = _case(5, "mixed", seed=4)
    wrapped = lambda x: tp(x)  # noqa: E731 - no out_features attribute
    want = _reference(embed, hidden, mask, wrapped, hp)
    got = build_user_part(
        thinker_embed=embed, thinker_hidden=hidden, multimodal_mask=mask,
        text_projection=wrapped, hidden_projection=hp,
    )
    assert torch.equal(got, want)
