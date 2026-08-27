# SPDX-License-Identifier: Apache-2.0
"""Equivalence tests for the single-token embedding fast path.

The fast path skips torch.unique, the CPU index build and two host-to-device
transfers. It must return exactly what the batched gather returns, must not
hand out a view into the table, and must populate/extend the table on a
first-touch token exactly as the batched path would -- that last one is the
place a fast path can silently fork cache state.
"""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from sglang_omni.models.qwen3_omni.components import talker_prefill
from sglang_omni.models.qwen3_omni.components.talker_prefill import TalkerPrefillBuilder

VOCAB, HIDDEN = 128, 16


@pytest.fixture()
def model_dir(tmp_path):
    weight = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
    shard = "model-00001-of-00001.safetensors"
    save_file({"thinker.model.embed_tokens.weight": weight}, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"thinker.model.embed_tokens.weight": shard}})
    )
    talker_prefill._EMBED_SOURCE_CACHE.clear()
    talker_prefill._EMBED_HANDLE_CACHE.clear()
    try:
        yield tmp_path
    finally:
        talker_prefill._EMBED_SOURCE_CACHE.clear()
        talker_prefill._EMBED_HANDLE_CACHE.clear()


def _builder(model_dir):
    b = object.__new__(TalkerPrefillBuilder)
    b._model_path = str(model_dir)
    b._device = torch.device("cpu")
    b._dtype = torch.float32
    b._embed_slots = {}
    b._embed_table = None
    b._embed_table_used = 0
    return b


@pytest.mark.parametrize("token_id", [0, 1, 7, 63, 127])
def test_fast_path_matches_the_batched_gather_exactly(model_dir, token_id):
    fast = _builder(model_dir)
    slow = _builder(model_dir)
    t = torch.tensor([token_id], dtype=torch.long)
    got = fast._load_prompt_token_embeddings(t)
    want = slow._load_prompt_token_embeddings_batch([t])[0]
    assert got.shape == want.shape == (1, HIDDEN)
    assert torch.equal(got, want)


def test_fast_path_is_not_a_view_into_the_table(model_dir):
    """A bare slice would alias the cache; callers mutate prompt embeds."""
    b = _builder(model_dir)
    t = torch.tensor([5], dtype=torch.long)
    row = b._load_prompt_token_embeddings(t)
    row[0] = -999.0
    again = b._load_prompt_token_embeddings(t)
    assert not torch.equal(again, torch.full((1, HIDDEN), -999.0))
    assert torch.equal(again, b._load_prompt_token_embeddings_batch([t])[0])


def test_first_touch_of_a_new_token_populates_the_table(model_dir):
    """The cache-growth edge: the fast path must extend state, not fork it."""
    b = _builder(model_dir)
    assert b._embed_table is None and not b._embed_slots
    row = b._load_prompt_token_embeddings(torch.tensor([42], dtype=torch.long))
    assert 42 in b._embed_slots, "fast path did not register the new token"
    assert b._embed_table is not None and b._embed_table_used == 1
    # and the batched path must now agree with, and reuse, that same slot
    batched = b._load_prompt_token_embeddings_batch(
        [torch.tensor([42, 42], dtype=torch.long)]
    )[0]
    assert torch.equal(batched[0], row[0]) and torch.equal(batched[1], row[0])
    assert b._embed_table_used == 1, "batched path re-added an existing token"


def test_fast_and_batched_paths_share_one_cache(model_dir):
    """Interleaving the two must never produce two different rows for a token."""
    b = _builder(model_dir)
    for token_id in (3, 9, 3, 70, 9, 70):
        fast = b._load_prompt_token_embeddings(
            torch.tensor([token_id], dtype=torch.long)
        )
        batched = b._load_prompt_token_embeddings_batch(
            [torch.tensor([token_id], dtype=torch.long)]
        )[0]
        assert torch.equal(fast, batched), token_id
    # {3, 9, 70} = three distinct tokens, each cached exactly once
    assert b._embed_table_used == 3, b._embed_table_used


def test_multi_token_still_takes_the_batched_path(model_dir):
    b = _builder(model_dir)
    t = torch.tensor([1, 2, 1], dtype=torch.long)
    got = b._load_prompt_token_embeddings(t)
    want = b._load_prompt_token_embeddings_batch([t])[0]
    assert got.shape == (3, HIDDEN)
    assert torch.equal(got, want)


def test_fast_path_accepts_shaped_single_token_tensors(model_dir):
    """numel()==1 covers [[7]] and scalar-ish shapes, not just [7]."""
    b = _builder(model_dir)
    flat = b._load_prompt_token_embeddings(torch.tensor([7], dtype=torch.long))
    for shaped in (torch.tensor([[7]], dtype=torch.long),
                   torch.tensor(7, dtype=torch.long).reshape(1, 1)):
        assert torch.equal(b._load_prompt_token_embeddings(shaped), flat)
