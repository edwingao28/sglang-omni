# SPDX-License-Identifier: Apache-2.0
"""Exact-equivalence tests for the pooled token gather and modality mask.

These cover the layer OUTSIDE build_prefill_input: the embedding gather (21.5%
of build time by profile) now reads from a dense table via one index_select
instead of stacking N individual device tensors, and it serves a whole batch
from one torch.unique. Both are pure data movement, so the bar is torch.equal
against the pre-change implementation, reproduced verbatim below.
"""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from sglang_omni.models.qwen3_omni.components import talker_prefill
from sglang_omni.models.qwen3_omni.components.talker_prefill import (
    TalkerPrefillBuilder,
    load_thinker_embedding_rows,
)

VOCAB, HIDDEN = 128, 16
AUDIO, IMAGE, VIDEO = 5, 6, 7


@pytest.fixture()
def model_dir(tmp_path):
    weight = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
    shard = "model-00001-of-00001.safetensors"
    save_file({"thinker.model.embed_tokens.weight": weight}, str(tmp_path / shard))
    index = {"weight_map": {"thinker.model.embed_tokens.weight": shard}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    talker_prefill._EMBED_SOURCE_CACHE.clear()
    talker_prefill._EMBED_HANDLE_CACHE.clear()
    try:
        yield tmp_path
    finally:
        talker_prefill._EMBED_SOURCE_CACHE.clear()
        talker_prefill._EMBED_HANDLE_CACHE.clear()


def _builder(model_dir):
    """A builder with only the state these two methods touch."""
    builder = object.__new__(TalkerPrefillBuilder)
    builder._model_path = str(model_dir)
    builder._device = torch.device("cpu")
    builder._dtype = torch.float32
    builder._embed_slots = {}
    builder._embed_table = None
    builder._embed_table_used = 0
    builder._audio_token_id = AUDIO
    builder._image_token_id = IMAGE
    builder._video_token_id = VIDEO
    return builder


def _reference_gather(model_path, token_ids, cache):
    """The pre-change _load_prompt_token_embeddings, verbatim."""
    token_ids = token_ids.to(dtype=torch.long).view(-1).cpu()
    unique_ids, inverse = torch.unique(token_ids, sorted=False, return_inverse=True)
    missing_ids = [
        int(t) for t in unique_ids.tolist() if int(t) not in cache
    ]
    if missing_ids:
        loaded = load_thinker_embedding_rows(model_path, missing_ids).to(
            device=torch.device("cpu"), dtype=torch.float32
        )
        for token_id, row in zip(missing_ids, loaded):
            cache[int(token_id)] = row.detach().clone()
    unique_rows = torch.stack(
        [cache[int(t)] for t in unique_ids.tolist()], dim=0
    )
    gathered = unique_rows.index_select(0, inverse.to(device=unique_rows.device))
    return gathered.view(token_ids.shape[0], unique_rows.shape[-1])


SEQUENCES = [
    ("single", [3]),
    ("no-repeats", [1, 2, 3, 4, 5]),
    ("heavy-repeats", [9, 9, 9, 2, 9, 2, 9]),
    ("all-same", [11] * 20),
    ("long", list(range(0, 120, 3)) * 2),
    ("descending", list(range(40, 0, -1))),
]


@pytest.mark.parametrize("name,ids", SEQUENCES)
def test_single_gather_matches_the_previous_implementation(model_dir, name, ids):
    builder = _builder(model_dir)
    tokens = torch.tensor(ids, dtype=torch.long)
    got = builder._load_prompt_token_embeddings(tokens)
    want = _reference_gather(str(model_dir), tokens, {})
    assert got.shape == want.shape, name
    assert torch.equal(got, want), name


def test_batched_gather_matches_per_sequence_gathers(model_dir):
    """One unique + one index_select for the batch must equal N separate calls."""
    builder = _builder(model_dir)
    tensors = [torch.tensor(ids, dtype=torch.long) for _, ids in SEQUENCES]
    got = builder._load_prompt_token_embeddings_batch(tensors)
    assert len(got) == len(tensors)
    cache: dict = {}
    for (name, _ids), tensor, actual in zip(SEQUENCES, tensors, got):
        want = _reference_gather(str(model_dir), tensor, cache)
        assert actual.shape == want.shape, name
        assert torch.equal(actual, want), name


def test_batched_gather_matches_repeated_single_calls(model_dir):
    """Batching must not depend on the batch: same rows, one call or many."""
    batched_builder = _builder(model_dir)
    serial_builder = _builder(model_dir)
    tensors = [torch.tensor(ids, dtype=torch.long) for _, ids in SEQUENCES]
    batched = batched_builder._load_prompt_token_embeddings_batch(tensors)
    for tensor, actual in zip(tensors, batched):
        assert torch.equal(actual, serial_builder._load_prompt_token_embeddings(tensor))


def test_cache_reuse_across_calls_stays_exact(model_dir):
    """Second call hits the table instead of the loader; rows must be identical."""
    builder = _builder(model_dir)
    tokens = torch.tensor([4, 8, 15, 16, 23, 42], dtype=torch.long)
    first = builder._load_prompt_token_embeddings(tokens).clone()
    again = builder._load_prompt_token_embeddings(tokens)
    assert torch.equal(first, again)
    # a superset must reuse the cached slots and still be exact
    bigger = torch.tensor([4, 8, 15, 16, 23, 42, 99, 100], dtype=torch.long)
    got = builder._load_prompt_token_embeddings(bigger)
    assert torch.equal(got[:6], first)


def test_table_growth_preserves_every_earlier_row(model_dir):
    """Geometric growth recopies the table; nothing may shift or be lost."""
    builder = _builder(model_dir)
    seen = {}
    for start in range(0, VOCAB, 7):
        ids = list(range(start, min(start + 7, VOCAB)))
        rows = builder._load_prompt_token_embeddings(
            torch.tensor(ids, dtype=torch.long)
        )
        for token_id, row in zip(ids, rows):
            seen.setdefault(token_id, row.clone())
    # every id ever loaded must still gather back to the same row
    all_ids = sorted(seen)
    gathered = builder._load_prompt_token_embeddings(
        torch.tensor(all_ids, dtype=torch.long)
    )
    for token_id, row in zip(all_ids, gathered):
        assert torch.equal(row, seen[token_id]), f"row for id {token_id} moved"


def test_gathered_rows_are_not_views_into_the_table(model_dir):
    """Callers mutate prompt embeds in place; a view would corrupt the cache."""
    builder = _builder(model_dir)
    tokens = torch.tensor([1, 1, 2], dtype=torch.long)
    first = builder._load_prompt_token_embeddings(tokens)
    first[0] = -999.0
    again = builder._load_prompt_token_embeddings(tokens)
    assert not torch.equal(again[0], torch.full((HIDDEN,), -999.0))
    assert torch.equal(again[0], again[1]), "id 1 must still gather one row"


def test_batched_mask_matches_per_request_masks():
    builder = object.__new__(TalkerPrefillBuilder)
    builder._device = torch.device("cpu")
    builder._audio_token_id = AUDIO
    builder._image_token_id = IMAGE
    builder._video_token_id = VIDEO
    sequences = [
        torch.tensor([1, AUDIO, 2, IMAGE, 3], dtype=torch.long),
        torch.tensor([AUDIO] * 6, dtype=torch.long),
        torch.tensor([9, 9, 9], dtype=torch.long),
        torch.tensor([VIDEO, 1, AUDIO, IMAGE], dtype=torch.long),
    ]
    got = builder.build_multimodal_mask_batch(sequences)
    assert len(got) == len(sequences)
    for tensor, actual in zip(sequences, got):
        assert torch.equal(actual, builder.build_multimodal_mask(tensor))


def test_batched_mask_handles_the_empty_batch():
    builder = object.__new__(TalkerPrefillBuilder)
    builder._device = torch.device("cpu")
    assert builder.build_multimodal_mask_batch([]) == []
