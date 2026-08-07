# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang_omni.models.qwen3_omni.components.sglang_thinker import (
    Qwen3OmniThinkerForCausalLM,
)


class _CountingEmbedding(nn.Embedding):
    def __init__(self) -> None:
        super().__init__(8, 4)
        self.calls = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(input_ids)


class _FakeTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = _CountingEmbedding()
        self.pp_group = SimpleNamespace(is_first_rank=True)
        self.seen_input_ids: torch.Tensor | None = None
        self.seen_input_embeds: torch.Tensor | None = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: object,
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: object | None = None,
        input_deepstack_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions, forward_batch, input_deepstack_embeds
        self.seen_input_ids = input_ids
        self.seen_input_embeds = input_embeds
        if input_embeds is not None:
            return input_embeds
        if input_ids is not None:
            return self.embed_tokens(input_ids)
        assert pp_proxy_tensors is not None
        return pp_proxy_tensors


class _FakeLogitsProcessor(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        forward_batch: object,
    ) -> str:
        del input_ids, hidden_states, lm_head, forward_batch
        return "logits"


def _make_outer() -> tuple[Qwen3OmniThinkerForCausalLM, _FakeTextModel]:
    outer = Qwen3OmniThinkerForCausalLM.__new__(Qwen3OmniThinkerForCausalLM)
    nn.Module.__init__(outer)
    text_model = _FakeTextModel()
    outer.model = text_model
    outer.lm_head = nn.Identity()
    outer.logits_processor = _FakeLogitsProcessor()
    return outer, text_model


def _forward_batch() -> SimpleNamespace:
    return SimpleNamespace(mrope_positions=None)


def test_outer_forward_materializes_text_embeddings_before_inner_model() -> None:
    outer, text_model = _make_outer()
    input_ids = torch.tensor([1, 2])
    expected = text_model.embed_tokens.weight.detach()[input_ids]

    output = outer(input_ids, torch.tensor([0, 1]), _forward_batch())

    assert output == "logits"
    assert text_model.seen_input_ids is None
    torch.testing.assert_close(text_model.seen_input_embeds, expected)
    assert text_model.embed_tokens.calls == 1


def test_outer_forward_preserves_caller_supplied_multimodal_embeddings() -> None:
    outer, text_model = _make_outer()
    input_embeds = torch.randn(2, 4)

    output = outer(
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        _forward_batch(),
        input_embeds=input_embeds,
    )

    assert output == "logits"
    assert text_model.seen_input_ids is None
    assert text_model.seen_input_embeds is input_embeds
    assert text_model.embed_tokens.calls == 0


def test_outer_forward_skips_embedding_on_non_first_pp_rank() -> None:
    outer, text_model = _make_outer()
    text_model.pp_group.is_first_rank = False
    input_ids = torch.tensor([1, 2])
    pp_proxy_tensors = torch.zeros(2, 4)

    output = outer(
        input_ids,
        torch.tensor([0, 1]),
        _forward_batch(),
        pp_proxy_tensors=pp_proxy_tensors,
    )

    assert output == "logits"
    assert text_model.seen_input_ids is None
    assert text_model.seen_input_embeds is None
    assert text_model.embed_tokens.calls == 0
