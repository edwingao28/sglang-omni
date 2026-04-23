# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

import sglang_omni_v1.models.qwen3_omni.components.talker as talker_module
from sglang_omni_v1.models.qwen3_omni.components.talker import Qwen3OmniTalker


def test_sample_code_predictor_token_uses_top_k_top_p(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_sampler(
        probs: torch.Tensor, top_k: int, top_p: float
    ) -> torch.Tensor:
        captured["probs"] = probs.clone()
        captured["top_k"] = top_k
        captured["top_p"] = top_p
        return torch.tensor([2, 1], device=probs.device, dtype=torch.long)

    monkeypatch.setattr(
        talker_module,
        "top_k_top_p_sampling_from_probs",
        fake_sampler,
    )

    logits = torch.tensor(
        [
            [[0.0, 1.0, 2.0]],
            [[2.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    result = Qwen3OmniTalker._sample_code_predictor_token(logits)

    assert result.shape == (2, 1)
    assert result[:, 0].tolist() == [2, 1]
    assert captured["top_k"] == 50
    assert captured["top_p"] == 0.8
    assert torch.allclose(
        captured["probs"],
        torch.softmax(logits[:, -1, :], dim=-1),
    )
