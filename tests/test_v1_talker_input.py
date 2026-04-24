# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from torch import nn

from sglang_omni_v1.models.qwen3_omni.components.talker_input import (
    build_assistant_part,
)


class _ZeroCodecEmbedding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (token_ids.shape[0], self.hidden_size),
            device=token_ids.device,
            dtype=torch.float32,
        )


def test_build_assistant_part_tolerates_short_prefix() -> None:
    assistant_embed = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=torch.float32,
    )
    codec_embed = _ZeroCodecEmbedding(hidden_size=2)

    result = build_assistant_part(
        assistant_embed=assistant_embed,
        text_projection=nn.Identity(),
        codec_embed_fn=codec_embed,
        tts_bos_embed=torch.tensor([[10.0, 11.0]], dtype=torch.float32),
        tts_eos_embed=torch.tensor([[12.0, 13.0]], dtype=torch.float32),
        tts_pad_embed=torch.tensor([[7.0, 8.0]], dtype=torch.float32),
        speaker_id=1,
        codec_nothink_id=2,
        codec_think_bos_id=3,
        codec_think_eos_id=4,
        codec_pad_id=5,
        codec_bos_id=6,
        tts_pad_token_id=99,
    )

    assert result["input_embeds"].shape == (9, 2)
    assert result["input_ids"].tolist() == [99] * 9
    assert torch.equal(result["input_embeds"][:3], assistant_embed)
    assert torch.equal(
        result["input_embeds"][3:7],
        torch.tensor(
            [[7.0, 8.0], [7.0, 8.0], [7.0, 8.0], [7.0, 8.0]],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(result["input_embeds"][7], torch.tensor([10.0, 11.0]))
    assert torch.equal(result["input_embeds"][8], torch.zeros(2, dtype=torch.float32))
    assert torch.equal(
        result["trailing_text_hidden"],
        torch.tensor([[12.0, 13.0]], dtype=torch.float32),
    )
