# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni_v1.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner


def _sched_req(**data_kwargs):
    data = SimpleNamespace(**data_kwargs)
    return SimpleNamespace(data=data)


def test_combine_feedback_embed_matches_legacy_step_index_for_projected_prefill() -> None:
    sched_req = _sched_req(
        feedback_embeds=torch.tensor([1.0, 2.0]),
        feedback_step_index=1,
        generation_steps=2,
        input_embeds_are_projected=True,
        trailing_text_hidden=[
            torch.tensor([10.0, 10.0]),
            torch.tensor([20.0, 20.0]),
        ],
        tts_pad_embed=torch.tensor([7.0, 8.0]),
        thinker_chunks_done=False,
    )

    combined = QwenTalkerModelRunner._combine_feedback_embed(
        sched_req=sched_req,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(combined, torch.tensor([21.0, 22.0]))
def test_combine_feedback_embed_falls_back_to_pad_when_stream_done() -> None:
    sched_req = _sched_req(
        feedback_embeds=torch.tensor([1.0, 2.0]),
        feedback_step_index=3,
        generation_steps=4,
        input_embeds_are_projected=True,
        trailing_text_hidden=[torch.tensor([10.0, 10.0])],
        tts_pad_embed=torch.tensor([7.0, 8.0]),
        thinker_chunks_done=True,
    )

    combined = QwenTalkerModelRunner._combine_feedback_embed(
        sched_req=sched_req,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(combined, torch.tensor([8.0, 10.0]))
