# SPDX-License-Identifier: Apache-2.0
"""Ming image-generation thinker-input merge tests."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.ming_omni.io import MingOmniPipelineState
from sglang_omni.models.ming_omni.pipeline.merge import build_thinker_inputs


def test_build_thinker_inputs_routes_image_gen_query_tokens_to_image_embeds() -> None:
    state = MingOmniPipelineState(
        mm_inputs={
            "image_gen": {
                "query_tokens": [[1.0, 2.0], [3.0, 4.0]],
                "gen_mask": [0, 1, 1, 0],
                "prefill_only": True,
                "image_patch_token_id": 22,
                "image_gen_params": {"size": "512x512"},
            }
        }
    )

    thinker_inputs = build_thinker_inputs(state, {})
    model_inputs = thinker_inputs["model_inputs"]

    assert set(model_inputs) == {"image_embeds"}
    assert model_inputs["image_embeds"].shape == (2, 2)
    assert model_inputs["image_embeds"].dtype == torch.bfloat16
    assert thinker_inputs["capture_model_output_keys"] == ("hidden_states",)


def test_build_thinker_inputs_rejects_input_image_and_image_gen_mix() -> None:
    state = MingOmniPipelineState(
        mm_inputs={
            "image_gen": {
                "query_tokens": [[1.0, 2.0]],
                "gen_mask": [1],
                "prefill_only": True,
                "image_patch_token_id": 22,
                "image_gen_params": {},
            }
        }
    )
    encoder_outs = {
        "image_encoder": {
            "image_embeds": torch.ones((1, 2), dtype=torch.float32),
        }
    }

    with pytest.raises(ValueError, match="input image"):
        build_thinker_inputs(state, encoder_outs)


def test_image_gen_state_ignores_non_dict_without_tensor_truthiness() -> None:
    state = MingOmniPipelineState(mm_inputs={"image_gen": torch.ones((2, 2))})
    assert build_thinker_inputs(state, {}) == {}


def test_build_thinker_inputs_preserves_media_cache_keys_with_image_gen() -> None:
    state = MingOmniPipelineState(
        mm_inputs={
            "image_gen": {
                "query_tokens": [[1.0, 2.0]],
                "gen_mask": [1],
                "prefill_only": True,
                "image_patch_token_id": 22,
                "image_gen_params": {},
            }
        },
        encoder_inputs={"audio_encoder": {"cache_key": "aud-1"}},
    )

    thinker_inputs = build_thinker_inputs(state, {})

    assert thinker_inputs["media_cache_keys"] == {"audio": "audio:aud-1"}
    assert thinker_inputs["capture_model_output_keys"] == ("hidden_states",)
