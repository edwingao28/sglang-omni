# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni.config import build_stage_placement_plan
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechColocatedPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)

_COLOCATED_FRACTIONS = {
    "image_encoder": 0.025,
    "audio_encoder": 0.025,
    "thinker": 0.75,
    "talker_ar": 0.12,
    "code2wav": 0.02,
}


class _CustomerColocatedConfig(Qwen3OmniSpeechColocatedPipelineConfig):
    pass


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def _set_colocated_budgets(config) -> None:
    for stage_name, fraction in _COLOCATED_FRACTIONS.items():
        _stage(config, stage_name).gpu_memory_fraction = fraction


def test_topology_flag_is_declared_by_config_class() -> None:
    assert Qwen3OmniPipelineConfig.colocated_speech_topology is False
    assert Qwen3OmniSpeechPipelineConfig.colocated_speech_topology is False
    assert Qwen3OmniSpeechColocatedPipelineConfig.colocated_speech_topology is True
    assert _CustomerColocatedConfig.colocated_speech_topology is True


def test_colocated_subclass_inherits_colocated_validation() -> None:
    config = _CustomerColocatedConfig(model_path="dummy")
    _set_colocated_budgets(config)

    plan = build_stage_placement_plan(config)

    assert plan.stages["thinker"].gpu_ids == (0,)
    assert plan.stages["talker_ar"].gpu_ids == (0,)


def test_split_speech_config_rejects_shared_ar_gpu() -> None:
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    _stage(config, "talker_ar").gpu = _stage(config, "thinker").gpu

    with pytest.raises(ValueError, match="colocated_speech_topology"):
        build_stage_placement_plan(config)


def test_text_config_is_unaffected() -> None:
    config = Qwen3OmniPipelineConfig(model_path="dummy")

    plan = build_stage_placement_plan(config)

    assert "talker_ar" not in plan.stages
