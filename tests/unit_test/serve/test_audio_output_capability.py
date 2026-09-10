# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.models.auk.config import AuKPipelineConfig
from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig
from sglang_omni.models.ming_omni import config as ming_omni_config
from sglang_omni.models.qwen3_omni import config as qwen3_omni_config
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from sglang_omni.models.whisper_asr.config import WhisperASRPipelineConfig


def test_pipeline_config_leaves_audio_output_undeclared() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="engine",
                process="engine",
                factory_path="tests.fake.create_stage",
                terminal=True,
            )
        ],
    )

    assert config.supports_audio_output() is None


@pytest.mark.parametrize(
    ("variants", "variant", "expected"),
    [
        (qwen3_omni_config.Variants, "text", False),
        (qwen3_omni_config.Variants, "speech", True),
        (qwen3_omni_config.Variants, "speech-colocated", True),
        (ming_omni_config.Variants, "text", False),
        (ming_omni_config.Variants, "speech", True),
        (ming_omni_config.Variants, "streaming_speech", True),
    ],
)
def test_omni_variants_declare_audio_output_by_talker_stage(
    variants: dict[str, type[PipelineConfig]], variant: str, expected: bool
) -> None:
    config = variants[variant](model_path="dummy")

    assert config.supports_audio_output() is expected


@pytest.mark.parametrize(
    ("config_cls", "model_path"),
    [
        (AuKPipelineConfig, "dummy"),
        (HiggsTtsPipelineConfig, "dummy"),
        (Qwen3TTSPipelineConfig, "dummy"),
        (WhisperASRPipelineConfig, "openai/whisper-large-v3"),
    ],
)
def test_other_pipelines_leave_audio_output_undeclared(
    config_cls: type[PipelineConfig], model_path: str
) -> None:
    config = config_cls(model_path=model_path)

    assert config.supports_audio_output() is None
