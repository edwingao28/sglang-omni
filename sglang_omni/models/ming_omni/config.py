# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Ming-Omni V1."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.models.ming_omni.pipeline.next_stage import (
    AGGREGATE_STAGE,
    AUDIO_STAGE,
    DECODE_STAGE,
    IMAGE_STAGE,
    PREPROCESSING_STAGE,
    TALKER_STAGE,
    THINKER_STAGE,
)

_PKG = "sglang_omni.models.ming_omni"


def _stage_by_name(stages: list[StageConfig], name: str) -> StageConfig | None:
    return next((stage for stage in stages if stage.name == name), None)


def _stage_gpu_set(gpu: int | list[int] | None, tp_size: int) -> set[int]:
    """Return GPUs occupied by a stage.

    Explicit list placement is authoritative; scalar placement preserves the
    legacy contiguous TP range interpretation.
    """
    if isinstance(gpu, list):
        return {int(gpu_id) for gpu_id in gpu}
    if gpu is None:
        return set()
    return set(range(int(gpu), int(gpu) + tp_size))


def _ming_text_stages() -> list[StageConfig]:
    return [
        StageConfig(
            name=PREPROCESSING_STAGE,
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next=[AUDIO_STAGE, IMAGE_STAGE, AGGREGATE_STAGE],
            project_payload={
                AUDIO_STAGE: f"{_PKG}.stages.project_preprocessing_to_audio_encoder",
                IMAGE_STAGE: f"{_PKG}.stages.project_preprocessing_to_image_encoder",
                AGGREGATE_STAGE: (
                    f"{_PKG}.stages.project_preprocessing_to_mm_aggregate"
                ),
            },
        ),
        StageConfig(
            name=AUDIO_STAGE,
            factory=f"{_PKG}.stages.create_audio_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            next=AGGREGATE_STAGE,
            project_payload={
                AGGREGATE_STAGE: f"{_PKG}.stages.project_encoder_to_mm_aggregate"
            },
        ),
        StageConfig(
            name=IMAGE_STAGE,
            factory=f"{_PKG}.stages.create_image_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            next=AGGREGATE_STAGE,
            project_payload={
                AGGREGATE_STAGE: f"{_PKG}.stages.project_encoder_to_mm_aggregate"
            },
        ),
        StageConfig(
            name=AGGREGATE_STAGE,
            factory=f"{_PKG}.stages.create_aggregate_executor",
            wait_for=[PREPROCESSING_STAGE, AUDIO_STAGE, IMAGE_STAGE],
            merge_fn=f"{_PKG}.pipeline.merge.merge_for_thinker",
            next=THINKER_STAGE,
        ),
        StageConfig(
            name=THINKER_STAGE,
            factory=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
            factory_args={"thinker_max_seq_len": 8192},
            gpu=0,
            next=DECODE_STAGE,
        ),
        StageConfig(
            name=DECODE_STAGE,
            factory=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
    ]


def _ming_speech_stages() -> list[StageConfig]:
    stages = _ming_text_stages()
    thinker = _stage_by_name(stages, THINKER_STAGE)
    if thinker is not None:
        thinker.next = [DECODE_STAGE, TALKER_STAGE]

    stages.append(
        StageConfig(
            name=TALKER_STAGE,
            factory=f"{_PKG}.stages.create_talker_executor",
            factory_args={"device": "cuda", "voice": "DB30"},
            gpu=1,
            terminal=True,
        )
    )
    return stages


class MingOmniPipelineConfig(PipelineConfig):
    """6-stage text pipeline."""

    architecture: ClassVar[str] = "BailingMoeV2ForCausalLM"

    model_path: str
    entry_stage: str = PREPROCESSING_STAGE
    stages: list[StageConfig] = _ming_text_stages()


class MingOmniSpeechPipelineConfig(PipelineConfig):
    """7-stage speech pipeline."""

    architecture: ClassVar[str] = "BailingMoeV2ForCausalLM"

    model_path: str
    entry_stage: str = PREPROCESSING_STAGE
    stages: list[StageConfig] = _ming_speech_stages()

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        self._validate_talker_gpu_not_in_thinker_tp_range()

    def _validate_talker_gpu_not_in_thinker_tp_range(self) -> None:
        thinker = _stage_by_name(self.stages, THINKER_STAGE)
        talker = _stage_by_name(self.stages, TALKER_STAGE)
        if thinker is None or talker is None:
            return

        thinker_gpus = _stage_gpu_set(thinker.gpu, thinker.tp_size)
        talker_gpus = _stage_gpu_set(talker.gpu, talker.tp_size)
        collisions = thinker_gpus & talker_gpus
        if not collisions:
            return

        raise ValueError(
            "Ming-Omni speech talker GPU collides with thinker TP range: "
            f"talker gpus={sorted(talker_gpus)}, "
            f"thinker gpus={sorted(thinker_gpus)}, "
            f"collisions={sorted(collisions)}"
        )


EntryClass = MingOmniPipelineConfig

Variants = {
    "text": MingOmniPipelineConfig,
    "speech": MingOmniSpeechPipelineConfig,
}
