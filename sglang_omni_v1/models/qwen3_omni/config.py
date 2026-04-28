# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Qwen3-Omni."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni_v1.config import PipelineConfig, StageConfig

_PKG = "sglang_omni_v1.models.qwen3_omni"


def _validate_qwen3_speech_gpu_placement(
    gpu_placement: dict[str, int | list[int]],
    *,
    tp_size: int,
) -> None:
    """Reject configs where speech stages collide with the thinker's TP rank range.

    A scalar thinker GPU expands to [thinker_gpu, thinker_gpu + tp_size). A
    list thinker placement is treated as the explicit rank GPU list.
    """
    thinker_placement = gpu_placement.get("thinker", 0)
    if isinstance(thinker_placement, list):
        if len(thinker_placement) != tp_size:
            raise ValueError(
                "GPU placement for Qwen3 speech pipeline is invalid: "
                f"'thinker' has {len(thinker_placement)} GPU(s), but "
                f"thinker tp_size is {tp_size}."
            )
        thinker_gpus = set(thinker_placement)
        thinker_gpu_desc = str(thinker_placement)
    else:
        thinker_gpu_end = thinker_placement + tp_size
        thinker_gpus = set(range(thinker_placement, thinker_gpu_end))
        thinker_gpu_desc = f"[{thinker_placement}, {thinker_gpu_end})"

    for stage_name in ("talker_ar", "code_predictor", "code2wav"):
        stage_placement = gpu_placement.get(stage_name)
        if stage_placement is None:
            continue
        stage_gpus = (
            set(stage_placement)
            if isinstance(stage_placement, list)
            else {stage_placement}
        )
        colliding_gpus = sorted(stage_gpus & thinker_gpus)
        if colliding_gpus:
            raise ValueError(
                "GPU placement for Qwen3 speech pipeline is invalid: "
                f"{stage_name!r} is on GPU(s) {colliding_gpus}, which collides "
                f"with the thinker TP rank GPU placement {thinker_gpu_desc}. "
                f"Move {stage_name!r} to a GPU outside that range or reduce "
                "the thinker tp_size."
            )


class Qwen3OmniPipelineConfig(PipelineConfig):
    """6-stage text-only pipeline."""

    architecture: ClassVar[str] = "Qwen3OmniMoeForConditionalGeneration"

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next=["image_encoder", "audio_encoder", "mm_aggregate"],
        ),
        StageConfig(
            name="image_encoder",
            factory=f"{_PKG}.stages.create_image_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            next="mm_aggregate",
        ),
        StageConfig(
            name="audio_encoder",
            factory=f"{_PKG}.stages.create_audio_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            next="mm_aggregate",
        ),
        StageConfig(
            name="mm_aggregate",
            factory=f"{_PKG}.stages.create_aggregate_executor",
            wait_for=["preprocessing", "image_encoder", "audio_encoder"],
            merge_fn=f"{_PKG}.merge.merge_for_thinker",
            next="thinker",
        ),
        StageConfig(
            name="thinker",
            factory=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
            factory_args={"thinker_max_seq_len": 8192},
            gpu=0,
            next="decode",
        ),
        StageConfig(
            name="decode",
            factory=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
    ]


class Qwen3OmniSpeechPipelineConfig(PipelineConfig):
    """8-stage speech pipeline (text + audio output)."""

    architecture: ClassVar[str] = "Qwen3OmniMoeForConditionalGeneration"

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next=["image_encoder", "audio_encoder", "mm_aggregate"],
        ),
        StageConfig(
            name="image_encoder",
            factory=f"{_PKG}.stages.create_image_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            next="mm_aggregate",
        ),
        StageConfig(
            name="audio_encoder",
            factory=f"{_PKG}.stages.create_audio_encoder_executor",
            factory_args={"device": "cuda", "dtype": None},
            gpu=0,
            next="mm_aggregate",
        ),
        StageConfig(
            name="mm_aggregate",
            factory=f"{_PKG}.stages.create_aggregate_executor",
            wait_for=["preprocessing", "image_encoder", "audio_encoder"],
            merge_fn=f"{_PKG}.merge.merge_for_thinker",
            next="thinker",
        ),
        StageConfig(
            name="thinker",
            factory=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
            factory_args={"thinker_max_seq_len": 8192, "speech_enabled": True},
            gpu=0,
            next=["decode", "talker_ar"],
            stream_to=["talker_ar"],
        ),
        StageConfig(
            name="decode",
            factory=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
        StageConfig(
            name="talker_ar",
            factory=f"{_PKG}.stages.create_talker_ar_executor_from_config",
            factory_args={
                # Note (Xuesong): must exceed talker_max_new_tokens (4096) +
                # prefill, else req_to_token_pool OOBs and crashes talker_ar.
                "talker_max_seq_len": 8192,
                "speech_enabled": True,
                "feedback_enabled": True,
            },
            gpu=1,
            next="code2wav",
            stream_to=["code2wav"],
        ),
        StageConfig(
            name="code2wav",
            factory=f"{_PKG}.components.code2wav_scheduler.create_code2wav_scheduler",
            factory_args={"device": "cuda"},
            gpu=1,
            terminal=True,
        ),
    ]

    def validate_thinker_tp_gpu_placement(self, *, tp_size: int) -> None:
        _validate_qwen3_speech_gpu_placement(self.gpu_placement, tp_size=tp_size)


EntryClass = Qwen3OmniSpeechPipelineConfig

Variants = {
    "text": Qwen3OmniPipelineConfig,
    "speech": Qwen3OmniSpeechPipelineConfig,
}
