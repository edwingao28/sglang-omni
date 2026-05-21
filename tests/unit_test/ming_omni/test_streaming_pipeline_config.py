# SPDX-License-Identifier: Apache-2.0
"""Ming streaming TTS topology and non-regression tests."""

from __future__ import annotations

import pytest

from sglang_omni.models.ming_omni.pipeline.next_stage import (
    AUDIO_STAGE,
    AGGREGATE_STAGE,
    DECODE_STAGE,
    IMAGE_STAGE,
    PREPROCESSING_STAGE,
    TALKER_STAGE,
    THINKER_STAGE,
)


def _stage(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def test_default_speech_pipeline_stays_non_streaming() -> None:
    from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

    config = MingOmniSpeechPipelineConfig(model_path="dummy")

    assert [stage.name for stage in config.stages] == [
        PREPROCESSING_STAGE,
        AUDIO_STAGE,
        IMAGE_STAGE,
        AGGREGATE_STAGE,
        THINKER_STAGE,
        DECODE_STAGE,
        TALKER_STAGE,
    ]
    assert config.terminal_stages == [DECODE_STAGE, TALKER_STAGE]

    thinker = _stage(config, THINKER_STAGE)
    talker = _stage(config, TALKER_STAGE)

    assert thinker.next == [DECODE_STAGE, TALKER_STAGE]
    assert thinker.stream_to == []
    assert thinker.factory_args == {"thinker_max_seq_len": 8192}
    assert talker.terminal is True
    assert talker.factory == "sglang_omni.models.ming_omni.stages.create_talker_executor"


def test_streaming_speech_pipeline_is_opt_in_and_v1_native() -> None:
    from sglang_omni.models.ming_omni.config import (
        MingOmniSpeechPipelineConfig,
        MingOmniStreamingSpeechPipelineConfig,
        Variants,
    )
    from sglang_omni.models.ming_omni.pipeline.next_stage import (
        SEGMENTER_STAGE,
        TALKER_STREAM_STAGE,
    )

    assert Variants["speech"] is MingOmniSpeechPipelineConfig
    assert Variants["streaming_speech"] is MingOmniStreamingSpeechPipelineConfig

    config = MingOmniStreamingSpeechPipelineConfig(model_path="dummy")
    stages = {stage.name: stage for stage in config.stages}

    assert list(stages) == [
        PREPROCESSING_STAGE,
        AUDIO_STAGE,
        IMAGE_STAGE,
        AGGREGATE_STAGE,
        THINKER_STAGE,
        DECODE_STAGE,
        SEGMENTER_STAGE,
        TALKER_STREAM_STAGE,
    ]
    assert config.terminal_stages == [DECODE_STAGE, TALKER_STREAM_STAGE]

    thinker = stages[THINKER_STAGE]
    decode = stages[DECODE_STAGE]
    segmenter = stages[SEGMENTER_STAGE]
    talker_stream = stages[TALKER_STREAM_STAGE]

    assert thinker.next == [DECODE_STAGE, SEGMENTER_STAGE]
    assert thinker.stream_to == [DECODE_STAGE, SEGMENTER_STAGE]
    assert thinker.factory_args == {
        "thinker_max_seq_len": 8192,
        "enable_streaming_outputs": True,
    }
    assert decode.factory == (
        "sglang_omni.models.ming_omni.stages."
        "create_streaming_decode_scheduler"
    )
    assert decode.terminal is True
    assert decode.can_accept_stream_before_payload is True
    assert segmenter.next == TALKER_STREAM_STAGE
    assert segmenter.stream_to == [TALKER_STREAM_STAGE]
    assert segmenter.can_accept_stream_before_payload is True
    assert talker_stream.terminal is True
    assert talker_stream.can_accept_stream_before_payload is True
    assert segmenter.factory == (
        "sglang_omni.models.ming_omni.stages."
        "create_streaming_segmenter_scheduler"
    )
    assert talker_stream.factory == (
        "sglang_omni.models.ming_omni.stages."
        "create_streaming_talker_scheduler"
    )


def test_streaming_talker_gpu_must_not_overlap_thinker_tp() -> None:
    from sglang_omni.models.ming_omni.config import (
        MingOmniStreamingSpeechPipelineConfig,
    )
    from sglang_omni.models.ming_omni.pipeline.next_stage import TALKER_STREAM_STAGE

    config = MingOmniStreamingSpeechPipelineConfig(model_path="dummy")
    thinker = _stage(config, THINKER_STAGE)
    talker_stream = _stage(config, TALKER_STREAM_STAGE)
    thinker.gpu = [0, 1]
    thinker.tp_size = 2
    talker_stream.gpu = 1

    with pytest.raises(ValueError, match="talker.*GPU.*collides"):
        config._validate_talker_gpu_not_in_thinker_tp_range()
