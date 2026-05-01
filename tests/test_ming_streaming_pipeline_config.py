# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


def _stage_by_name(config, name: str):
    return next(stage for stage in config.stages if stage.name == name)


def test_launcher_selects_streaming_speech_config_and_talker_stream_gpu() -> None:
    from examples.run_ming_omni_server import (
        resolve_ming_gpu_placement,
        resolve_ming_pipeline_config,
    )
    from sglang_omni.models.ming_omni.config import (
        MingOmniStreamingSpeechPipelineConfig,
    )
    from sglang_omni.models.ming_omni.pipeline.next_stage import TALKER_STREAM_STAGE

    assert (
        resolve_ming_pipeline_config(
            enable_speech=True,
            enable_streaming_tts=True,
        )
        is MingOmniStreamingSpeechPipelineConfig
    )
    assert resolve_ming_gpu_placement(
        enable_speech=True,
        enable_streaming_tts=True,
        gpu_thinker=0,
        gpu_talker=2,
    ) == {"thinker": 0, TALKER_STREAM_STAGE: 2}


def test_streaming_speech_topology_routes_text_segments_to_talker_stream() -> None:
    from sglang_omni.models.ming_omni.config import MingOmniStreamingSpeechPipelineConfig
    from sglang_omni.models.ming_omni.pipeline.next_stage import (
        DECODE_STAGE,
        SEGMENTER_STAGE,
        TALKER_STREAM_STAGE,
        THINKER_STAGE,
        segmenter_next,
        thinker_next_streaming_speech,
    )

    config = MingOmniStreamingSpeechPipelineConfig(model_path="/tmp/model")
    stage_names = {stage.name for stage in config.stages}

    assert TALKER_STREAM_STAGE in stage_names
    assert SEGMENTER_STAGE in stage_names
    assert "code_predictor" not in stage_names
    assert "code2wav" not in stage_names
    assert config.terminal_stages == [DECODE_STAGE, TALKER_STREAM_STAGE]
    assert thinker_next_streaming_speech("req-1", {}) == [DECODE_STAGE, SEGMENTER_STAGE]
    assert segmenter_next("req-1", {}) == TALKER_STREAM_STAGE
    assert _stage_by_name(config, THINKER_STAGE).stream_to[0].to_stage == SEGMENTER_STAGE
    assert _stage_by_name(config, SEGMENTER_STAGE).stream_to[0].to_stage == TALKER_STREAM_STAGE


def test_streaming_speech_config_validates_talker_stream_outside_thinker_tp_range() -> None:
    from sglang_omni.models.ming_omni.config import (
        MingOmniStreamingSpeechPipelineConfig,
    )
    from sglang_omni.models.ming_omni.pipeline.next_stage import (
        TALKER_STREAM_STAGE,
        THINKER_STAGE,
    )

    config = MingOmniStreamingSpeechPipelineConfig(
        model_path="/tmp/model",
        gpu_placement={"thinker": 0, TALKER_STREAM_STAGE: 2},
    )
    config.apply_server_args_overrides(
        stage_name=THINKER_STAGE,
        overrides={"tp_size": 2},
    )

    bad_config = MingOmniStreamingSpeechPipelineConfig(
        model_path="/tmp/model",
        gpu_placement={"thinker": 0, TALKER_STREAM_STAGE: 1},
    )
    try:
        bad_config.apply_server_args_overrides(
            stage_name=THINKER_STAGE,
            overrides={"tp_size": 2},
        )
    except ValueError as exc:
        assert "collides" in str(exc).lower()
    else:
        raise AssertionError("tp_size=2 should reject talker_stream on GPU 1")
