# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock

import pytest

from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechColocatedPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)

SPEAKERS = {"ethan": 2302, "chelsie": 2303, "aiden": 2304}


def _write_checkpoint(directory, speakers) -> None:
    directory.mkdir(exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps({"talker_config": {"speaker_id": speakers}})
    )


@pytest.mark.parametrize(
    "config_cls",
    [Qwen3OmniSpeechPipelineConfig, Qwen3OmniSpeechColocatedPipelineConfig],
)
def test_speech_pipeline_lists_checkpoint_speakers_without_task_type(
    tmp_path, monkeypatch, config_cls
) -> None:
    _write_checkpoint(tmp_path, SPEAKERS)
    download = Mock(side_effect=AssertionError("Local checkpoint must not use HF"))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)

    custom_voice_config = config_cls(
        model_path=str(tmp_path)
    ).resolve_custom_voice_config()

    assert custom_voice_config.speakers == ("ethan", "chelsie", "aiden")
    assert custom_voice_config.task_type is None
    download.assert_not_called()


def test_speech_pipeline_reads_hub_checkpoint_once(tmp_path, monkeypatch) -> None:
    _write_checkpoint(tmp_path, {"ethan": 2302})
    download = Mock(return_value=str(tmp_path / "config.json"))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)

    config = Qwen3OmniSpeechPipelineConfig(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )

    assert config.resolve_custom_voice_config().speakers == ("ethan",)
    download.assert_called_once_with(
        repo_id="Qwen/Qwen3-Omni-30B-A3B-Instruct", filename="config.json"
    )


def test_speech_pipeline_honours_talker_stage_model_path(tmp_path) -> None:
    _write_checkpoint(tmp_path / "root", {"ethan": 2302})
    _write_checkpoint(tmp_path / "talker", {"vivian": 7})
    config = Qwen3OmniSpeechPipelineConfig(model_path=str(tmp_path / "root"))
    talker_stage = config.stage_named("talker_ar")
    talker_stage.factory = talker_stage.factory.model_copy(
        update={"model_path": str(tmp_path / "talker")}
    )

    assert config.resolve_custom_voice_config().speakers == ("vivian",)


def test_typed_talker_model_path_beats_pipeline_authored_kwargs(
    tmp_path, monkeypatch
) -> None:
    _write_checkpoint(tmp_path / "root", {"ethan": 2302})
    _write_checkpoint(tmp_path / "authored", {"ryan": 5})
    _write_checkpoint(tmp_path / "typed", {"vivian": 7})
    config = Qwen3OmniSpeechPipelineConfig(model_path=str(tmp_path / "root"))
    monkeypatch.setattr(
        Qwen3OmniSpeechPipelineConfig,
        "stage_factory_kwargs",
        lambda self, name: {"model_path": str(tmp_path / "authored")},
    )
    assert config.resolve_custom_voice_config().speakers == ("ryan",)

    talker_stage = config.stage_named("talker_ar")
    talker_stage.factory = talker_stage.factory.model_copy(
        update={"model_path": str(tmp_path / "typed")}
    )
    assert config.resolve_custom_voice_config().speakers == ("vivian",)


@pytest.mark.parametrize("speakers", [None, {}, "ethan"])
def test_speech_pipeline_without_speaker_table_lists_nothing(
    tmp_path, speakers
) -> None:
    _write_checkpoint(tmp_path, speakers)

    config = Qwen3OmniSpeechPipelineConfig(model_path=str(tmp_path))

    assert config.resolve_custom_voice_config() is None


def test_text_only_pipeline_lists_no_voices(tmp_path, monkeypatch) -> None:
    _write_checkpoint(tmp_path, SPEAKERS)
    download = Mock(side_effect=AssertionError("text-only pipeline must not read"))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)

    assert (
        Qwen3OmniPipelineConfig(model_path=str(tmp_path)).resolve_custom_voice_config()
        is None
    )
