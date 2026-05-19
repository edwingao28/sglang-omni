def _stage_by_name(config, name):
    for stage in config.stages:
        if stage.name == name:
            return stage
    available_names = [stage.name for stage in config.stages]
    raise KeyError(f"{name!r} not found in stages: {available_names!r}")


def test_text_pipeline_factory_args_carry_thinker_max_seq_len():
    from sglang_omni.config.runtime import resolve_stage_factory_args
    from sglang_omni.models.ming_omni.config import MingOmniPipelineConfig

    config = MingOmniPipelineConfig(model_path="test/model")
    stage = _stage_by_name(config, "thinker")

    assert stage.factory_args["thinker_max_seq_len"] == 8192

    stage.factory_args = dict(stage.factory_args or {})
    stage.factory_args["thinker_max_seq_len"] = 12345

    resolved_args = resolve_stage_factory_args(stage, config)
    assert resolved_args["thinker_max_seq_len"] == 12345


def test_speech_pipeline_factory_args_carry_thinker_max_seq_len():
    from sglang_omni.config.runtime import resolve_stage_factory_args
    from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

    config = MingOmniSpeechPipelineConfig(model_path="test/model")
    stage = _stage_by_name(config, "thinker")

    assert stage.factory_args["thinker_max_seq_len"] == 8192

    stage.factory_args = dict(stage.factory_args or {})
    stage.factory_args["thinker_max_seq_len"] = 67890

    resolved_args = resolve_stage_factory_args(stage, config)
    assert resolved_args["thinker_max_seq_len"] == 67890
