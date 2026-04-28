from typer.testing import CliRunner

from sglang_omni_v1.cli.cli import app
from sglang_omni_v1.cli.serve import apply_thinker_tp_size_cli_override
from sglang_omni_v1.config import PipelineConfig, StageConfig


def test_thinker_tp_size_flag_appears_in_help():
    runner = CliRunner()

    result = runner.invoke(app, ["serve", "--help"])

    assert result.exit_code == 0
    assert "--thinker-tp-size" in result.output


def test_thinker_tp_size_override_sets_factory_and_stage_gpu_list():
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="thinker",
                factory="tests.fake.create_thinker",
                gpu=2,
                terminal=True,
            )
        ],
    )

    apply_thinker_tp_size_cli_override(config, thinker_tp_size=3)

    stage = config.stages[0]
    assert stage.tp_size == 3
    assert stage.gpu == [2, 3, 4]
    assert stage.factory_args["server_args_overrides"]["tp_size"] == 3


def test_thinker_tp_size_override_leaves_gpu_list_unchanged():
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            StageConfig(
                name="thinker",
                factory="tests.fake.create_thinker",
                gpu=[4, 6],
                tp_size=2,
                terminal=True,
            )
        ],
    )

    apply_thinker_tp_size_cli_override(config, thinker_tp_size=2)

    assert config.stages[0].gpu == [4, 6]
