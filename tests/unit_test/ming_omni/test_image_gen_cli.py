# SPDX-License-Identifier: Apache-2.0
"""CLI tests for the Ming-Omni image-generation serve flags.

Covers the five ``sgl-omni serve`` image-generation flags wired in
``sglang_omni/cli/serve.py``:

    --image-gen / --diffusion-model-path / --image-gen-gpu /
    --enable-standalone-semantic-encoder / --enable-byt5-text-rendering
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import typer

from sglang_omni.cli.serve import apply_image_gen_cli_overrides, serve
from sglang_omni.config import PipelineConfig
from sglang_omni.models.ming_omni.config import (
    MingOmniImagePipelineConfig,
    MingOmniSpeechPipelineConfig,
)


def _stage(config: PipelineConfig, name: str):
    return next(stage for stage in config.stages if stage.name == name)


class _DummyManager:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def parse_extra_args(self, args):
        return {}

    def merge_config(self, extra_args):
        return self.config


def _serve_kwargs(**overrides):
    data = dict(
        ctx=SimpleNamespace(args=[]),
        model_path="dummy",
        config=None,
        text_only=False,
        colocate=False,
        thinker_cuda_graph="default",
        image_gen=False,
        diffusion_model_path=None,
        image_gen_gpu=None,
        enable_standalone_semantic_encoder=False,
        enable_byt5_text_rendering=False,
    )
    data.update(overrides)
    return data


# --------------------------------------------------------------------------
# serve() variant selection (mirrors test_cli_text_only_selects_text_variant)
# --------------------------------------------------------------------------


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
def test_cli_image_gen_selects_image_variant(from_model_path, launch_server):
    from_model_path.return_value = _DummyManager(
        MingOmniImagePipelineConfig(model_path="dummy")
    )

    serve(**_serve_kwargs(image_gen=True, diffusion_model_path="/models/zimage"))

    from_model_path.assert_called_once_with("dummy", variant="image")
    launch_server.assert_called_once()


# --------------------------------------------------------------------------
# serve() rejection rules (pure CLI validation, raised before resolution)
# --------------------------------------------------------------------------


def test_cli_image_gen_requires_diffusion_model_path():
    with pytest.raises(typer.BadParameter, match="--diffusion-model-path"):
        serve(**_serve_kwargs(image_gen=True))


def test_cli_rejects_image_gen_with_text_only():
    with pytest.raises(typer.BadParameter, match="--text-only"):
        serve(
            **_serve_kwargs(
                image_gen=True,
                diffusion_model_path="/models/zimage",
                text_only=True,
            )
        )


def test_cli_rejects_image_gen_with_colocate():
    with pytest.raises(typer.BadParameter, match="--colocate"):
        serve(**_serve_kwargs(image_gen=True, colocate=True))


def test_cli_rejects_image_gen_with_thinker_cuda_graph_on():
    with pytest.raises(typer.BadParameter, match="--thinker-cuda-graph"):
        serve(
            **_serve_kwargs(
                image_gen=True,
                diffusion_model_path="/models/zimage",
                thinker_cuda_graph="on",
            )
        )


# --------------------------------------------------------------------------
# apply_image_gen_cli_overrides() (post-resolution, config-aware)
# --------------------------------------------------------------------------


def test_apply_image_gen_overrides_sets_dit_factory_args_and_gpu():
    config = MingOmniImagePipelineConfig(model_path="dummy")

    updated = apply_image_gen_cli_overrides(
        config,
        diffusion_model_path="/models/zimage",
        image_gen_gpu=2,
        enable_standalone_semantic_encoder=True,
        enable_byt5_text_rendering=False,
    )

    image_gen = _stage(updated, "image_gen")
    assert image_gen.gpu == 2
    assert image_gen.factory_args["dit_type"] == "zimage"
    assert image_gen.factory_args["dit_model_path"] == "/models/zimage"
    assert image_gen.factory_args["enable_standalone_semantic_encoder"] is True
    assert "enable_byt5_text_rendering" not in image_gen.factory_args


def test_apply_image_gen_overrides_rejects_image_flags_on_speech_pipeline():
    config = MingOmniSpeechPipelineConfig(model_path="dummy")

    with pytest.raises(
        typer.BadParameter,
        match="--diffusion-model-path is not supported by MingOmniSpeechPipelineConfig",
    ):
        apply_image_gen_cli_overrides(
            config,
            diffusion_model_path="/models/zimage",
            image_gen_gpu=None,
            enable_standalone_semantic_encoder=False,
            enable_byt5_text_rendering=False,
        )


def test_apply_image_gen_overrides_noop_on_speech_pipeline_without_flags():
    config = MingOmniSpeechPipelineConfig(model_path="dummy")

    result = apply_image_gen_cli_overrides(
        config,
        diffusion_model_path=None,
        image_gen_gpu=None,
        enable_standalone_semantic_encoder=False,
        enable_byt5_text_rendering=False,
    )

    assert result is config


def test_apply_image_gen_overrides_rejects_gpu_colliding_with_thinker():
    config = MingOmniImagePipelineConfig(model_path="dummy")

    # The thinker stage defaults to GPU 0; placing image_gen there must collide.
    with pytest.raises(typer.BadParameter, match="image_gen.*thinker"):
        apply_image_gen_cli_overrides(
            config,
            diffusion_model_path="/models/zimage",
            image_gen_gpu=0,
            enable_standalone_semantic_encoder=False,
            enable_byt5_text_rendering=False,
        )


# --------------------------------------------------------------------------
# Full CLI integration (Typer -> serve -> launch_server)
# --------------------------------------------------------------------------


def test_omni_serve_builds_ming_image_config(monkeypatch):
    from typer.testing import CliRunner

    from sglang_omni.cli import app

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "sglang_omni.config.manager.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            architectures=["BailingMM2NativeForConditionalGeneration"]
        ),
    )

    def fake_launch_server(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs

    monkeypatch.setattr("sglang_omni.cli.serve.launch_server", fake_launch_server)

    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--model-path",
            "inclusionAI/Ming-flash-omni-2.0",
            "--image-gen",
            "--diffusion-model-path",
            "/models/zimage",
            "--image-gen-gpu",
            "2",
            "--enable-byt5-text-rendering",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--model-name",
            "ming-omni",
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert type(config).__name__ == "MingOmniImagePipelineConfig"
    image_gen = _stage(config, "image_gen")
    assert image_gen.gpu == 2
    assert image_gen.factory_args["dit_model_path"] == "/models/zimage"
    assert image_gen.factory_args["dit_type"] == "zimage"
    assert image_gen.factory_args["enable_byt5_text_rendering"] is True
    assert captured["kwargs"]["model_name"] == "ming-omni"
