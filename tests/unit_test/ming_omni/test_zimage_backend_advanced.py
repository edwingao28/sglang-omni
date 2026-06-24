# SPDX-License-Identifier: Apache-2.0
"""ZImage advanced-path guard tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.models.ming_omni.diffusion.backend import ImageGenParams


class FakePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=["image"])


class FakeTextEncoder:
    def __init__(self, torch_module):
        self.torch = torch_module
        self.calls = []

    def encode(self, text, *, tokenizer, device, max_length):
        self.calls.append((text, tokenizer, device, max_length))
        return (
            [self.torch.full((1, 3), 2.0)],
            [self.torch.full((1, 3), -2.0)],
        )


class FakeSemanticEncoder:
    def __init__(self, torch_module):
        self.torch = torch_module
        self.calls = []
        self.unloaded = False

    def encode(self, prompt):
        self.calls.append(prompt)
        return (
            [self.torch.full((2, 3), 4.0)],
            [self.torch.zeros(2, 3)],
        )

    def unload(self):
        self.unloaded = True


def _backend_with_fake_pipe():
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

    backend = ZImageBackend()
    backend._pipe = FakePipe()
    backend._device = "cpu"
    return backend


def test_extract_render_text_uses_last_quoted_span() -> None:
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import (
        _extract_render_text,
    )

    assert _extract_render_text('make a sign saying "SALE"') == "SALE"
    assert _extract_render_text('first "A" then "B"') == "B"
    assert _extract_render_text("no quoted text") == ""


def test_generate_requires_standalone_encoder_when_condition_embeds_missing() -> None:
    backend = _backend_with_fake_pipe()

    with pytest.raises(RuntimeError, match="standalone semantic encoder unavailable"):
        backend.generate("draw", ImageGenParams())


def test_generate_uses_explicit_standalone_encoder_when_loaded() -> None:
    torch = pytest.importorskip("torch")
    backend = _backend_with_fake_pipe()
    backend._semantic_encoder = FakeSemanticEncoder(torch)

    image = backend.generate("draw from reference", ImageGenParams())

    assert image == "image"
    assert backend._semantic_encoder.calls == ["draw from reference"]
    prompt_embeds = backend._pipe.calls[0]["prompt_embeds"]
    torch.testing.assert_close(prompt_embeds[0], torch.full((2, 3), 4.0))


def test_generate_concats_byt5_only_when_text_rendering_is_enabled() -> None:
    torch = pytest.importorskip("torch")
    backend = _backend_with_fake_pipe()
    backend._text_encoder = FakeTextEncoder(torch)
    backend._tokenizer = object()
    sem = torch.ones(2, 3)
    neg = torch.zeros(2, 3)

    backend.generate(
        'make a sign saying "SALE"',
        ImageGenParams(enable_text_rendering=False),
        condition_embeds=[sem],
        negative_condition_embeds=[neg],
    )
    backend.generate(
        'make a sign saying "SALE"',
        ImageGenParams(enable_text_rendering=True),
        condition_embeds=[sem],
        negative_condition_embeds=[neg],
    )

    first_prompt = backend._pipe.calls[0]["prompt_embeds"][0]
    second_prompt = backend._pipe.calls[1]["prompt_embeds"][0]
    torch.testing.assert_close(first_prompt, sem)
    assert second_prompt.shape == (3, 3)
    torch.testing.assert_close(second_prompt[-1], torch.full((3,), 2.0))
    assert backend._text_encoder.calls[0][0] == "SALE"


def test_generate_fails_when_text_rendering_requested_but_byt5_unloaded() -> None:
    torch = pytest.importorskip("torch")
    backend = _backend_with_fake_pipe()
    sem = torch.ones(2, 3)

    with pytest.raises(RuntimeError, match="ByT5 text rendering unavailable"):
        backend.generate(
            'make a sign saying "SALE"',
            ImageGenParams(enable_text_rendering=True),
            condition_embeds=[sem],
        )


def test_load_models_semantic_encoder_without_ming_model_path_raises() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

    with pytest.raises(
        ValueError, match="load_semantic_encoder=True requires ming_model_path"
    ):
        ZImageBackend().load_models(
            "/fake/dit",
            torch.device("cpu"),
            load_semantic_encoder=True,
        )


def test_load_models_byt5_without_ming_model_path_raises() -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

    with pytest.raises(
        ValueError, match="load_byt5_text_encoder=True requires ming_model_path"
    ):
        ZImageBackend().load_models(
            "/fake/dit",
            torch.device("cpu"),
            load_byt5_text_encoder=True,
        )


def test_load_models_byt5_missing_dir_raises(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

    with pytest.raises(RuntimeError, match="ByT5 text rendering requested but"):
        ZImageBackend().load_models(
            "/fake/dit",
            torch.device("cpu"),
            ming_model_path=str(tmp_path),
            load_byt5_text_encoder=True,
        )


def test_image_gen_params_production_defaults() -> None:
    """Lock the production-facing defaults a request inherits when fields are omitted."""
    params = ImageGenParams()
    assert params.width == 1024
    assert params.height == 1024
    assert params.num_inference_steps == 28
    # Z-Image-Turbo is distilled/low-CFG; SD's 7.0 default washes images out.
    assert params.guidance_scale == 2.0
    assert params.seed is None
    assert params.negative_prompt == ""
    assert params.semantic_source is None
    assert params.enable_text_rendering is False
