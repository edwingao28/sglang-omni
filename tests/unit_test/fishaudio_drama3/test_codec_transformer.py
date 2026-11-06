# SPDX-License-Identifier: Apache-2.0
"""The fused codec transformer forward against the codec's own, on the real weights."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.fishaudio_drama3.codec_transformer import CodecTransformer
from sglang_omni.models.fishaudio_drama3.config import DEFAULT_CODEC_PATH
from sglang_omni.models.fishaudio_drama3.stages import resolve_codec_directory
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import DAC
from sglang_omni.models.fishaudio_s2_pro.stages import load_codec

pytestmark = pytest.mark.accelerator


@pytest.fixture(scope="module")
def codec() -> DAC:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    else:
        pass
    codec = load_codec(resolve_codec_directory(DEFAULT_CODEC_PATH), "cuda")
    codec.fold_weight_norm()
    codec.quantizer.to(torch.float16)
    return codec


def quantized_latent(codec: DAC, batch_size: int, frame_count: int) -> torch.Tensor:
    """The post_module input of random codes: the sum of the codebook vectors."""
    quantizer = codec.quantizer
    torch.manual_seed(frame_count)
    codes = torch.randint(0, 1024, (batch_size, 10, frame_count), device="cuda")
    with torch.no_grad():
        semantic = quantizer.semantic_quantizer.from_codes(codes[:, :1])[0]
        return semantic + quantizer.quantizer.from_codes(codes[:, 1:])[0]


@pytest.mark.parametrize(
    ("batch_size", "frame_count"),
    [(1, 44), (4, 96), (1, 220)],
    ids=["1x44", "4x96", "1x220-beyond-the-128-frame-window"],
)
def test_the_fused_post_module_decodes_like_the_codec_in_float16(
    codec: DAC, batch_size: int, frame_count: int
) -> None:
    module = codec.quantizer.post_module
    latent = quantized_latent(codec, batch_size, frame_count)
    assert latent.dtype == torch.float16

    with torch.no_grad():
        expected = module(latent.clone())
    actual = CodecTransformer(module)(latent.clone())

    assert actual.dtype == torch.float16
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)
