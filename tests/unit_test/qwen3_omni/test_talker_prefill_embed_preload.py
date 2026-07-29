# SPDX-License-Identifier: Apache-2.0
import json
import types

import pytest
import torch
from safetensors.torch import save_file

from sglang_omni.models.qwen3_omni.components import talker_prefill

VOCAB, HIDDEN = 64, 8
TTS_BOS, TTS_EOS, TTS_PAD = 51, 52, 53


@pytest.fixture()
def model_dir(tmp_path):
    weight = torch.linspace(-3.0, 3.0, VOCAB * HIDDEN, dtype=torch.float32).reshape(
        VOCAB, HIDDEN
    )
    shard = "model-00001-of-00001.safetensors"
    save_file({"thinker.model.embed_tokens.weight": weight}, str(tmp_path / shard))
    index = {"weight_map": {"thinker.model.embed_tokens.weight": shard}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    talker_prefill._EMBED_SOURCE_CACHE.clear()
    return tmp_path


def _fake_model():
    codec_embedding = types.SimpleNamespace(
        weight=torch.zeros(1, 1, dtype=torch.float32)
    )
    return types.SimpleNamespace(
        model=types.SimpleNamespace(codec_embedding=codec_embedding),
        activation_dtype=torch.bfloat16,
        text_projection=lambda tensor: tensor,
        hidden_projection=lambda tensor: tensor,
        get_input_embeddings=lambda: None,
    )


def _build(model_dir, model=None):
    return talker_prefill.TalkerPrefillBuilder(
        model=model if model is not None else _fake_model(),
        model_path=str(model_dir),
        audio_token_id=1,
        image_token_id=2,
        video_token_id=3,
        tts_bos_token_id=TTS_BOS,
        tts_eos_token_id=TTS_EOS,
        tts_pad_token_id=TTS_PAD,
        im_start_token_id=4,
        im_end_token_id=5,
        system_token_id=6,
        user_token_id=7,
        assistant_token_id=8,
        codec_bos_id=9,
        codec_nothink_id=10,
        codec_think_bos_id=11,
        codec_think_eos_id=12,
        codec_pad_id=13,
        speaker_map={"ethan": 0},
    )


SAMPLE_IDS = torch.tensor(
    [0, 7, 7, 31, 63, TTS_BOS, TTS_EOS, TTS_PAD], dtype=torch.long
)


def test_preload_rows_bitwise_match_lazy(model_dir, monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_TALKER_EMBED_PRELOAD", raising=False)
    lazy = _build(model_dir)
    assert lazy._embed_table is None
    expected = lazy._load_prompt_token_embeddings(SAMPLE_IDS)
    expected_special = torch.cat(lazy.get_tts_special_embeds(), dim=0)

    monkeypatch.setenv("SGLANG_OMNI_TALKER_EMBED_PRELOAD", "device")
    preloaded = _build(model_dir)
    assert preloaded._embed_table is not None
    got = preloaded._load_prompt_token_embeddings(SAMPLE_IDS)

    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    assert torch.equal(got, expected)
    assert torch.equal(
        torch.cat(preloaded.get_tts_special_embeds(), dim=0), expected_special
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_preload_rows_bitwise_match_lazy_cuda(model_dir, monkeypatch):
    def cuda_model():
        model = _fake_model()
        model.model.codec_embedding.weight = torch.zeros(1, 1, device="cuda")
        return model

    monkeypatch.delenv("SGLANG_OMNI_TALKER_EMBED_PRELOAD", raising=False)
    lazy = _build(model_dir, model=cuda_model())
    expected = lazy._load_prompt_token_embeddings(SAMPLE_IDS)

    monkeypatch.setenv("SGLANG_OMNI_TALKER_EMBED_PRELOAD", "device")
    preloaded = _build(model_dir, model=cuda_model())
    got = preloaded._load_prompt_token_embeddings(SAMPLE_IDS)
    assert got.device.type == "cuda"
    assert torch.equal(got, expected)


def test_off_mode_does_not_read_weights_at_init(model_dir, monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_TALKER_EMBED_PRELOAD", raising=False)
    opens = {"n": 0}
    real_safe_open = talker_prefill.safe_open

    def counting_safe_open(*args, **kwargs):
        opens["n"] += 1
        return real_safe_open(*args, **kwargs)

    monkeypatch.setattr(talker_prefill, "safe_open", counting_safe_open)
    builder = _build(model_dir)
    assert opens["n"] == 0
    assert builder._embed_table is None

    rows = {"n": 0}
    real_rows = talker_prefill.load_thinker_embedding_rows

    def counting_rows(*args, **kwargs):
        rows["n"] += 1
        return real_rows(*args, **kwargs)

    monkeypatch.setattr(talker_prefill, "load_thinker_embedding_rows", counting_rows)
    builder._load_prompt_token_embeddings(SAMPLE_IDS)
    assert rows["n"] == 1


def test_preload_never_loads_rows_after_init(model_dir, monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_TALKER_EMBED_PRELOAD", "device")
    builder = _build(model_dir)

    def fail_rows(*args, **kwargs):
        raise AssertionError("load_thinker_embedding_rows called under preload")

    monkeypatch.setattr(talker_prefill, "load_thinker_embedding_rows", fail_rows)
    builder._load_prompt_token_embeddings(SAMPLE_IDS)
    builder.get_tts_special_embeds()


def test_preload_failure_falls_back_to_lazy(model_dir, monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_OMNI_TALKER_EMBED_PRELOAD", "device")
    (model_dir / "model-00001-of-00001.safetensors").unlink()
    talker_prefill._EMBED_SOURCE_CACHE.clear()

    with caplog.at_level("WARNING", logger=talker_prefill.__name__):
        builder = _build(model_dir)

    assert builder._embed_table is None
    warnings = [record.getMessage() for record in caplog.records]
    assert any("preload failed" in message for message in warnings)
    assert any("memory fraction" in message for message in warnings)


def test_preload_failure_warning_reports_table_size(model_dir, monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_OMNI_TALKER_EMBED_PRELOAD", "device")
    model = _fake_model()
    model.model.codec_embedding.weight = types.SimpleNamespace(
        device=torch.device("cuda", 99)
    )

    with caplog.at_level("WARNING", logger=talker_prefill.__name__):
        builder = _build(model_dir, model=model)

    assert builder._embed_table is None
    message = "\n".join(record.getMessage() for record in caplog.records)
    assert str((VOCAB, HIDDEN)) in message
    assert "GiB" in message


@pytest.mark.parametrize("mode", ["gpu", "cpu"])
def test_invalid_mode_is_off(model_dir, monkeypatch, caplog, mode):
    monkeypatch.setenv("SGLANG_OMNI_TALKER_EMBED_PRELOAD", mode)
    with caplog.at_level("WARNING", logger=talker_prefill.__name__):
        builder = _build(model_dir)
    assert builder._embed_table is None
