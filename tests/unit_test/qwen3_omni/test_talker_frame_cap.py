# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni.models.qwen3_omni.frame_cap import (
    TalkerFrameCap,
    resolve_talker_frame_cap,
    talker_frame_limit,
)
from sglang_omni.models.qwen3_omni.pending_text_queue import (
    PendingTextTensorQueue,
    coerce_pending_text_queue,
)
from sglang_omni.models.qwen3_omni.request_builders import (
    _build_talker_request_data,
    build_sglang_talker_request,
    make_talker_scheduler_adapters,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from tests.unit_test.fixtures.qwen_fakes import (
    FakeQwenTokenizer,
    make_talker_decode_prep_fake,
)

VOCAB = 8
EOS = 3
CAP = TalkerFrameCap(floor=1, per_text_token=1, eos_id=EOS)


def _row(value: float) -> torch.Tensor:
    return torch.full((2,), value)


def test_pending_text_queue_counts_every_row_ever_appended() -> None:
    queue = PendingTextTensorQueue.from_tensor(torch.zeros((3, 2)))
    assert queue.appended_total == 3
    queue.popleft()
    queue.popleft()
    queue.append(_row(1.0))
    queue.append_rows(torch.zeros((2, 2)))
    assert len(queue) == 4
    assert queue.appended_total == 6
    while queue:
        queue.popleft()
    queue.append(_row(2.0))
    assert queue.appended_total == 7
    assert queue.copy().appended_total == 7
    assert (
        PendingTextTensorQueue(rows=torch.zeros((4, 2)), cursor=1).appended_total == 4
    )
    assert coerce_pending_text_queue(deque([_row(1.0), _row(2.0)])).appended_total == 2


def test_resolve_frame_cap_is_off_without_slope_or_usable_eos() -> None:
    def _resolve(**kwargs: Any) -> TalkerFrameCap | None:
        base = dict(floor=40, per_text_token=12, eos_id=EOS, vocab_size=VOCAB)
        return resolve_talker_frame_cap(**{**base, **kwargs})

    assert _resolve(per_text_token=0) is None
    assert _resolve(eos_id=None) is None
    assert _resolve(eos_id=-1) is None
    assert _resolve(eos_id=VOCAB) is None
    assert _resolve() == TalkerFrameCap(floor=40, per_text_token=12, eos_id=EOS)
    assert _resolve(floor=-5).floor == 0


def _data(cap: TalkerFrameCap | None, *, rows: int, done: bool) -> SGLangARRequestData:
    data = SGLangARRequestData()
    data.talker_frame_cap = cap
    data.thinker_chunks_done = done
    data.pending_text_queue = PendingTextTensorQueue.from_tensor(torch.zeros((rows, 2)))
    return data


def test_frame_limit_waits_for_thinker_and_scales_with_text_rows() -> None:
    cap = TalkerFrameCap(floor=40, per_text_token=12, eos_id=EOS)
    assert talker_frame_limit(_data(None, rows=5, done=True)) is None
    assert talker_frame_limit(_data(cap, rows=5, done=False)) is None
    data = _data(cap, rows=5, done=True)
    assert talker_frame_limit(data) == 40 + 12 * 5
    data.pending_text_queue.append(_row(1.0))
    data.pending_text_queue.popleft()
    assert talker_frame_limit(data) == 40 + 12 * 6


def test_frame_limit_is_off_for_untyped_text_queue() -> None:
    data = _data(
        TalkerFrameCap(floor=40, per_text_token=12, eos_id=EOS), rows=5, done=True
    )
    data.pending_text_queue = deque([_row(1.0)])
    assert talker_frame_limit(data) is None


def _sched_req(
    rid: str,
    *,
    out_len: int,
    cap: TalkerFrameCap | None,
    rows: int = 2,
    done: bool = True,
    suppress: tuple[int, ...] = (5,),
) -> SimpleNamespace:
    sp = SimpleNamespace(
        repetition_penalty=1.05,
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        min_p=0.0,
        sampling_seed=7,
    )
    req = SimpleNamespace(
        sampling_params=sp,
        output_ids=[0] * out_len,
        _codec_suppress_tokens=None,
        rid=rid,
    )
    data = SGLangARRequestData(req=req, output_ids=req.output_ids)
    data.suppress_tokens = list(suppress)
    data.talker_frame_cap = cap
    data.thinker_chunks_done = done
    data.pending_text_queue = PendingTextTensorQueue.from_tensor(torch.zeros((rows, 2)))
    return SimpleNamespace(data=data)


def _forced_row() -> torch.Tensor:
    row = torch.ones(VOCAB, dtype=torch.bool)
    row[EOS] = False
    return row


def _suppress_only_row() -> torch.Tensor:
    row = torch.zeros(VOCAB, dtype=torch.bool)
    row[5] = True
    return row


def test_prepare_decode_buffers_forces_eos_row_at_frame_limit(caplog) -> None:
    fake = make_talker_decode_prep_fake(vocab=VOCAB)
    below = _sched_req("below", out_len=2, cap=CAP)
    at = _sched_req("at", out_len=3, cap=CAP)
    streaming = _sched_req("streaming", out_len=9, cap=CAP, done=False)
    uncapped = _sched_req("free", out_len=9, cap=None)

    with caplog.at_level(logging.WARNING, logger=Qwen3OmniTalker.__module__):
        Qwen3OmniTalker.prepare_decode_buffers(fake, [below, at, streaming, uncapped])
        Qwen3OmniTalker.prepare_decode_buffers(
            make_talker_decode_prep_fake(vocab=VOCAB), [below, at, streaming, uncapped]
        )

    assert torch.equal(fake._suppress_mask[0], _suppress_only_row())
    assert torch.equal(fake._suppress_mask[1], _forced_row())
    assert torch.equal(fake._suppress_mask[2], _suppress_only_row())
    assert torch.equal(fake._suppress_mask[3], _suppress_only_row())
    assert below.data.talker_frame_cap_hit_at is None
    assert at.data.talker_frame_cap_hit_at == 3
    assert streaming.data.talker_frame_cap_hit_at is None
    hits = [rec for rec in caplog.records if "talker_frame_cap" in rec.getMessage()]
    assert len(hits) == 1 and "rid=at" in hits[0].getMessage()


def test_frame_limit_reached_on_steady_state_path_matches_fresh_rebuild() -> None:
    fake = make_talker_decode_prep_fake(vocab=VOCAB)
    reused: list[bool] = []
    inner = fake._reuse_decode_buffers

    def _spy(requests: list) -> bool:
        result = inner(requests)
        reused.append(result)
        return result

    fake._reuse_decode_buffers = _spy
    requests = [
        _sched_req("a", out_len=2, cap=CAP),
        _sched_req("b", out_len=2, cap=None),
    ]
    Qwen3OmniTalker.prepare_decode_buffers(fake, requests)
    assert torch.equal(fake._suppress_mask[0], _suppress_only_row())

    for row_idx, sched_req in enumerate(requests):
        fake._sampled_token_ids[row_idx] = 2
        sched_req.data.req.output_ids.append(2)
    Qwen3OmniTalker.prepare_decode_buffers(fake, requests)

    assert reused == [False, True]
    assert torch.equal(fake._suppress_mask[0], _forced_row())
    assert torch.equal(fake._suppress_mask[1], _suppress_only_row())
    assert requests[0].data.talker_frame_cap_hit_at == 3
    assert bool(fake._repetition_mask[0, 2]) and bool(fake._repetition_mask[1, 2])

    fresh = make_talker_decode_prep_fake(vocab=VOCAB)
    Qwen3OmniTalker.prepare_decode_buffers(fresh, requests)
    assert torch.equal(fake._suppress_mask, fresh._suppress_mask)
    assert torch.equal(fake._repetition_mask, fresh._repetition_mask)


def test_frame_limit_applies_once_thinker_stream_closes_on_fast_path() -> None:
    fake = make_talker_decode_prep_fake(vocab=VOCAB)
    streaming = _sched_req("open", out_len=5, cap=CAP, rows=2, done=False)
    Qwen3OmniTalker.prepare_decode_buffers(fake, [streaming])
    assert torch.equal(fake._suppress_mask[0], _suppress_only_row())
    assert streaming.data.talker_frame_cap_hit_at is None

    streaming.data.pending_text_queue.append(_row(1.0))
    streaming.data.thinker_chunks_done = True
    fake._sampled_token_ids[0] = 2
    streaming.data.req.output_ids.append(2)
    Qwen3OmniTalker.prepare_decode_buffers(fake, [streaming])

    assert fake._decode_prep_out_lens == [6]
    assert torch.equal(fake._suppress_mask[0], _forced_row())
    assert streaming.data.talker_frame_cap_hit_at == 6


def test_forced_row_samples_codec_eos() -> None:
    fake = make_talker_decode_prep_fake(vocab=VOCAB)
    at = _sched_req("at", out_len=3, cap=CAP)
    Qwen3OmniTalker.prepare_decode_buffers(fake, [at])

    logits = torch.arange(VOCAB, dtype=torch.float32).unsqueeze(0)
    logits[0, EOS] = -50.0
    sampled = Qwen3OmniTalker._sample_decode_tokens(fake, logits, forward_batch=None)
    assert sampled.tolist() == [EOS]

    masked = logits.masked_fill(fake._suppress_mask[:1], float("-inf"))
    probs = torch.softmax(masked, dim=-1)
    assert probs[0, EOS] == pytest.approx(1.0)


def test_build_sglang_talker_request_attaches_frame_cap() -> None:
    data = build_sglang_talker_request(
        thinker_hidden_states=torch.zeros((4, 2)),
        tokenizer=FakeQwenTokenizer(),
        codec_vocab_size=4096,
        codec_eos_id=2150,
        frame_cap_floor=40,
        frame_cap_per_text_token=12,
    )
    assert data.talker_frame_cap == TalkerFrameCap(
        floor=40, per_text_token=12, eos_id=2150
    )
    assert data.talker_frame_cap_hit_at is None

    off = build_sglang_talker_request(
        thinker_hidden_states=torch.zeros((4, 2)),
        tokenizer=FakeQwenTokenizer(),
        codec_vocab_size=4096,
        codec_eos_id=2150,
    )
    assert off.talker_frame_cap is None


class _StubPrefillBuilder:
    def build_prompt_prefill(
        self, _payload: Any, thinker_chunks: list[Any], *, thinker_done: bool
    ) -> dict[str, Any]:
        del thinker_chunks, thinker_done
        return {
            "input_embeds": torch.zeros((9, 2), dtype=torch.float32),
            "input_ids": torch.zeros((9,), dtype=torch.long),
            "pending_text_queue": deque([torch.zeros((2,), dtype=torch.float32)]),
            "tts_eos_embed": torch.full((2,), 0.5, dtype=torch.float32),
            "tts_pad_embed": torch.full((2,), 0.25, dtype=torch.float32),
            "prompt_model_inputs": {"audio_embeds": None},
        }


def _payload(request_id: str = "req-1") -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        request=SimpleNamespace(params={}),
        prefetched_chunks=[object()] * 3,
        prefetched_stream_done=True,
    )


def test_builder_threads_frame_cap_from_sampling_config(monkeypatch) -> None:
    from sglang_omni.models.qwen3_omni import request_builders as rb_mod

    captured: dict[str, Any] = {}

    def _fake_build(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(req=SimpleNamespace(rid="req-1"))

    def _resolve(_params: dict[str, Any]) -> dict[str, Any]:
        return {
            "max_new_tokens": 4096,
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 1.0,
            "repetition_penalty": 1.05,
            "codec_eos_id": 7,
            "suppress_tokens": [],
            "seed": 1,
            "frame_cap_floor": 48,
            "frame_cap_per_text_token": 10,
        }

    monkeypatch.setattr(rb_mod, "build_sglang_talker_request", _fake_build)
    _build_talker_request_data(
        _payload(),
        prefill_builder=_StubPrefillBuilder(),
        tokenizer=SimpleNamespace(),
        codec_vocab_size=4096,
        codec_bos_id=2149,
        audio_token_id=151646,
        image_token_id=151647,
        video_token_id=151648,
        thinker_config=SimpleNamespace(),
        resolve_sampling_config=_resolve,
    )
    assert captured["frame_cap_floor"] == 48
    assert captured["frame_cap_per_text_token"] == 10


def test_adapters_resolve_frame_cap_defaults(monkeypatch, tmp_path) -> None:
    from sglang_omni.models.qwen3_omni import request_builders as rb_mod

    captured: dict[str, Any] = {}

    def _fake_build_data(payload: Any, **kwargs: Any) -> Any:
        captured["resolve"] = kwargs["resolve_sampling_config"]
        return SimpleNamespace(req=SimpleNamespace(rid=payload.request_id))

    monkeypatch.setattr(rb_mod, "_build_talker_request_data", _fake_build_data)
    model = SimpleNamespace(
        config=SimpleNamespace(codec_eos_token_id=2150),
        model=SimpleNamespace(codec_embedding=SimpleNamespace(weight=torch.zeros(1))),
        activation_dtype=torch.float32,
    )
    common = dict(
        tokenizer=SimpleNamespace(),
        codec_vocab_size=4096,
        model=model,
        model_path=str(tmp_path),
        thinker_config=SimpleNamespace(),
        required_aux_hidden_key=24,
    )

    request_builder, _, _, _ = make_talker_scheduler_adapters(**common)
    request_builder(_payload())
    cfg = captured["resolve"]({})
    assert cfg["frame_cap_per_text_token"] == 0
    assert cfg["frame_cap_floor"] == 40

    request_builder, _, _, _ = make_talker_scheduler_adapters(
        **common, talker_frame_cap_floor=64, talker_frame_cap_per_text_token=12
    )
    request_builder(_payload())
    cfg = captured["resolve"]({})
    assert cfg["frame_cap_per_text_token"] == 12
    assert cfg["frame_cap_floor"] == 64


@pytest.mark.parametrize(
    ("overrides", "expected_floor", "expected_slope"),
    [
        ({}, 40, 0),
        ({"talker_frame_cap_floor": 64, "talker_frame_cap_per_text_token": 12}, 64, 12),
    ],
)
def test_bootstrap_forwards_frame_cap_knobs_to_adapters(
    monkeypatch, overrides: dict[str, int], expected_floor: int, expected_slope: int
) -> None:
    from sglang.srt.utils import hf_transformers_utils

    from sglang_omni.models.qwen3_omni import bootstrap, request_builders
    from sglang_omni.models.qwen3_omni import talker_model_runner as runner_mod
    from sglang_omni.models.qwen3_omni import talker_scheduler as sched_mod
    from sglang_omni.scheduling import bootstrap as scheduling_bootstrap
    from sglang_omni.scheduling import sglang_backend

    adapter_kwargs: list[dict[str, Any]] = []
    talker_config = SimpleNamespace(
        text_config=SimpleNamespace(vocab_size=4096),
        accept_hidden_layer=24,
        codec_bos_id=2149,
        codec_eos_token_id=2150,
        codec_nothink_id=2155,
        codec_think_bos_id=2156,
        codec_think_eos_id=2157,
        codec_pad_id=2148,
        speaker_id={},
    )
    hf_config = SimpleNamespace(
        talker_config=talker_config,
        thinker_config=SimpleNamespace(
            audio_token_id=1, image_token_id=2, video_token_id=3
        ),
        tts_bos_token_id=4,
        tts_eos_token_id=5,
        tts_pad_token_id=6,
        im_start_token_id=7,
        im_end_token_id=8,
        system_token_id=9,
        user_token_id=10,
        assistant_token_id=11,
    )
    model_config = SimpleNamespace(
        model_path="model", vocab_size=10, hf_config=hf_config
    )
    model_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model_config=model_config, model=SimpleNamespace(), sampler=object()
        )
    )

    def _fake_adapters(**kwargs: Any) -> tuple[object, object, object, object]:
        adapter_kwargs.append(kwargs)
        return object(), object(), object(), object()

    monkeypatch.setattr(
        sched_mod, "configure_talker_server_args", lambda *a, **k: False
    )
    monkeypatch.setattr(
        scheduling_bootstrap,
        "create_sglang_infrastructure",
        lambda *a, **k: (model_worker, object(), object(), object(), model_config),
    )
    monkeypatch.setattr(
        hf_transformers_utils, "get_tokenizer", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        request_builders, "make_talker_scheduler_adapters", _fake_adapters
    )
    monkeypatch.setattr(sglang_backend, "SGLangOutputProcessor", lambda **k: object())
    monkeypatch.setattr(
        sched_mod,
        "QwenTalkerScheduler",
        lambda **k: SimpleNamespace(outbox=object(), bind_model_runner=lambda r: None),
    )
    monkeypatch.setattr(runner_mod, "QwenTalkerModelRunner", lambda *a, **k: object())

    bootstrap.create_talker_scheduler(SimpleNamespace(), **overrides)

    assert adapter_kwargs[-1]["talker_frame_cap_floor"] == expected_floor
    assert adapter_kwargs[-1]["talker_frame_cap_per_text_token"] == expected_slope
