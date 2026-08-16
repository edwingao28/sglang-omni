# SPDX-License-Identifier: Apache-2.0
"""Two-phase talker prefill must reproduce the monolithic build byte for byte."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.talker_prefill import (
    TalkerPrefillBuilder,
    TalkerPromptSegment,
)

IM_START = 151644
IM_END = 151645
SYSTEM = 8948
USER = 872
ASSISTANT = 77091
NEWLINE = 198
AUDIO = 151646
TTS_BOS = 151672
TTS_EOS = 151673
TTS_PAD = 151671

THINKER_DIM = 4
TALKER_DIM = 6
CODEC_VOCAB = 32


def _fake_model() -> SimpleNamespace:
    torch.manual_seed(0)
    text_projection = torch.nn.Linear(THINKER_DIM, TALKER_DIM)
    hidden_projection = torch.nn.Linear(THINKER_DIM, TALKER_DIM)
    codec_embedding = torch.nn.Embedding(CODEC_VOCAB, TALKER_DIM)
    return SimpleNamespace(
        text_projection=text_projection,
        hidden_projection=hidden_projection,
        get_input_embeddings=lambda: codec_embedding,
    )


def _prompt_ids() -> torch.Tensor:
    return torch.tensor(
        [
            IM_START,
            SYSTEM,
            100,
            NEWLINE,
            IM_START,
            USER,
            AUDIO,
            AUDIO,
            101,
            NEWLINE,
            IM_START,
            ASSISTANT,
            NEWLINE,
        ],
        dtype=torch.long,
    )


def _builder(model: SimpleNamespace) -> TalkerPrefillBuilder:
    builder = object.__new__(TalkerPrefillBuilder)
    builder._model = model
    builder._model_path = "<unused>"
    builder._device = torch.device("cpu")
    builder._dtype = torch.float32
    builder._audio_token_id = AUDIO
    builder._image_token_id = None
    builder._video_token_id = None
    builder._tts_bos_token_id = TTS_BOS
    builder._tts_eos_token_id = TTS_EOS
    builder._tts_pad_token_id = TTS_PAD
    builder._im_start_token_id = IM_START
    builder._im_end_token_id = IM_END
    builder._system_token_id = SYSTEM
    builder._user_token_id = USER
    builder._assistant_token_id = ASSISTANT
    builder._codec_bos_id = 20
    builder._codec_nothink_id = 21
    builder._codec_think_bos_id = 22
    builder._codec_think_eos_id = 23
    builder._codec_pad_id = 24
    builder._speaker_map = {"ethan": 7}

    torch.manual_seed(1)
    table = {
        token_id: torch.randn(THINKER_DIM)
        for token_id in (
            IM_START,
            IM_END,
            SYSTEM,
            USER,
            ASSISTANT,
            NEWLINE,
            AUDIO,
            100,
            101,
            200,
            201,
            202,
            203,
            204,
            TTS_BOS,
            TTS_EOS,
            TTS_PAD,
        )
    }
    builder._thinker_embed_cache = dict(table)
    special = torch.stack([table[TTS_BOS], table[TTS_EOS], table[TTS_PAD]], dim=0)
    builder._tts_special_cache = tuple(model.text_projection(special).chunk(3, dim=0))
    return builder


def _payload() -> SimpleNamespace:
    prompt_ids = _prompt_ids()
    audio_rows = torch.arange(2 * THINKER_DIM, dtype=torch.float32).reshape(
        2, THINKER_DIM
    )
    return SimpleNamespace(
        request_id="req-two-phase",
        request=SimpleNamespace(params={"speaker": "Ethan"}),
        data={
            "prompt": {"input_ids": prompt_ids},
            "thinker_inputs": {"model_inputs": {"audio_embeds": audio_rows}},
        },
    )


def _chunks(token_ids: list[int]) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            data=torch.zeros(THINKER_DIM),
            metadata={"token_id": token_id, "layer_hidden": torch.randn(THINKER_DIM)},
        )
        for token_id in token_ids
    ]


@pytest.mark.parametrize(
    "token_ids, thinker_done",
    [
        ([200, 201, 202, 203, 204, IM_END], True),
        ([200, 201, 202, 203, 204], False),
        ([200, 201, 202], False),
        ([200], False),
    ],
)
def test_two_phase_build_matches_monolithic(token_ids, thinker_done) -> None:
    """The split point is the prompt/assistant boundary, so nothing shifts."""
    model = _fake_model()
    builder = _builder(model)
    payload = _payload()
    chunks = _chunks(token_ids)

    monolithic = builder.build_prompt_prefill(
        payload, chunks, thinker_done=thinker_done
    )
    segment = builder.build_prompt_segment(payload)
    tail = builder.build_assistant_tail(segment, chunks, thinker_done=thinker_done)
    two_phase = builder.compose_two_phase_prefill(segment, tail)

    assert torch.equal(two_phase["input_embeds"], monolithic["input_embeds"])
    assert torch.equal(two_phase["input_ids"], monolithic["input_ids"])
    assert torch.equal(two_phase["tts_pad_embed"], monolithic["tts_pad_embed"])
    assert torch.equal(two_phase["tts_eos_embed"], monolithic["tts_eos_embed"])
    expected_queue = monolithic["pending_text_queue"]
    actual_queue = two_phase["pending_text_queue"]
    assert len(actual_queue) == len(expected_queue)
    for row in range(len(expected_queue)):
        assert torch.equal(actual_queue[row], expected_queue[row])


def test_prompt_segment_covers_everything_before_the_assistant_header() -> None:
    """Phase 1 must stop at the final <|im_start|> and leave 9 rows to phase 2."""
    builder = _builder(_fake_model())
    segment = builder.build_prompt_segment(_payload())

    assert segment.tail_rows == 9
    assert segment.tail_input_ids.tolist() == [TTS_PAD] * 9
    # user segment only: [im_start user audio audio 101 nl]; system is dropped.
    assert segment.input_ids.tolist() == [IM_START, USER, AUDIO, AUDIO, 101, NEWLINE]
    assert segment.header_embed.shape[0] == 3


def test_prompt_segment_consumes_no_thinker_hidden_state() -> None:
    """Records today's behaviour: multimodal rows are projected from zeros."""
    model = _fake_model()
    builder = _builder(model)
    seen: list[torch.Tensor] = []
    real_hidden_projection = model.hidden_projection
    model.hidden_projection = lambda rows: seen.append(rows) or real_hidden_projection(
        rows
    )

    builder.build_prompt_segment(_payload())

    assert seen, "expected the audio rows to route through hidden_projection"
    assert all(bool(torch.count_nonzero(rows) == 0) for rows in seen)


def test_assistant_tail_rejects_a_generated_im_start() -> None:
    """A generated <|im_start|> would move the monolithic split point."""
    builder = _builder(_fake_model())
    segment = builder.build_prompt_segment(_payload())

    with pytest.raises(ValueError, match="im_start"):
        builder.build_assistant_tail(
            segment, _chunks([200, IM_START, 201]), thinker_done=False
        )


def test_assistant_tail_requires_a_thinker_chunk() -> None:
    builder = _builder(_fake_model())
    segment = builder.build_prompt_segment(_payload())

    with pytest.raises(ValueError, match="requires thinker chunks"):
        builder.build_assistant_tail(segment, [], thinker_done=False)


def test_prompt_segment_rejects_a_prompt_without_an_assistant_header() -> None:
    builder = _builder(_fake_model())
    payload = _payload()
    payload.data["prompt"]["input_ids"] = torch.tensor(
        [IM_START, USER, 101, NEWLINE], dtype=torch.long
    )

    with pytest.raises(ValueError, match="no trailing assistant segment"):
        builder.build_prompt_segment(payload)


def test_prompt_segment_is_a_frozen_record() -> None:
    builder = _builder(_fake_model())
    segment = builder.build_prompt_segment(_payload())

    assert isinstance(segment, TalkerPromptSegment)
    assert segment.prompt_rows == segment.input_ids.shape[0]
    with pytest.raises(Exception):
        segment.speaker_id = 3


def _two_phase_scheduler(prebuilder, *, active=True):
    from concurrent.futures import ThreadPoolExecutor

    from sglang_omni.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler

    scheduler = object.__new__(QwenTalkerScheduler)
    if active:
        scheduler._two_phase_prefill = True
        scheduler._prompt_segment_prebuilder = prebuilder
        scheduler._prompt_segment_executor = ThreadPoolExecutor(max_workers=1)
        scheduler._prompt_segment_futures = {}
    return scheduler


def test_deferred_payload_prebuilds_the_prompt_segment_once() -> None:
    """Re-deferring a payload on every stream chunk must not requeue the build."""
    calls: list[str] = []
    scheduler = _two_phase_scheduler(lambda payload: calls.append(payload.request_id))
    payload = _payload()

    scheduler._prebuild_deferred_payload(payload)
    scheduler._prebuild_deferred_payload(payload)
    scheduler._prompt_segment_executor.shutdown(wait=True)

    assert calls == [payload.request_id]
    assert payload._talker_prompt_segment_future is not None


def test_prebuild_stays_off_without_the_two_phase_flag() -> None:
    scheduler = _two_phase_scheduler(lambda payload: None, active=False)
    payload = _payload()

    scheduler._prebuild_deferred_payload(payload)

    assert getattr(payload, "_talker_prompt_segment_future", None) is None


def test_releasing_a_payload_drops_its_prebuild_future() -> None:
    """Abort and admission share this path; neither may leak the future map."""
    scheduler = _two_phase_scheduler(lambda payload: None)
    payload = _payload()
    scheduler._prebuild_deferred_payload(payload)

    scheduler._release_prebuilt_payload(payload.request_id)
    scheduler._prompt_segment_executor.shutdown(wait=True)

    assert scheduler._prompt_segment_futures == {}


def test_releasing_an_unknown_payload_is_a_no_op() -> None:
    scheduler = _two_phase_scheduler(lambda payload: None, active=False)

    scheduler._release_prebuilt_payload("never-seen")


def _completed_future(fn):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn)


def test_prefill_consumes_the_prebuilt_segment_and_matches_the_whole_build() -> None:
    """The flag must move where the prompt rows are built, not what they are."""
    from sglang_omni.models.qwen3_omni.request_builders import _talker_prompt_prefill

    model = _fake_model()
    builder = _builder(model)
    chunks = _chunks([200, 201, 202, 203, 204, IM_END])

    expected = builder.build_prompt_prefill(_payload(), chunks, thinker_done=True)

    payload = _payload()
    payload._talker_prompt_segment_future = _completed_future(
        lambda: builder.build_prompt_segment(payload)
    )
    actual = _talker_prompt_prefill(
        payload, chunks, prefill_builder=builder, thinker_done=True
    )

    assert torch.equal(actual["input_embeds"], expected["input_embeds"])
    assert torch.equal(actual["input_ids"], expected["input_ids"])
    assert payload._talker_prompt_segment_future is None


def test_prefill_falls_back_when_the_prebuild_raised() -> None:
    """A prebuild bug must cost latency, never a request."""
    from sglang_omni.models.qwen3_omni.request_builders import _talker_prompt_prefill

    builder = _builder(_fake_model())
    chunks = _chunks([200, 201, 202, 203, 204, IM_END])
    payload = _payload()
    expected = builder.build_prompt_prefill(_payload(), chunks, thinker_done=True)

    def _boom():
        raise RuntimeError("prebuild exploded")

    payload._talker_prompt_segment_future = _completed_future(_boom)
    actual = _talker_prompt_prefill(
        payload, chunks, prefill_builder=builder, thinker_done=True
    )

    assert torch.equal(actual["input_embeds"], expected["input_embeds"])


def test_prefill_falls_back_when_the_tail_cannot_be_split() -> None:
    from sglang_omni.models.qwen3_omni.request_builders import _talker_prompt_prefill

    builder = _builder(_fake_model())
    chunks = _chunks([200, IM_START, 201])
    payload = _payload()
    expected = builder.build_prompt_prefill(_payload(), chunks, thinker_done=False)

    payload._talker_prompt_segment_future = _completed_future(
        lambda: builder.build_prompt_segment(payload)
    )
    actual = _talker_prompt_prefill(
        payload, chunks, prefill_builder=builder, thinker_done=False
    )

    assert torch.equal(actual["input_embeds"], expected["input_embeds"])
