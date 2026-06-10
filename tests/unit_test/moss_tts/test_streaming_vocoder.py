# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS streaming vocoder tests.

All tests are CPU-only and drive the scheduler hooks synchronously in the real
pipeline order (chunks -> stream_done -> terminal payload). The headline
assertions are deterministic equivalence: concatenated streamed PCM must equal
the non-streaming ``_vocode`` waveform of the same terminal payload, and the
online frame/segment view must equal ``split_moss_audio_segments`` over the
EOS delayed codes.
"""

from __future__ import annotations

import copy
import queue
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from sglang_omni.models.moss_tts import stages
from sglang_omni.models.moss_tts.codec import (
    apply_de_delay_pattern,
    split_moss_audio_segments,
)
from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.models.moss_tts.request_builders import (
    apply_sglang_moss_tts_result,
    build_moss_stream_metadata,
)
from sglang_omni.models.moss_tts.streaming_vocoder import (
    MossStreamingVocoderScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage

PAD_CODE = 8
TEXT_FILLER = 2
AUDIO_START = 10
AUDIO_END = 11
GEN_SLOT = 12
DELAY_SLOT = 13

SMALL_STREAM_KWARGS = dict(
    stream_stride=8,
    stream_followup_stride=3,
    stream_overlap_frames=4,
    stream_holdback_frames=2,
)


class FakeMossProcessor:
    """Deterministic position-free fake codec.

    Each frame decodes to ``samples_per_frame`` samples derived from the frame
    itself and its ``receptive_field`` predecessors *within the decoded
    segment* (distance-weighted, so values do not depend on the frame's index
    inside the window). ``tail_samples`` appends a per-decode codec tail.
    """

    def __init__(
        self,
        *,
        samples_per_frame: int = 5,
        receptive_field: int = 0,
        tail_samples: int = 0,
    ) -> None:
        self.samples_per_frame = samples_per_frame
        self.receptive_field = receptive_field
        self.tail_samples = tail_samples
        self.model_config = SimpleNamespace(
            sampling_rate=24000,
            audio_pad_code=PAD_CODE,
            downsample_rate=samples_per_frame,
        )
        self.audio_tokenizer = SimpleNamespace(
            config=SimpleNamespace(sampling_rate=24000)
        )
        self.decode_windows: list[torch.Tensor] = []

    def decode_audio_codes(self, segments: list[Any]) -> list[torch.Tensor]:
        return [self._decode_one(segment) for segment in segments]

    def _decode_one(self, segment: Any) -> torch.Tensor:
        codes = torch.as_tensor(segment, dtype=torch.long)
        self.decode_windows.append(codes.detach().clone())
        pieces: list[torch.Tensor] = []
        offsets = torch.arange(self.samples_per_frame, dtype=torch.float32) / 100.0
        for t in range(int(codes.shape[0])):
            context = codes[max(0, t - self.receptive_field) : t + 1].to(torch.float32)
            weights = torch.tensor(
                [
                    0.5 ** (int(context.shape[0]) - 1 - j)
                    for j in range(context.shape[0])
                ],
                dtype=torch.float32,
            )
            base = float((context.sum(dim=1) * weights).sum())
            pieces.append(base + offsets)
        body = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.float32)
        if self.tail_samples:
            last = float(codes[-1].sum()) if codes.shape[0] else 0.0
            tail = (
                1000.0
                + last
                + torch.arange(self.tail_samples, dtype=torch.float32) / 10.0
            )
            body = torch.cat([body, tail])
        return body


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        audio_pad_code=PAD_CODE,
        audio_start_token_id=AUDIO_START,
        audio_end_token_id=AUDIO_END,
        audio_assistant_gen_slot_token_id=GEN_SLOT,
        audio_assistant_delay_slot_token_id=DELAY_SLOT,
    )


def _member_frames(count: int, *, n_vq: int = 4, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, PAD_CODE, (count, n_vq), generator=generator)


def _pad_frames(count: int, *, n_vq: int = 4) -> torch.Tensor:
    return torch.full((count, n_vq), PAD_CODE, dtype=torch.long)


def _apply_moss_delay(raw_FN: torch.Tensor) -> torch.Tensor:
    frames, n_vq = raw_FN.shape
    delayed = torch.full((frames + n_vq - 1, n_vq), PAD_CODE, dtype=torch.long)
    for channel in range(n_vq):
        delayed[channel : channel + frames, channel] = raw_FN[:, channel]
    return delayed


def _build_rows(
    raw_FN: torch.Tensor,
    *,
    lead_text_rows: int = 2,
    include_audio_end: bool = True,
    drop_tail_rows: int = 0,
) -> torch.Tensor:
    """Full [L, n_vq + 1] row matrix as the engine would emit it."""
    n_vq = int(raw_FN.shape[1])
    delayed = _apply_moss_delay(raw_FN)
    if drop_tail_rows:
        delayed = delayed[: delayed.shape[0] - drop_tail_rows]
    rows: list[list[int]] = []
    for _ in range(lead_text_rows):
        rows.append([TEXT_FILLER] + [PAD_CODE] * n_vq)
    rows.append([AUDIO_START] + [PAD_CODE] * n_vq)
    for index in range(int(delayed.shape[0])):
        slot = GEN_SLOT if index % 2 == 0 else DELAY_SLOT
        rows.append([slot] + delayed[index].tolist())
    if include_audio_end:
        rows.append([AUDIO_END] + [PAD_CODE] * n_vq)
    return torch.tensor(rows, dtype=torch.long)


def _build_two_span_rows(span1_frames: int, span2_frames: int) -> torch.Tensor:
    """Gen-slot-led first span (no audio_start) then a second audio_start span."""
    n_vq = 4

    def slot_rows(raw_FN: torch.Tensor) -> list[list[int]]:
        delayed = _apply_moss_delay(raw_FN)
        return [
            [GEN_SLOT if index % 2 == 0 else DELAY_SLOT] + delayed[index].tolist()
            for index in range(int(delayed.shape[0]))
        ]

    rows = slot_rows(_member_frames(span1_frames, seed=21))
    rows.append([AUDIO_END] + [PAD_CODE] * n_vq)
    rows.append([TEXT_FILLER] + [PAD_CODE] * n_vq)
    rows.append([AUDIO_START] + [PAD_CODE] * n_vq)
    rows += slot_rows(_member_frames(span2_frames, seed=22))
    rows.append([AUDIO_END] + [PAD_CODE] * n_vq)
    return torch.tensor(rows, dtype=torch.long)


def _terminal_payload(
    full_rows: torch.Tensor,
    prefix_len: int,
    *,
    params: dict[str, Any],
    request_id: str = "req",
) -> StagePayload:
    n_vq = int(full_rows.shape[1]) - 1
    prefix = full_rows[:prefix_len].clone() if prefix_len else None
    state = MossTTSState()
    state.assistant_start_length = prefix_len
    data = SimpleNamespace(
        state=state,
        assistant_prefix_rows=prefix,
        output_rows=[row.clone() for row in full_rows[prefix_len:]],
        model_config=_cfg(),
        prompt_rows=torch.zeros((1, n_vq + 1), dtype=torch.long),
        input_ids=list(range(5)),
        engine_start_s=time.perf_counter(),
    )
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="", params=dict(params)),
        data={},
    )
    return apply_sglang_moss_tts_result(payload, data)


def _stream_metadata(
    *,
    n_vq: int = 4,
    prefix_rows: torch.Tensor | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = MossTTSState()
    state.assistant_start_length = (
        int(prefix_rows.shape[0]) if prefix_rows is not None else 0
    )
    data = SimpleNamespace(
        prompt_rows=torch.zeros((1, n_vq + 1), dtype=torch.long),
        state=state,
        assistant_prefix_rows=prefix_rows,
    )
    payload = StagePayload(
        request_id="req",
        request=OmniRequest(inputs="", params={"stream": True, **(params or {})}),
        data={},
    )
    metadata = build_moss_stream_metadata(payload, data, _cfg())
    assert metadata is not None
    return metadata


def _make_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    processor: FakeMossProcessor,
    **stream_kwargs: int,
) -> MossStreamingVocoderScheduler:
    monkeypatch.setattr(
        stages,
        "_load_moss_processor",
        lambda model_path, *, device, dtype: processor,
    )
    scheduler = stages.create_vocoder_executor(
        "fake-model", device="cpu", **stream_kwargs
    )
    assert isinstance(scheduler, MossStreamingVocoderScheduler)
    return scheduler


def _drain(scheduler: MossStreamingVocoderScheduler) -> list:
    messages = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            return messages


def _stream_item(row: torch.Tensor, metadata: dict[str, Any], chunk_id: int = 0):
    return StreamItem(
        chunk_id=chunk_id,
        data=row.clone(),
        from_stage="tts_engine",
        metadata=metadata,
    )


def _run_stream(
    scheduler: MossStreamingVocoderScheduler,
    metadata: dict[str, Any],
    generated_rows: list[torch.Tensor],
    terminal_payload: StagePayload,
    *,
    request_id: str = "req",
) -> list:
    for index, row in enumerate(generated_rows):
        scheduler._on_chunk(request_id, _stream_item(row, metadata, index))
    # (wenyao) Real pipeline order: chunks -> stream_done -> terminal payload replay.
    scheduler._on_done(request_id)
    scheduler._on_streaming_new_request(request_id, terminal_payload)
    return _drain(scheduler)


def _reference_waveform(
    scheduler: MossStreamingVocoderScheduler, terminal_payload: StagePayload
) -> np.ndarray:
    result = scheduler._fn(copy.deepcopy(terminal_payload))
    return np.frombuffer(result.data["audio_waveform"], dtype=np.float32)


def _stream_waveform(messages: list) -> np.ndarray:
    chunks = [
        np.frombuffer(message.data["audio_waveform"], dtype=np.float32)
        for message in messages
        if message.type == "stream"
    ]
    assert chunks, "expected at least one streamed audio chunk"
    return np.concatenate(chunks)


def _assert_stream_equals_reference(
    monkeypatch: pytest.MonkeyPatch,
    processor: FakeMossProcessor,
    full_rows: torch.Tensor,
    *,
    prefix_len: int = 0,
    min_stream_messages: int = 1,
    stream_kwargs: dict[str, int] | None = None,
) -> list:
    scheduler = _make_scheduler(
        monkeypatch, processor, **(stream_kwargs or SMALL_STREAM_KWARGS)
    )
    payload = _terminal_payload(full_rows, prefix_len, params={"stream": True})
    reference = _reference_waveform(scheduler, payload)
    processor.decode_windows.clear()

    metadata = _stream_metadata(
        n_vq=int(full_rows.shape[1]) - 1,
        prefix_rows=full_rows[:prefix_len].clone() if prefix_len else None,
    )
    messages = _run_stream(scheduler, metadata, list(full_rows[prefix_len:]), payload)

    stream_messages = [message for message in messages if message.type == "stream"]
    assert len(stream_messages) >= min_stream_messages
    np.testing.assert_array_equal(_stream_waveform(messages), reference)
    return messages


def test_moss_streaming_matches_non_streaming_single_segment(monkeypatch) -> None:
    messages = _assert_stream_equals_reference(
        monkeypatch,
        FakeMossProcessor(),
        _build_rows(_member_frames(30)),
        min_stream_messages=2,
    )

    result_messages = [message for message in messages if message.type == "result"]
    assert len(result_messages) == 1


def test_moss_streaming_matches_non_streaming_multi_segment_pad_gap(
    monkeypatch,
) -> None:
    raw = torch.cat(
        [_member_frames(12, seed=1), _pad_frames(5), _member_frames(10, seed=2)]
    )
    _assert_stream_equals_reference(
        monkeypatch,
        FakeMossProcessor(),
        _build_rows(raw),
        min_stream_messages=2,
    )


def test_moss_streaming_matches_non_streaming_with_incomplete_frames(
    monkeypatch,
) -> None:
    # (wenyao) A partially padded frame is "not all pad" but incomplete: it must split
    # segments exactly like split_moss_audio_segments does.
    raw = _member_frames(20, seed=3)
    raw[8, 1] = PAD_CODE
    _assert_stream_equals_reference(
        monkeypatch,
        FakeMossProcessor(),
        _build_rows(raw),
        min_stream_messages=2,
    )


def test_moss_streaming_matches_non_streaming_length_abort(monkeypatch) -> None:
    # (wenyao) No audio_end and the delay ramp is cut short: EOS resolves bounds via the
    # last gen/delay slot fallback, and the online view must agree.
    rows = _build_rows(
        _member_frames(24, seed=4), include_audio_end=False, drop_tail_rows=2
    )
    _assert_stream_equals_reference(
        monkeypatch,
        FakeMossProcessor(),
        rows,
        min_stream_messages=2,
    )


def test_moss_streaming_matches_non_streaming_with_continuation_prefix(
    monkeypatch,
) -> None:
    # (wenyao) First 8 rows (lead + audio_start + 5 slot rows) ride in stream metadata
    # as assistant_prefix_rows, exactly like continuation prompts.
    rows = _build_rows(_member_frames(26, seed=5))
    _assert_stream_equals_reference(
        monkeypatch,
        FakeMossProcessor(),
        rows,
        prefix_len=8,
        min_stream_messages=2,
    )


def test_moss_streaming_relatches_start_at_late_audio_start(monkeypatch) -> None:
    # (wenyao) EOS bounds anchor at the first audio_start even when gen-slot rows
    # precede it: a gen-slot-led prefix plus a later audio_start span must re-anchor the
    # online scan and stream only the audio_start span.
    rows = _build_two_span_rows(3, 20)
    _assert_stream_equals_reference(
        monkeypatch,
        FakeMossProcessor(),
        rows,
        prefix_len=6,  # span-1 slot rows only: no audio_start in the prefix
        min_stream_messages=2,
    )


def test_moss_streaming_errors_on_audio_start_after_emission(monkeypatch) -> None:
    # (wenyao) Audio already streamed under the gen-slot anchor cannot match EOS bounds
    # that re-anchor at a later audio_start: the chunk handler must error out instead of
    # delivering divergent audio.
    rows = _build_two_span_rows(16, 8)
    prefix_len = 6
    scheduler = _make_scheduler(monkeypatch, FakeMossProcessor(), **SMALL_STREAM_KWARGS)
    metadata = _stream_metadata(prefix_rows=rows[:prefix_len].clone())

    with pytest.raises(RuntimeError, match="audio_start"):
        for index, row in enumerate(rows[prefix_len:]):
            scheduler._on_chunk("req", _stream_item(row, metadata, index))

    assert scheduler._stream_states["req"].has_emitted
    assert any(message.type == "stream" for message in _drain(scheduler))


def test_moss_streaming_matches_non_streaming_with_codec_tail(monkeypatch) -> None:
    raw = torch.cat(
        [_member_frames(14, seed=6), _pad_frames(4), _member_frames(9, seed=7)]
    )
    _assert_stream_equals_reference(
        monkeypatch,
        FakeMossProcessor(tail_samples=3),
        _build_rows(raw),
        min_stream_messages=2,
    )


def test_moss_streaming_matches_non_streaming_with_receptive_field(
    monkeypatch,
) -> None:
    # (wenyao) K-frame receptive field with overlap >= K: the overlap window must give
    # every emitted frame its full left context (a 1-frame fake cannot prove the seam
    # handling).
    processor = FakeMossProcessor(receptive_field=4)
    _assert_stream_equals_reference(
        monkeypatch,
        processor,
        _build_rows(_member_frames(36, seed=8)),
        min_stream_messages=3,
    )
    # (wenyao) The streamed decode actually used trimmed mid-segment windows.
    assert len(processor.decode_windows) >= 3


@pytest.mark.parametrize(
    "branch",
    ["bounds_resolved", "length_abort_fallback", "continuation_prefix"],
)
def test_moss_streaming_online_segments_match_eos_split(monkeypatch, branch) -> None:
    raw = torch.cat(
        [_member_frames(13, seed=9), _pad_frames(6), _member_frames(11, seed=10)]
    )
    if branch == "length_abort_fallback":
        rows = _build_rows(raw, include_audio_end=False, drop_tail_rows=1)
    else:
        rows = _build_rows(raw)
    prefix_len = 8 if branch == "continuation_prefix" else 0
    processor = FakeMossProcessor()
    scheduler = _make_scheduler(monkeypatch, processor, **SMALL_STREAM_KWARGS)
    payload = _terminal_payload(rows, prefix_len, params={"stream": True})
    metadata = _stream_metadata(
        prefix_rows=rows[:prefix_len].clone() if prefix_len else None
    )

    for index, row in enumerate(rows[prefix_len:]):
        scheduler._on_chunk("req", _stream_item(row, metadata, index))

    state = scheduler._stream_states["req"]
    eos_state = MossTTSState.from_dict(payload.data)
    assert eos_state.assistant_start_length == 0
    delayed = torch.as_tensor(eos_state.delayed_audio_codes, dtype=torch.long)
    expected_frames = apply_de_delay_pattern(delayed)
    online_frames = torch.stack(state.frame_values, dim=0)
    assert torch.equal(online_frames[: expected_frames.shape[0]], expected_frames)

    expected_segments = split_moss_audio_segments(
        delayed,
        audio_pad_code=PAD_CODE,
        assistant_start_length=eos_state.assistant_start_length,
    )
    assert len(state.segments) == len(expected_segments)
    for segment, expected in zip(state.segments, expected_segments):
        online = torch.stack(
            state.frame_values[segment.start_frame : segment.end_frame], dim=0
        )
        assert torch.equal(online, expected)

    reference = _reference_waveform(scheduler, payload)
    scheduler._on_done("req")
    scheduler._on_streaming_new_request("req", payload)
    np.testing.assert_array_equal(_stream_waveform(_drain(scheduler)), reference)


def test_moss_streaming_emits_compact_chunks_and_slim_final(monkeypatch) -> None:
    rows = _build_rows(_member_frames(20, seed=11))
    processor = FakeMossProcessor()
    scheduler = _make_scheduler(monkeypatch, processor, **SMALL_STREAM_KWARGS)
    payload = _terminal_payload(rows, 0, params={"stream": True})
    generated = list(rows)

    messages = _run_stream(scheduler, _stream_metadata(), generated, payload)

    stream_messages = [message for message in messages if message.type == "stream"]
    assert stream_messages
    first_chunk = stream_messages[0].data
    assert set(first_chunk) == {
        "audio_waveform",
        "audio_waveform_shape",
        "audio_waveform_dtype",
        "sample_rate",
        "modality",
    }
    assert first_chunk["sample_rate"] == 24000
    assert first_chunk["modality"] == "audio"
    assert first_chunk["audio_waveform_dtype"] == "float32"
    assert stream_messages[0].metadata == {"modality": "audio"}

    result_messages = [message for message in messages if message.type == "result"]
    assert len(result_messages) == 1
    final_data = result_messages[0].data.data
    assert "audio_waveform" not in final_data
    assert final_data["modality"] == "audio"
    assert final_data["sample_rate"] == 24000
    usage = final_data["usage"]
    assert usage["prompt_tokens"] == 5
    assert usage["completion_tokens"] == len(generated)
    assert usage["total_tokens"] == 5 + len(generated)
    assert usage["engine_time_s"] > 0

    assert scheduler._stream_states == {}
    assert scheduler._stream_payloads == {}
    assert "req" not in scheduler._pending_done


def _real_dims_rows() -> torch.Tensor:
    return _build_rows(_member_frames(60, n_vq=32, seed=12))


def test_moss_initial_codec_chunk_frames_controls_first_chunk(monkeypatch) -> None:
    rows = _real_dims_rows()
    processor = FakeMossProcessor()
    scheduler = _make_scheduler(monkeypatch, processor)  # default 40/25/16/4
    params = {"stream": True, "initial_codec_chunk_frames": 1}
    payload = _terminal_payload(rows, 0, params=params)
    metadata = _stream_metadata(n_vq=32, params={"initial_codec_chunk_frames": 1})
    assert metadata["initial_codec_chunk_frames"] == 1

    emissions: list[tuple[int, int]] = []
    for index, row in enumerate(rows):
        scheduler._on_chunk("req", _stream_item(row, metadata, index))
        for message in _drain(scheduler):
            samples = int(
                np.frombuffer(message.data["audio_waveform"], dtype=np.float32).size
            )
            emissions.append((index + 1, samples))

    # (wenyao) start_row = 3 (two lead rows + audio_start). First decode fires at 32
    # payload rows (row 35) and emits exactly 1 frame; the next emission waits for the
    # followup boundary at payload row max(32, 40) + 25 = 65 (row 68) and emits frames
    # 1..(65 - 31 - 4) = 29 frames.
    assert emissions[0] == (35, 1 * 5)
    assert emissions[1] == (68, 29 * 5)

    reference = _reference_waveform(scheduler, payload)
    scheduler._on_done("req")
    scheduler._on_streaming_new_request("req", payload)
    tail_samples = sum(
        np.frombuffer(message.data["audio_waveform"], dtype=np.float32).size
        for message in _drain(scheduler)
        if message.type == "stream"
    )
    body_samples = sum(samples for _, samples in emissions)
    assert body_samples + tail_samples == reference.size


def test_moss_default_first_chunk_waits_for_stride(monkeypatch) -> None:
    rows = _real_dims_rows()
    processor = FakeMossProcessor()
    scheduler = _make_scheduler(monkeypatch, processor)  # default 40/25/16/4
    metadata = _stream_metadata(n_vq=32)

    emissions: list[tuple[int, int]] = []
    for index, row in enumerate(rows):
        scheduler._on_chunk("req", _stream_item(row, metadata, index))
        for message in _drain(scheduler):
            samples = int(
                np.frombuffer(message.data["audio_waveform"], dtype=np.float32).size
            )
            emissions.append((index + 1, samples))

    # (wenyao) Without the param the first decode waits for stream_stride=40 payload
    # rows (row 43) and emits (40 - 31) - holdback(4) = 5 frames.
    assert emissions[0] == (43, 5 * 5)


def test_moss_streaming_never_emitted_falls_back_to_full_decode(monkeypatch) -> None:
    # (wenyao) Too short to cross the first decode boundary: the terminal payload is
    # vocoded by the unchanged non-streaming closure and sent as one chunk.
    rows = _build_rows(_member_frames(3, seed=13))
    processor = FakeMossProcessor()
    scheduler = _make_scheduler(monkeypatch, processor, **SMALL_STREAM_KWARGS)
    payload = _terminal_payload(rows, 0, params={"stream": True})
    reference = _reference_waveform(scheduler, payload)

    messages = _run_stream(scheduler, _stream_metadata(), list(rows), payload)

    stream_messages = [message for message in messages if message.type == "stream"]
    assert len(stream_messages) == 1
    np.testing.assert_array_equal(_stream_waveform(messages), reference)
    result_messages = [message for message in messages if message.type == "result"]
    assert len(result_messages) == 1
    assert "audio_waveform" not in result_messages[0].data.data


def test_moss_streaming_done_before_payload_replays_via_pending_done(
    monkeypatch,
) -> None:
    rows = _build_rows(_member_frames(16, seed=14))
    scheduler = _make_scheduler(monkeypatch, FakeMossProcessor(), **SMALL_STREAM_KWARGS)
    payload = _terminal_payload(rows, 0, params={"stream": True})
    metadata = _stream_metadata()

    for index, row in enumerate(rows):
        scheduler._on_chunk("req", _stream_item(row, metadata, index))
    scheduler._on_done("req")
    assert "req" in scheduler._pending_done
    assert not [message for message in _drain(scheduler) if message.type == "result"]

    scheduler._on_streaming_new_request("req", payload)
    messages = _drain(scheduler)
    assert [message.type for message in messages][-1] == "result"
    assert "req" not in scheduler._pending_done
    assert scheduler._stream_states == {}


def test_moss_streaming_abort_clears_state_and_suppresses_output(monkeypatch) -> None:
    rows = _build_rows(_member_frames(20, seed=15))
    scheduler = _make_scheduler(monkeypatch, FakeMossProcessor(), **SMALL_STREAM_KWARGS)
    metadata = _stream_metadata()

    for index, row in enumerate(rows[:6]):
        scheduler._on_chunk("req", _stream_item(row, metadata, index))
    assert "req" in scheduler._stream_states

    scheduler.abort("req")
    assert "req" not in scheduler._stream_states
    assert scheduler._is_aborted("req")

    # (wenyao) In-flight chunks after abort produce no outbox messages.
    for index, row in enumerate(rows):
        scheduler._on_chunk("req", _stream_item(row, metadata, index))
    assert _drain(scheduler) == []

    # (wenyao) Re-using the request id un-flags the abort on the next new_request.
    payload = _terminal_payload(rows, 0, params={"stream": True})
    scheduler._on_streaming_new_request("req", payload)
    assert not scheduler._is_aborted("req")


def test_moss_streaming_metadata_contract_validation(monkeypatch) -> None:
    rows = _build_rows(_member_frames(10, seed=16))
    scheduler = _make_scheduler(monkeypatch, FakeMossProcessor(), **SMALL_STREAM_KWARGS)
    metadata = _stream_metadata()

    scheduler._on_chunk("req", _stream_item(rows[0], metadata, 0))

    changed = dict(metadata)
    changed["n_vq"] = 5
    with pytest.raises(ValueError, match="contract changed"):
        scheduler._on_chunk("req", _stream_item(rows[1], changed, 1))

    partial = dict(metadata)
    del partial["audio_pad_code"]
    partial["n_vq"] = 5
    with pytest.raises(ValueError, match="contract changed"):
        scheduler._on_chunk("req", _stream_item(rows[1], partial, 1))

    no_stream = dict(metadata)
    no_stream["stream"] = False
    with pytest.raises(RuntimeError, match="stream"):
        scheduler._on_chunk("req", _stream_item(rows[1], no_stream, 1))

    bad_modality = dict(metadata)
    bad_modality["modality"] = "audio_codes"
    with pytest.raises(ValueError, match="modality"):
        scheduler._on_chunk("req", _stream_item(rows[1], bad_modality, 1))

    with pytest.raises(ValueError, match="channels"):
        scheduler._on_chunk(
            "req",
            StreamItem(
                chunk_id=1,
                data=torch.zeros(3, dtype=torch.long),
                from_stage="tts_engine",
                metadata=metadata,
            ),
        )

    with pytest.raises(TypeError, match="torch.Tensor"):
        scheduler._on_chunk(
            "req",
            StreamItem(
                chunk_id=1,
                data=[0, 1, 2, 3, 4],
                from_stage="tts_engine",
                metadata=metadata,
            ),
        )

    with pytest.raises(RuntimeError, match="missing metadata fields"):
        scheduler._on_chunk(
            "other",
            StreamItem(
                chunk_id=0,
                data=rows[0].clone(),
                from_stage="tts_engine",
                metadata={"modality": "moss_delayed_audio_row", "stream": True},
            ),
        )

    bad_params = StagePayload(
        request_id="bad",
        request=OmniRequest(inputs="", params=None),
        data={},
    )
    with pytest.raises(TypeError, match="params"):
        scheduler.is_streaming_payload(bad_params)


def test_moss_non_streaming_requests_use_stage_closures_unchanged(
    monkeypatch,
) -> None:
    rows_a = _build_rows(_member_frames(12, seed=17))
    rows_b = _build_rows(_member_frames(15, seed=18))
    scheduler = _make_scheduler(monkeypatch, FakeMossProcessor(), **SMALL_STREAM_KWARGS)
    payload_a = _terminal_payload(rows_a, 0, params={"stream": False}, request_id="a")
    payload_b = _terminal_payload(rows_b, 0, params={}, request_id="b")
    expected_a = scheduler._fn(copy.deepcopy(payload_a))
    expected_b = scheduler._fn(copy.deepcopy(payload_b))

    assert scheduler.is_streaming_payload(payload_a) is False
    assert scheduler.is_streaming_payload(payload_b) is False

    scheduler._handle_new_request_batch(
        [
            IncomingMessage(request_id="a", type="new_request", data=payload_a),
            IncomingMessage(request_id="b", type="new_request", data=payload_b),
        ]
    )

    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["result", "result"]
    by_id = {message.request_id: message.data for message in messages}
    assert by_id["a"].data["audio_waveform"] == expected_a.data["audio_waveform"]
    assert by_id["b"].data["audio_waveform"] == expected_b.data["audio_waveform"]
    assert by_id["a"].data["usage"]["completion_tokens"] == len(rows_a)
    assert scheduler._stream_states == {}


def test_moss_streaming_decode_windows_are_bounded(monkeypatch) -> None:
    rows = _build_rows(_member_frames(80, seed=19))
    processor = FakeMossProcessor()
    scheduler = _make_scheduler(monkeypatch, processor, **SMALL_STREAM_KWARGS)
    payload = _terminal_payload(rows, 0, params={"stream": True})
    metadata = _stream_metadata()

    for index, row in enumerate(rows):
        scheduler._on_chunk("req", _stream_item(row, metadata, index))
        state = scheduler._stream_states["req"]
        assert state.scanned_rows == index + 1
    scheduler._on_done("req")
    scheduler._on_streaming_new_request("req", payload)
    _drain(scheduler)

    bound = (
        SMALL_STREAM_KWARGS["stream_overlap_frames"]
        + SMALL_STREAM_KWARGS["stream_followup_stride"]
        + SMALL_STREAM_KWARGS["stream_holdback_frames"]
        + 4  # n_vq
    )
    assert processor.decode_windows
    assert max(int(window.shape[0]) for window in processor.decode_windows) <= bound
