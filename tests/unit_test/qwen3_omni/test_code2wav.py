# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from tests.unit_test.fixtures.qwen_fakes import FakeCode2WavModel, make_qwen_payload


def _make_scheduler(model: FakeCode2WavModel) -> Code2WavScheduler:
    return Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
    )


def _feed(
    scheduler: Code2WavScheduler,
    request_id: str,
    codes: tuple[int, ...],
    *,
    stream: bool,
) -> None:
    meta = {"stream": stream}
    for i, code in enumerate(codes):
        scheduler._handle_stream_chunk(
            request_id,
            StreamItem(i, torch.tensor([code, code * 10]), "talker", metadata=meta),
        )


def test_qwen_code2wav_streams_incrementally_and_abort_clears_state() -> None:
    """Preserves incremental waveform windows and request-state cleanup on abort."""
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = _make_scheduler(model)
    scheduler._stream_payloads["req-1"] = make_qwen_payload(request_id="req-1")
    _feed(scheduler, "req-1", (1, 2, 3), stream=False)
    scheduler._on_done("req-1")

    message = scheduler.outbox.get_nowait()
    audio = np.frombuffer(message.data.data["audio_waveform"], dtype=np.float32)
    assert model.calls == [(1, 2, 2), (1, 2, 2)]
    assert audio.shape == (6,)

    scheduler._stream_payloads["req-2"] = make_qwen_payload(request_id="req-2")
    scheduler._get_or_create_stream_state("req-2")
    scheduler.abort("req-2")
    assert "req-2" not in scheduler._stream_states


def test_streaming_client_gets_stream_chunks_and_metadata_final() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = _make_scheduler(model)
    scheduler._stream_payloads["req-1"] = make_qwen_payload(request_id="req-1")
    _feed(scheduler, "req-1", (1, 2, 3), stream=True)
    scheduler._on_done("req-1")

    first = scheduler.outbox.get_nowait()
    assert first.type == "stream"
    first_audio = np.frombuffer(first.data["audio_waveform"], dtype=np.float32)
    assert first_audio.shape == (4,)

    remainder = scheduler.outbox.get_nowait()
    assert remainder.type == "stream"
    remainder_audio = np.frombuffer(remainder.data["audio_waveform"], dtype=np.float32)
    assert remainder_audio.shape == (2,)

    result = scheduler.outbox.get_nowait()
    assert result.type == "result"
    assert result.data.data == {"modality": "audio", "sample_rate": 24000}
    assert model.calls == [(1, 2, 2), (1, 2, 2)]


def test_eos_chunk_is_skipped_and_never_decoded() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = _make_scheduler(model)
    scheduler._stream_payloads["req-1"] = make_qwen_payload(request_id="req-1")
    _feed(scheduler, "req-1", (1, 2), stream=False)
    assert model.calls == [(1, 2, 2)]

    scheduler._handle_stream_chunk(
        "req-1",
        StreamItem(2, torch.tensor([2150, 0]), "talker", metadata={"stream": False}),
    )
    assert model.calls == [(1, 2, 2)]

    scheduler._on_done("req-1")
    message = scheduler.outbox.get_nowait()
    audio = np.frombuffer(message.data.data["audio_waveform"], dtype=np.float32)
    assert model.calls == [(1, 2, 2)]
    assert audio.shape == (4,)
