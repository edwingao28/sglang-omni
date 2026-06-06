# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest
import torch

from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.models.qwen3_tts.streaming_vocoder import (
    Qwen3TTSStreamingVocoderScheduler,
)
from sglang_omni.models.tts_streaming import INITIAL_CODEC_CHUNK_FRAMES_PARAM
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload


class FakeQwen3TTSTokenizer:
    def __init__(self) -> None:
        self.decode_inputs: list[torch.Tensor] = []

    def decode(self, encoded):
        wavs = []
        for item in encoded:
            codes = torch.as_tensor(item["audio_codes"], dtype=torch.long)
            self.decode_inputs.append(codes.clone())
            frames = int(codes.shape[0])
            offset = int(codes[0, 0].item()) if frames else 0
            wavs.append(torch.arange(offset, offset + frames * 2, dtype=torch.float32))
        return wavs, 24000


def _payload(
    *,
    request_id: str = "req",
    stream: bool,
    audio_codes: torch.Tensor | None = None,
    non_streaming_mode: bool = False,
    initial_codec_chunk_frames: int | None = None,
) -> StagePayload:
    params = {"stream": stream}
    if initial_codec_chunk_frames is not None:
        params[INITIAL_CODEC_CHUNK_FRAMES_PARAM] = initial_codec_chunk_frames
    state = Qwen3TTSState(
        audio_codes=audio_codes,
        sample_rate=24000,
        non_streaming_mode=non_streaming_mode,
        prompt_tokens=2,
        completion_tokens=0 if audio_codes is None else int(audio_codes.shape[0]),
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params=params),
        data=state.to_dict(),
    )


def _item(row: torch.Tensor, *, chunk_id: int = 0) -> StreamItem:
    return StreamItem(
        from_stage="tts_engine",
        data=row,
        metadata={
            "modality": "audio_codes",
            "stream": True,
            "codec": "qwen3_tts",
            "sample_rate": 24000,
            "num_codebooks": int(row.numel()),
        },
        chunk_id=chunk_id,
    )


def _drain(scheduler: Qwen3TTSStreamingVocoderScheduler):
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    return messages


def test_qwen3_tts_streaming_vocoder_emits_incremental_audio() -> None:
    tokenizer = FakeQwen3TTSTokenizer()
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        stream_chunk_frames=3,
        left_context_frames=0,
    )
    payload = _payload(stream=True)
    rows = [
        torch.tensor([1, 11], dtype=torch.long),
        torch.tensor([2, 12], dtype=torch.long),
        torch.tensor([3, 13], dtype=torch.long),
    ]

    scheduler._on_streaming_new_request("req", payload)
    for idx, row in enumerate(rows):
        scheduler._on_chunk("req", _item(row, chunk_id=idx))

    stream_messages = [msg for msg in _drain(scheduler) if msg.type == "stream"]
    assert len(stream_messages) == 1
    first = stream_messages[0]
    assert first.metadata == {"modality": "audio"}
    assert first.data["sample_rate"] == 24000
    audio = np.frombuffer(first.data["audio_waveform"], dtype=np.float32)
    assert audio.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert tokenizer.decode_inputs[0].tolist() == [[1, 11], [2, 12], [3, 13]]


def test_qwen3_tts_initial_codec_chunk_frames_controls_first_audio() -> None:
    tokenizer = FakeQwen3TTSTokenizer()
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        stream_chunk_frames=4,
        left_context_frames=0,
    )
    payload = _payload(stream=True, initial_codec_chunk_frames=1)

    scheduler._on_streaming_new_request("req", payload)
    scheduler._on_chunk("req", _item(torch.tensor([7, 17], dtype=torch.long)))

    stream_messages = [msg for msg in _drain(scheduler) if msg.type == "stream"]
    assert len(stream_messages) == 1
    audio = np.frombuffer(stream_messages[0].data["audio_waveform"], dtype=np.float32)
    assert audio.tolist() == [7.0, 8.0]


def test_qwen3_tts_streaming_vocoder_final_result_is_metadata_only() -> None:
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        FakeQwen3TTSTokenizer(),
        stream_chunk_frames=2,
        left_context_frames=0,
    )
    payload = _payload(stream=True)

    scheduler._on_streaming_new_request("req", payload)
    scheduler._on_chunk("req", _item(torch.tensor([1, 11], dtype=torch.long)))
    scheduler._on_chunk("req", _item(torch.tensor([2, 12], dtype=torch.long)))
    scheduler._on_done("req")

    results = [msg for msg in _drain(scheduler) if msg.type == "result"]
    assert len(results) == 1
    final_data = results[0].data.data
    assert final_data["modality"] == "audio"
    assert final_data["sample_rate"] == 24000
    assert "audio_waveform" not in final_data
    assert final_data["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 0,
        "total_tokens": 2,
    }


def test_qwen3_tts_non_streaming_vocoder_batches_decode_requests() -> None:
    tokenizer = FakeQwen3TTSTokenizer()
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        max_batch_size=2,
        max_batch_wait_ms=3,
    )
    first = _payload(
        request_id="first",
        stream=False,
        audio_codes=torch.tensor([[1, 11], [2, 12]], dtype=torch.long),
    )
    second = _payload(
        request_id="second",
        stream=False,
        audio_codes=torch.tensor([[5, 15], [6, 16]], dtype=torch.long),
    )

    results = scheduler._batch_fn([first, second])

    assert scheduler._max_batch_size == 2
    assert scheduler._max_batch_wait_s == pytest.approx(0.003)
    assert len(tokenizer.decode_inputs) == 2
    assert results[0].data["sample_rate"] == 24000
    first_audio = np.frombuffer(results[0].data["audio_waveform"], dtype=np.float32)
    second_audio = np.frombuffer(results[1].data["audio_waveform"], dtype=np.float32)
    assert first_audio.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert second_audio.tolist() == [5.0, 6.0, 7.0, 8.0]
