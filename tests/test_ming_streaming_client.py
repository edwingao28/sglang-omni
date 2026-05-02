# SPDX-License-Identifier: Apache-2.0
"""Tests for Ming streaming TTS client behavior."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from typing import Any

import numpy as np

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk, GenerateRequest
from sglang_omni.proto import StreamMessage
from sglang_omni.serve.openai_api import _chat_stream


class _FakeStreamingCoordinator:
    def __init__(self, *, segmenter_first_emit_ms: float | None = 42.0):
        self._segmenter_first_emit_ms = segmenter_first_emit_ms
        self.aborted_request_ids: list[str] = []

    async def abort(self, request_id: str) -> bool:
        self.aborted_request_ids.append(request_id)
        return True

    async def stream(self, request_id: str, request: Any):
        yield StreamMessage(
            request_id=request_id,
            from_stage="talker",
            chunk=GenerateChunk(
                request_id=request_id,
                modality="audio",
                audio_data=np.zeros(4410, dtype=np.float32),
                sample_rate=44100,
                stage_name="talker_stream",
                talker_queue_depth=2,
                segmenter_first_emit_ms=self._segmenter_first_emit_ms,
            ),
        )
        yield StreamMessage(
            request_id=request_id,
            from_stage="talker",
            chunk=GenerateChunk(request_id=request_id, finish_reason="stop"),
        )


class _FakeStreamingCoordinatorWithEmptyAudio:
    def __init__(self) -> None:
        self.aborted_request_ids: list[str] = []

    async def abort(self, request_id: str) -> bool:
        self.aborted_request_ids.append(request_id)
        return True

    async def stream(self, request_id: str, request: Any):
        yield StreamMessage(
            request_id=request_id,
            from_stage="thinker",
            chunk=GenerateChunk(request_id=request_id, modality="text", text="hi"),
        )
        yield StreamMessage(
            request_id=request_id,
            from_stage="talker",
            chunk=GenerateChunk(
                request_id=request_id,
                modality="audio",
                audio_data=np.zeros(0, dtype=np.float32),
                sample_rate=44100,
                stage_name="talker_stream",
            ),
        )
        yield StreamMessage(
            request_id=request_id,
            from_stage="talker",
            chunk=GenerateChunk(
                request_id=request_id,
                modality="audio",
                audio_data=np.zeros(4410, dtype=np.float32),
                sample_rate=44100,
                stage_name="talker_stream",
            ),
        )
        yield StreamMessage(
            request_id=request_id,
            from_stage="decode",
            chunk=GenerateChunk(request_id=request_id, finish_reason="stop"),
        )


class _FakeMingTalkerMergedCoordinator:
    async def submit(self, request_id: str, request: Any):
        waveform = np.zeros(2205, dtype=np.float32)
        return {
            "decode": {
                "text": "The sun rises in the east.",
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 7,
                    "total_tokens": 10,
                },
            },
            "talker": {
                "audio_waveform": waveform.tobytes(),
                "audio_waveform_dtype": str(waveform.dtype),
                "audio_waveform_shape": list(waveform.shape),
                "sample_rate": 44100,
            },
        }


async def _collect_stream_chunks():
    client = Client(_FakeStreamingCoordinator())
    chunks = []
    async for chunk in client.completion_stream(
        GenerateRequest(prompt="hello", stream=True),
        request_id="req-sr",
    ):
        chunks.append(chunk)
    return chunks


def _wav_sample_rate(audio_b64: str) -> int:
    wav = base64.b64decode(audio_b64)
    return struct.unpack("<I", wav[24:28])[0]


def _wav_frame_count(audio_b64: str) -> int:
    wav = base64.b64decode(audio_b64)
    data_size = struct.unpack("<I", wav[40:44])[0]
    return data_size // 2


def test_completion_stream_uses_chunk_sample_rate_for_wav_header():
    chunks = asyncio.run(_collect_stream_chunks())

    audio_chunk = next(chunk for chunk in chunks if chunk.audio_b64)

    assert _wav_sample_rate(audio_chunk.audio_b64) == 44100
    assert audio_chunk.sample_rate == 44100
    assert audio_chunk.talker_queue_depth == 2


def test_completion_builds_audio_from_ming_talker_merged_result():
    client = Client(_FakeMingTalkerMergedCoordinator())

    result = asyncio.run(
        client.completion(
            GenerateRequest(prompt="hello", stream=False),
            request_id="req-ming-nonstream-audio",
        )
    )

    assert result.audio is not None
    assert result.audio.transcript == "The sun rises in the east."
    assert _wav_sample_rate(result.audio.data) == 44100
    assert _wav_frame_count(result.audio.data) == 2205
    assert result.usage is not None
    assert result.usage.total_tokens == 10


def test_chat_stream_emits_audio_observability_extensions():
    coordinator = _FakeStreamingCoordinator()
    client = Client(coordinator)
    request = type("Req", (), {"modalities": ["text", "audio"]})()
    gen = _chat_stream(
        client,
        GenerateRequest(prompt="hello", stream=True),
        "req-observe",
        "chatcmpl",
        0,
        "ming",
        request,
        "wav",
    )

    async def run():
        first = await gen.__anext__()
        await gen.aclose()
        return json.loads(first[len("data: ") :])

    payload = asyncio.run(run())

    assert payload["stage_name"] == "talker_stream"
    assert payload["sample_rate"] == 44100
    assert payload["talker_queue_depth"] == 2
    assert payload["stage_times_ms"]["talker_first_audio"] >= 0
    assert payload["stage_times_ms"]["segmenter_first_emit"] == 42.0
    assert coordinator.aborted_request_ids == ["req-observe"]


def test_chat_stream_suppresses_empty_audio_chunks():
    client = Client(_FakeStreamingCoordinatorWithEmptyAudio())
    request = type("Req", (), {"modalities": ["text", "audio"]})()
    gen = _chat_stream(
        client,
        GenerateRequest(prompt="hello", stream=True),
        "req-empty-audio",
        "chatcmpl",
        0,
        "ming",
        request,
        "wav",
    )

    async def run():
        events = []
        async for raw in gen:
            payload = raw[len("data: ") :].strip()
            events.append(payload)
            if payload == "[DONE]":
                break
        return events

    events = asyncio.run(run())
    payloads = [json.loads(event) for event in events if event != "[DONE]"]
    audio_chunks = [
        choice["delta"]["audio"]["data"]
        for payload in payloads
        for choice in payload["choices"]
        if "audio" in choice["delta"]
    ]

    assert [_wav_frame_count(audio) for audio in audio_chunks] == [4410]
    assert events[-1] == "[DONE]"
