# SPDX-License-Identifier: Apache-2.0
"""Unit tests for raw PCM streaming timing in the TTS serving HTTP client."""

from __future__ import annotations

import asyncio

from benchmarks.tts_serving.http_client import _handle_speech_raw_pcm_stream
from benchmarks.tts_serving.metrics import ScenarioResult


class _FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunks(self):
        for chunk in self._chunks:
            yield chunk, True


class _FakeResponse:
    def __init__(self, headers: dict[str, str], chunks: list[bytes]) -> None:
        self.status = 200
        self.headers = headers
        self.content = _FakeContent(chunks)


def _new_result() -> ScenarioResult:
    return ScenarioResult(
        scenario_id="s",
        endpoint="speech_stream_audio",
        category="speech_stream_audio",
        capability_key="speech.stream_audio",
        response_format="pcm",
    )


def test_raw_pcm_stream_records_ttfa_on_first_body_chunk() -> None:
    # 24 kHz mono 16-bit PCM => 48000 bytes/s. Two chunks of 4800 bytes => 0.2s audio.
    chunk = b"\x01\x00" * 2400
    response = _FakeResponse(
        {"x-sample-rate": "24000", "x-channels": "1", "x-bit-depth": "16"},
        [chunk, chunk],
    )
    result = _new_result()
    start = 0.0

    asyncio.run(_handle_speech_raw_pcm_stream(response, result, start))

    assert result.success is True
    assert result.ttfa_s is not None
    assert result.ttfa_s > 0.0
    assert result.audio_bytes == 2 * len(chunk)
    assert abs(result.audio_duration_s - 0.2) < 1e-6


def test_raw_pcm_stream_marks_error_on_empty_body() -> None:
    response = _FakeResponse(
        {"x-sample-rate": "24000", "x-channels": "1", "x-bit-depth": "16"},
        [],
    )
    result = _new_result()

    asyncio.run(_handle_speech_raw_pcm_stream(response, result, 0.0))

    assert result.success is False
    assert result.audio_duration_s == 0.0
