# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Realtime audio-output event adaptation."""

from __future__ import annotations

import json
from typing import Any

import pytest
from starlette.websockets import WebSocketState

from sglang_omni.client import CompletionStreamChunk
from sglang_omni.serve.realtime.events import SessionUpdate
from sglang_omni.serve.realtime.session import RealtimeSession


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.application_state = WebSocketState.CONNECTED
        self.client_state = WebSocketState.CONNECTED

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self) -> None:
        self.application_state = WebSocketState.DISCONNECTED
        self.client_state = WebSocketState.DISCONNECTED


class FakeRealtimeClient:
    def __init__(self, chunks: list[CompletionStreamChunk]) -> None:
        self.chunks = chunks
        self.requests: list[Any] = []
        self.audio_formats: list[str] = []
        self.aborted: list[str] = []

    async def completion_stream(
        self,
        request: Any,
        *,
        request_id: str,
        audio_format: str = "wav",
    ):
        self.requests.append(request)
        self.audio_formats.append(audio_format)
        for chunk in self.chunks:
            yield chunk

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


def _session(client: Any | None = None) -> RealtimeSession:
    return RealtimeSession(
        FakeWebSocket(),
        client=client or FakeRealtimeClient([]),
        model_name="qwen3-omni",
        session_id="sess_test",
    )


@pytest.mark.asyncio
async def test_session_update_accepts_audio_output_config_and_builds_audio_request() -> None:
    session = _session()

    await session.handle_session_update(
        SessionUpdate.model_validate(
            {
                "type": "session.update",
                "session": {
                    "modalities": ["audio"],
                    "voice": "Ethan",
                    "output_audio_format": "pcm16",
                    "instructions": "Answer briefly.",
                },
            }
        )
    )

    assert session.session_object.modalities == ["audio"]
    assert session.session_object.voice == "Ethan"
    assert session.session_object.output_audio_format == "pcm16"
    assert session._wants_audio_output() is True
    assert session._client_audio_format() == "pcm"

    request = session.build_response_request("data:audio/wav;base64,AAAA")

    assert request.output_modalities == ["text", "audio"]
    assert request.metadata["audios"] == ["data:audio/wav;base64,AAAA"]
    assert request.metadata["audio_config"] == {"voice": "Ethan"}
    assert request.extra_params["speaker"] == "Ethan"


@pytest.mark.asyncio
async def test_text_only_session_still_builds_text_request() -> None:
    session = _session()

    await session.handle_session_update(
        SessionUpdate.model_validate(
            {
                "type": "session.update",
                "session": {"modalities": ["text"]},
            }
        )
    )

    request = session.build_response_request("data:audio/wav;base64,AAAA")

    assert session._wants_audio_output() is False
    assert request.output_modalities == ["text"]
    assert "audio_config" not in request.metadata
    assert "speaker" not in request.extra_params


@pytest.mark.asyncio
async def test_text_only_session_with_voice_omits_audio_output_config() -> None:
    session = _session()

    await session.handle_session_update(
        SessionUpdate.model_validate(
            {
                "type": "session.update",
                "session": {"modalities": ["text"], "voice": "Ethan"},
            }
        )
    )

    request = session.build_response_request("data:audio/wav;base64,AAAA")

    assert request.output_modalities == ["text"]
    assert "audio_config" not in request.metadata
    assert "speaker" not in request.extra_params


def _types(events: list[dict[str, Any]]) -> list[str]:
    return [event["type"] for event in events]


@pytest.mark.asyncio
async def test_audio_response_streams_output_audio_and_transcript_events() -> None:
    client = FakeRealtimeClient(
        [
            CompletionStreamChunk(request_id="req", modality="text", text="Hello "),
            CompletionStreamChunk(request_id="req", modality="audio", audio_b64="AAEC"),
            CompletionStreamChunk(request_id="req", modality="text", text="world"),
            CompletionStreamChunk(request_id="req", modality="audio", audio_b64="AwQF"),
            CompletionStreamChunk(request_id="req", finish_reason="stop"),
        ]
    )
    session = _session(client)
    session.session_object.modalities = ["audio"]
    session.session_object.output_audio_format = "pcm16"

    response_text = await session.run_response("data:audio/wav;base64,AAAA")

    assert response_text == "Hello world"
    assert client.audio_formats == ["pcm"]
    assert client.requests[0].output_modalities == ["text", "audio"]

    events = session.websocket.sent
    event_types = _types(events)

    assert event_types == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_audio_transcript.delta",
        "response.output_audio.delta",
        "response.output_audio_transcript.delta",
        "response.output_audio.delta",
        "response.output_audio.done",
        "response.output_audio_transcript.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.done",
    ]

    audio_deltas = [
        event["delta"]
        for event in events
        if event["type"] == "response.output_audio.delta"
    ]
    assert audio_deltas == ["AAEC", "AwQF"]

    transcript_deltas = [
        event["delta"]
        for event in events
        if event["type"] == "response.output_audio_transcript.delta"
    ]
    assert transcript_deltas == ["Hello ", "world"]

    response_done = events[-1]["response"]
    assert response_done["status"] == "completed"
    assert response_done["output"][0]["content"] == [
        {"type": "audio", "transcript": "Hello world"}
    ]
    assert "data" not in response_done["output"][0]["content"][0]


@pytest.mark.asyncio
async def test_text_response_keeps_existing_text_event_names() -> None:
    client = FakeRealtimeClient(
        [
            CompletionStreamChunk(request_id="req", modality="text", text="Hi"),
            CompletionStreamChunk(request_id="req", finish_reason="stop"),
        ]
    )
    session = _session(client)
    session.session_object.modalities = ["text"]

    response_text = await session.run_response("data:audio/wav;base64,AAAA")

    assert response_text == "Hi"
    assert client.audio_formats == ["wav"]
    assert client.requests[0].output_modalities == ["text"]

    event_types = _types(session.websocket.sent)
    assert event_types == [
        "response.created",
        "response.text.delta",
        "response.text.done",
        "response.done",
    ]
    assert "response.output_audio.delta" not in event_types
