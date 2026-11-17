# SPDX-License-Identifier: Apache-2.0
"""Loopback OpenAI-compatible and realtime services for the cascade tests."""

import base64
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import yaml
from websockets.sync.server import serve

from sglang_omni.models.streaming_cascade.config import StreamingCascadePipelineConfig

SPEECH_SAMPLES = np.array([0, 16384, -16384], dtype="<i2")
CHAT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# note (wenyao): a warm-up prompt of more words than this is long: it is over 64 tokens, and a chat service that checkpoints its state every 64 tokens takes another prefill kernel for it.
SHORT_PROMPT_MAX_WORDS = 64
# note (wenyao): a warm-up batch needs this many prompts in flight at once, so that each of 2 data-parallel ranks still prefills 3 together after prefilling one alone; a model with 32 state heads needs 3 sequences for the fused prefill kernel on an H200.
WARM_UP_BATCH_MIN_PROMPTS = 8


@pytest.fixture
def backend_requests():
    return []


@pytest.fixture(params=["streaming"])
def backends(request, backend_requests):
    """Loopback transcription, chat SSE, streaming speech and forced alignment on one HTTP server.

    In the streaming scenario the chat holds its reply until the
    speech service has sent audio, so a cascade that waits for the whole reply fails.
    """
    scenario = request.param
    audio_sent = threading.Event()
    received_audio, received_words, received_messages = [], [], []
    batching = threading.Condition()
    batched_kinds: set[str] = set()
    in_flight = {"short": 0, "long": 0}
    warm_up_prompts: set[str] = set()

    class Handler(BaseHTTPRequestHandler):
        def send(self, content_type: str, body: bytes = b"", **headers) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def do_GET(self):
            if self.path == "/health":
                self.send("application/json", b'{"status": "ready"}')
            else:
                assert self.path == "/v1/models"
                self.send("application/json", json.dumps({"data": []}).encode())

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/v1/audio/transcriptions":
                assert b'name="file"' in body and b"RIFF" in body
                received_audio.append(body)
                self.send("application/json", b'{"text": " test question "}')
            elif self.path == "/v1/chat/completions":
                self.chat(json.loads(body))
            elif self.path == "/v1/audio/speech":
                request = json.loads(body)
                assert request["response_format"] == "pcm" and request["stream"] is True
                reference = {"audio_path": "/tmp/ref.wav", "text": "ref"}
                assert request["references"] == [reference]
                received_words.append(request["input"])
                if scenario == "english_tts":
                    spoken_chinese = any(
                        "\u4e00" <= char <= "\u9fff" for char in request["input"]
                    )
                    assert request["language"] == (
                        "Chinese" if spoken_chinese else "English"
                    )
                else:
                    assert "language" not in request
                labeled = scenario != "unlabeled_rate"
                rate = {"X-Sample-Rate": "24000"} if labeled else {}
                pcm = SPEECH_SAMPLES.tobytes()
                if scenario == "long_audio":
                    pcm *= 8000
                else:
                    pass
                self.send("audio/pcm", pcm[:3], **rate)
                self.wfile.write(pcm[3:])
                audio_sent.set()
            elif self.path == "/align" and scenario == "failed_align":
                self.send_response(500)
                self.end_headers()
            elif self.path == "/align":
                backend_requests.append(json.loads(body))
                self.send("application/json", b'{"boundaries": []}')
            else:
                self.send_response(404)
                self.end_headers()

        def chat(self, request: dict) -> None:
            assert request["model"] == CHAT_MODEL
            if request.get("min_tokens", 0) > request["max_tokens"]:
                self.send_response(400)
                self.end_headers()
                return
            else:
                pass
            backend_requests.append(request)
            self.send("text/event-stream")
            if request["max_tokens"] == 1:
                prompt = request["messages"][-1]["content"]
                finish_reason = {
                    "failed_probe": "abort",
                    "length_chat": "length",
                }.get(scenario, "stop")
                with nullcontext() if prompt == "Hi" else self.batched(prompt):
                    self.event({"delta": {"content": "Hi"}, "finish_reason": None})
                    self.event({"delta": {}, "finish_reason": finish_reason})
                self.wfile.write(b"data: [DONE]\n\n")
                return
            else:
                received_messages.append(request["messages"])
            deltas = [{"content": text} for text in ("Hel", "lo ", "world.")]
            finish_reason = {"failed_chat": "abort", "length_chat": "length"}.get(
                scenario, "stop"
            )
            for delta in deltas:
                self.event({"delta": delta, "finish_reason": None})
                if (
                    delta.get("content") == "Hel"
                    and scenario == "slow_first"
                    and len(received_messages) == 1
                ):
                    time.sleep(0.5)
                elif delta.get("content") == "lo " and scenario == "streaming":
                    assert audio_sent.wait(3), "TTS must start before LLM finishes"
                else:
                    pass
            self.event({"delta": {}, "finish_reason": finish_reason})
            self.wfile.write(b"data: [DONE]\n\n")

        @contextmanager
        def batched(self, prompt: str) -> Iterator[None]:
            """Answer the first warm-up prompts of a kind, short or long, only once WARM_UP_BATCH_MIN_PROMPTS of that kind are in flight, and refuse a repeated prompt and one that meets the other kind in flight. A prompt is in flight until the last line of its answer."""
            if len(prompt.split()) > SHORT_PROMPT_MAX_WORDS:
                kind, other = "long", "short"
            else:
                kind, other = "short", "long"
            with batching:
                assert prompt not in warm_up_prompts, "a warm-up prompt must be new"
                warm_up_prompts.add(prompt)
                assert not in_flight[
                    other
                ], f"a {kind} warm-up prompt must not be in flight with a {other} one"
                in_flight[kind] += 1
                batching.notify_all()
                assert batching.wait_for(
                    lambda: in_flight[kind] >= WARM_UP_BATCH_MIN_PROMPTS
                    or kind in batched_kinds,
                    3,
                ), f"{WARM_UP_BATCH_MIN_PROMPTS} {kind} warm-up prompts must be in flight at once"
                batched_kinds.add(kind)
            yield
            with batching:
                in_flight[kind] -= 1

        def event(self, choice: dict) -> None:
            data = json.dumps({"choices": [choice]})
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()

        def log_message(self, *arguments):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    settings = loopback_graph(f"http://127.0.0.1:{server.server_port}/v1")
    try:
        yield settings, received_audio, received_words, received_messages
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.fixture
def realtime_transcription_server():
    """A realtime transcription service that says one word per two appended frames."""
    appended: list[tuple[int, bytes]] = []
    client_events: list[str] = []
    spoken = ["hello", "there", "general", "kenobi"]

    def transcribe(websocket):
        path = "/v1/realtime?model=nvidia%2Fnemotron-3.5-asr-streaming-0.6b"
        assert websocket.request.path == path

        def send(kind: str, **fields) -> None:
            websocket.send(json.dumps({"type": kind, **fields}))

        send("session.created", event_id="e0")
        for encoded in websocket:
            event = json.loads(encoded)
            assert event["event_id"]
            client_events.append(event["type"])
            if event["type"] == "session.update":
                assert event["session"] == {"output_modalities": ["text"]}
                send("session.updated")
            elif event["type"] == "input_audio_buffer.append":
                seq = event["sglang"]["seq"]
                appended.append((seq, base64.b64decode(event["audio"])))
                send("sglang.input_audio.accepted", seq=seq)
                if len(appended) % 2 == 0:
                    word = spoken[len(appended) // 2 - 1]
                    delta = word if len(appended) == 2 else " " + word
                    send("response.output_text.delta", delta=delta)
                else:
                    pass
            elif event["type"] == "sglang.input_audio.end":
                text = " ".join(spoken[: len(appended) // 2])
                send("response.output_text.done", text=text)
                for kind in ("response.done", "sglang.input_audio.ended"):
                    send(kind)
                send("sglang.input_audio.drained")
            elif event["type"] == "session.close":
                send("session.closed")
                return
            else:
                raise AssertionError(event)

    with serve(transcribe, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            port = server.socket.getsockname()[1]
            yield f"ws://127.0.0.1:{port}", appended, client_events
        finally:
            server.shutdown()
            thread.join(5)


@pytest.fixture
def transcription_session_server():
    """A transcription-intent session that refreshes its partial every two appended frames and names the language in its final transcript."""
    appended: list[bytes] = []
    client_events: list[dict[str, object]] = []
    spoken = ["hello", "there", "general", "kenobi"]

    def transcribe(websocket):
        path = "/v1/realtime?intent=transcription&model=Qwen%2FQwen3-ASR-1.7B"
        assert websocket.request.path == path

        def send(kind: str, **fields) -> None:
            websocket.send(json.dumps({"type": kind, **fields}))

        send("session.created", session={"id": "s0", "decode_interval_ms": 2000})
        for encoded in websocket:
            event = json.loads(encoded)
            assert event["event_id"]
            client_events.append(event)
            if event["type"] == "session.update":
                send("session.updated", session={"id": "s0", **event["session"]})
            elif event["type"] == "input_audio_buffer.append":
                appended.append(base64.b64decode(event["audio"]))
                if len(appended) % 2 == 0:
                    text = " ".join(spoken[: len(appended) // 2])
                    send(
                        "transcription.segment",
                        segment_id=0,
                        audio_start_ms=0,
                        audio_end_ms=len(appended) * 20,
                        text=text,
                        is_final=False,
                        language="English",
                    )
                else:
                    pass
            elif event["type"] == "transcription.done":
                text = " ".join(spoken[: len(appended) // 2])
                send("input_audio_buffer.committed", segment_id=0)
                send(
                    "transcription.segment",
                    segment_id=0,
                    audio_start_ms=0,
                    audio_end_ms=len(appended) * 20,
                    text=text,
                    is_final=True,
                    language="English",
                )
                send("transcription.completed", text=text, language="English")
            else:
                raise AssertionError(event)

    with serve(transcribe, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            port = server.socket.getsockname()[1]
            yield f"ws://127.0.0.1:{port}", appended, client_events
        finally:
            server.shutdown()
            thread.join(5)


LOOPBACK_GRAPH = """
session_nodes:
  asr:
    kind: openai_transcription
    inputs: [input]
    base_url: URL
    model: openai/whisper-large-v3
    timeout_s: 5
  dialog:
    kind: dialog
    inputs: ["input:session", {kind: control, source: input, event: interrupt}, asr, {source: "llm:text_delta", feedback: true, overflow: drop_oldest, capacity: 1024}, {source: "tts:session", feedback: true, overflow: drop_oldest, capacity: 1024}, {kind: control, source: tts, event: reply.progress}]
  llm:
    kind: openai_chat
    inputs: [dialog, "input:session", {kind: control, source: dialog, event: interrupt}]
    base_url: URL
    model: Qwen/Qwen3-4B-Instruct-2507
    timeout_s: 5
  tts:
    kind: openai_speech
    inputs: [llm, "input:session", {kind: control, source: dialog, event: interrupt}]
    base_url: URL
    model: Qwen/Qwen3-TTS-12Hz-0.6B-Base
    reference_audio_path: /tmp/ref.wav
    reference_text: ref
    max_sentence_words: 1
    timeout_s: 5
session_publish:
  {asr: [transcript], dialog: [session], llm: [text_delta], tts: [pcm_f32le, session]}
"""


def loopback_graph(base_url: str) -> StreamingCascadePipelineConfig:
    """A commit-driven cascade of the OpenAI-compatible kinds, every one served at `base_url`."""
    declaration = yaml.safe_load(LOOPBACK_GRAPH.replace("URL", base_url))
    return StreamingCascadePipelineConfig(
        model_path="loopback", timeout_s=5, **declaration
    )
