# SPDX-License-Identifier: Apache-2.0
"""Streaming text segmenter scheduler for Ming streaming TTS."""

from __future__ import annotations

import logging
import queue as _queue_mod
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sglang_omni.models.ming_omni.components.streaming_text import (
    SegmenterConfig,
    SegmenterState,
    TextSegment,
    TokenCountFn,
    text_to_uint8_tensor,
    uint8_tensor_to_text,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

DEFAULT_TARGET_STAGE = "talker_stream"

_DONE_SEEN_MAX = 10000
_DONE_SEEN_EVICT_TO = 5000
_ABORTED_SEEN_MAX = 10000
_ABORTED_SEEN_EVICT_TO = 5000
_FINISHED_SEEN_MAX = 10000
_FINISHED_SEEN_EVICT_TO = 5000
_TIMEOUT_POLL_SECONDS = 0.1


def _fallback_token_count(text: str) -> int:
    return len(text.split()) or len(text)


@dataclass
class _RequestState:
    segmenter: SegmenterState
    payload: StagePayload | None = None
    done: bool = False
    pending_text: list[str] = field(default_factory=list)
    segment_count: int = 0
    first_text_ms: int | None = None
    first_emit_ms: int | None = None
    audio_enabled: bool | None = None


class MingStreamingSegmenterScheduler:
    """Current scheduler-style text segmenter for Ming streaming TTS."""

    def __init__(
        self,
        config: SegmenterConfig | None = None,
        token_count_fn: TokenCountFn | None = None,
        target_stage: str = DEFAULT_TARGET_STAGE,
        stage_name: str = "segmenter",
        now_ms_fn: Callable[[], int] | None = None,
    ) -> None:
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self._config = config or SegmenterConfig()
        self._token_count_fn = token_count_fn or _fallback_token_count
        self.target_stage = target_stage
        self.stage_name = stage_name
        self._now_ms_fn = now_ms_fn or self._default_now_ms
        self._running = False
        self._state: dict[str, _RequestState] = {}
        self._done_seen: OrderedDict[str, None] = OrderedDict()
        self._aborted_seen: OrderedDict[str, None] = OrderedDict()
        self._finished_seen: OrderedDict[str, None] = OrderedDict()

    def start(self) -> None:
        self._running = True
        while self._running:
            self._emit_first_segment_timeouts()
            try:
                msg = self.inbox.get(timeout=self._queue_timeout_seconds())
            except _queue_mod.Empty:
                continue

            try:
                if self._is_inactive(msg.request_id):
                    continue
                if msg.type == "new_request":
                    self._on_new_request(msg.request_id, msg.data)
                elif msg.type == "stream_chunk":
                    self._on_stream_chunk(msg.request_id, msg.data)
                elif msg.type == "stream_done":
                    self._on_stream_done(msg.request_id)
                else:
                    raise ValueError(f"Unexpected message type: {msg.type!r}")
            except Exception as exc:
                logger.exception(
                    "MingStreamingSegmenterScheduler failed request %s",
                    msg.request_id,
                )
                self.abort(msg.request_id)
                self.outbox.put(
                    OutgoingMessage(
                        request_id=msg.request_id,
                        type="error",
                        data=exc,
                    )
                )

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        self._state.pop(request_id, None)
        self._done_seen.pop(request_id, None)
        self._remember_aborted(request_id)

    def _ensure_state(self, request_id: str) -> _RequestState:
        state = self._state.get(request_id)
        if state is None:
            state = _RequestState(
                segmenter=SegmenterState(self._config, self._token_count_fn)
            )
            self._state[request_id] = state
        return state

    def _on_new_request(self, request_id: str, payload: StagePayload) -> None:
        if not isinstance(payload, StagePayload):
            raise TypeError("new_request data must be a StagePayload")

        state = self._ensure_state(request_id)
        state.payload = payload
        state.audio_enabled = self._payload_includes_audio(payload)
        if request_id in self._done_seen:
            state.done = True
            self._done_seen.pop(request_id, None)

        if not state.audio_enabled:
            state.pending_text.clear()
            self._finalize(request_id, aborted=False)
            return

        for text in list(state.pending_text):
            self._push_text(request_id, state, text)
        state.pending_text.clear()

        if state.done or not self._payload_is_streaming(payload):
            self._finalize(request_id, aborted=False)

    def _on_stream_chunk(self, request_id: str, item: Any) -> None:
        if not isinstance(item, StreamItem):
            raise TypeError(f"Unexpected stream item type: {type(item)!r}")

        text = uint8_tensor_to_text(item.data)
        state = self._ensure_state(request_id)
        if state.payload is None:
            state.pending_text.append(text)
            return
        if not state.audio_enabled:
            return
        self._push_text(request_id, state, text)

    def _on_stream_done(self, request_id: str) -> None:
        state = self._state.get(request_id)
        if state is None:
            self._remember_done(request_id)
            return
        state.done = True
        if state.payload is not None:
            self._finalize(request_id, aborted=False)

    def _push_text(self, request_id: str, state: _RequestState, text: str) -> None:
        now_ms = self._now_ms()
        if (
            text
            and state.segment_count == 0
            and state.first_text_ms is None
            and state.segmenter.buffer_token_count() == 0
        ):
            state.first_text_ms = now_ms

        for segment in state.segmenter.push(text, now_ms=now_ms):
            self._emit_segment(request_id, state, segment)

    def _finalize(self, request_id: str, *, aborted: bool) -> None:
        state = self._state.pop(request_id, None)
        self._done_seen.pop(request_id, None)
        if state is None or state.payload is None:
            return
        self._remember_finished(request_id)

        if state.audio_enabled and not aborted:
            for segment in state.segmenter.flush():
                self._emit_segment(request_id, state, segment)

        self._set_payload_result(
            state.payload,
            segment_count=state.segment_count if state.audio_enabled else 0,
            aborted=aborted,
        )
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=state.payload,
            )
        )

    def _emit_segment(
        self, request_id: str, state: _RequestState, segment: TextSegment
    ) -> None:
        if not segment.text.strip():
            if state.segmenter.buffer_token_count() == 0:
                state.first_text_ms = None
            return

        data = text_to_uint8_tensor(segment.text)
        now_ms = self._now_ms()
        metadata: dict[str, Any] = {
            "modality": "text",
            "stage_name": self.stage_name,
            "segment_id": segment.segment_id,
            "is_final_segment": bool(segment.is_final_segment),
            "text_len": int(data.numel()),
        }
        if state.first_emit_ms is None:
            state.first_emit_ms = now_ms
            metadata["segmenter_first_emit_ms"] = now_ms

        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=data,
                target=self.target_stage,
                metadata=metadata,
            )
        )
        state.segment_count += 1
        if state.segment_count > 0:
            state.first_text_ms = None

    def _emit_first_segment_timeouts(self) -> None:
        for request_id, state in list(self._state.items()):
            if not self._first_segment_timeout_ready(state):
                continue
            try:
                for segment in state.segmenter.push("", now_ms=self._now_ms()):
                    self._emit_segment(request_id, state, segment)
            except Exception as exc:
                logger.exception(
                    "MingStreamingSegmenterScheduler timeout failed request %s",
                    request_id,
                )
                self.abort(request_id)
                self.outbox.put(
                    OutgoingMessage(
                        request_id=request_id,
                        type="error",
                        data=exc,
                    )
                )

    def _first_segment_timeout_ready(self, state: _RequestState) -> bool:
        if state.payload is None or not state.audio_enabled:
            return False
        if state.segment_count != 0 or state.first_text_ms is None:
            return False
        if state.segmenter.buffer_token_count() < self._config.first_segment_min_tokens:
            return False
        elapsed_ms = self._now_ms() - state.first_text_ms
        return elapsed_ms >= self._config.first_segment_max_wait_ms

    def _queue_timeout_seconds(self) -> float:
        remaining_ms: int | None = None
        now_ms = self._now_ms()
        for state in self._state.values():
            if (
                state.payload is None
                or not state.audio_enabled
                or state.segment_count != 0
                or state.first_text_ms is None
                or state.segmenter.buffer_token_count()
                < self._config.first_segment_min_tokens
            ):
                continue
            state_remaining_ms = max(
                self._config.first_segment_max_wait_ms - (now_ms - state.first_text_ms),
                0,
            )
            remaining_ms = (
                state_remaining_ms
                if remaining_ms is None
                else min(remaining_ms, state_remaining_ms)
            )

        if remaining_ms is None:
            return _TIMEOUT_POLL_SECONDS
        return min(_TIMEOUT_POLL_SECONDS, remaining_ms / 1000)

    def _payload_includes_audio(self, payload: StagePayload) -> bool:
        modalities = (payload.request.metadata or {}).get("output_modalities")
        if modalities is None:
            return True
        return "audio" in modalities

    def _payload_is_streaming(self, payload: StagePayload) -> bool:
        return bool((payload.request.params or {}).get("stream", False))

    def _set_payload_result(
        self,
        payload: StagePayload,
        *,
        segment_count: int,
        aborted: bool,
    ) -> None:
        result = {
            "segment_count": segment_count,
            "aborted": aborted,
            "modality": "text",
        }
        if isinstance(payload.data, dict):
            payload.data = {**payload.data, **result}
        else:
            payload.data = result

    def _remember_done(self, request_id: str) -> None:
        self._done_seen[request_id] = None
        self._done_seen.move_to_end(request_id)
        if len(self._done_seen) > _DONE_SEEN_MAX:
            for _ in range(len(self._done_seen) - _DONE_SEEN_EVICT_TO):
                self._done_seen.popitem(last=False)

    def _remember_aborted(self, request_id: str) -> None:
        self._aborted_seen[request_id] = None
        self._aborted_seen.move_to_end(request_id)
        if len(self._aborted_seen) > _ABORTED_SEEN_MAX:
            for _ in range(len(self._aborted_seen) - _ABORTED_SEEN_EVICT_TO):
                self._aborted_seen.popitem(last=False)

    def _remember_finished(self, request_id: str) -> None:
        self._finished_seen[request_id] = None
        self._finished_seen.move_to_end(request_id)
        if len(self._finished_seen) > _FINISHED_SEEN_MAX:
            for _ in range(len(self._finished_seen) - _FINISHED_SEEN_EVICT_TO):
                self._finished_seen.popitem(last=False)

    def _is_inactive(self, request_id: str) -> bool:
        return request_id in self._aborted_seen or request_id in self._finished_seen

    def _now_ms(self) -> int:
        return int(self._now_ms_fn())

    def _default_now_ms(self) -> int:
        return int(time.monotonic() * 1000)
