# SPDX-License-Identifier: Apache-2.0
"""Streaming decode scheduler for Ming-Omni."""

from __future__ import annotations

import logging
import queue as _queue_mod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from sglang_omni.models.ming_omni.io import OmniEvent, PipelineState
from sglang_omni.models.ming_omni.pipeline.merge import decode_events
from sglang_omni.models.ming_omni.pipeline.next_stage import THINKER_STAGE
from sglang_omni.models.ming_omni.pipeline.state_io import load_state
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

_DONE_SEEN_MAX = 10000
_DONE_SEEN_EVICT_TO = 5000


def _event_to_dict(event: OmniEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "modality": event.modality,
        "payload": dict(event.payload),
        "is_final": bool(event.is_final),
    }


@dataclass
class _RequestState:
    stream_state: PipelineState = field(default_factory=PipelineState)
    pending_deltas: list[str] = field(default_factory=list)
    payload: StagePayload | None = None
    done: bool = False


class MingStreamingDecodeScheduler:
    """Stream-aware Ming decode stage."""

    def __init__(
        self,
        tokenizer: Any,
        eos_token_id: int | None,
        stage_name: str = "decode",
    ):
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self._tokenizer = tokenizer
        self._eos_token_id = eos_token_id
        self.stage_name = stage_name
        self._running = False
        self._state: dict[str, _RequestState] = {}
        self._done_seen: OrderedDict[str, None] = OrderedDict()

    def start(self) -> None:
        self._running = True
        while self._running:
            try:
                msg = self.inbox.get(timeout=0.1)
            except _queue_mod.Empty:
                continue

            try:
                if msg.type == "new_request":
                    self._on_new_request(msg.request_id, msg.data)
                elif msg.type == "stream_chunk":
                    self._on_stream_chunk(msg.request_id, msg.data)
                elif msg.type == "stream_done":
                    self._on_stream_done(msg.request_id)
            except Exception as exc:
                logger.exception(
                    "MingStreamingDecodeScheduler failed request %s",
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

    def _ensure_state(self, request_id: str) -> _RequestState:
        state = self._state.get(request_id)
        if state is None:
            state = _RequestState()
            self._state[request_id] = state
        return state

    def _on_stream_chunk(self, request_id: str, item: Any) -> None:
        data = item.data
        token_id = int(data.item()) if hasattr(data, "item") else int(data)
        state = self._ensure_state(request_id)

        step = len(state.stream_state.stream_state.get("token_ids", [])) + 1
        events = decode_events(
            thinker_out={
                "output_ids": [token_id],
                "step": step,
                "is_final": False,
                "extra_model_outputs": {},
            },
            state=state.stream_state,
            tokenizer=self._tokenizer,
            eos_token_id=self._eos_token_id,
            step=step,
        )
        for event in events:
            if event.type != "text_delta":
                continue
            delta = event.payload.get("text")
            if not delta:
                continue
            if state.payload is None:
                state.pending_deltas.append(delta)
            elif self._is_streaming_request(state.payload):
                self._emit_delta(request_id, delta)

    def _on_stream_done(self, request_id: str) -> None:
        state = self._state.get(request_id)
        if state is None:
            self._done_seen[request_id] = None
            if len(self._done_seen) > _DONE_SEEN_MAX:
                for _ in range(len(self._done_seen) - _DONE_SEEN_EVICT_TO):
                    self._done_seen.popitem(last=False)
            return

        state.done = True
        if state.payload is not None and self._is_streaming_request(state.payload):
            self._finalize(request_id)

    def _on_new_request(self, request_id: str, payload: StagePayload) -> None:
        state = self._ensure_state(request_id)
        state.payload = payload
        if self._is_streaming_request(payload):
            for delta in state.pending_deltas:
                self._emit_delta(request_id, delta)
        state.pending_deltas.clear()
        if request_id in self._done_seen:
            state.done = True
            self._done_seen.pop(request_id, None)
        if state.done or not self._is_streaming_request(payload):
            self._finalize(request_id)

    def _finalize(self, request_id: str) -> None:
        state = self._state.pop(request_id, None)
        self._done_seen.pop(request_id, None)
        if state is None or state.payload is None:
            return

        is_streaming = self._is_streaming_request(state.payload)
        result = self._build_result(state.payload, is_streaming=is_streaming)
        state.payload.data = result
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=state.payload,
            )
        )

    def _build_result(
        self, payload: StagePayload, *, is_streaming: bool = False
    ) -> dict[str, Any]:
        state = load_state(payload)
        thinker_out = state.thinker_out or state.engine_outputs.get(THINKER_STAGE)
        if not isinstance(thinker_out, dict):
            thinker_out = {
                "output_ids": [],
                "step": 0,
                "is_final": True,
                "extra_model_outputs": {},
            }

        step = int(thinker_out.get("step") or len(thinker_out.get("output_ids", [])))
        events = list(
            decode_events(
                thinker_out=thinker_out,
                state=state,
                tokenizer=self._tokenizer,
                eos_token_id=self._eos_token_id,
                step=step,
            )
        )
        result: dict[str, Any] = {"events": [_event_to_dict(event) for event in events]}
        final_event = next(
            (
                event
                for event in reversed(events)
                if event.is_final or event.type in {"text_final", "final"}
            ),
            None,
        )
        if final_event is not None:
            result.update(final_event.payload)
            result.setdefault("modality", final_event.modality)

        if is_streaming:
            result.pop("text", None)
        elif "text" not in result:
            output_ids = thinker_out.get("output_ids")
            if (
                callable(getattr(self._tokenizer, "decode", None))
                and isinstance(output_ids, list)
                and output_ids
            ):
                result["text"] = self._tokenizer.decode(
                    output_ids, skip_special_tokens=True
                )
                result.setdefault("modality", "text")

        return result

    def _emit_delta(self, request_id: str, delta: str) -> None:
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target=None,
                data={
                    "text": delta,
                    "modality": "text",
                    "stage_name": self.stage_name,
                },
                metadata={"modality": "text"},
            )
        )

    @staticmethod
    def _is_streaming_request(payload: StagePayload | None) -> bool:
        if payload is None:
            return False
        return bool((payload.request.params or {}).get("stream") is True)
