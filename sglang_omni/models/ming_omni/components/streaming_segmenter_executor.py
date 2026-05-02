# SPDX-License-Identifier: Apache-2.0
"""Streaming text segmenter executor for Ming-Omni TTS."""

from __future__ import annotations

import asyncio
import collections
import time
from dataclasses import dataclass
from typing import Any

from sglang_omni.executors.interface import Executor
from sglang_omni.models.ming_omni.components.streaming_text import (
    CompletedResult,
    SegmenterConfig,
    SegmenterState,
    TextSegment,
    TokenCountFn,
    is_done_signal,
    text_to_uint8_tensor,
    uint8_tensor_to_text,
)
from sglang_omni.models.ming_omni.pipeline.next_stage import TALKER_STREAM_STAGE
from sglang_omni.pipeline.stage.stream_queue import StreamItem, StreamQueue, StreamSignal
from sglang_omni.proto import StagePayload


@dataclass
class _RequestState:
    payload: StagePayload
    segmenter: SegmenterState
    segment_count: int = 0
    first_text_ms: int | None = None


def _fallback_token_count(text: str) -> int:
    return len(text.split()) or len(text)


_FIRST_SEGMENT_TIMEOUT = object()


class MingStreamingSegmenterExecutor(Executor):
    """Consume streamed thinker text and fan out speakable text segments."""

    def __init__(
        self,
        *,
        config: SegmenterConfig | None = None,
        token_count_fn: TokenCountFn | None = None,
    ) -> None:
        self._config = config or SegmenterConfig()
        self._token_count_fn = token_count_fn or _fallback_token_count
        self._stream_queue: StreamQueue | None = None
        self._stream_fn: Any | None = None
        self._results: asyncio.Queue[CompletedResult] = asyncio.Queue()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._states: dict[str, _RequestState] = {}
        self._aborted: set[str] = set()
        self._pre_aborted: collections.OrderedDict[str, None] = (
            collections.OrderedDict()
        )
        self._max_pre_aborted = 4096

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        for request_id, task in list(self._tasks.items()):
            self._aborted.add(request_id)
            if not task.done():
                task.cancel()
            self._close_stream_queue(request_id)
        self._tasks.clear()
        self._states.clear()

    def set_stream_fn(self, fn) -> None:
        self._stream_fn = fn

    def set_feedback_mailbox(self, mailbox: Any) -> None:
        del mailbox

    async def add_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if request_id in self._pre_aborted:
            self._pre_aborted.pop(request_id, None)
            self._set_payload_result(payload, segment_count=0, aborted=True)
            self._results.put_nowait(
                CompletedResult(request_id=request_id, payload=payload)
            )
            self._close_stream_queue(request_id)
            return
        if self._stream_queue is None:
            raise RuntimeError("Ming streaming segmenter requires a stream queue")
        if request_id in self._tasks:
            raise RuntimeError(f"Request {request_id} is already running")

        state = _RequestState(
            payload=payload,
            segmenter=SegmenterState(self._config, self._token_count_fn),
        )
        self._states[request_id] = state
        task = asyncio.create_task(self._run_request(request_id))
        self._tasks[request_id] = task

    async def get_result(self) -> StagePayload:
        completed = await self._results.get()
        self._aborted.discard(completed.request_id)
        if completed.error is not None:
            completed.error.request_id = completed.request_id
            raise completed.error
        if completed.payload is None:
            raise RuntimeError(f"Missing result payload for {completed.request_id}")
        return completed.payload

    async def abort(self, request_id: str) -> None:
        state = self._states.pop(request_id, None)
        task = self._tasks.pop(request_id, None)
        if state is None and task is None:
            self._remember_pre_aborted(request_id)
            self._close_stream_queue(request_id)
            return

        self._aborted.add(request_id)
        if task is not None and not task.done():
            task.cancel()
        self._close_stream_queue(request_id)
        if state is not None:
            self._set_payload_result(
                state.payload,
                segment_count=state.segment_count,
                aborted=True,
            )
            self._results.put_nowait(
                CompletedResult(request_id=request_id, payload=state.payload)
            )

    async def _run_request(self, request_id: str) -> None:
        state = self._states[request_id]
        try:
            while request_id not in self._aborted:
                item = await self._get_inbound_with_timeout(request_id, state)
                if item is _FIRST_SEGMENT_TIMEOUT:
                    for segment in state.segmenter.push("", now_ms=self._now_ms()):
                        self._emit_segment(request_id, segment)
                        state.segment_count += 1
                        state.first_text_ms = None
                    continue
                if is_done_signal(item):
                    break
                if isinstance(item, StreamSignal):
                    if item.error is not None:
                        raise item.error
                    continue
                if not isinstance(item, StreamItem):
                    raise TypeError(f"Unexpected stream item type: {type(item)!r}")

                text = uint8_tensor_to_text(item.data)
                now_ms = self._now_ms()
                if text and state.segment_count == 0 and state.first_text_ms is None:
                    state.first_text_ms = now_ms
                for segment in state.segmenter.push(text, now_ms=now_ms):
                    self._emit_segment(request_id, segment)
                    state.segment_count += 1
                    state.first_text_ms = None

            if request_id in self._aborted:
                return

            final_segments = state.segmenter.flush()
            if not final_segments:
                final_segments = [self._empty_final_segment(state.segment_count)]
            for segment in final_segments:
                self._emit_segment(request_id, segment)
                state.segment_count += 1

            self._set_payload_result(
                state.payload,
                segment_count=state.segment_count,
                aborted=False,
            )
            await self._results.put(
                CompletedResult(request_id=request_id, payload=state.payload)
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            exc.request_id = request_id
            await self._results.put(CompletedResult(request_id=request_id, error=exc))
        finally:
            self._tasks.pop(request_id, None)
            self._states.pop(request_id, None)
            self._close_stream_queue(request_id)

    async def _get_inbound_with_timeout(
        self, request_id: str, state: _RequestState
    ) -> StreamItem | StreamSignal | None | object:
        timeout = self._first_segment_timeout_seconds(state)
        if timeout is None:
            return await self._get_inbound(request_id)
        if timeout <= 0:
            return _FIRST_SEGMENT_TIMEOUT
        try:
            return await asyncio.wait_for(
                self._get_inbound(request_id), timeout=timeout
            )
        except asyncio.TimeoutError:
            return _FIRST_SEGMENT_TIMEOUT

    async def _get_inbound(self, request_id: str) -> StreamItem | StreamSignal | None:
        if self._stream_queue is None:
            raise RuntimeError("Ming streaming segmenter requires a stream queue")
        return await self._stream_queue.get_with_source(request_id)

    def _emit_segment(self, request_id: str, segment: TextSegment) -> None:
        if self._stream_fn is None:
            return
        data = text_to_uint8_tensor(segment.text)
        self._stream_fn(
            request_id,
            data,
            TALKER_STREAM_STAGE,
            metadata={
                "segment_id": segment.segment_id,
                "is_final_segment": bool(segment.is_final_segment),
                "text_len": int(data.numel()),
            },
        )

    def _first_segment_timeout_seconds(self, state: _RequestState) -> float | None:
        if state.segment_count != 0 or state.first_text_ms is None:
            return None
        if state.segmenter.buffer_token_count() < self._config.first_segment_min_tokens:
            return None
        elapsed_ms = self._now_ms() - state.first_text_ms
        remaining_ms = self._config.first_segment_max_wait_ms - elapsed_ms
        return max(remaining_ms, 0) / 1000

    def _close_stream_queue(self, request_id: str) -> None:
        if self._stream_queue is not None:
            self._stream_queue.close(request_id)

    def _empty_final_segment(self, segment_id: int) -> TextSegment:
        return TextSegment(segment_id=segment_id, text="", is_final_segment=True)

    def _now_ms(self) -> int:
        return int(time.monotonic() * 1000)

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
        }
        if isinstance(payload.data, dict):
            payload.data = {**payload.data, **result}
        else:
            payload.data = result

    def _remember_pre_aborted(self, request_id: str) -> None:
        self._pre_aborted[request_id] = None
        self._pre_aborted.move_to_end(request_id)
        while len(self._pre_aborted) > self._max_pre_aborted:
            self._pre_aborted.popitem(last=False)
