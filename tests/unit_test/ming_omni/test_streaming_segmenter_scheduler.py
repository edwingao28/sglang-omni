# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Ming streaming segmenter scheduler."""

from __future__ import annotations

import queue
import threading

import pytest
import torch

from sglang_omni.models.ming_omni.components.streaming_segmenter_scheduler import (
    MingStreamingSegmenterScheduler,
)
from sglang_omni.models.ming_omni.components.streaming_text import (
    SegmenterConfig,
    text_to_uint8_tensor,
    uint8_tensor_to_text,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage


def _count_words(text: str) -> int:
    return len(text.split()) or len(text)


def _payload(
    request_id: str = "req",
    *,
    stream: bool = True,
    modalities: list[str] | None = None,
    data: dict | None = None,
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs=[],
            params={"stream": stream},
            metadata={"output_modalities": modalities or ["audio"]},
        ),
        data=data if data is not None else {"existing": "value"},
    )


def _chunk(text: str, chunk_id: int = 0) -> StreamItem:
    return StreamItem(
        chunk_id=chunk_id,
        data=text_to_uint8_tensor(text),
        from_stage="thinker",
    )


def _start_scheduler(
    scheduler: MingStreamingSegmenterScheduler,
) -> threading.Thread:
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    return thread


def _next_out(
    scheduler: MingStreamingSegmenterScheduler,
    *,
    timeout: float = 1.0,
) -> OutgoingMessage:
    return scheduler.outbox.get(timeout=timeout)


def _assert_no_outbox_message(
    scheduler: MingStreamingSegmenterScheduler,
    *,
    timeout: float = 0.05,
) -> None:
    with pytest.raises(queue.Empty):
        scheduler.outbox.get(timeout=timeout)


def test_stream_chunk_emits_segment_stream_message_for_punctuation_text() -> None:
    scheduler = MingStreamingSegmenterScheduler(
        config=SegmenterConfig(segment_min_tokens=3),
        token_count_fn=_count_words,
        now_ms_fn=lambda: 10,
    )
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        scheduler.inbox.put(
            IncomingMessage("req", "stream_chunk", _chunk("one two three."))
        )

        msg = _next_out(scheduler)

        assert msg.type == "stream"
        assert msg.request_id == "req"
        assert msg.target == "talker_stream"
        assert uint8_tensor_to_text(msg.data) == "one two three."
        assert msg.metadata == {
            "modality": "text",
            "stage_name": "segmenter",
            "segment_id": 0,
            "is_final_segment": False,
            "text_len": int(msg.data.numel()),
            "segmenter_first_emit_ms": 10,
        }
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_stream_done_before_new_request_finalizes_after_payload_arrives() -> None:
    scheduler = MingStreamingSegmenterScheduler(
        config=SegmenterConfig(segment_min_tokens=3),
        token_count_fn=_count_words,
        now_ms_fn=lambda: 20,
    )
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(
            IncomingMessage("req", "stream_chunk", _chunk("one two three."))
        )
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        _assert_no_outbox_message(scheduler)

        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        stream_msg = _next_out(scheduler)
        result_msg = _next_out(scheduler)

        assert stream_msg.type == "stream"
        assert result_msg.type == "result"
        assert result_msg.data.request_id == "req"
        assert result_msg.data.request.metadata["output_modalities"] == ["audio"]
        assert result_msg.data.data["existing"] == "value"
        assert result_msg.data.data["segment_count"] == 1
        assert result_msg.data.data["aborted"] is False
        assert result_msg.data.data["modality"] == "text"
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_non_audio_payload_suppresses_buffered_segments_and_finalizes() -> None:
    scheduler = MingStreamingSegmenterScheduler(
        config=SegmenterConfig(segment_min_tokens=3),
        token_count_fn=_count_words,
    )
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(
            IncomingMessage("req", "stream_chunk", _chunk("one two three."))
        )
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        scheduler.inbox.put(
            IncomingMessage(
                "req",
                "new_request",
                _payload(modalities=["text"]),
            )
        )

        msg = _next_out(scheduler)

        assert msg.type == "result"
        assert msg.data.data["segment_count"] == 0
        assert msg.data.data["aborted"] is False
        assert msg.data.data["modality"] == "text"
        _assert_no_outbox_message(scheduler)
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_whitespace_only_timeout_never_emits_stream_message() -> None:
    times = [0, 0, 0]

    def now_ms() -> int:
        if times:
            return times.pop(0)
        return 2

    scheduler = MingStreamingSegmenterScheduler(
        config=SegmenterConfig(
            segment_min_tokens=8,
            segment_max_tokens=40,
            first_segment_min_tokens=4,
            first_segment_max_wait_ms=1,
        ),
        token_count_fn=_count_words,
        now_ms_fn=now_ms,
    )
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk("    ")))

        _assert_no_outbox_message(scheduler, timeout=0.2)

        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        result_msg = _next_out(scheduler)

        assert result_msg.type == "result"
        assert result_msg.data.data["segment_count"] == 0
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_abort_drops_buffered_state_and_done_latch() -> None:
    scheduler = MingStreamingSegmenterScheduler(
        config=SegmenterConfig(segment_min_tokens=3),
        token_count_fn=_count_words,
    )
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(
            IncomingMessage("req", "stream_chunk", _chunk("buffered text"))
        )
        scheduler.inbox.put(
            IncomingMessage("barrier", "new_request", _payload("barrier"))
        )
        scheduler.inbox.put(
            IncomingMessage(
                "barrier",
                "stream_chunk",
                _chunk("one two three.", chunk_id=1),
            )
        )

        barrier_msg = _next_out(scheduler)
        assert barrier_msg.request_id == "barrier"
        assert barrier_msg.type == "stream"

        scheduler.abort("req")
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))

        _assert_no_outbox_message(scheduler)
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_malformed_chunk_emits_error_and_scheduler_keeps_running() -> None:
    scheduler = MingStreamingSegmenterScheduler(
        config=SegmenterConfig(segment_min_tokens=3),
        token_count_fn=_count_words,
    )
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("bad", "new_request", _payload("bad")))
        scheduler.inbox.put(
            IncomingMessage(
                "bad",
                "stream_chunk",
                StreamItem(
                    chunk_id=0,
                    data=torch.tensor([1, 2, 3], dtype=torch.int64),
                    from_stage="thinker",
                ),
            )
        )

        error_msg = _next_out(scheduler)
        assert error_msg.request_id == "bad"
        assert error_msg.type == "error"
        assert isinstance(error_msg.data, TypeError)

        scheduler.inbox.put(IncomingMessage("good", "new_request", _payload("good")))
        scheduler.inbox.put(
            IncomingMessage(
                "good",
                "stream_chunk",
                _chunk("one two three.", chunk_id=1),
            )
        )

        stream_msg = _next_out(scheduler)
        assert stream_msg.request_id == "good"
        assert stream_msg.type == "stream"
        assert uint8_tensor_to_text(stream_msg.data) == "one two three."
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_first_segment_timeout_emits_early() -> None:
    times = [0, 0, 0]

    def now_ms() -> int:
        if times:
            return times.pop(0)
        return 2

    scheduler = MingStreamingSegmenterScheduler(
        config=SegmenterConfig(
            segment_min_tokens=8,
            segment_max_tokens=40,
            first_segment_min_tokens=4,
            first_segment_max_wait_ms=1,
        ),
        token_count_fn=_count_words,
        now_ms_fn=now_ms,
    )
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        scheduler.inbox.put(
            IncomingMessage("req", "stream_chunk", _chunk("one two three four"))
        )

        msg = _next_out(scheduler, timeout=1.0)

        assert msg.type == "stream"
        assert uint8_tensor_to_text(msg.data) == "one two three four"
        assert msg.metadata["segment_id"] == 0
        assert msg.metadata["is_final_segment"] is False
    finally:
        scheduler.stop()
        thread.join(timeout=1)
