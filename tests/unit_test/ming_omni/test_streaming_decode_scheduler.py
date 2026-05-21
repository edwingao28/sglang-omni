# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Ming streaming decode scheduler."""

from __future__ import annotations

import queue
import sys
import threading
import types

import pytest
import torch

from sglang_omni.models.ming_omni.components.streaming_decode_scheduler import (
    MingStreamingDecodeScheduler,
)
from sglang_omni.models.ming_omni.io import PipelineState
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self._tokens = {1: "h", 2: "i", 3: "!", 4: " ", 5: "o", 6: "k"}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = [
            int(token_id)
            for token_id in ids
            if not skip_special_tokens or int(token_id) != self.eos_token_id
        ]
        return "".join(
            self._tokens.get(token_id, f"<{token_id}>") for token_id in tokens
        )


def _scheduler(stage_name: str = "decode") -> MingStreamingDecodeScheduler:
    return MingStreamingDecodeScheduler(
        tokenizer=FakeTokenizer(),
        eos_token_id=FakeTokenizer.eos_token_id,
        stage_name=stage_name,
    )


def _payload(
    request_id: str = "req",
    *,
    stream: bool = True,
    output_ids: list[int] | None = None,
    extra_state: dict | None = None,
) -> StagePayload:
    state = PipelineState(
        thinker_out={
            "output_ids": output_ids if output_ids is not None else [1, 2, 3],
            "step": 3,
            "is_final": True,
            "extra_model_outputs": {},
        }
    ).to_dict()
    if extra_state:
        state.update(extra_state)
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={}, params={"stream": stream}, metadata={}),
        data=state,
    )


def _chunk(token_id: int | torch.Tensor, chunk_id: int = 0) -> StreamItem:
    return StreamItem(
        chunk_id=chunk_id,
        data=token_id,
        from_stage="thinker",
    )


def _start_scheduler(scheduler: MingStreamingDecodeScheduler) -> threading.Thread:
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    return thread


def _next_out(
    scheduler: MingStreamingDecodeScheduler,
    *,
    timeout: float = 1.0,
) -> OutgoingMessage:
    return scheduler.outbox.get(timeout=timeout)


def _assert_no_outbox_message(
    scheduler: MingStreamingDecodeScheduler,
    *,
    timeout: float = 0.05,
) -> None:
    with pytest.raises(queue.Empty):
        scheduler.outbox.get(timeout=timeout)


def test_streaming_token_chunks_emit_text_deltas() -> None:
    scheduler = _scheduler()
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        scheduler.inbox.put(
            IncomingMessage("req", "stream_chunk", _chunk(torch.tensor(1)))
        )
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(2, 1)))

        first = _next_out(scheduler)
        second = _next_out(scheduler)

        assert first == OutgoingMessage(
            request_id="req",
            type="stream",
            target=None,
            data={"text": "h", "modality": "text", "stage_name": "decode"},
            metadata={"modality": "text"},
        )
        assert second == OutgoingMessage(
            request_id="req",
            type="stream",
            target=None,
            data={"text": "i", "modality": "text", "stage_name": "decode"},
            metadata={"modality": "text"},
        )
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_stream_done_before_new_request_is_latched_until_payload_arrives() -> None:
    scheduler = _scheduler()
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(1)))
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        _assert_no_outbox_message(scheduler)

        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        stream_msg = _next_out(scheduler)
        result_msg = _next_out(scheduler)

        assert stream_msg.type == "stream"
        assert stream_msg.data["text"] == "h"
        assert result_msg.type == "result"
        assert result_msg.request_id == "req"
        assert result_msg.data.request_id == "req"
        assert result_msg.data.data["events"][0]["type"] == "text_final"
        assert result_msg.data.data["modality"] == "text"
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_streaming_final_result_omits_duplicate_text_but_keeps_events_and_modality() -> (
    None
):
    scheduler = _scheduler()
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(1)))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(2, 1)))
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))

        assert _next_out(scheduler).type == "stream"
        assert _next_out(scheduler).type == "stream"
        result = _next_out(scheduler)

        assert result.type == "result"
        assert "text" not in result.data.data
        assert result.data.data["modality"] == "text"
        assert result.data.data["events"] == [
            {
                "type": "text_final",
                "modality": "text",
                "payload": {"text": "hi!"},
                "is_final": True,
            }
        ]
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_non_streaming_request_emits_no_streams_and_final_result_includes_full_text() -> (
    None
):
    scheduler = _scheduler()
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(
            IncomingMessage(
                "req",
                "new_request",
                _payload(stream=False, output_ids=[1, 2, 0]),
            )
        )

        msg = _next_out(scheduler)

        assert msg.type == "result"
        assert msg.data.data["text"] == "hi"
        assert msg.data.data["modality"] == "text"
        assert msg.data.data["events"][0]["payload"] == {"text": "hi"}
        _assert_no_outbox_message(scheduler)
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_abort_drops_buffered_state_and_done_latch() -> None:
    scheduler = _scheduler()
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        scheduler.inbox.put(
            IncomingMessage("barrier", "new_request", _payload("barrier"))
        )
        scheduler.inbox.put(IncomingMessage("barrier", "stream_done"))

        assert _next_out(scheduler).request_id == "barrier"

        scheduler.abort("req")
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))

        _assert_no_outbox_message(scheduler)
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_malformed_chunk_emits_error_and_scheduler_keeps_running() -> None:
    scheduler = _scheduler()
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(
            IncomingMessage(
                "bad",
                "stream_chunk",
                _chunk(torch.tensor([1, 2], dtype=torch.int64)),
            )
        )

        error_msg = _next_out(scheduler)
        assert error_msg.request_id == "bad"
        assert error_msg.type == "error"
        assert isinstance(error_msg.data, RuntimeError)

        scheduler.inbox.put(IncomingMessage("good", "new_request", _payload("good")))
        scheduler.inbox.put(IncomingMessage("good", "stream_chunk", _chunk(6)))

        stream_msg = _next_out(scheduler)
        assert stream_msg.request_id == "good"
        assert stream_msg.type == "stream"
        assert stream_msg.data["text"] == "k"
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_factory_uses_ming_tokenizer_and_returns_scheduler_contract(
    monkeypatch,
) -> None:
    from sglang_omni.models.ming_omni import stages

    tokenizer = FakeTokenizer()

    def fake_load_ming_tokenizer(model_path: str) -> FakeTokenizer:
        assert model_path == "fake-model"
        return tokenizer

    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.ming_omni.components.common",
        types.SimpleNamespace(load_ming_tokenizer=fake_load_ming_tokenizer),
    )

    scheduler = stages.create_streaming_decode_scheduler(
        "fake-model", stage_name="custom_decode"
    )

    assert isinstance(scheduler, MingStreamingDecodeScheduler)
    assert scheduler.stage_name == "custom_decode"
    assert isinstance(scheduler.inbox, queue.Queue)
    assert isinstance(scheduler.outbox, queue.Queue)
