# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Ming streaming talker scheduler."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

import numpy as np
import pytest
import torch

from sglang_omni.models.ming_omni.components.streaming_text import (
    text_to_uint8_tensor,
)
from sglang_omni.models.ming_omni.components import (
    streaming_talker_scheduler as talker_scheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage


class FakeTalker:
    def __init__(self, outputs: list[Any] | None = None) -> None:
        self.outputs = outputs if outputs is not None else [np.array([0.1, 0.2])]
        self.calls: list[dict[str, Any]] = []

    def omni_audio_generation(
        self,
        *,
        tts_text: str,
        voice_name: str,
        audio_detokenizer: Any,
        stream: bool,
        abort_event: threading.Event,
    ):
        self.calls.append(
            {
                "tts_text": tts_text,
                "voice_name": voice_name,
                "audio_detokenizer": audio_detokenizer,
                "stream": stream,
                "abort_event": abort_event,
            }
        )
        for output in self.outputs:
            yield output


def _scheduler(
    talker: FakeTalker | None = None,
    *,
    sample_rate: int = 16000,
    now_ms_values: list[int] | None = None,
) -> talker_scheduler.MingStreamingTalkerScheduler:
    values = list(now_ms_values or [100, 125, 150, 175, 200, 225])

    def now_ms() -> int:
        if values:
            return values.pop(0)
        return 250

    return talker_scheduler.MingTalkerStreamScheduler(
        talker=talker or FakeTalker(),
        audio_detokenizer=object(),
        sample_rate=sample_rate,
        now_ms_fn=now_ms,
    )


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
        from_stage="segmenter",
        metadata={
            "segment_id": chunk_id,
            "is_final_segment": False,
            "segmenter_first_emit_ms": 42,
        },
    )


def _start_scheduler(
    scheduler: talker_scheduler.MingStreamingTalkerScheduler,
) -> threading.Thread:
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    return thread


def _next_out(
    scheduler: talker_scheduler.MingStreamingTalkerScheduler,
    *,
    timeout: float = 1.0,
) -> OutgoingMessage:
    return scheduler.outbox.get(timeout=timeout)


def _assert_no_outbox_message(
    scheduler: talker_scheduler.MingStreamingTalkerScheduler,
    *,
    timeout: float = 0.05,
) -> None:
    with pytest.raises(queue.Empty):
        scheduler.outbox.get(timeout=timeout)


def test_requested_scheduler_class_name_is_exported() -> None:
    assert talker_scheduler.MingTalkerStreamScheduler is not None


def test_start_and_non_audio_request_do_not_load_talker() -> None:
    loader_calls = 0

    def loader() -> dict[str, Any]:
        nonlocal loader_calls
        loader_calls += 1
        return {"talker": FakeTalker(), "sample_rate": 16000}

    scheduler = talker_scheduler.MingTalkerStreamScheduler(loader=loader)
    thread = _start_scheduler(scheduler)
    try:
        time.sleep(0.15)
        assert loader_calls == 0

        scheduler.inbox.put(
            IncomingMessage("req", "new_request", _payload(modalities=["text"]))
        )

        msg = _next_out(scheduler)

        assert msg.type == "result"
        assert msg.data.data["skipped"] is True
        assert loader_calls == 0
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_blocking_generation_does_not_block_scheduler_inbox_loop() -> None:
    class BlockingTalker(FakeTalker):
        def __init__(self) -> None:
            super().__init__(outputs=[])
            self.started = threading.Event()

        def omni_audio_generation(
            self,
            *,
            tts_text: str,
            voice_name: str,
            audio_detokenizer: Any,
            stream: bool,
            abort_event: threading.Event,
        ):
            self.calls.append({"tts_text": tts_text, "abort_event": abort_event})
            self.started.set()
            while not abort_event.is_set():
                time.sleep(0.005)
            yield np.array([9.0], dtype=np.float32)

    talker = BlockingTalker()
    scheduler = _scheduler(talker)
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("blocked", "new_request", _payload()))
        scheduler.inbox.put(IncomingMessage("blocked", "stream_chunk", _chunk("wait")))
        assert talker.started.wait(timeout=1)

        scheduler.inbox.put(
            IncomingMessage(
                "skip",
                "new_request",
                _payload("skip", modalities=["text"]),
            )
        )

        msg = _next_out(scheduler, timeout=0.3)
        assert msg.request_id == "skip"
        assert msg.type == "result"
        assert msg.data.data["skipped"] is True

        scheduler.abort("blocked")
        _assert_no_outbox_message(scheduler, timeout=0.2)
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_concurrent_request_workers_serialize_shared_talker_generation() -> None:
    class SerializingTalker(FakeTalker):
        def __init__(self) -> None:
            super().__init__(outputs=[])
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0
            self.calls: list[dict[str, Any]] = []
            self.first_entered = threading.Event()
            self.second_entered = threading.Event()
            self.release_first = threading.Event()

        def omni_audio_generation(
            self,
            *,
            tts_text: str,
            voice_name: str,
            audio_detokenizer: Any,
            stream: bool,
            abort_event: threading.Event,
        ):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                self.calls.append({"tts_text": tts_text})
                call_index = len(self.calls)
                if call_index == 1:
                    self.first_entered.set()
                elif call_index == 2:
                    self.second_entered.set()
            try:
                if call_index == 1:
                    self.release_first.wait(timeout=1)
                yield np.array([float(call_index)], dtype=np.float32)
            finally:
                with self.lock:
                    self.active -= 1

    talker = SerializingTalker()
    scheduler = _scheduler(talker)
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req1", "new_request", _payload("req1")))
        scheduler.inbox.put(
            IncomingMessage("req1", "stream_chunk", _chunk("one", 1))
        )
        scheduler.inbox.put(IncomingMessage("req2", "new_request", _payload("req2")))
        scheduler.inbox.put(
            IncomingMessage("req2", "stream_chunk", _chunk("two", 2))
        )

        assert talker.first_entered.wait(timeout=1)
        assert not talker.second_entered.wait(timeout=0.1)

        talker.release_first.set()
        first = _next_out(scheduler)
        second = _next_out(scheduler)

        assert {first.request_id, second.request_id} == {"req1", "req2"}
        assert {call["tts_text"] for call in talker.calls} == {"one", "two"}
        assert talker.max_active == 1
    finally:
        talker.release_first.set()
        scheduler.stop()
        thread.join(timeout=1)


def test_multiple_segments_generate_and_emit_in_segment_order() -> None:
    class OrderedTalker(FakeTalker):
        def omni_audio_generation(
            self,
            *,
            tts_text: str,
            voice_name: str,
            audio_detokenizer: Any,
            stream: bool,
            abort_event: threading.Event,
        ):
            self.calls.append({"tts_text": tts_text})
            yield np.array([float(len(self.calls))], dtype=np.float32)

    talker = OrderedTalker(outputs=[])
    scheduler = _scheduler(talker)
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk("one", 1)))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk("two", 2)))
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))

        first = _next_out(scheduler)
        second = _next_out(scheduler)
        result = _next_out(scheduler)

        assert [call["tts_text"] for call in talker.calls] == ["one", "two"]
        assert first.type == "stream"
        assert second.type == "stream"
        assert [first.data["segment_id"], second.data["segment_id"]] == [1, 2]
        first_audio = np.frombuffer(first.data["audio_waveform"], dtype=np.float32)
        second_audio = np.frombuffer(second.data["audio_waveform"], dtype=np.float32)
        assert first_audio.tolist() == [1.0]
        assert second_audio.tolist() == [2.0]
        assert result.type == "result"
        assert result.data.data["audio_chunk_count"] == 2
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_pending_stream_chunk_and_done_are_processed_when_payload_arrives() -> None:
    talker = FakeTalker(
        outputs=[
            (np.array([0.1, 0.2], dtype=np.float32), None, None, None),
        ],
    )
    scheduler = _scheduler(talker, now_ms_values=[100, 130, 160, 190])
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk("hello", 7)))
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        _assert_no_outbox_message(scheduler)

        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        stream_msg = _next_out(scheduler)
        result_msg = _next_out(scheduler)

        assert stream_msg.type == "stream"
        assert stream_msg.target is None
        assert stream_msg.metadata["modality"] == "audio"
        assert stream_msg.data["audio_waveform_shape"] == [2]
        assert stream_msg.data["audio_waveform_dtype"] == "float32"
        assert stream_msg.data["sample_rate"] == 16000
        assert stream_msg.data["stage_name"] == "talker_stream"
        assert stream_msg.data["segment_id"] == 7
        assert stream_msg.data["segmenter_first_emit_ms"] == 42
        assert stream_msg.data["talker_first_audio_ms"] == 30
        assert stream_msg.data["talker_queue_depth"] == 0

        assert result_msg.type == "result"
        assert result_msg.data.data["modality"] == "audio"
        assert result_msg.data.data["sample_rate"] == 16000
        assert result_msg.data.data["stage_name"] == "talker_stream"
        assert result_msg.data.data["audio_chunk_count"] == 1
        assert result_msg.data.data["aborted"] is False
        assert "audio_waveform" not in result_msg.data.data
        assert talker.calls[0]["tts_text"] == "hello"
        assert talker.calls[0]["stream"] is True
        assert isinstance(talker.calls[0]["abort_event"], threading.Event)
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_non_streaming_request_emits_no_streams_and_returns_concatenated_audio() -> None:
    talker = FakeTalker(
        outputs=[
            np.array([0.1, 0.2], dtype=np.float32),
            torch.tensor([0.3, 0.4], dtype=torch.float32),
        ]
    )
    scheduler = _scheduler(talker)
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(
            IncomingMessage("req", "new_request", _payload(stream=False))
        )
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk("hello")))
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))

        msg = _next_out(scheduler)

        assert msg.type == "result"
        assert msg.data.data["audio_waveform_shape"] == [4]
        assert msg.data.data["audio_waveform_dtype"] == "float32"
        assert msg.data.data["sample_rate"] == 16000
        assert msg.data.data["modality"] == "audio"
        audio = np.frombuffer(msg.data.data["audio_waveform"], dtype=np.float32)
        assert audio.tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
        _assert_no_outbox_message(scheduler)
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_non_streaming_concatenates_audio_on_last_dimension() -> None:
    talker = FakeTalker(
        outputs=[
            np.array([[0.1, 0.2]], dtype=np.float32),
            torch.tensor([[0.3, 0.4, 0.5]], dtype=torch.float32),
        ]
    )
    scheduler = _scheduler(talker)
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(
            IncomingMessage("req", "new_request", _payload(stream=False))
        )
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk("hello")))
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))

        msg = _next_out(scheduler)

        assert msg.type == "result"
        assert msg.data.data["audio_waveform_shape"] == [1, 5]
        audio = np.frombuffer(msg.data.data["audio_waveform"], dtype=np.float32)
        assert audio.reshape(1, 5).tolist() == [
            pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
        ]
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_empty_generated_waveforms_are_skipped() -> None:
    talker = FakeTalker(
        outputs=[
            np.array([], dtype=np.float32),
            None,
            np.array([0.5], dtype=np.float32),
        ]
    )
    scheduler = _scheduler(talker)
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk("hello")))
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))

        stream_msg = _next_out(scheduler)
        result_msg = _next_out(scheduler)

        assert stream_msg.type == "stream"
        assert stream_msg.data["audio_waveform_shape"] == [1]
        assert result_msg.type == "result"
        assert result_msg.data.data["audio_chunk_count"] == 1
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_all_empty_streaming_audio_finalizes_as_metadata_only_skipped() -> None:
    talker = FakeTalker(
        outputs=[
            np.array([], dtype=np.float32),
            None,
        ]
    )
    scheduler = _scheduler(talker)
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk("hello")))
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))

        msg = _next_out(scheduler)

        assert msg.type == "result"
        assert msg.data.data["skipped"] is True
        assert msg.data.data["audio_chunk_count"] == 0
        assert "audio_waveform" not in msg.data.data
        _assert_no_outbox_message(scheduler)
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_streaming_request_does_not_retain_audio_history() -> None:
    scheduler = _scheduler(FakeTalker(outputs=[np.array([0.1], dtype=np.float32)]))
    original_build = scheduler._build_streaming_result
    retained_piece_counts: list[int] = []

    def capturing_build(state: Any) -> dict[str, Any]:
        retained_piece_counts.append(len(state.audio_pieces))
        return original_build(state)

    scheduler._build_streaming_result = capturing_build
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk("hello")))
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))

        stream_msg = _next_out(scheduler)
        result_msg = _next_out(scheduler)

        assert stream_msg.type == "stream"
        assert result_msg.type == "result"
        assert retained_piece_counts == [0]
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_abort_before_payload_prevents_stale_stream_and_final_output() -> None:
    scheduler = _scheduler()
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk("stale")))
        scheduler.inbox.put(
            IncomingMessage("barrier", "new_request", _payload("barrier"))
        )
        scheduler.inbox.put(
            IncomingMessage("barrier", "stream_chunk", _chunk("fresh", 1))
        )

        barrier_stream = _next_out(scheduler)
        assert barrier_stream.request_id == "barrier"
        assert barrier_stream.type == "stream"

        scheduler.abort("req")
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))

        _assert_no_outbox_message(scheduler)
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_non_audio_streaming_request_skips_with_metadata_only_final() -> None:
    talker = FakeTalker()
    scheduler = _scheduler(talker)
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(
            IncomingMessage("req", "stream_chunk", _chunk("should not speak"))
        )
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        scheduler.inbox.put(
            IncomingMessage("req", "new_request", _payload(modalities=["text"]))
        )

        msg = _next_out(scheduler)

        assert msg.type == "result"
        assert msg.data.data["skipped"] is True
        assert msg.data.data["modality"] == "audio"
        assert msg.data.data["audio_chunk_count"] == 0
        assert "audio_waveform" not in msg.data.data
        assert talker.calls == []
        _assert_no_outbox_message(scheduler)
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_non_audio_non_streaming_request_keeps_empty_audio_shape() -> None:
    talker = FakeTalker()
    scheduler = _scheduler(talker)
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(
            IncomingMessage(
                "req",
                "new_request",
                _payload(stream=False, modalities=["text"]),
            )
        )

        msg = _next_out(scheduler)

        assert msg.type == "result"
        assert msg.data.data["skipped"] is True
        assert msg.data.data["audio_waveform"] is None
        assert msg.data.data["audio_waveform_shape"] == []
        assert msg.data.data["audio_chunk_count"] == 0
        assert talker.calls == []
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_abort_during_finalization_suppresses_stale_final_result() -> None:
    scheduler = _scheduler(FakeTalker(outputs=[np.array([0.1], dtype=np.float32)]))
    original_build = scheduler._build_streaming_result

    def aborting_build(state: Any) -> dict[str, Any]:
        scheduler.abort("req")
        return original_build(state)

    scheduler._build_streaming_result = aborting_build
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk("hello")))
        stream_msg = _next_out(scheduler)
        assert stream_msg.type == "stream"

        scheduler.inbox.put(IncomingMessage("req", "stream_done"))

        _assert_no_outbox_message(scheduler, timeout=0.2)
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_abort_between_stream_check_and_put_suppresses_stale_stream() -> None:
    scheduler = _scheduler(FakeTalker(outputs=[np.array([0.1], dtype=np.float32)]))
    original_is_active = scheduler._is_active_state

    def aborting_is_active(request_id: str, state: Any) -> bool:
        active = original_is_active(request_id, state)
        if active and request_id == "req":
            scheduler.abort(request_id)
        return active

    scheduler._is_active_state = aborting_is_active
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk("hello")))

        _assert_no_outbox_message(scheduler, timeout=0.2)
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_abort_marker_prevents_state_recreation_after_inactive_check_race() -> None:
    scheduler = _scheduler()
    original_is_inactive = scheduler._is_inactive
    injected = False

    def aborting_is_inactive(request_id: str) -> bool:
        nonlocal injected
        if request_id == "req" and not injected:
            injected = True
            scheduler.abort(request_id)
            return False
        return original_is_inactive(request_id)

    scheduler._is_inactive = aborting_is_inactive
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        time.sleep(0.1)

        assert "req" not in scheduler._state
        _assert_no_outbox_message(scheduler)
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_malformed_text_tensor_emits_error_and_scheduler_recovers() -> None:
    scheduler = _scheduler()
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
                    from_stage="segmenter",
                ),
            )
        )

        error_msg = _next_out(scheduler)
        assert error_msg.request_id == "bad"
        assert error_msg.type == "error"
        assert isinstance(error_msg.data, TypeError)

        scheduler.inbox.put(IncomingMessage("good", "new_request", _payload("good")))
        scheduler.inbox.put(IncomingMessage("good", "stream_chunk", _chunk("hello")))

        stream_msg = _next_out(scheduler)
        assert stream_msg.request_id == "good"
        assert stream_msg.type == "stream"
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_abort_during_generation_sets_event_and_drops_late_outputs() -> None:
    class BlockingTalker(FakeTalker):
        def omni_audio_generation(
            self,
            *,
            tts_text: str,
            voice_name: str,
            audio_detokenizer: Any,
            stream: bool,
            abort_event: threading.Event,
        ):
            self.calls.append({"abort_event": abort_event})
            while not abort_event.is_set():
                time.sleep(0.005)
            yield np.array([9.0], dtype=np.float32)

    talker = BlockingTalker()
    scheduler = _scheduler(talker)
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk("hello")))
        while not talker.calls:
            time.sleep(0.005)

        scheduler.abort("req")

        _assert_no_outbox_message(scheduler, timeout=0.2)
        assert talker.calls[0]["abort_event"].is_set()
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_stop_sets_abort_and_joins_cooperative_worker_thread() -> None:
    class SlowExitTalker(FakeTalker):
        def omni_audio_generation(
            self,
            *,
            tts_text: str,
            voice_name: str,
            audio_detokenizer: Any,
            stream: bool,
            abort_event: threading.Event,
        ):
            self.calls.append({"abort_event": abort_event})
            while not abort_event.is_set():
                time.sleep(0.005)
            time.sleep(0.05)
            yield np.array([1.0], dtype=np.float32)

    talker = SlowExitTalker()
    scheduler = _scheduler(talker)
    thread = _start_scheduler(scheduler)
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload()))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk("hello")))
        while not talker.calls:
            time.sleep(0.005)
        worker = scheduler._state["req"].worker_thread
        assert worker is not None

        scheduler.stop()

        assert not worker.is_alive()
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_stop_gate_prevents_late_state_and_worker_creation() -> None:
    scheduler = _scheduler()
    thread = _start_scheduler(scheduler)
    try:
        scheduler.stop()
        thread.join(timeout=1)

        assert scheduler._ensure_state("late") is None

        state = talker_scheduler._RequestState(payload=_payload("late"))
        with scheduler._state_lock:
            scheduler._state["late"] = state
        scheduler._start_worker("late", state)

        assert state.worker_thread is None
        scheduler.inbox.put(IncomingMessage("late", "new_request", _payload("late")))
        scheduler.inbox.put(IncomingMessage("late", "stream_chunk", _chunk("hello")))
        _assert_no_outbox_message(scheduler)
    finally:
        scheduler.stop()
        thread.join(timeout=1)


def test_stop_does_not_join_unstarted_worker_thread() -> None:
    scheduler = _scheduler()
    state = talker_scheduler._RequestState(payload=_payload("req"))
    state.worker_thread = threading.Thread(target=lambda: None)
    with scheduler._state_lock:
        scheduler._state["req"] = state

    scheduler.stop()

    assert state.abort_event.is_set()
