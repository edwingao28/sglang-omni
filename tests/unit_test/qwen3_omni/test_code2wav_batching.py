# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time

import numpy as np
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    Code2WavRunResult,
    GraphKey,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.scheduling.messages import IncomingMessage
from tests.unit_test.fixtures.qwen_fakes import FakeCode2WavModel


def _make_batching_scheduler(**kwargs) -> Code2WavScheduler:
    return Code2WavScheduler(
        FakeCode2WavModel(total_upsample=2),
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
        enable_batching=True,
        **kwargs,
    )


def _chunk(request_id: str) -> IncomingMessage:
    return IncomingMessage(request_id=request_id, type="stream_chunk", data=None)


def _put_later(scheduler: Code2WavScheduler, msg: IncomingMessage, delay: float):
    timer = threading.Timer(delay, scheduler.inbox.put, args=(msg,))
    timer.start()
    return timer


def _stream_item(code: int, *, stream: bool = True) -> StreamItem:
    return StreamItem(
        0, torch.tensor([code, code * 10]), "talker", metadata={"stream": stream}
    )


def _feed_batch(
    scheduler: Code2WavScheduler,
    entries: list[tuple[str, int]],
    *,
    stream_flags: dict[str, bool] | None = None,
) -> None:
    items = []
    for rid, code in entries:
        stream = True if stream_flags is None else stream_flags[rid]
        items.append((rid, _stream_item(code, stream=stream)))
    scheduler.on_stream_chunk_batch(items)


def _drain_outbox(scheduler: Code2WavScheduler) -> list:
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    return messages


def test_collector_waits_until_deadline() -> None:
    scheduler = _make_batching_scheduler()
    scheduler._batch_deadline = lambda: time.monotonic() + 0.2
    timer = _put_later(scheduler, _chunk("req-2"), 0.05)
    try:
        batch = scheduler._collect_stream_chunk_batch(_chunk("req-1"))
    finally:
        timer.join()
    assert [m.request_id for m in batch] == ["req-1", "req-2"]


def test_collector_no_wait_when_nothing_due() -> None:
    scheduler = _make_batching_scheduler()
    assert scheduler._batch_deadline() is None
    timer = _put_later(scheduler, _chunk("req-2"), 0.05)
    try:
        batch = scheduler._collect_stream_chunk_batch(_chunk("req-1"))
    finally:
        timer.join()
    assert [m.request_id for m in batch] == ["req-1"]


def test_collector_pushback_non_chunk() -> None:
    scheduler = _make_batching_scheduler()
    scheduler._batch_deadline = lambda: time.monotonic() + 0.2
    done = IncomingMessage(request_id="req-1", type="stream_done", data=None)
    timer = _put_later(scheduler, done, 0.05)
    try:
        batch = scheduler._collect_stream_chunk_batch(_chunk("req-1"))
    finally:
        timer.join()
    assert [m.request_id for m in batch] == ["req-1"]
    assert scheduler._pending_messages[0] is done


def test_decompose_batch() -> None:
    assert Code2WavScheduler._decompose_batch(1) == [1]
    assert Code2WavScheduler._decompose_batch(3) == [2, 1]
    assert Code2WavScheduler._decompose_batch(5) == [4, 1]
    assert Code2WavScheduler._decompose_batch(6) == [4, 2]
    assert Code2WavScheduler._decompose_batch(7) == [4, 2, 1]
    assert Code2WavScheduler._decompose_batch(8) == [8]


def test_first_window_fires_immediately() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=1000, batch_floor=4)
    _feed_batch(scheduler, [("req-1", 1), ("req-1", 2)])
    messages = _drain_outbox(scheduler)
    assert [m.type for m in messages] == ["stream"]
    assert scheduler._model.calls == [(1, 2, 2)]


def test_floor_fires_without_deadline() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=1000, batch_floor=2)
    _feed_batch(scheduler, [("req-a", 1), ("req-a", 2)])
    _feed_batch(scheduler, [("req-b", 3), ("req-b", 4)])
    _drain_outbox(scheduler)
    _feed_batch(scheduler, [("req-a", 5), ("req-a", 6), ("req-b", 7), ("req-b", 8)])
    messages = _drain_outbox(scheduler)
    assert sorted(m.request_id for m in messages) == ["req-a", "req-b"]
    assert scheduler._model.calls == [(1, 2, 2), (1, 2, 2), (2, 2, 3)]


def test_deadline_fires_single() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    _feed_batch(scheduler, [("req-1", 1), ("req-1", 2)])
    _drain_outbox(scheduler)
    _feed_batch(scheduler, [("req-1", 3), ("req-1", 4)])
    messages = _drain_outbox(scheduler)
    assert [m.request_id for m in messages] == ["req-1"]
    assert scheduler._model.calls == [(1, 2, 2), (1, 2, 3)]


def test_bucket_isolation() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    _feed_batch(scheduler, [("req-a", 1), ("req-a", 2)])
    _feed_batch(scheduler, [("req-b", 3), ("req-b", 4)])
    _drain_outbox(scheduler)
    _feed_batch(
        scheduler,
        [
            ("req-a", 5),
            ("req-a", 6),
            ("req-b", 7),
            ("req-b", 8),
            ("req-b", 9),
            ("req-b", 10),
        ],
    )
    steady_calls = scheduler._model.calls[2:]
    assert all(call[0] == 1 for call in steady_calls)
    assert sorted(steady_calls) == [(1, 2, 3), (1, 2, 5)]
    assert scheduler._stream_states["req-a"].emitted == 4
    assert scheduler._stream_states["req-b"].emitted == 6


def test_bitwise_equivalence() -> None:
    schedule = {
        "req-1": [1, 2, 3, 4, 5, 6],
        "req-2": [7, 8, 9, 10, 11, 12],
        "req-3": [13, 14, 15, 16, 17, 18],
    }

    control = Code2WavScheduler(
        FakeCode2WavModel(total_upsample=2),
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
    )
    for rid, codes in schedule.items():
        for code in codes:
            control._handle_stream_chunk(rid, _stream_item(code))

    batched = _make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    for round_start in range(0, 6, 2):
        entries = []
        for rid, codes in schedule.items():
            entries.append((rid, codes[round_start]))
            entries.append((rid, codes[round_start + 1]))
        _feed_batch(batched, entries)

    assert any(call[0] > 1 for call in batched._model.calls)
    for rid in schedule:
        control_state = control._stream_states[rid]
        batched_state = batched._stream_states[rid]
        assert batched_state.emitted == 6
        assert np.array_equal(
            np.concatenate(control_state.audio_parts),
            np.concatenate(batched_state.audio_parts),
        )


def test_mixed_stream_enabled() -> None:
    scheduler = _make_batching_scheduler()
    _feed_batch(
        scheduler,
        [("req-a", 1), ("req-a", 2), ("req-b", 3), ("req-b", 4)],
        stream_flags={"req-a": True, "req-b": False},
    )
    messages = _drain_outbox(scheduler)
    assert [(m.type, m.request_id) for m in messages] == [("stream", "req-a")]
    assert scheduler._model.calls == [(2, 2, 2)]
    for rid in ("req-a", "req-b"):
        state = scheduler._stream_states[rid]
        assert state.emitted == 2
        assert len(state.audio_parts) == 1


def test_step_failure_isolates_participants() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    _feed_batch(scheduler, [("req-a", 1), ("req-a", 2)])
    _feed_batch(scheduler, [("req-b", 3), ("req-b", 4)])
    _drain_outbox(scheduler)

    real_forward = scheduler._forward_codes

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    scheduler._forward_codes = _boom
    _feed_batch(
        scheduler,
        [
            ("req-a", 5),
            ("req-a", 6),
            ("req-b", 7),
            ("req-b", 8),
            ("req-c", 9),
        ],
    )
    assert "req-a" not in scheduler._stream_states
    assert "req-b" not in scheduler._stream_states
    assert scheduler._is_aborted("req-a") and scheduler._is_aborted("req-b")
    assert "req-c" in scheduler._stream_states

    scheduler._forward_codes = real_forward
    _drain_outbox(scheduler)
    _feed_batch(scheduler, [("req-c", 10)])
    messages = _drain_outbox(scheduler)
    assert [(m.type, m.request_id) for m in messages] == [("stream", "req-c")]
    assert scheduler._stream_states["req-c"].emitted == 2


def test_one_participation_per_pump() -> None:
    scheduler = _make_batching_scheduler(max_batch_wait_ms=0, batch_floor=2)
    _feed_batch(scheduler, [("req-1", 1), ("req-1", 2)])
    _drain_outbox(scheduler)

    selections: list[list[str]] = []
    original_select = scheduler.select_step_participants

    def recording_select():
        participants = original_select()
        if participants:
            selections.append([rid for rid, _ in participants])
        return participants

    scheduler.select_step_participants = recording_select
    _feed_batch(scheduler, [("req-1", 3), ("req-1", 4), ("req-1", 5), ("req-1", 6)])
    assert selections == [["req-1"]]
    state = scheduler._stream_states["req-1"]
    assert state.emitted == 6
    assert len(state.chunks) - state.emitted == 0


class _StubGraphRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], bool]] = []

    def run(self, codes: torch.Tensor, *, eligible: bool = True) -> Code2WavRunResult:
        self.calls.append((tuple(codes.shape), eligible))
        return Code2WavRunResult(
            output=torch.zeros(codes.shape[0], 1, int(codes.shape[-1]) * 2),
            execution_mode="cuda_graph",
            key=GraphKey(
                batch_size=int(codes.shape[0]), frames=int(codes.shape[-1])
            ),
            fallback_reason=None,
        )


def test_factory_flags_reach_scheduler(monkeypatch) -> None:
    import sglang_omni.models.qwen3_omni.components.code2wav_scheduler as mod

    monkeypatch.setattr(
        mod,
        "load_code2wav_model",
        lambda path, *, device, dtype: FakeCode2WavModel(total_upsample=2),
    )
    scheduler = mod.create_code2wav_scheduler(
        "fake-path",
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        enable_batching=True,
        max_batch_wait_ms=250,
        batch_floor=3,
        batch_ceiling=4,
    )
    assert scheduler._enable_batching is True
    assert scheduler._max_batch_wait_s == 0.25
    assert scheduler._batch_floor == 3
    assert scheduler._batch_ceiling == 4
    assert scheduler._graph_runner is None
    assert scheduler._can_batch_stream_chunks is True


def test_forward_codes_uses_graph_runner() -> None:
    runner = _StubGraphRunner()
    scheduler = _make_batching_scheduler(graph_runner=runner)
    codes = torch.zeros(1, 2, 2, dtype=torch.long)
    wav, meta = scheduler._forward_codes(codes, graph_eligible=True)
    assert runner.calls == [((1, 2, 2), True)]
    assert meta == {
        "execution_mode": "cuda_graph",
        "graph_key": {"batch_size": 1, "frames": 2},
        "fallback_reason": None,
    }
    assert tuple(wav.shape) == (1, 1, 4)
    scheduler._forward_codes(codes, graph_eligible=False)
    assert runner.calls[-1] == ((1, 2, 2), False)


def test_forward_codes_eager_without_runner() -> None:
    scheduler = _make_batching_scheduler()
    codes = torch.zeros(1, 2, 2, dtype=torch.long)
    _, meta = scheduler._forward_codes(codes, graph_eligible=True)
    assert meta == {
        "execution_mode": "eager",
        "graph_key": None,
        "fallback_reason": None,
    }
