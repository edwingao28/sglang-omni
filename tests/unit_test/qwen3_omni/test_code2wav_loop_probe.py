# SPDX-License-Identifier: Apache-2.0
"""Tests for the code2wav loop probe.

The probe exists to answer one question the existing event set CANNOT answer:
every code2wav event is emitted inside the pump, so nothing records how deep
the inbox was when the collector arrived, nor how long the queue went
unattended between drains. Those two readings are what these tests pin.

The load-bearing properties, each with its own test:
  - the depth sample is taken BEFORE the drain (a sample after it reads zero
    and would have "proved" an empty queue on a backlogged loop);
  - the arrival rate is MEASURED (drained count per window), not assumed --
    assuming it is what invalidated the previous round's pump-cluster model;
  - ``gap`` covers the interval BETWEEN drains, which is the interval during
    which arrivals accumulate unattended;
  - with the knob off the loop is byte-for-byte the unprobed loop and the
    probe object is not even constructed.
"""
from __future__ import annotations

import json
import threading
import time

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
    _LoopProbe,
)
from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.scheduling.messages import IncomingMessage
from tests.unit_test.fixtures.qwen_fakes import FakeCode2WavModel


def _scheduler(**kwargs) -> Code2WavScheduler:
    return Code2WavScheduler(
        FakeCode2WavModel(total_upsample=2),
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
        enable_batching=True,
        **kwargs,
    )


def _stream_item(code: int) -> StreamItem:
    return StreamItem(
        0, torch.tensor([code, code * 10]), "talker", metadata={"stream": True}
    )


def _chunk(request_id: str, code: int) -> IncomingMessage:
    return IncomingMessage(
        request_id=request_id, type="stream_chunk", data=_stream_item(code)
    )


def _done(request_id: str) -> IncomingMessage:
    return IncomingMessage(request_id=request_id, type="stream_done", data=None)


def _drain_outbox(scheduler: Code2WavScheduler) -> list:
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    return messages


# ---------------------------------------------------------------- knob is off


def test_probe_is_absent_by_default() -> None:
    """Default-off means the object is never built, not merely unused."""
    assert _scheduler()._loop_probe is None
    assert _scheduler(loop_probe_interval_ms=0)._loop_probe is None
    assert _scheduler(loop_probe_interval_ms=None)._loop_probe is None


def _script(scheduler: Code2WavScheduler) -> tuple:
    """One fixed message script; returns everything observable about how the
    loop handled it."""
    scheduler.inbox.put(_chunk("req-a", 1))
    scheduler.inbox.put(_chunk("req-a", 2))
    scheduler.inbox.put(_done("req-b"))
    first = scheduler._next_message()
    scheduler.inbox.put(_chunk("req-a", 3))
    scheduler.inbox.put(_chunk("req-a", 4))
    second = scheduler._next_message()
    return (
        None if first is None else (first.request_id, first.type),
        None if second is None else (second.request_id, second.type),
        scheduler._stream_states["req-a"].emitted,
        len(scheduler._stream_states["req-a"].chunks),
        [m.type for m in _drain_outbox(scheduler)],
        [m.request_id for m in scheduler._pending_messages],
    )


def test_probe_does_not_change_what_the_loop_does() -> None:
    """An instrument that changes the behaviour it measures is not an
    instrument. Same script, both arms, identical observations."""
    off = _script(_scheduler(max_batch_wait_ms=0, batch_floor=2))
    on = _script(
        _scheduler(max_batch_wait_ms=0, batch_floor=2, loop_probe_interval_ms=50)
    )
    assert off == on
    # Pin the script itself, so an equal-but-wrong pair cannot pass silently.
    assert off[0] == ("req-b", "stream_done")
    assert off[1] is None
    assert off[2] == 4


# ---------------------------------------------------------------- the readings


def test_depth_is_sampled_before_the_drain_not_after() -> None:
    """THE reading no existing gauge gives. A sample taken after the drain
    reads 0 on the same queue, so this test fails loudly if the sample ever
    migrates below the drain loop."""
    scheduler = _scheduler(
        max_batch_wait_ms=0, batch_floor=2, loop_probe_interval_ms=50
    )
    for code in range(1, 7):
        scheduler.inbox.put(_chunk("req-a", code))
    assert scheduler.inbox.qsize() == 6

    scheduler._next_message()

    probe = scheduler._loop_probe
    assert probe._gauges["depth_at_drain"] == 6
    assert probe._counts["depth_at_drain_sum"] == 6
    assert scheduler.inbox.qsize() == 0, "drain did not actually empty the queue"


def test_drained_count_measures_arrivals_rather_than_assuming_them() -> None:
    """Arrival rate is derived from this count over the window. It must equal
    what the drain really took, across BOTH destinations (new-stream first
    chunks and everything appended to _pending_messages)."""
    scheduler = _scheduler(
        max_batch_wait_ms=0, batch_floor=2, loop_probe_interval_ms=50
    )
    scheduler.inbox.put(_chunk("req-a", 1))  # first chunk -> first_chunks
    scheduler.inbox.put(_chunk("req-b", 1))  # first chunk -> first_chunks
    scheduler.inbox.put(_done("req-c"))      # -> _pending_messages
    scheduler.inbox.put(_done("req-d"))      # -> _pending_messages

    scheduler._next_message()

    assert scheduler._loop_probe._counts["drained"] == 4
    assert scheduler._loop_probe._counts["drains"] == 1


def test_gap_covers_the_interval_between_drains() -> None:
    """The unattended interval, not the drain itself. This is the quantity a
    depth prediction multiplies the measured arrival rate by."""
    scheduler = _scheduler(
        max_batch_wait_ms=0, batch_floor=2, loop_probe_interval_ms=50
    )
    scheduler.inbox.put(_chunk("req-a", 1))
    scheduler._next_message()
    assert "gap" not in scheduler._loop_probe._secs, "first drain has no predecessor"

    time.sleep(0.05)
    scheduler.inbox.put(_chunk("req-a", 2))
    scheduler._next_message()

    gap = scheduler._loop_probe._secs["gap"]
    assert 0.04 <= gap <= 0.2, f"gap {gap:.4f}s does not track the real interval"


def test_pump_time_is_nested_inside_batch_time() -> None:
    """The report derives ingest as batch minus pump, so the nesting is a
    contract, not an incidental."""
    scheduler = _scheduler(
        max_batch_wait_ms=0, batch_floor=2, loop_probe_interval_ms=50
    )
    scheduler.inbox.put(_chunk("req-a", 1))
    scheduler.inbox.put(_chunk("req-a", 2))
    scheduler._next_message()

    secs = scheduler._loop_probe._secs
    assert secs["pump"] <= secs["batch"]
    assert scheduler._loop_probe._counts["steps"] >= 1
    assert scheduler._loop_probe._counts["step_participants"] >= 1


def test_lock_wait_is_recorded_and_disjoint_from_reap() -> None:
    """Both are recorded so a lock-contention reading can never be inferred
    from the reap cost it sits next to."""
    scheduler = _scheduler(
        max_batch_wait_ms=0, batch_floor=2, loop_probe_interval_ms=50
    )
    scheduler.inbox.put(_chunk("req-a", 1))
    scheduler._next_message()

    secs = scheduler._loop_probe._secs
    assert "lock_wait" in secs and "reap" in secs
    assert secs["lock_wait"] >= 0.0


def test_lock_wait_sees_a_lock_another_thread_is_holding() -> None:
    """If the pump ever blocked intake from a sibling thread, this is the key
    that would carry it -- so it has to actually measure contention."""
    scheduler = _scheduler(
        max_batch_wait_ms=0, batch_floor=2, loop_probe_interval_ms=50
    )
    held = threading.Event()
    release = threading.Event()

    def _hold() -> None:
        with scheduler._state_lock:
            held.set()
            release.wait(2.0)

    holder = threading.Thread(target=_hold)
    holder.start()
    assert held.wait(2.0)
    threading.Timer(0.05, release.set).start()
    scheduler._next_message()
    holder.join(2.0)

    wait = scheduler._loop_probe._secs["lock_wait"]
    assert wait >= 0.04, f"lock_wait {wait:.4f}s missed a held lock"


# ---------------------------------------------------------------- aggregation


def test_window_withholds_until_the_interval_elapses() -> None:
    probe = _LoopProbe(50.0)
    now = time.perf_counter()
    assert probe.end_turn(now, 0.001) is None
    assert probe.end_turn(now + 0.049, 0.001) is None
    payload = probe.end_turn(now + 0.051, 0.001)
    assert payload is not None
    assert payload["turns"] == 3
    assert payload["window_ms"] == pytest.approx(51.0, abs=1.0)


def test_window_resets_so_counters_do_not_accumulate_across_windows() -> None:
    """A window that carried its predecessor's totals would read as unbounded
    growth on a perfectly steady loop."""
    probe = _LoopProbe(10.0)
    now = time.perf_counter()
    probe.bump("drained", 7)
    probe.add("pump", 0.003)
    probe.gauge_max("depth_at_drain", 40)
    first = probe.end_turn(now + 0.011, 0.001)
    assert first["n_drained"] == 7
    assert first["pump_ms"] == pytest.approx(3.0)
    assert first["max_depth_at_drain"] == 40

    probe.bump("drained", 2)
    second = probe.end_turn(now + 0.030, 0.001)
    assert second["n_drained"] == 2
    assert "pump_ms" not in second
    assert "max_depth_at_drain" not in second


def test_gauge_keeps_the_peak_and_counts_keep_the_sum() -> None:
    """Peak and mean answer different questions: a spiky queue and a uniformly
    deep one can share a mean, and only the peak separates them."""
    probe = _LoopProbe(10.0)
    for depth in (3, 91, 5, 5):
        probe.gauge_max("depth_at_drain", depth)
        probe.bump("depth_at_drain_sum", depth)
        probe.bump("drains")
    payload = probe.end_turn(time.perf_counter() + 1.0, 0.001)
    assert payload["max_depth_at_drain"] == 91
    assert payload["n_depth_at_drain_sum"] == 104
    assert payload["n_drains"] == 4


def test_zero_interval_never_emits() -> None:
    """The disabled spelling must not emit through a probe someone built by
    hand with 0."""
    probe = _LoopProbe(0.0)
    assert probe.end_turn(time.perf_counter() + 10.0, 0.001) is None


def test_window_reaches_the_event_tree(tmp_path) -> None:
    """The whole point of the round is a table read off these events, so the
    emit path is tested end to end. A silent null here would be
    indistinguishable from a loop with nothing to report -- which is the exact
    failure the arming gate exists to rule out."""
    recorder = get_recorder()
    recorder.start("probe-test", str(tmp_path), "code2wav")
    try:
        scheduler = _scheduler(
            max_batch_wait_ms=0, batch_floor=2, loop_probe_interval_ms=1
        )
        for code in (1, 2, 3, 4):
            scheduler.inbox.put(_chunk("req-a", code))
            scheduler._next_message()
            time.sleep(0.002)
        scheduler.inbox.put(_chunk("req-a", 5))
        scheduler._next_message()
    finally:
        recorder.stop()

    lines = []
    for path in tmp_path.glob("*.jsonl"):
        lines.extend(path.read_text().splitlines())
    events = [
        json.loads(line)
        for line in lines
        if "code2wav_loop_summary" in line
    ]
    assert events, "loop probe emitted nothing into the event tree"
    meta = events[0]["metadata"]
    assert meta["turns"] >= 1
    assert meta["window_ms"] > 0.0
    assert "n_drained" in meta and "max_depth_at_drain" in meta


def test_emitted_window_names_every_recorded_key() -> None:
    """A key recorded but not emitted is a silent hole in the accounting."""
    probe = _LoopProbe(1.0)
    probe.add("drain", 0.001)
    probe.add("batch", 0.002)
    probe.add("pump", 0.0015)
    probe.add("gap", 0.004)
    probe.add("lock_wait", 0.0001)
    probe.add("reap", 0.0002)
    probe.add("idle", 0.01)
    probe.bump("drained", 12)
    probe.bump("steps", 2)
    probe.gauge_max("depth_at_drain", 64)
    payload = probe.end_turn(time.perf_counter() + 1.0, 0.02)
    for key in (
        "drain_ms",
        "batch_ms",
        "pump_ms",
        "gap_ms",
        "lock_wait_ms",
        "reap_ms",
        "idle_ms",
        "turn_ms",
        "n_drained",
        "n_steps",
        "max_depth_at_drain",
        "window_ms",
        "turns",
    ):
        assert key in payload, f"{key} recorded but not emitted"
