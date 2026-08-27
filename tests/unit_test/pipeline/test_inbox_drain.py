# SPDX-License-Identifier: Apache-2.0
"""Tests for the bounded inbox drain.

The bound exists to stop one pass absorbing an entire backlog before readiness
is evaluated once. It must therefore INTERLEAVE work, never LOSE or REORDER it:
the same messages come out, in the same order, just across more passes.
"""
from __future__ import annotations

import queue as _queue_mod

import pytest

from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def _sched(limit: int, n_msgs: int):
    s = object.__new__(OmniScheduler)
    s.inbox = _queue_mod.Queue()
    s.inbox_drain_max = limit
    for i in range(n_msgs):
        s.inbox.put(f"m{i}")
    return s


def test_unbounded_default_drains_everything():
    """limit 0 is the historical behaviour and must stay byte-for-byte it."""
    s = _sched(0, 250)
    got = s._drain_local_inbox()
    assert got == [f"m{i}" for i in range(250)]
    assert s.inbox.empty()
    assert s._inbox_has_pending() is False


@pytest.mark.parametrize("limit", [1, 2, 32, 64])
def test_bound_is_respected(limit):
    s = _sched(limit, 500)
    got = s._drain_local_inbox()
    assert len(got) == limit


def test_repeated_drains_lose_nothing_and_preserve_order():
    """The whole backlog must still arrive, FIFO, across successive passes."""
    s = _sched(32, 1000)
    seen = []
    passes = 0
    while True:
        batch = s._drain_local_inbox()
        if not batch:
            break
        passes += 1
        seen.extend(batch)
        assert len(batch) <= 32
    assert seen == [f"m{i}" for i in range(1000)], "messages lost or reordered"
    assert passes == 1000 // 32 + 1


def test_drain_shorter_than_bound_returns_what_is_there():
    s = _sched(64, 5)
    assert s._drain_local_inbox() == [f"m{i}" for i in range(5)]
    assert s._drain_local_inbox() == []


def test_pending_flag_tracks_the_leftover():
    """A bounded drain that left work behind must not look like an idle loop."""
    s = _sched(10, 25)
    s._drain_local_inbox()
    assert s._inbox_has_pending() is True
    s._drain_local_inbox()
    assert s._inbox_has_pending() is True
    s._drain_local_inbox()
    assert s._inbox_has_pending() is False


def test_unbounded_never_reports_pending():
    """With no bound there is no leftover, so the idle path is unchanged."""
    s = _sched(0, 25)
    s._drain_local_inbox()
    assert s._inbox_has_pending() is False


def test_idle_sleep_is_short_while_the_inbox_still_has_work(monkeypatch):
    import sglang_omni.scheduling.omni_scheduler as mod

    s = _sched(4, 20)
    s._drain_local_inbox()
    slept: list[float] = []
    monkeypatch.setattr(mod.time, "sleep", slept.append)
    s._sleep_during_idle()
    assert slept == [0.0001], "long idle sleep on top of a non-empty inbox"
