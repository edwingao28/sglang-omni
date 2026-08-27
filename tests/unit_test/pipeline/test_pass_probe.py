# SPDX-License-Identifier: Apache-2.0
"""Tests for the scheduler pass probe's aggregation bound.

The probe rides the hot loop, so the properties that matter are: it never
emits more than once per window no matter how fast the loop spins, it accounts
for a pass with no unnamed remainder, and it resets cleanly so consecutive
windows are independent rather than cumulative.
"""
from __future__ import annotations

import pytest

from sglang_omni.scheduling.omni_scheduler import _PassProbe


def test_zero_interval_never_emits():
    """interval 0 is the OFF switch: it must not emit, ever."""
    probe = _PassProbe(0.0)
    now = 0.0
    for _ in range(10_000):
        now += 1.0
        assert probe.end_pass(now, 0.5) is None


def test_emits_at_most_once_per_window():
    """A loop spinning far faster than the window still emits once per window."""
    probe = _PassProbe(50.0)  # 50ms
    base = probe._window_start
    emits = []
    # 10_000 passes spread over 500ms => 10 windows, not 10_000 events
    for i in range(1, 10_001):
        now = base + (i * 0.5 / 10_000)
        payload = probe.end_pass(now, 0.00005)
        if payload is not None:
            emits.append(payload)
    assert 8 <= len(emits) <= 11, f"expected ~10 emits, got {len(emits)}"
    assert sum(e["passes"] for e in emits) <= 10_000


def test_accumulates_activities_then_resets():
    probe = _PassProbe(10.0)
    base = probe._window_start
    probe.add("ingest", 0.004)
    probe.add("decode", 0.006)
    probe.bump("recv", 3)
    probe.bump("decode_steps")
    assert probe.end_pass(base + 0.005, 0.010) is None  # window not due

    probe.add("ingest", 0.002)
    probe.bump("recv", 2)
    payload = probe.end_pass(base + 0.020, 0.010)
    assert payload is not None
    assert payload["passes"] == 2
    assert payload["ingest_ms"] == pytest.approx(6.0)   # 4ms + 2ms
    assert payload["decode_ms"] == pytest.approx(6.0)
    assert payload["pass_ms"] == pytest.approx(20.0)    # both passes
    assert payload["n_recv"] == 5
    assert payload["n_decode_steps"] == 1
    assert payload["window_ms"] == pytest.approx(20.0, abs=1e-6)

    # the next window must be independent, not cumulative
    probe.add("ingest", 0.001)
    nxt = probe.end_pass(base + 0.040, 0.001)
    assert nxt is not None
    assert nxt["ingest_ms"] == pytest.approx(1.0)
    assert nxt["passes"] == 1
    assert "decode_ms" not in nxt


def test_pass_wall_is_the_accounting_total():
    """pass_ms is the denominator: activities must be comparable against it."""
    probe = _PassProbe(1.0)
    base = probe._window_start
    for key, secs in (("admin", 0.001), ("ingest", 0.004), ("input", 0.002),
                      ("decode", 0.010), ("result", 0.003)):
        probe.add(key, secs)
    payload = probe.end_pass(base + 0.050, 0.020)
    named = sum(v for k, v in payload.items()
                if k.endswith("_ms") and k not in ("pass_ms", "window_ms"))
    assert payload["pass_ms"] == pytest.approx(20.0)
    assert named == pytest.approx(20.0), "activities must sum to the pass wall"


def test_bump_ignores_zero():
    probe = _PassProbe(1.0)
    probe.bump("recv", 0)
    payload = probe.end_pass(probe._window_start + 0.01, 0.001)
    assert "n_recv" not in payload
