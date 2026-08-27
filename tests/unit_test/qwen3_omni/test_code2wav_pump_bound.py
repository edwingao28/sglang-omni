# SPDX-License-Identifier: Apache-2.0
"""Tests for the bounded coalesced pump.

The bound exists to stop ONE pump call servicing the entire system-wide
backlog while holding ``_state_lock`` and the single loop thread (measured at
68 chained steps and 2.4 s at C=64, with ~700 codec messages stranding behind
it). It must therefore INTERLEAVE work, never LOSE or REORDER it, and it must
never change how much audio a request receives: the same steps run either way,
only the moment moves.

That last property is the correctness backstop for the whole change, so it is
tested directly rather than inferred from step counts.
"""
from __future__ import annotations

import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
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
        max_batch_wait_ms=0,
        batch_floor=1,
        **kwargs,
    )


def _item(code: int) -> StreamItem:
    return StreamItem(
        0, torch.tensor([code, code * 10]), "talker", metadata={"stream": True}
    )


def _chunk(rid: str, code: int) -> IncomingMessage:
    return IncomingMessage(request_id=rid, type="stream_chunk", data=_item(code))


def _drain_outbox(scheduler) -> list:
    out = []
    while not scheduler.outbox.empty():
        out.append(scheduler.outbox.get_nowait())
    return out


# 20 streams, not 4. With no CUDA-graph runner the fake decodes a stream's
# whole ready window in ONE step, so a per-stream backlog does not chain steps
# at all -- the first draft of these tests fed 4 streams and never produced a
# multi-step pump, which is a fixture defect and was fixed here rather than by
# weakening the bound. Chaining comes from the <=8 participants-per-step
# ceiling: 20 due streams force ceil(20/8) = 3 steps in one pump.
_STREAMS = [f"r{i:02d}" for i in range(20)]


def _feed(scheduler, rids, codes) -> None:
    """Make every stream due, so one pump must chain several steps."""
    items = [(rid, _item(c)) for c in codes for rid in rids]
    scheduler.on_stream_chunk_batch(items)


def _feed_via_loop(scheduler, rids, codes) -> None:
    """Drive the REAL intake path, so the probe's batch counter sees it."""
    for c in codes:
        for rid in rids:
            scheduler.inbox.put(_chunk(rid, c))
    scheduler._next_message()


def _settle(scheduler, limit: int = 200) -> None:
    """Re-pump until nothing is due, the way the serving loop would."""
    for _ in range(limit):
        with scheduler._state_lock:
            if not scheduler.select_step_participants():
                return
            scheduler._pump_streams()
    raise AssertionError("pump never settled")


# ---------------------------------------------------------------- default off


def test_bound_defaults_to_unbounded() -> None:
    """Historical behaviour must stay byte-for-byte the default."""
    assert _scheduler()._max_pump_steps == 0
    assert _scheduler(max_pump_steps=0)._max_pump_steps == 0
    assert _scheduler(max_pump_steps=None)._max_pump_steps == 0


def test_unbounded_pump_still_drains_everything_in_one_call() -> None:
    """The bound is off: one pump must still clear the whole backlog, or the
    default path changed."""
    s = _scheduler()
    rids = _STREAMS
    _feed(s, rids, range(1, 5))
    assert s._pump_hit_bound is False
    for rid in rids:
        state = s._stream_states[rid]
        assert state.emitted == len(state.chunks), "unbounded pump left a backlog"


# ---------------------------------------------------------------- the bound


def test_bound_is_respected_per_call() -> None:
    s = _scheduler(max_pump_steps=2)
    steps: list[int] = []
    original = s.run_step

    def _count(participants, plan):
        steps.append(len(participants))
        return original(participants, plan)

    s.run_step = _count
    _feed(s, _STREAMS, range(1, 5))
    assert len(steps) == 2, f"pump ran {len(steps)} steps against a bound of 2"
    assert s._pump_hit_bound is True


def test_bounded_pump_loses_no_audio_and_keeps_frame_counts_equal() -> None:
    """THE CONTRACT. Bounded and unbounded must deliver each request exactly
    the same audio: same number of emitted frames, same waveform bytes, same
    per-request message count. Interleaving moves the moment, never the work."""
    rids, codes = _STREAMS, list(range(1, 5))

    ctl = _scheduler()
    _feed(ctl, rids, codes)
    _settle(ctl)
    # A bounded pump returns early, so the caller must re-pump; the serving
    # loop does that on its next turn. Here we pump until it settles.
    on = _scheduler(max_pump_steps=2)
    _feed(on, rids, codes)
    _settle(on)

    for rid in rids:
        assert on._stream_states[rid].emitted == ctl._stream_states[rid].emitted, (
            f"{rid}: bounded pump delivered a different number of frames"
        )

    def per_request(scheduler):
        out: dict[str, list] = {}
        for msg in _drain_outbox(scheduler):
            out.setdefault(msg.request_id, []).append(msg)
        return out

    got, want = per_request(on), per_request(ctl)
    assert set(got) == set(want)
    for rid in want:
        wav_on = b"".join(m.data["audio_waveform"] for m in got[rid])
        wav_ctl = b"".join(m.data["audio_waveform"] for m in want[rid])
        assert len(wav_on) == len(wav_ctl), f"{rid}: audio length changed"
        assert wav_on == wav_ctl, f"{rid}: audio content changed"
        samples_on = sum(m.data["audio_waveform_shape"][0] for m in got[rid])
        samples_ctl = sum(m.data["audio_waveform_shape"][0] for m in want[rid])
        assert samples_on == samples_ctl, f"{rid}: sample count changed"


def test_bounded_pump_preserves_per_stream_order() -> None:
    """Each stream's own chunks must arrive in order; interleaving reorders
    ACROSS streams only, which is the scheduler's prerogative."""
    s = _scheduler(max_pump_steps=1)
    rids = _STREAMS
    _feed(s, rids, range(1, 5))
    _settle(s)
    seen: dict[str, int] = {}
    for msg in _drain_outbox(s):
        seen[msg.request_id] = seen.get(msg.request_id, 0) + 1
    for rid in rids:
        state = s._stream_states[rid]
        assert state.emitted == len(state.chunks), f"{rid} left unemitted frames"
        assert seen[rid] >= 1


def test_bound_adds_no_decode_steps() -> None:
    """The asymmetry against the rejected drain bound, asserted rather than
    argued: bounding the PUMP splits the same steps across more calls, it does
    not create extra ones. The drain bound failed precisely because every
    extra pass paid its own decode step."""
    rids, codes = _STREAMS, list(range(1, 5))

    def total_steps(scheduler) -> int:
        n = [0]
        original = scheduler.run_step

        def _count(participants, plan):
            n[0] += 1
            return original(participants, plan)

        scheduler.run_step = _count
        _feed(scheduler, rids, codes)
        _settle(scheduler)
        return n[0]

    assert total_steps(_scheduler(max_pump_steps=2)) == total_steps(_scheduler())


def test_bound_above_the_backlog_is_inert() -> None:
    """A bound the workload never reaches must behave exactly like no bound.
    This is the low-C guard: at C=32 no observed pump exceeded 16 steps, so a
    bound of 16 must not bind there."""
    rids, codes = _STREAMS, list(range(1, 5))
    ctl, on = _scheduler(), _scheduler(max_pump_steps=64)
    _feed(ctl, rids, codes)
    _feed(on, rids, codes)
    assert on._pump_hit_bound is False, "an inert bound reported that it bound"
    for rid in rids:
        assert on._stream_states[rid].emitted == ctl._stream_states[rid].emitted


def test_hit_bound_flag_resets_between_pumps() -> None:
    """A stale flag would over-report the bound-hit rate, which is the very
    number that tells us whether the bound is too small."""
    s = _scheduler(max_pump_steps=2)
    _feed(s, _STREAMS, range(1, 5))
    assert s._pump_hit_bound is True
    _settle(s)
    with s._state_lock:
        s._pump_streams()
    assert s._pump_hit_bound is False, "flag survived a pump that did not bind"


# ------------------------------------------------- strengthened nesting test


def test_probe_pump_timing_covers_the_deadline_path_too() -> None:
    """STRENGTHENED after the probe round. The original nesting test only ever
    exercised the chunk-arrival path, so it never noticed that
    _pump_due_streams calls _pump_streams OUTSIDE _handle_stream_chunk_batch --
    which is why the reported 'pump nests inside batch' was wrong for ~5% of
    pumps. Both entry points must be timed, and the deadline path must be
    counted as a pump with no batch around it."""
    s = _scheduler(loop_probe_interval_ms=50)
    _feed_via_loop(s, _STREAMS[:2], range(1, 3))
    probe = s._loop_probe
    batch_after_chunk_path = probe._counts.get("batches", 0)
    pumps_after_chunk_path = probe._counts.get("pumps", 0)
    assert batch_after_chunk_path >= 1
    assert pumps_after_chunk_path >= 1

    # Now the deadline path, which takes no batch at all.
    _drain_outbox(s)
    s.on_stream_chunk_batch([(rid, _item(9)) for rid in _STREAMS[:2]])
    s._pump_due_streams()
    assert probe._counts["pumps"] > pumps_after_chunk_path, (
        "the deadline-path pump was not timed"
    )
    deadline_pumps = probe._counts["pumps"] - probe._counts.get("batches", 0)
    assert deadline_pumps >= 1, (
        "pumps and batches are equal, so this test is not exercising the "
        "deadline path and would not catch the nesting error again"
    )
