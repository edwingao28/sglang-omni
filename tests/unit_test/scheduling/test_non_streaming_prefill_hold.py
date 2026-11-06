# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the non-streaming prefill hold.

After a prefill, upstream defers the next one for ``prefill_decode_interval``
decode steps. A queue holding only non-streaming requests then keeps waiting up
to ``non_streaming_prefill_decode_interval`` steps in all, so their admissions
share one prefill; a streaming arrival, a lone waiting request, or a running
batch below ``prefill_decode_interval_min_bs`` releases at the base interval.
Tested on a stub scheduler with upstream's own countdown patched to a public
counter.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("sglang")

from sglang_omni.scheduling import omni_scheduler  # noqa: E402
from sglang_omni.scheduling.omni_scheduler import OmniScheduler  # noqa: E402

BASE_INTERVAL = 4
NON_STREAMING_INTERVAL = 8
UPSTREAM = omni_scheduler._Upstream  # noqa: leading-underscore  # production name


def make_req(*, stream: bool) -> SimpleNamespace:
    request = SimpleNamespace(params={"stream": stream})
    payload = SimpleNamespace(request=request)
    return SimpleNamespace(omni_data=SimpleNamespace(stage_payload=payload))


class StubScheduler(OmniScheduler):
    """The attribute surface the hold reads; nothing of the real constructor."""

    def __init__(self, *, running_bs: int, min_bs: int = 2) -> None:
        self.prefill_decode_interval = BASE_INTERVAL
        self.prefill_decode_interval_min_bs = min_bs
        self.non_streaming_prefill_decode_interval = NON_STREAMING_INTERVAL
        self.non_streaming_hold_remaining = 0
        self.base_remaining = 0
        self.require_mlp_sync = False
        self.running_batch = SimpleNamespace(batch_size=lambda: running_bs)
        self.waiting_queue: list[SimpleNamespace] = []

    def deferred_steps(self) -> int:
        """Decode steps the scheduler defers the next prefill, from a fresh arm."""
        extend_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True)
        )
        self._arm_prefill_decode_interval(extend_batch)  # noqa: leading-underscore
        steps = 0
        while self._should_defer_prefill(  # noqa: leading-underscore
            self.running_batch
        ):
            steps += 1
            assert steps <= 2 * NON_STREAMING_INTERVAL, "the hold never released"
        return steps


def upstream_should_defer(self: StubScheduler, running_batch) -> bool:
    if self.base_remaining == 0:
        return False
    elif running_batch.batch_size() < self.prefill_decode_interval_min_bs:
        self.base_remaining = 0
        return False
    else:
        self.base_remaining -= 1
        return True


def upstream_arm(self: StubScheduler, batch) -> None:
    self.base_remaining = self.prefill_decode_interval


@pytest.fixture(autouse=True)
def upstream_countdown():
    with (
        mock.patch.object(UPSTREAM, "_should_defer_prefill", upstream_should_defer),
        mock.patch.object(UPSTREAM, "_arm_prefill_decode_interval", upstream_arm),
    ):
        yield


def test_a_non_streaming_queue_waits_the_longer_interval():
    sched = StubScheduler(running_bs=16)
    sched.waiting_queue = [make_req(stream=False), make_req(stream=False)]
    assert sched.deferred_steps() == NON_STREAMING_INTERVAL


def test_a_streaming_request_in_the_queue_is_admitted_at_the_base_interval():
    sched = StubScheduler(running_bs=16)
    sched.waiting_queue = [make_req(stream=False), make_req(stream=True)]
    assert sched.deferred_steps() == BASE_INTERVAL


def test_a_lone_non_streaming_request_is_admitted_at_the_base_interval():
    sched = StubScheduler(running_bs=16)
    sched.waiting_queue = [make_req(stream=False)]
    assert sched.deferred_steps() == BASE_INTERVAL


def test_a_running_batch_below_min_bs_never_holds():
    sched = StubScheduler(running_bs=1)
    sched.waiting_queue = [make_req(stream=False)]
    assert sched.deferred_steps() == 0
