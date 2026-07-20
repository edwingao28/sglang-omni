# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
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
