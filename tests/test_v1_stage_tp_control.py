# SPDX-License-Identifier: Apache-2.0
"""B5 control regressions: Stage control messages enter scheduler inbox."""
from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang_omni_v1.pipeline.stage.runtime import Stage
from sglang_omni_v1.scheduling.messages import IncomingMessage


def _make_stage(scheduler):
    relay = SimpleNamespace(cleanup=MagicMock(), close=MagicMock())
    return Stage(
        name="thinker",
        get_next=lambda request_id, result: None,
        gpu_id=None,
        recv_endpoint="ipc:///tmp/test-stage-tp-control-recv.sock",
        coordinator_endpoint="ipc:///tmp/test-stage-tp-control-complete.sock",
        abort_endpoint="ipc:///tmp/test-stage-tp-control-abort.sock",
        endpoints={},
        scheduler=scheduler,
        relay=relay,
    )


def test_on_abort_enqueues_abort_for_tp_inbox_scheduler():
    scheduler = SimpleNamespace(
        inbox=queue.Queue(),
        abort=MagicMock(),
        stop=MagicMock(),
        tp_size=2,
        _apply_envelope=MagicMock(),
        _tp_shutdown_requested=False,
    )
    stage = _make_stage(scheduler)
    stage._active_requests.add("r0")

    stage._on_abort("r0")

    msg = scheduler.inbox.get_nowait()
    assert msg == IncomingMessage(request_id="r0", type="abort")
    scheduler.abort.assert_not_called()
    assert "r0" in stage._aborted
    assert "r0" not in stage._active_requests
    stage.relay.cleanup.assert_called_once_with("r0")


def test_on_abort_falls_back_for_tp1_inbox_scheduler():
    scheduler = SimpleNamespace(
        inbox=queue.Queue(),
        abort=MagicMock(),
        stop=MagicMock(),
        tp_size=1,
        _apply_envelope=MagicMock(),
        _tp_shutdown_requested=False,
    )
    stage = _make_stage(scheduler)

    stage._on_abort("r0")

    assert scheduler.inbox.empty()
    scheduler.abort.assert_called_once_with("r0")


def test_on_abort_falls_back_for_scheduler_without_inbox():
    scheduler = SimpleNamespace(abort=MagicMock(), stop=MagicMock())
    stage = _make_stage(scheduler)

    stage._on_abort("r0")

    scheduler.abort.assert_called_once_with("r0")


def test_enqueue_scheduler_control_puts_shutdown_in_inbox():
    scheduler = SimpleNamespace(
        inbox=queue.Queue(),
        tp_size=2,
        _apply_envelope=MagicMock(),
        _tp_shutdown_requested=False,
    )
    stage = _make_stage(scheduler)

    assert stage._enqueue_scheduler_control("__tp__", "shutdown") is True

    msg = scheduler.inbox.get_nowait()
    assert msg == IncomingMessage(request_id="__tp__", type="shutdown")
    assert stage._scheduler_shutdown_enqueued is True


def test_enqueue_scheduler_control_rejects_non_omni_inbox_scheduler():
    scheduler = SimpleNamespace(inbox=queue.Queue(), tp_size=2)
    stage = _make_stage(scheduler)

    assert stage._enqueue_scheduler_control("__tp__", "shutdown") is False
    assert scheduler.inbox.empty()


def test_stop_does_not_preempt_pending_tp_shutdown_envelope():
    scheduler = SimpleNamespace(
        inbox=queue.Queue(),
        stop=MagicMock(),
        tp_size=2,
        _apply_envelope=MagicMock(),
        _tp_shutdown_requested=False,
    )
    stage = _make_stage(scheduler)
    stage._scheduler_shutdown_enqueued = True

    asyncio.run(stage.stop())

    scheduler.stop.assert_not_called()
