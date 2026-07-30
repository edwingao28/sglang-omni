# SPDX-License-Identifier: Apache-2.0
"""Talker async-decode wiring and the retract drain that protects it.

``enable_async_decode`` has to reach two places to do anything: the scheduler's
``enable_async_decode`` (which selects the lookahead event loop) and the model
runner's ``_async_enabled``, which the runner attaches too late for the
scheduler constructor to set. With the loop live, a retract must not find a
launched-but-unresolved step: ``retract_decode`` frees the KV and flags the request
before handing it back, after which the resolve drops that row and its token.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest
from sglang.srt.managers import scheduler as _upstream_scheduler
from sglang.srt.managers.scheduler import Scheduler as _Upstream

from sglang_omni.models.qwen3_omni import bootstrap as qwen_bootstrap
from sglang_omni.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler


class _Config:
    """Stands in for the HF config tree: any attribute answers with another bag."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def __getattr__(self, name: str) -> Any:
        child = _Config()
        setattr(self, name, child)
        return child


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    model_config = _Config(model_path="dummy")
    worker = _Config()

    def _infrastructure(*args: Any, **kwargs: Any):
        return (worker, None, None, None, None, None, model_config)

    class _Scheduler:
        def __init__(self, **kwargs: Any) -> None:
            seen["scheduler_kwargs"] = kwargs
            self.outbox = SimpleNamespace()

    class _Runner:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._async_enabled = None

    monkeypatch.setattr(
        "sglang_omni.scheduling.bootstrap.create_sglang_infrastructure",
        _infrastructure,
    )
    monkeypatch.setattr(
        "sglang.srt.utils.hf_transformers_utils.get_tokenizer",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.request_builders.make_talker_scheduler_adapters",
        lambda **kwargs: (None, None, None, None),
    )
    monkeypatch.setattr(
        "sglang_omni.scheduling.sglang_backend.SGLangOutputProcessor",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.talker_scheduler.QwenTalkerScheduler",
        _Scheduler,
    )
    monkeypatch.setattr(
        "sglang_omni.models.qwen3_omni.talker_model_runner.QwenTalkerModelRunner",
        _Runner,
    )
    return seen


def _server_args() -> SimpleNamespace:
    return SimpleNamespace(
        disable_cuda_graph=True,
        disable_overlap_schedule=False,
        disable_radix_cache=False,
        chunked_prefill_size=8192,
    )


def test_enable_async_decode_defaults_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_stubs(monkeypatch)

    scheduler = qwen_bootstrap.create_talker_scheduler(_server_args())

    assert seen["scheduler_kwargs"]["enable_async_decode"] is False
    assert scheduler._model_runner._async_enabled is False


@pytest.mark.parametrize("enable_async_decode", [False, True])
def test_enable_async_decode_reaches_scheduler_and_runner(
    monkeypatch: pytest.MonkeyPatch, enable_async_decode: bool
) -> None:
    seen = _install_stubs(monkeypatch)

    scheduler = qwen_bootstrap.create_talker_scheduler(
        _server_args(), enable_async_decode=enable_async_decode
    )

    assert seen["scheduler_kwargs"]["enable_async_decode"] is enable_async_decode
    assert scheduler._model_runner._async_enabled is enable_async_decode


def test_feedback_disabled_keeps_async_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without the feedback path there is no launch/resolve split to run.
    seen = _install_stubs(monkeypatch)

    scheduler = qwen_bootstrap.create_talker_scheduler(
        _server_args(), feedback_enabled=False, enable_async_decode=True
    )

    assert seen["scheduler_kwargs"]["enable_async_decode"] is False
    assert scheduler._model_runner._async_enabled is False


def _new_scheduler_for_start(*, enable_async_decode: bool) -> QwenTalkerScheduler:
    # Note (wenyao): __new__ + minimal attrs, not a real __init__, which needs a
    # full server_args/model_config/tp_worker fixture this test doesn't need.
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler.enable_async_decode = enable_async_decode
    scheduler.enable_overlap = False
    scheduler._shutdown_lock = threading.Lock()
    scheduler._shutdown_callback = None
    scheduler._request_build_executor = None
    return scheduler


@pytest.mark.parametrize(
    ("enable_async_decode", "expected_loop"),
    [(True, "_event_loop_async_decode"), (False, "_event_loop_normal")],
)
def test_start_selects_loop_from_enable_async_decode(
    monkeypatch: pytest.MonkeyPatch, enable_async_decode: bool, expected_loop: str
) -> None:
    scheduler = _new_scheduler_for_start(enable_async_decode=enable_async_decode)
    calls: list[str] = []
    monkeypatch.setattr(
        QwenTalkerScheduler,
        "_event_loop_async_decode",
        lambda self: calls.append("_event_loop_async_decode"),
    )
    monkeypatch.setattr(
        QwenTalkerScheduler,
        "_event_loop_normal",
        lambda self: calls.append("_event_loop_normal"),
    )

    scheduler.start()

    assert calls == [expected_loop]


def _drain_scheduler(monkeypatch: pytest.MonkeyPatch, *, pending: bool) -> Any:
    calls: list[str] = []
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler._async_pending = ("batch", "sched_output", "step") if pending else None
    scheduler.forward_ct = 1
    scheduler._resolve_pending_async = lambda: calls.append("drain")
    monkeypatch.setattr(
        _Upstream,
        "update_running_batch",
        lambda self, batch: calls.append("upstream") or batch,
    )
    monkeypatch.setattr(_upstream_scheduler, "TEST_RETRACT", False)
    monkeypatch.setattr(_upstream_scheduler, "TEST_RETRACT_INTERVAL", 1)
    return scheduler, calls


def _batch(*, decode_mem_ok: bool, probes: list[str] | None = None) -> SimpleNamespace:
    def _check_decode_mem() -> bool:
        if probes is not None:
            probes.append("check_decode_mem")
        return decode_mem_ok

    return SimpleNamespace(reqs=[object()], check_decode_mem=_check_decode_mem)


def test_drain_runs_before_upstream_can_retract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, calls = _drain_scheduler(monkeypatch, pending=True)

    scheduler.update_running_batch(_batch(decode_mem_ok=False))

    assert calls == ["drain", "upstream"]


def test_no_drain_when_memory_is_fine(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler, calls = _drain_scheduler(monkeypatch, pending=True)

    scheduler.update_running_batch(_batch(decode_mem_ok=True))

    assert calls == ["upstream"]


def test_test_retract_interval_also_drains(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler, calls = _drain_scheduler(monkeypatch, pending=True)
    monkeypatch.setattr(_upstream_scheduler, "TEST_RETRACT", True)
    scheduler.forward_ct = 4
    monkeypatch.setattr(_upstream_scheduler, "TEST_RETRACT_INTERVAL", 2)

    scheduler.update_running_batch(_batch(decode_mem_ok=True))

    assert calls == ["drain", "upstream"]


def test_sync_talker_never_probes_memory_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With the flag off there is nothing in flight, so the pre-check must not add a
    # second check_decode_mem to every decode step.
    scheduler, calls = _drain_scheduler(monkeypatch, pending=False)
    probes: list[str] = []

    scheduler.update_running_batch(_batch(decode_mem_ok=False, probes=probes))

    assert calls == ["upstream"]
    assert probes == []


def test_empty_batch_is_not_a_retract_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, calls = _drain_scheduler(monkeypatch, pending=True)

    scheduler.update_running_batch(SimpleNamespace(reqs=[]))

    assert calls == ["upstream"]
