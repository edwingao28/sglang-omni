# SPDX-License-Identifier: Apache-2.0
"""Tests for the talker prefill quota (decode-priority mixing) knobs.

Two opt-in levers keep ongoing talker decodes at their steady cadence while
new requests' prefills trickle in:

- ``prefill_max_requests`` (upstream ServerArgs field, plumbed through the
  talker factory) caps how many whole-sequence prefills one batch admits.
- ``prefill_decode_interleave`` (OmniScheduler knob) guarantees a decode step
  between prefill steps, overriding upstream's strict prefill-first policy.

Both default to current behavior (off).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("sglang")

from sglang.srt.managers.schedule_batch import NextBatchPlan  # noqa: E402

from sglang_omni.cli.serve import (  # noqa: E402
    apply_talker_prefill_quota_cli_overrides,
)
from sglang_omni.config import resolve_stage_factory_args  # noqa: E402
from sglang_omni.models.qwen3_omni.config import (  # noqa: E402
    Qwen3OmniSpeechPipelineConfig,
)
from sglang_omni.scheduling import omni_scheduler  # noqa: E402
from sglang_omni.scheduling.generation_batch_policy import (  # noqa: E402
    build_generation_batch_overrides,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler  # noqa: E402

_UPSTREAM_BATCH = object()


class _StubScheduler:
    """The attribute surface get_new_batch_prefill touches."""

    def __init__(
        self,
        *,
        prefill_decode_interleave: bool = False,
        running_is_empty: bool = False,
        running_is_prefill_only: bool = False,
    ) -> None:
        self.prefill_decode_interleave = prefill_decode_interleave
        self._interleave_defer_prefill = False
        # Coalesce gate disabled: quota tests exercise the interleave wrapper.
        self.prefill_coalesce_requests = 0
        self.prefill_coalesce_wait_s = 0.06
        self.prefill_coalesce_when_idle = False
        self.prefill_coalesce_requires_pending_builds = False
        self.prefill_coalesce_after_builds_during_decode = False
        self.chunked_req = None
        self.waiting_queue: list = []
        self.running_batch = SimpleNamespace(
            is_empty=lambda: running_is_empty,
            is_prefill_only=running_is_prefill_only,
        )
        self._request_admission_lock = threading.RLock()
        self._pending_request_builds: dict = {}
        self._backlogged_request_build_payloads: list = []

    def _get_new_batch_prefill_coalesced(self, running_batch):
        return OmniScheduler._get_new_batch_prefill_coalesced(self, running_batch)

    def get_new_batch_prefill(self):
        plan = OmniScheduler.get_new_batch_prefill(self, self.running_batch)
        return plan.batch_to_run


@pytest.fixture()
def upstream():
    with mock.patch.object(
        omni_scheduler._Upstream,
        "get_new_batch_prefill",
        return_value=NextBatchPlan(batch_to_run=_UPSTREAM_BATCH, running_batch=None),
    ) as patched:
        yield patched


def test_interleave_off_never_defers(upstream):
    sched = _StubScheduler(prefill_decode_interleave=False)
    for _ in range(3):
        assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
    assert upstream.call_count == 3


def test_interleave_alternates_prefill_and_decode(upstream):
    sched = _StubScheduler(prefill_decode_interleave=True)

    # Step 1: prefill scheduled, defer flag armed.
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
    # Step 2: prefill deferred so the decode running batch gets the step.
    assert sched.get_new_batch_prefill() is None
    assert upstream.call_count == 1
    # Step 3: prefill admitted again.
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
    assert upstream.call_count == 2


def test_interleave_defer_preserves_running_batch(upstream):
    sched = _StubScheduler(prefill_decode_interleave=True)
    sched.get_new_batch_prefill()
    plan = OmniScheduler.get_new_batch_prefill(sched, sched.running_batch)
    assert plan.batch_to_run is None
    assert plan.running_batch is sched.running_batch


def test_interleave_skips_defer_without_decode_work(upstream):
    sched = _StubScheduler(prefill_decode_interleave=True, running_is_empty=True)
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
    # No decode step to protect: the next prefill goes straight through.
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
    assert upstream.call_count == 2


def test_interleave_skips_defer_for_prefill_only_running_batch(upstream):
    sched = _StubScheduler(
        prefill_decode_interleave=True, running_is_prefill_only=True
    )
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
    assert upstream.call_count == 2


def test_interleave_chunked_prefill_bypasses_defer(upstream):
    sched = _StubScheduler(prefill_decode_interleave=True)
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
    sched.chunked_req = object()
    # A chunked prefill in flight must proceed or its KV leaks.
    assert sched.get_new_batch_prefill() is _UPSTREAM_BATCH
    assert upstream.call_count == 2


def test_interleave_empty_upstream_plan_does_not_arm_defer():
    sched = _StubScheduler(prefill_decode_interleave=True)
    with mock.patch.object(
        omni_scheduler._Upstream,
        "get_new_batch_prefill",
        return_value=NextBatchPlan(batch_to_run=None, running_batch=None),
    ) as patched:
        assert sched.get_new_batch_prefill() is None
        assert sched.get_new_batch_prefill() is None
        assert patched.call_count == 2
    assert sched._interleave_defer_prefill is False


def test_scheduler_default_keeps_interleave_off():
    # OmniScheduler.__init__ defaults must preserve current behavior; assert
    # the signature default rather than constructing GPU infrastructure.
    import inspect

    sig = inspect.signature(OmniScheduler.__init__)
    assert sig.parameters["prefill_decode_interleave"].default is False


def test_stage_factory_rejects_invalid_prefill_max_requests():
    from sglang_omni.models.qwen3_omni.stages import (
        create_talker_ar_executor_from_config,
    )

    with pytest.raises(ValueError, match="prefill_max_requests"):
        create_talker_ar_executor_from_config(
            "dummy-model-path",
            prefill_max_requests=0,
        )


def test_prefill_max_requests_stage_default_and_operator_precedence():
    # The talker factory passes prefill_max_requests as a stage default;
    # an operator server_args_overrides value must win.
    overrides = build_generation_batch_overrides(
        max_running_requests=32,
        prefill_max_requests=2,
        server_args_overrides=None,
    )
    assert overrides["prefill_max_requests"] == 2

    overrides = build_generation_batch_overrides(
        max_running_requests=32,
        prefill_max_requests=2,
        server_args_overrides={"prefill_max_requests": 8},
    )
    assert overrides["prefill_max_requests"] == 8


def test_cli_override_sets_talker_factory_args():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    apply_talker_prefill_quota_cli_overrides(
        config,
        talker_prefill_max_requests=2,
        talker_prefill_decode_interleave="on",
    )
    talker = next(stage for stage in config.stages if stage.name == "talker_ar")
    talker_args = resolve_stage_factory_args(talker, config)
    assert talker_args["prefill_max_requests"] == 2
    assert talker_args["prefill_decode_interleave"] is True


def test_cli_override_off_disables_interleave_only():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    apply_talker_prefill_quota_cli_overrides(
        config,
        talker_prefill_max_requests=None,
        talker_prefill_decode_interleave="off",
    )
    talker = next(stage for stage in config.stages if stage.name == "talker_ar")
    talker_args = resolve_stage_factory_args(talker, config)
    assert talker_args["prefill_decode_interleave"] is False
    assert "prefill_max_requests" not in talker_args


def test_cli_override_default_is_noop():
    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    apply_talker_prefill_quota_cli_overrides(
        config,
        talker_prefill_max_requests=None,
        talker_prefill_decode_interleave="default",
    )
    talker = next(stage for stage in config.stages if stage.name == "talker_ar")
    talker_args = resolve_stage_factory_args(talker, config)
    assert "prefill_max_requests" not in talker_args
    assert "prefill_decode_interleave" not in talker_args
