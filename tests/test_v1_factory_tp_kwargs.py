# SPDX-License-Identifier: Apache-2.0
"""Stage factory entry points must accept tp_rank / tp_size kwargs."""
from __future__ import annotations

import inspect

import pytest

from sglang_omni_v1.models.qwen3_omni.stages import (
    create_sglang_thinker_executor_from_config,
    create_talker_ar_executor_from_config,
)


def test_thinker_stage_accepts_tp_kwargs():
    sig = inspect.signature(create_sglang_thinker_executor_from_config)
    assert "tp_rank" in sig.parameters
    assert "tp_size" in sig.parameters


def test_talker_stage_accepts_tp_kwargs():
    sig = inspect.signature(create_talker_ar_executor_from_config)
    assert "tp_rank" in sig.parameters
    assert "tp_size" in sig.parameters


def test_create_thinker_scheduler_accepts_tp_kwargs():
    from sglang_omni_v1.models.qwen3_omni.bootstrap import create_thinker_scheduler

    sig = inspect.signature(create_thinker_scheduler)
    assert "tp_rank" in sig.parameters
    assert "tp_size" in sig.parameters


def test_create_talker_scheduler_accepts_tp_kwargs():
    from sglang_omni_v1.models.qwen3_omni.bootstrap import create_talker_scheduler

    sig = inspect.signature(create_talker_scheduler)
    assert "tp_rank" in sig.parameters
    assert "tp_size" in sig.parameters


def test_create_talker_scheduler_rejects_tp_size_gt_1():
    from sglang_omni_v1.models.qwen3_omni.bootstrap import create_talker_scheduler

    fake_server_args = type("S", (), {"tp_size": 2})()
    with pytest.raises((NotImplementedError, ValueError, AssertionError)):
        create_talker_scheduler(fake_server_args, gpu_id=0, tp_rank=0, tp_size=2)


def test_create_sglang_infrastructure_accepts_tp_kwargs():
    from sglang_omni_v1.scheduling.bootstrap import create_sglang_infrastructure

    sig = inspect.signature(create_sglang_infrastructure)
    assert "tp_rank" in sig.parameters


def test_thinker_stage_overrides_align_tp_size_with_server_args(monkeypatch):
    from sglang_omni_v1.models.qwen3_omni import stages as qwen3_stages

    captured = {}

    def fake_build(model_path, *, context_length, **overrides):
        del model_path, context_length
        captured.update(overrides)
        return type("SA", (), {"tp_size": overrides.get("tp_size", 1)})()

    monkeypatch.setattr(qwen3_stages, "build_sglang_server_args", fake_build)
    monkeypatch.setattr(
        "sglang_omni_v1.models.qwen3_omni.bootstrap.create_thinker_scheduler",
        lambda *args, **kwargs: None,
    )

    qwen3_stages.create_sglang_thinker_executor_from_config(
        "dummy",
        gpu_id=0,
        tp_rank=0,
        tp_size=2,
    )
    assert captured.get("tp_size") == 2
