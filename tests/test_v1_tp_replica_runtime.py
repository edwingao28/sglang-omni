# SPDX-License-Identifier: Apache-2.0
"""Rank 0 owns the external Stage; rank >=1 runs scheduler-replica only."""
from __future__ import annotations

import logging
import os
import sys
import types
from unittest.mock import MagicMock, patch

from sglang_omni_v1.pipeline.stage_process import (
    StageProcessSpec,
    _init_torch_distributed,
    stage_process_main,
)
from sglang_omni_v1.scheduling.messages import IncomingMessage


def _spec(tp_rank: int) -> StageProcessSpec:
    return StageProcessSpec(
        stage_name="thinker",
        tp_rank=tp_rank,
        tp_size=2,
        gpu_id=tp_rank,
        nccl_port=29501,
        factory="tests.fake.factory",
        factory_args={"tp_rank": tp_rank, "tp_size": 2, "gpu_id": tp_rank},
        recv_endpoint="ipc:///tmp/thinker.sock",
        coordinator_endpoint="ipc:///tmp/completion.sock",
        abort_endpoint="ipc:///tmp/abort.sock",
    )


def test_stage_process_main_runs_full_stage_on_rank_0() -> None:
    ready = MagicMock()
    with (
        patch("sglang_omni_v1.pipeline.stage_process._setup_cuda_device"),
        patch("sglang_omni_v1.pipeline.stage_process._init_torch_distributed"),
        patch("sglang_omni_v1.pipeline.stage_process._run_stage") as run_stage,
        patch(
            "sglang_omni_v1.pipeline.stage_process._run_tp_replica"
        ) as run_replica,
    ):
        stage_process_main(_spec(0), ready)
    run_stage.assert_called_once()
    run_replica.assert_not_called()


def test_stage_process_main_runs_scheduler_replica_on_rank_1() -> None:
    ready = MagicMock()
    with (
        patch("sglang_omni_v1.pipeline.stage_process._setup_cuda_device"),
        patch("sglang_omni_v1.pipeline.stage_process._init_torch_distributed"),
        patch("sglang_omni_v1.pipeline.stage_process._run_stage") as run_stage,
        patch(
            "sglang_omni_v1.pipeline.stage_process._run_tp_replica"
        ) as run_replica,
    ):
        stage_process_main(_spec(1), ready)
    run_stage.assert_not_called()
    run_replica.assert_called_once()


def test_incoming_message_supports_abort_and_shutdown() -> None:
    abort = IncomingMessage(request_id="r1", type="abort")
    shutdown = IncomingMessage(request_id="__tp__", type="shutdown")
    assert abort.type == "abort"
    assert shutdown.type == "shutdown"


def test_init_torch_distributed_keeps_visible_devices_and_uses_spec_gpu(
    monkeypatch,
) -> None:
    set_device_calls = []
    init_process_group_calls = []

    class FakeCuda:
        def set_device(self, device: int) -> None:
            set_device_calls.append(device)

    fake_torch = types.ModuleType("torch")
    fake_torch.__path__ = []
    fake_torch.cuda = FakeCuda()

    fake_dist = types.ModuleType("torch.distributed")

    def fake_init_process_group(**kwargs):
        init_process_group_calls.append(kwargs)

    fake_dist.init_process_group = fake_init_process_group
    fake_torch.distributed = fake_dist

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    spec = _spec(tp_rank=1)
    _init_torch_distributed(spec, logging.getLogger("test.v1.tp"))

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert set_device_calls == [spec.gpu_id]
    assert init_process_group_calls == [
        {"backend": "nccl", "world_size": spec.tp_size, "rank": spec.tp_rank}
    ]
