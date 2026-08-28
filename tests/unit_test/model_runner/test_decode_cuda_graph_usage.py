# SPDX-License-Identifier: Apache-2.0
"""Decode-graph replay attestation: the census must be positive replay
evidence, because a silent fall back to eager fails nothing.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from sglang_omni.model_runner.model_worker import (
    ModelWorker,
    _DecodeCudaGraphUsage,
    _PrefillCudaGraphUsage,
)


def _forward_batch(
    num_tokens: int,
    *,
    forward_mode: ForwardMode = ForwardMode.DECODE,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        forward_mode=forward_mode,
    )


def _worker(
    *,
    decode_runner: object | None,
    outcomes: list[bool],
) -> SimpleNamespace:
    results = iter(outcomes)

    def forward(*, forward_batch: object) -> object:
        del forward_batch
        return SimpleNamespace(
            logits_output="out",
            can_run_graph=next(results),
            expert_distribution_metrics=None,
        )

    worker = object.__new__(ModelWorker)
    worker.dllm_algorithm = None
    worker.model_runner = SimpleNamespace(
        forward=forward,
        prefill_cuda_graph_runner=None,
        decode_cuda_graph_runner=decode_runner,
    )
    worker._prefill_cuda_graph_usage = _PrefillCudaGraphUsage()
    worker._decode_cuda_graph_usage = _DecodeCudaGraphUsage()
    worker.server_args = SimpleNamespace(
        model_path="model",
        load_format="auto",
        weight_version=None,
        tp_size=1,
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend="breakable", bs=[16, 32])
        ),
    )
    worker.tp_rank = 0
    worker.model_arch_override = "Qwen3OmniTalker"
    return worker


def test_decode_usage_instances_do_not_share_buckets() -> None:
    first = _DecodeCudaGraphUsage()
    second = _DecodeCudaGraphUsage()

    first.replay_buckets[8] += 1

    assert first.replay_buckets == {8: 1}
    assert second.replay_buckets == {}


def test_reports_decode_replays_by_bucket_and_ignores_prefill() -> None:
    decode_runner = SimpleNamespace(
        capture_bs=[1, 8, 32],
        compile_bs=[],
        enable_torch_compile=False,
        bs=8,
    )
    worker = _worker(decode_runner=decode_runner, outcomes=[True, False, True, False])

    ModelWorker.forward_batch_generation(worker, _forward_batch(8))
    ModelWorker.forward_batch_generation(worker, _forward_batch(8))
    decode_runner.bs = 32
    ModelWorker.forward_batch_generation(worker, _forward_batch(32))
    # Note (wenyao): extend must not touch the decode census; it runs eager so
    # the runner-less stub stays out of the prefill census internals.
    ModelWorker.forward_batch_generation(
        worker, _forward_batch(40, forward_mode=ForwardMode.EXTEND)
    )

    stats = ModelWorker.model_info(worker)["decode_cuda_graph"]

    assert stats["replay_count"] == 2
    assert stats["eager_count"] == 1
    assert stats["replay_buckets"] == {"8": 1, "32": 1}
    assert stats["capture_bs"] == [1, 8, 32]
    assert json.loads(json.dumps(stats)) == stats


def test_first_eager_decode_step_warns_exactly_once(caplog) -> None:
    worker = _worker(
        decode_runner=SimpleNamespace(
            capture_bs=[1], compile_bs=[], enable_torch_compile=False, bs=1
        ),
        outcomes=[False, False, False],
    )

    with caplog.at_level(
        logging.WARNING, logger="sglang_omni.model_runner.model_worker"
    ):
        for _ in range(3):
            ModelWorker.forward_batch_generation(worker, _forward_batch(1))

    misses = [r for r in caplog.records if "[decode-graph miss]" in r.getMessage()]
    assert len(misses) == 1
    assert "ran EAGER" in misses[0].getMessage()
    assert ModelWorker.model_info(worker)["decode_cuda_graph"]["eager_count"] == 3


def test_attests_once_on_first_replay(caplog) -> None:
    worker = _worker(
        decode_runner=SimpleNamespace(
            capture_bs=[1], compile_bs=[1], enable_torch_compile=True, bs=1
        ),
        outcomes=[True, True],
    )

    with caplog.at_level(logging.INFO, logger="sglang_omni.model_runner.model_worker"):
        ModelWorker.forward_batch_generation(worker, _forward_batch(1))
        ModelWorker.forward_batch_generation(worker, _forward_batch(1))

    attestations = [
        r for r in caplog.records if "[decode-graph attestation]" in r.getMessage()
    ]
    assert len(attestations) == 1


def test_reports_torch_compile_state() -> None:
    worker = _worker(
        decode_runner=SimpleNamespace(
            capture_bs=[1, 8], compile_bs=[1], enable_torch_compile=True, bs=1
        ),
        outcomes=[],
    )

    stats = ModelWorker.model_info(worker)["decode_cuda_graph"]

    assert stats["enable_torch_compile"] is True
    assert stats["compile_bs"] == [1]


def test_compile_mode_defaults_to_the_no_cudagraphs_mode(monkeypatch) -> None:
    monkeypatch.delenv("SGLANG_TORCH_COMPILE_MODE", raising=False)
    worker = _worker(decode_runner=None, outcomes=[])

    stats = ModelWorker.model_info(worker)["decode_cuda_graph"]

    assert stats["compile_mode"] == "max-autotune-no-cudagraphs"
    assert stats["compile_mode_from_env"] is False


def test_compile_mode_override_is_reported_as_from_env(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_TORCH_COMPILE_MODE", "reduce-overhead")
    worker = _worker(decode_runner=None, outcomes=[])

    stats = ModelWorker.model_info(worker)["decode_cuda_graph"]

    assert stats["compile_mode"] == "reduce-overhead"
    assert stats["compile_mode_from_env"] is True


def test_missing_decode_runner_reports_null_rather_than_raising() -> None:
    worker = _worker(decode_runner=None, outcomes=[True])

    ModelWorker.forward_batch_generation(worker, _forward_batch(1))
    stats = ModelWorker.model_info(worker)["decode_cuda_graph"]

    assert stats["runner"] is None
    assert stats["capture_bs"] is None
    assert stats["replay_count"] == 1
    assert stats["replay_buckets"] == {}


def test_decode_mode_is_graph_eligible_which_must_not_gate_the_census() -> None:
    assert ForwardMode.DECODE.is_decode() is True
    assert ForwardMode.DECODE.is_cuda_graph() is True

    worker = _worker(
        decode_runner=SimpleNamespace(
            capture_bs=[1], compile_bs=[], enable_torch_compile=False, bs=1
        ),
        outcomes=[True],
    )
    ModelWorker.forward_batch_generation(worker, _forward_batch(1))

    assert ModelWorker.model_info(worker)["decode_cuda_graph"]["replay_count"] == 1
