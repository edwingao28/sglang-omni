# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from sglang_omni.model_runner.model_worker import ModelWorker


class _PrefillRunner:
    capture_num_tokens = [16, 32]

    def __init__(self) -> None:
        self._static_num_tokens = 0
        self.backend = SimpleNamespace(_omni_qualification_eager_replay=True)
        self.buffer_registry = SimpleNamespace(
            has_slot=lambda name: name == "input_embeds"
        )


def _forward_batch(
    num_tokens: int,
    *,
    forward_mode: ForwardMode = ForwardMode.EXTEND,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        forward_mode=forward_mode,
    )


def test_model_worker_reports_actual_prefill_graph_replays_by_bucket() -> None:
    prefill_runner = _PrefillRunner()
    outcomes = iter(
        [
            (
                16,
                SimpleNamespace(
                    logits_output="graph-16",
                    can_run_graph=True,
                    expert_distribution_metrics=None,
                ),
            ),
            (
                None,
                SimpleNamespace(
                    logits_output="eager",
                    can_run_graph=False,
                    expert_distribution_metrics=None,
                ),
            ),
            (
                32,
                SimpleNamespace(
                    logits_output="graph-32",
                    can_run_graph=True,
                    expert_distribution_metrics=None,
                ),
            ),
            # Decode graphs must not be reported as prefill replays.
            (
                None,
                SimpleNamespace(
                    logits_output="decode-graph",
                    can_run_graph=True,
                    expert_distribution_metrics=None,
                ),
            ),
            # TARGET_VERIFY is an extend-like mode but belongs to the decode
            # graph runner and must not consume a stale prefill bucket.
            (
                None,
                SimpleNamespace(
                    logits_output="target-verify-graph",
                    can_run_graph=True,
                    expert_distribution_metrics=None,
                ),
            ),
            # DRAFT_EXTEND_V2 is a prefill-runner mode even though the default
            # ForwardMode.is_extend() call excludes it.
            (
                16,
                SimpleNamespace(
                    logits_output="draft-prefill-graph",
                    can_run_graph=True,
                    expert_distribution_metrics=None,
                ),
            ),
        ]
    )

    def forward(*, forward_batch: object) -> object:
        del forward_batch
        static_num_tokens, outcome = next(outcomes)
        if static_num_tokens is not None:
            prefill_runner._static_num_tokens = static_num_tokens
        return outcome

    runner = SimpleNamespace(
        forward=forward,
        prefill_cuda_graph_runner=prefill_runner,
    )
    worker = object.__new__(ModelWorker)
    worker.dllm_algorithm = None
    worker.model_runner = runner
    worker.server_args = SimpleNamespace(
        model_path="model",
        load_format="auto",
        weight_version=None,
        tp_size=1,
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(
                backend="breakable",
                bs=[16, 32],
            )
        ),
    )
    worker.tp_rank = 0
    worker.model_arch_override = None
    worker._prefill_cuda_graph_debug_snapshot_provider = lambda: {
        "requests": [{"request_id": "debug-request"}]
    }

    # Raw length intentionally does not predict bucket 16; the metric must read
    # the runner's executed static bucket rather than bisecting input length.
    ModelWorker.forward_batch_generation(worker, _forward_batch(5))
    ModelWorker.forward_batch_generation(worker, _forward_batch(40))
    ModelWorker.forward_batch_generation(worker, _forward_batch(31))
    ModelWorker.forward_batch_generation(
        worker,
        _forward_batch(1, forward_mode=ForwardMode.DECODE),
    )
    ModelWorker.forward_batch_generation(
        worker,
        _forward_batch(1, forward_mode=ForwardMode.TARGET_VERIFY),
    )
    ModelWorker.forward_batch_generation(
        worker,
        _forward_batch(3, forward_mode=ForwardMode.DRAFT_EXTEND_V2),
    )
    ModelWorker.record_custom_prefill_eager(worker)

    stats = ModelWorker.model_info(worker)["prefill_cuda_graph"]

    assert stats["backend"] == "breakable"
    assert stats["capture_num_tokens"] == [16, 32]
    assert stats["runner"] == "_PrefillRunner"
    assert stats["backend_runner"] == "SimpleNamespace"
    assert stats["qualification_eager_replay"] is True
    assert stats["upstream_debug_eager"] is False
    assert stats["input_embeds_slot"] is True
    assert stats["replay_count"] == 3
    assert stats["standard_eager_count"] == 1
    assert stats["custom_eager_count"] == 1
    assert stats["replay_buckets"] == {"16": 2, "32": 1}
    assert stats["debug_snapshot"] == {"requests": [{"request_id": "debug-request"}]}
    assert json.loads(json.dumps(stats)) == stats
