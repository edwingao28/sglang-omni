# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from sglang.srt.model_executor import cuda_graph_config
from sglang.srt.model_executor.runner import prefill_cuda_graph_runner

from sglang_omni.model_runner import hybrid_prefill_router as hybrid


@pytest.mark.parametrize("structured", [False, True])
def test_hybrid_extraction_preserves_callers_configuration(monkeypatch, structured):
    monkeypatch.setattr(
        cuda_graph_config, "default_prefill_backend", lambda: "breakable"
    )
    config = {"prefill": {"backend": "hybrid", "bs": [8, 16]}}
    if structured:
        config = cuda_graph_config.CudaGraphConfig.from_dict(config)
    overrides = {
        "cuda_graph_config": config,
        "cuda_graph_bs_prefill_full": [32, 64],
    }
    before = deepcopy(overrides)

    rewritten, full_bs = hybrid.extract_hybrid_prefill_overrides(overrides)

    assert overrides == before
    assert rewritten["cuda_graph_config"]["prefill"]["backend"] == "breakable"
    assert rewritten["cuda_graph_config"]["prefill"]["bs"] == [8, 16]
    assert "cuda_graph_bs_prefill_full" not in rewritten
    assert full_bs == [32, 64]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"cuda_graph_backend_prefill": "hybrid"}, "must be set via"),
        ({"cuda_graph_bs_prefill_full": [32]}, "requires cuda_graph_config"),
        ({"cuda_graph_config": {"prefill": {"backend": "hybrid"}}}, "requires"),
    ],
)
def test_hybrid_extraction_rejects_incomplete_policy(overrides, message):
    with pytest.raises(ValueError, match=message):
        hybrid.extract_hybrid_prefill_overrides(overrides)


@pytest.mark.parametrize("overrides", [None, {}, {"chunked_prefill_size": 1024}])
def test_other_prefill_policies_are_returned_unchanged(overrides):
    rewritten, full_bs = hybrid.extract_hybrid_prefill_overrides(overrides)
    assert rewritten is overrides
    assert full_bs is None


def test_router_prefers_breakable_then_full_and_rejects_eager_shapes():
    def runner(buckets):
        return SimpleNamespace(
            capture_num_tokens=buckets,
            can_run_graph=lambda batch: batch <= buckets[-1],
            execute=lambda batch, **kwargs: (buckets, batch, kwargs),
        )

    router = hybrid.HybridPrefillGraphRouter(runner([8, 16]), runner([32, 64]))
    assert router.capture_num_tokens == [8, 16, 32, 64]
    assert router.execute(8, key="value") == ([8, 16], 8, {"key": "value"})
    assert router.execute(32) == ([32, 64], 32, {})
    assert not router.can_run_graph(65)
    with pytest.raises(AssertionError, match="can_run_graph"):
        router.execute(65)


@pytest.mark.parametrize("outcome", ["success", "capture_failure", "wrong_ladder"])
def test_install_restores_configuration_and_publishes_only_complete_capture(
    monkeypatch, outcome
):
    cfg = SimpleNamespace(backend="breakable", bs=[8, 16], full_prefill_max_req=None)
    model_cfg = SimpleNamespace(is_multimodal=False)
    model_runner = SimpleNamespace(
        model_config=model_cfg, req_to_token_pool=SimpleNamespace(size=3)
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_exec",
        lambda: SimpleNamespace(
            graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(prefill=cfg))
        ),
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_schedule",
        lambda: SimpleNamespace(chunked_prefill_size=4096),
    )
    worker = SimpleNamespace(
        model_runner=model_runner, enable_prefill_input_embeds=True
    )

    class CapturedRunner:
        def __init__(self, actual_runner):
            assert actual_runner is model_runner
            assert (cfg.backend, cfg.bs, cfg.full_prefill_max_req) == (
                "full",
                [32, 64],
                3,
            )
            assert model_cfg.is_multimodal
            if outcome == "capture_failure":
                raise RuntimeError("injected capture failure")
            self.capture_num_tokens = [32] if outcome == "wrong_ladder" else [32, 64]

    breakable = object.__new__(CapturedRunner)
    breakable.capture_num_tokens = [8, 16]
    breakable.max_num_tokens = 16
    model_runner.prefill_cuda_graph_runner = breakable
    monkeypatch.setattr(
        prefill_cuda_graph_runner, "PrefillCudaGraphRunner", CapturedRunner
    )

    if outcome == "success":
        hybrid.install_hybrid_full_prefill(worker, [32, 64])
        installed = model_runner.prefill_cuda_graph_runner
        assert isinstance(installed, hybrid.HybridPrefillGraphRouter)
        assert installed.breakable_runner is breakable
        assert installed.full_runner.capture_num_tokens == [32, 64]
    else:
        message = (
            "injected capture failure"
            if outcome == "capture_failure"
            else "shapes differ"
        )
        with pytest.raises(RuntimeError, match=message):
            hybrid.install_hybrid_full_prefill(worker, [32, 64])
        assert model_runner.prefill_cuda_graph_runner is breakable
    assert (cfg.backend, cfg.bs, cfg.full_prefill_max_req) == (
        "breakable",
        [8, 16],
        None,
    )
    assert not model_cfg.is_multimodal


def test_install_rejects_missing_breakable_capture_and_overlapping_ladders(monkeypatch):
    class CapturedRunner:
        max_num_tokens = 16

    monkeypatch.setattr(
        prefill_cuda_graph_runner, "PrefillCudaGraphRunner", CapturedRunner
    )
    model_runner = SimpleNamespace(prefill_cuda_graph_runner=None)
    worker = SimpleNamespace(model_runner=model_runner)
    with pytest.raises(RuntimeError, match="BREAKABLE runner"):
        hybrid.install_hybrid_full_prefill(worker, [32])
    model_runner.prefill_cuda_graph_runner = CapturedRunner()
    with pytest.raises(ValueError, match="must exceed"):
        hybrid.install_hybrid_full_prefill(worker, [16, 32])
