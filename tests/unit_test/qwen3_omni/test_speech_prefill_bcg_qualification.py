# SPDX-License-Identifier: Apache-2.0

import base64
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.test_model.qwen3_omni_speech_prefill_bcg_assertions import (
    assert_exact_prefill_embeddings,
    assert_prefill_parity,
    prefill_parity_diagnostics,
    snapshot,
)


def _tensor_payload(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "data": base64.b64encode(tensor.numpy().tobytes()).decode("ascii"),
    }


def _probe(
    *,
    content: str,
    request_id: str = "request-1",
    companion_request_id: str | None = None,
) -> dict[str, Any]:
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    logits = torch.tensor([0.5, 1.5, -0.25], dtype=torch.float32)
    requests = []
    if companion_request_id is not None:
        requests.append(
            {
                "request_id": companion_request_id,
                "logical_rows": 3,
                "next_token_id": 4,
                "hidden_states": {
                    "embed": _tensor_payload(torch.zeros((3, 2))),
                    "24": _tensor_payload(torch.zeros((3, 2))),
                },
                "next_token_logits": _tensor_payload(torch.zeros(3)),
            }
        )
    requests.append(
        {
            "request_id": request_id,
            "logical_rows": 2,
            "next_token_id": 9,
            "hidden_states": {
                "embed": _tensor_payload(hidden),
                "24": _tensor_payload(hidden),
            },
            "next_token_logits": _tensor_payload(logits),
        }
    )
    return {
        "body": {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 2},
        },
        "after": {
            "debug_snapshot": {
                "logical_rows": sum(item["logical_rows"] for item in requests),
                "requests": requests,
            }
        },
    }


def test_prefill_parity_ignores_post_prefill_decode_text_differences() -> None:
    errors = assert_prefill_parity(
        _probe(content="A person is coughing."),
        _probe(content="A person coughing."),
        request_id="request-1",
    )

    assert errors == {
        "hidden_embed": 0.0,
        "hidden_24": 0.0,
        "next_token_logits": 0.0,
    }


def test_prefill_parity_selects_request_from_mixed_batch() -> None:
    eager = _probe(
        content="same",
        request_id="selected-request",
        companion_request_id="companion-request",
    )
    graph = _probe(content="same", request_id="comparison-request")

    errors = assert_prefill_parity(
        eager,
        graph,
        eager_request_id="selected-request",
        graph_request_id="comparison-request",
    )

    assert snapshot(eager, "selected-request")["logical_rows"] == 2
    assert errors == {
        "hidden_embed": 0.0,
        "hidden_24": 0.0,
        "next_token_logits": 0.0,
    }


def test_prefill_parity_diagnostics_reports_mismatch_without_raising() -> None:
    eager = _probe(content="same")
    graph = _probe(content="same")
    graph_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.5]], dtype=torch.float32)
    graph_request = graph["after"]["debug_snapshot"]["requests"][0]
    graph_request["hidden_states"]["embed"] = _tensor_payload(graph_hidden)

    diagnostics = prefill_parity_diagnostics(
        eager,
        graph,
        request_id="request-1",
    )

    assert diagnostics["next_token_match"] is True
    assert diagnostics["hidden_states"]["embed"]["mismatch_count"] == 1
    assert diagnostics["hidden_states"]["embed"]["max_abs_error"] == 0.5
    assert diagnostics["hidden_states"]["embed"]["mean_abs_error"] == 0.125
    assert diagnostics["hidden_states"]["embed"]["root_mean_square_error"] == 0.25
    assert diagnostics["hidden_states"]["embed"]["mismatched_rows"] == [1]


def test_exact_prefill_embeddings_rejects_changed_copy() -> None:
    eager = _probe(content="same")
    graph = _probe(content="same")
    graph_request = graph["after"]["debug_snapshot"]["requests"][0]
    graph_request["hidden_states"]["embed"] = _tensor_payload(
        torch.tensor([[1.0, 2.0], [3.0, 4.0001]], dtype=torch.float32)
    )

    with pytest.raises(AssertionError):
        assert_exact_prefill_embeddings(
            eager,
            graph,
            request_id="request-1",
        )


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_prefill_parity_rejects_matching_infinity(nonfinite: float) -> None:
    eager = _probe(content="same")
    graph = _probe(content="same")
    hidden = torch.tensor([[1.0, nonfinite], [3.0, 4.0]])
    for probe in (eager, graph):
        request = probe["after"]["debug_snapshot"]["requests"][0]
        request["hidden_states"]["24"] = _tensor_payload(hidden)

    with pytest.raises(AssertionError, match="non-finite"):
        assert_prefill_parity(eager, graph, request_id="request-1")


def test_prefill_diagnostics_treat_matching_nan_as_mismatch() -> None:
    eager = _probe(content="same")
    graph = _probe(content="same")
    hidden = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])
    eager_request = eager["after"]["debug_snapshot"]["requests"][0]
    graph_request = graph["after"]["debug_snapshot"]["requests"][0]
    eager_request["hidden_states"]["24"] = _tensor_payload(hidden)
    graph_request["hidden_states"]["24"] = _tensor_payload(hidden)

    diagnostics = prefill_parity_diagnostics(
        eager,
        graph,
        request_id="request-1",
    )

    layer = diagnostics["hidden_states"]["24"]
    assert layer["mismatch_count"] == 1
    assert layer["eager_nonfinite_count"] == 1
    assert layer["graph_nonfinite_count"] == 1
    assert layer["max_abs_error"] == 0.0


def _qualification_eager_replay_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    from sglang_omni.model_runner import prefill_qualification

    calls: list[tuple[object, int]] = []
    observed_dp_padding_modes: list[object] = []
    default_mode_calls: list[None] = []
    capture_dp_padding_mode = SimpleNamespace(is_max_len=lambda: False)

    def cuda_graph_dp_padding_mode() -> object:
        default_mode_calls.append(None)
        return capture_dp_padding_mode

    monkeypatch.setattr(
        prefill_qualification,
        "_cuda_graph_dp_padding_mode",
        cuda_graph_dp_padding_mode,
    )
    backend_type = type(
        "BreakableCudaGraphBackend",
        (),
        {"replay": lambda self, *args, **kwargs: "captured"},
    )
    backend = backend_type()

    class LayerModel:
        forward_error: Exception | None = None

        def forward(self, batch: object, size: int) -> str:
            if self.forward_error is not None:
                raise self.forward_error
            calls.append((batch, size))
            return "eager"

    layer_model = LayerModel()

    @contextmanager
    def original_forward_context(*args, **kwargs):
        calls.append(("unexpected nested context", 0))
        yield

    def fake_run_forward(self, batch, size):
        observed_dp_padding_modes.append(batch.dp_padding_mode)
        batch.dp_padding_mode.is_max_len()
        with self._prefill_forward_context(batch):
            return self.layer_model.forward(batch, size)

    runner_type = type(
        "PrefillCudaGraphRunner",
        (),
        {
            "_run_forward": fake_run_forward,
            "_prefill_forward_context": original_forward_context,
        },
    )
    runner = runner_type()
    runner.backend = backend
    runner.layer_model = layer_model
    model_runner = SimpleNamespace(prefill_cuda_graph_runner=runner)

    prefill_qualification.enable_prefill_qualification_eager_replay(model_runner)
    prefill_qualification.enable_prefill_qualification_eager_replay(model_runner)

    def replay_layer_forward(*args, **kwargs):
        return "captured"

    layer_model.forward = replay_layer_forward
    return SimpleNamespace(
        backend=backend,
        calls=calls,
        capture_dp_padding_mode=capture_dp_padding_mode,
        default_mode_calls=default_mode_calls,
        layer_model=layer_model,
        observed_dp_padding_modes=observed_dp_padding_modes,
        replay_layer_forward=replay_layer_forward,
        runner=runner,
    )


def _qualification_replay(harness: SimpleNamespace, batch: object) -> object:
    return harness.backend.replay(
        SimpleNamespace(size=64),
        batch,
        input_embeds_slot=object(),
    )


def _assert_qualification_replay_state_restored(
    harness: SimpleNamespace,
) -> None:
    assert harness.layer_model.forward is harness.replay_layer_forward
    assert harness.runner._prefill_forward_context == (
        harness.backend._omni_original_prefill_forward_context
    )


def test_qualification_eager_replay_uses_live_static_batch_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _qualification_eager_replay_harness(monkeypatch)
    live_static_batch = SimpleNamespace(dp_padding_mode=None)
    results = []
    for _ in range(2):
        try:
            results.append(_qualification_replay(harness, live_static_batch))
        finally:
            assert live_static_batch.dp_padding_mode is None
            _assert_qualification_replay_state_restored(harness)

    assert results == ["eager", "eager"]
    assert harness.default_mode_calls == [None, None]
    assert harness.observed_dp_padding_modes == [
        harness.capture_dp_padding_mode,
        harness.capture_dp_padding_mode,
    ]
    assert harness.calls == [
        (live_static_batch, 64),
        (live_static_batch, 64),
    ]
    assert harness.backend._omni_qualification_eager_replay is True
    assert harness.backend._omni_original_replay is not harness.backend.replay


def test_qualification_eager_replay_restores_none_mode_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _qualification_eager_replay_harness(monkeypatch)
    live_static_batch = SimpleNamespace(dp_padding_mode=None)
    harness.layer_model.forward_error = RuntimeError("eager replay failed")

    with pytest.raises(RuntimeError, match="eager replay failed"):
        _qualification_replay(harness, live_static_batch)

    assert live_static_batch.dp_padding_mode is None
    assert harness.default_mode_calls == [None]
    assert harness.observed_dp_padding_modes == [harness.capture_dp_padding_mode]
    _assert_qualification_replay_state_restored(harness)


def test_qualification_eager_replay_preserves_non_none_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _qualification_eager_replay_harness(monkeypatch)
    live_dp_padding_mode = SimpleNamespace(is_max_len=lambda: True)
    live_static_batch = SimpleNamespace(dp_padding_mode=live_dp_padding_mode)

    assert _qualification_replay(harness, live_static_batch) == "eager"

    assert live_static_batch.dp_padding_mode is live_dp_padding_mode
    assert harness.default_mode_calls == []
    assert harness.observed_dp_padding_modes == [live_dp_padding_mode]
    _assert_qualification_replay_state_restored(harness)


def test_qualification_eager_replay_fails_closed_for_wrong_backend() -> None:
    from sglang_omni.model_runner.prefill_qualification import (
        enable_prefill_qualification_eager_replay,
    )

    runner_type = type("PrefillCudaGraphRunner", (), {})
    runner = runner_type()
    runner.backend = SimpleNamespace(replay=lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="BreakableCudaGraphBackend"):
        enable_prefill_qualification_eager_replay(
            SimpleNamespace(prefill_cuda_graph_runner=runner)
        )


def test_qualification_eager_replay_rejects_equivalent_original_bound_method() -> None:
    from sglang_omni.model_runner.prefill_qualification import (
        enable_prefill_qualification_eager_replay,
    )

    backend_type = type(
        "BreakableCudaGraphBackend",
        (),
        {"replay": lambda self, *args, **kwargs: "captured"},
    )
    backend = backend_type()

    class LayerModel:
        def forward(self, batch, size):
            return "eager"

    class PrefillCudaGraphRunner:
        def __init__(self):
            self.backend = backend
            self.layer_model = LayerModel()

        def _run_forward(self, batch, size):
            return self.layer_model.forward(batch, size)

        @contextmanager
        def _prefill_forward_context(self, *args, **kwargs):
            yield

    runner = PrefillCudaGraphRunner()
    enable_prefill_qualification_eager_replay(
        SimpleNamespace(prefill_cuda_graph_runner=runner)
    )
    # Accessing the descriptor creates a new bound-method object, but it still
    # denotes the same original method and must not bypass the outer-context gate.
    runner.layer_model.forward = LayerModel.forward.__get__(
        runner.layer_model, LayerModel
    )

    with pytest.raises(RuntimeError, match="prefill body replay closure"):
        backend.replay(SimpleNamespace(size=64), object())
