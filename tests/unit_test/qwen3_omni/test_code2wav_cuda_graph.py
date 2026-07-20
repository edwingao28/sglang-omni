# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
import json
from typing import Any
import weakref

import pytest
import torch

from sglang_omni.models.qwen3_omni.components import code2wav_cuda_graph
from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    CODE2WAV_GRAPH_KEYS,
    Code2WavCudaGraphRunner,
    GraphKey,
)


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        self.calls.append(codes.detach().clone())
        samples = int(codes.shape[-1]) * 2
        base = codes.float().sum(dim=(1, 2), keepdim=True)
        ramp = torch.arange(samples, dtype=torch.float32).view(1, 1, samples)
        return base + ramp


class _FakeGraph:
    def __init__(
        self,
        model: _FakeModel,
        static_input: torch.Tensor,
        static_output: torch.Tensor,
        *,
        corrupt: bool,
    ) -> None:
        self._model = model
        self.static_input = static_input
        self.static_output = static_output
        self.corrupt = corrupt
        self.fail_replay: Exception | None = None
        self.replay_inputs: list[torch.Tensor] = []

    def replay(self) -> None:
        if self.fail_replay is not None:
            raise self.fail_replay
        self.replay_inputs.append(self.static_input.clone())
        output = self._model(self.static_input)
        if self.corrupt:
            output = output + 1
        self.static_output.copy_(output)


class _FakeCudaBackend:
    def __init__(
        self,
        *,
        capture_error_at: int | None = None,
        corrupt_at: int | None = None,
        replay_error_at: int | None = None,
        retain_graphs: bool = True,
        after_allocated: int = 160,
        after_reserved: int = 200,
    ) -> None:
        self.capture_error_at = capture_error_at
        self.corrupt_at = corrupt_at
        self.replay_error_at = replay_error_at
        self.retain_graphs = retain_graphs
        self.capture_calls = 0
        self.pool_calls = 0
        self.capture_pools: list[Any] = []
        self.new_stream_devices: list[torch.device] = []
        self.warmup_streams: list[Any | None] = []
        self.capture_streams: list[Any | None] = []
        self.warmup_iterations: list[int] = []
        self.synchronize_calls = 0
        self.empty_cache_calls = 0
        self.outer_capture = False
        self.graphs: list[_FakeGraph] = []
        self.resolve_device_calls: list[torch.device] = []
        self._tensor_devices: dict[int, torch.device] = {}
        self._memory_snapshots = [
            {
                "allocated_bytes": 100,
                "reserved_bytes": 120,
                "max_reserved_bytes": 130,
                "free_bytes": 900,
                "total_bytes": 1000,
            },
            {
                "allocated_bytes": after_allocated,
                "reserved_bytes": after_reserved,
                "max_reserved_bytes": 250,
                "free_bytes": 820,
                "total_bytes": 1000,
            },
            {
                "allocated_bytes": 100,
                "reserved_bytes": 120,
                "max_reserved_bytes": 250,
                "free_bytes": 900,
                "total_bytes": 1000,
            },
        ]
        self._memory_index = 0

    def device_context(self, device: torch.device):
        del device
        return nullcontext()

    def resolve_device(self, device: torch.device) -> torch.device:
        self.resolve_device_calls.append(device)
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda:0")
        return device

    def memory_stats(self, device: torch.device) -> dict[str, int]:
        del device
        index = min(self._memory_index, len(self._memory_snapshots) - 1)
        self._memory_index += 1
        return dict(self._memory_snapshots[index])

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def new_static_input(
        self, shape: tuple[int, int, int], *, device: torch.device
    ) -> torch.Tensor:
        tensor = torch.zeros(shape, dtype=torch.long)
        self.mark_cuda(tensor, device=device)
        return tensor

    def warmup(
        self,
        model: _FakeModel,
        static_input: torch.Tensor,
        *,
        iterations: int,
        device: torch.device,
        stream: object | None = None,
    ) -> None:
        del device
        self.warmup_streams.append(stream)
        self.warmup_iterations.append(iterations)
        for _ in range(iterations):
            model(static_input)

    def graph_pool_handle(self) -> object:
        self.pool_calls += 1
        return object()

    def new_stream(self, device: torch.device) -> object:
        self.new_stream_devices.append(device)
        return object()

    def capture(
        self,
        model: _FakeModel,
        static_input: torch.Tensor,
        *,
        pool: object,
        stream: object | None = None,
    ) -> tuple[_FakeGraph, torch.Tensor]:
        call_index = self.capture_calls
        self.capture_calls += 1
        self.capture_pools.append(pool)
        self.capture_streams.append(stream)
        if call_index == self.capture_error_at:
            raise torch.OutOfMemoryError("fake capture OOM")
        static_output = model(static_input).detach().clone()
        graph = _FakeGraph(
            model,
            static_input,
            static_output,
            corrupt=call_index == self.corrupt_at,
        )
        if call_index == self.replay_error_at:
            graph.fail_replay = RuntimeError("fake build replay failed")
        if self.retain_graphs:
            self.graphs.append(graph)
        return graph, static_output

    def synchronize(self, device: torch.device) -> None:
        del device
        self.synchronize_calls += 1

    def is_cuda_tensor(self, tensor: torch.Tensor) -> bool:
        return id(tensor) in self._tensor_devices

    def tensor_device_matches(self, tensor: torch.Tensor, device: torch.device) -> bool:
        return self._tensor_devices.get(id(tensor)) == device

    def is_current_stream_capturing(self) -> bool:
        return self.outer_capture

    def mark_cuda(
        self, tensor: torch.Tensor, *, device: str | torch.device = "cuda:0"
    ) -> torch.Tensor:
        self._tensor_devices[id(tensor)] = torch.device(device)
        return tensor


class _LiveAllocationCudaBackend(_FakeCudaBackend):
    """Derive memory from weakly tracked live capture objects."""

    def __init__(self, *, replay_error_at: int) -> None:
        super().__init__(
            replay_error_at=replay_error_at,
            retain_graphs=False,
        )
        self._live_refs: list[weakref.ReferenceType[Any]] = []
        self.failed_refs: list[weakref.ReferenceType[Any]] = []
        self.failed_refs_dead_at_empty_cache: bool | None = None
        self._peak_reserved = 120

    def memory_stats(self, device: torch.device) -> dict[str, int]:
        del device
        live_count = sum(ref() is not None for ref in self._live_refs)
        allocated = 100 + live_count * 10
        reserved = 120 + live_count * 10
        self._peak_reserved = max(self._peak_reserved, reserved)
        return {
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "max_reserved_bytes": self._peak_reserved,
            "free_bytes": 1000 - reserved,
            "total_bytes": 1000,
        }

    def new_static_input(
        self, shape: tuple[int, int, int], *, device: torch.device
    ) -> torch.Tensor:
        tensor = super().new_static_input(shape, device=device)
        self._live_refs.append(weakref.ref(tensor))
        return tensor

    def capture(
        self,
        model: _FakeModel,
        static_input: torch.Tensor,
        *,
        pool: object,
        stream: object | None = None,
    ) -> tuple[_FakeGraph, torch.Tensor]:
        graph, static_output = super().capture(
            model,
            static_input,
            pool=pool,
            stream=stream,
        )
        refs = [
            weakref.ref(graph),
            weakref.ref(static_input),
            weakref.ref(static_output),
        ]
        self._live_refs.extend(refs)
        if self.capture_calls - 1 == self.replay_error_at:
            self.failed_refs = refs
        return graph, static_output

    def empty_cache(self) -> None:
        super().empty_cache()
        if self.failed_refs:
            self.failed_refs_dead_at_empty_cache = all(
                ref() is None for ref in self.failed_refs
            )


def _build_runner(
    *,
    backend: _FakeCudaBackend | None = None,
    model: _FakeModel | None = None,
    total_gpu_memory_fraction: float | None = 0.5,
) -> tuple[Code2WavCudaGraphRunner, _FakeCudaBackend, _FakeModel]:
    backend = backend or _FakeCudaBackend()
    model = model or _FakeModel()
    runner = Code2WavCudaGraphRunner.build(
        model,
        device="cuda:0",
        num_quantizers=16,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        cuda_api=backend,
    )
    return runner, backend, model


def _codes(
    backend: _FakeCudaBackend,
    batch_size: int,
    frames: int,
    *,
    num_quantizers: int = 16,
    dtype: torch.dtype = torch.long,
    device: str = "cuda:0",
) -> torch.Tensor:
    tensor = torch.arange(
        batch_size * num_quantizers * frames,
        dtype=dtype,
    ).reshape(batch_size, num_quantizers, frames)
    return backend.mark_cuda(tensor, device=device)


def test_fixed_graph_key_matrix_has_exactly_sixteen_cross_request_shapes() -> None:
    assert CODE2WAV_GRAPH_KEYS == tuple(
        GraphKey(batch_size=batch_size, frames=frames)
        for frames in (10, 20, 30, 35)
        for batch_size in (1, 2, 4, 8)
    )


def test_build_uses_three_warmups_one_private_pool_and_atomic_publication() -> None:
    runner, backend, _model = _build_runner()

    assert runner.enabled is True
    assert tuple(runner.graphs) == CODE2WAV_GRAPH_KEYS
    assert backend.warmup_iterations == [3] * 16
    assert backend.pool_calls == 1
    assert len({id(pool) for pool in backend.capture_pools}) == 1
    assert backend.synchronize_calls >= 1

    stats = runner.stats()
    assert stats["build"]["published_graph_count"] == 16
    expected_static_input_bytes = sum(
        key.batch_size
        * 16
        * key.frames
        * torch.tensor([], dtype=torch.long).element_size()
        for key in CODE2WAV_GRAPH_KEYS
    )
    expected_static_output_bytes = sum(
        key.batch_size
        * 1
        * (key.frames * 2)
        * torch.tensor([], dtype=torch.float32).element_size()
        for key in CODE2WAV_GRAPH_KEYS
    )
    assert stats["build"]["static_input_bytes"] == expected_static_input_bytes
    assert stats["build"]["static_output_bytes"] == expected_static_output_bytes
    assert stats["build"]["static_tensor_bytes"] == (
        expected_static_input_bytes + expected_static_output_bytes
    )
    assert stats["memory"] == {
        "total_gpu_memory_fraction": 0.5,
        "stage_budget_bytes": 500,
        "loaded_model_footprint_bytes": 100,
        "graph_budget_bytes": 400,
        "graph_footprint_bytes": 80,
        "before": {
            "allocated_bytes": 100,
            "reserved_bytes": 120,
            "max_reserved_bytes": 130,
            "free_bytes": 900,
            "total_bytes": 1000,
        },
        "after": {
            "allocated_bytes": 160,
            "reserved_bytes": 200,
            "max_reserved_bytes": 250,
            "free_bytes": 820,
            "total_bytes": 1000,
        },
        "after_rollback": None,
    }


def test_build_reuses_one_private_stream_for_all_warmups_and_captures() -> None:
    runner, backend, _model = _build_runner()

    assert runner.enabled is True
    assert backend.new_stream_devices == [torch.device("cuda:0")]
    assert len(backend.warmup_streams) == 16
    assert len(backend.capture_streams) == 16
    private_stream = backend.warmup_streams[0]
    assert private_stream is not None
    assert all(stream is private_stream for stream in backend.warmup_streams)
    assert all(stream is private_stream for stream in backend.capture_streams)


def test_cuda_api_restores_original_stream_when_capture_exit_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_stream = object()
    current = {"stream": original_stream}

    class _SideStream:
        def wait_stream(self, stream: object) -> None:
            assert stream is original_stream

    side_stream = _SideStream()

    class _FailingCaptureContext:
        def __enter__(self) -> None:
            current["stream"] = side_stream

        def __exit__(self, *_args: object) -> None:
            raise RuntimeError("fake capture_end failed")

    def graph_context(*_args: object, **kwargs: object) -> _FailingCaptureContext:
        assert kwargs["stream"] is side_stream
        return _FailingCaptureContext()

    monkeypatch.setattr(
        code2wav_cuda_graph.torch.cuda,
        "current_stream",
        lambda _device: current["stream"],
    )
    monkeypatch.setattr(
        code2wav_cuda_graph.torch.cuda,
        "set_stream",
        lambda stream: current.update(stream=stream),
    )
    monkeypatch.setattr(
        code2wav_cuda_graph.torch.cuda,
        "CUDAGraph",
        lambda: object(),
    )
    monkeypatch.setattr(code2wav_cuda_graph.torch.cuda, "graph", graph_context)

    with pytest.raises(RuntimeError, match="fake capture_end failed"):
        code2wav_cuda_graph._TorchCudaApi().capture(
            _FakeModel(),
            torch.zeros((1, 16, 10), dtype=torch.long),
            pool=object(),
            stream=side_stream,
        )

    assert current["stream"] is original_stream


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cuda_invalid_capture_preserves_current_stream() -> None:
    api = code2wav_cuda_graph._TorchCudaApi()
    device = torch.device("cuda", torch.cuda.current_device())
    original_stream = torch.cuda.current_stream(device)
    side_stream = api.new_stream(device)
    static_input = torch.zeros((1, 16, 10), dtype=torch.long, device=device)

    def invalidate_capture(codes: torch.Tensor) -> torch.Tensor:
        # Host synchronization is prohibited during graph capture. This marks
        # the capture invalid so capture_end itself exercises the failure path
        # that used to skip the graph context's normal stream restoration.
        torch.cuda.synchronize(device)
        return codes

    current_after: torch.cuda.Stream | None = None
    try:
        with pytest.raises(RuntimeError, match="(?i)captur"):
            api.capture(
                invalidate_capture,
                static_input,
                pool=torch.cuda.graph_pool_handle(),
                stream=side_stream,
            )
        current_after = torch.cuda.current_stream(device)
    finally:
        torch.cuda.set_stream(original_stream)

    assert current_after == original_stream


def test_unindexed_cuda_device_is_bound_to_current_concrete_device() -> None:
    backend = _FakeCudaBackend()
    model = _FakeModel()
    runner = Code2WavCudaGraphRunner.build(
        model,
        device="cuda",
        num_quantizers=16,
        total_gpu_memory_fraction=0.5,
        cuda_api=backend,
    )

    result = runner.run(_codes(backend, 1, 10, device="cuda:0"))

    assert backend.resolve_device_calls == [torch.device("cuda")]
    assert runner.device == torch.device("cuda:0")
    assert runner.stats()["binding"]["device"] == "cuda:0"
    assert result.execution_mode == "cuda_graph"
    assert result.fallback_reason is None


def test_run_copies_live_input_replays_and_returns_borrowed_output_metadata() -> None:
    runner, backend, _model = _build_runner()
    graph = next(
        graph
        for graph in backend.graphs
        if tuple(graph.static_input.shape) == (2, 16, 10)
    )
    first_codes = _codes(backend, 2, 10)

    first = runner.run(first_codes)
    first_snapshot = first.output.clone()

    assert first.execution_mode == "cuda_graph"
    assert first.key == GraphKey(batch_size=2, frames=10)
    assert first.fallback_reason is None
    assert first.output is graph.static_output
    assert torch.equal(graph.replay_inputs[-1], first_codes)

    second_codes = _codes(backend, 2, 10) + 7
    backend.mark_cuda(second_codes)
    second = runner.run(second_codes)

    assert second.output is first.output
    assert not torch.equal(first.output, first_snapshot)
    assert torch.equal(graph.replay_inputs[-1], second_codes)
    assert runner.stats()["runtime"]["graph_replays"] == 2


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("ineligible", "ineligible"),
        ("key_miss", "key_miss"),
        ("non_cuda", "non_cuda"),
        ("wrong_dtype", "wrong_dtype"),
        ("wrong_device", "wrong_device"),
        ("wrong_num_quantizers", "wrong_num_quantizers"),
        ("outer_cuda_capture", "outer_cuda_capture"),
    ],
)
def test_runtime_ineligible_inputs_fall_back_to_eager(
    case: str,
    expected_reason: str,
) -> None:
    runner, backend, model = _build_runner()
    codes = _codes(backend, 1, 10)
    eligible = True

    if case == "ineligible":
        eligible = False
    elif case == "key_miss":
        codes = _codes(backend, 3, 10)
    elif case == "non_cuda":
        codes = torch.zeros((1, 16, 10), dtype=torch.long)
    elif case == "wrong_dtype":
        codes = _codes(backend, 1, 10, dtype=torch.int32)
    elif case == "wrong_device":
        codes = _codes(backend, 1, 10, device="cuda:1")
    elif case == "wrong_num_quantizers":
        codes = _codes(backend, 1, 10, num_quantizers=15)
    elif case == "outer_cuda_capture":
        backend.outer_capture = True

    calls_before = len(model.calls)
    result = runner.run(codes, eligible=eligible)

    assert result.execution_mode == "eager"
    assert result.fallback_reason == expected_reason
    assert len(model.calls) == calls_before + 1
    assert runner.stats()["runtime"]["fallback_counts"] == {expected_reason: 1}
    if case == "key_miss":
        assert result.key == GraphKey(batch_size=3, frames=10)


@pytest.mark.parametrize(
    ("runner_state", "eligible"),
    [
        ("enabled", True),
        ("ineligible", False),
        ("disabled", True),
    ],
)
def test_pid_mismatch_fails_closed_before_any_eager_model_call(
    runner_state: str,
    eligible: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fraction = None if runner_state == "disabled" else 0.5
    runner, backend, model = _build_runner(total_gpu_memory_fraction=fraction)
    calls_before = len(model.calls)
    monkeypatch.setattr(
        code2wav_cuda_graph.os,
        "getpid",
        lambda: runner.owner_pid + 1,
    )

    with pytest.raises(RuntimeError, match="must be rebuilt in a spawned process"):
        runner.run(_codes(backend, 1, 10), eligible=eligible)

    assert len(model.calls) == calls_before
    runtime = runner.stats()["runtime"]
    assert runtime["fallback_counts"] == {}
    assert runtime["failure_counts"] == {"pid_mismatch": 1}
    json.dumps(runner.stats(), allow_nan=False)


@pytest.mark.parametrize(
    ("backend", "fraction", "reason_prefix"),
    [
        (_FakeCudaBackend(capture_error_at=3), 0.5, "capture_failed"),
        (_FakeCudaBackend(corrupt_at=3), 0.5, "equivalence_failed"),
        (_FakeCudaBackend(), 0.15, "memory_budget_exceeded"),
    ],
)
def test_any_build_failure_rolls_back_every_graph(
    backend: _FakeCudaBackend,
    fraction: float,
    reason_prefix: str,
) -> None:
    runner, backend, _model = _build_runner(
        backend=backend,
        total_gpu_memory_fraction=fraction,
    )

    assert runner.enabled is False
    assert runner.graphs == {}
    assert runner.disable_reason is not None
    assert runner.disable_reason.startswith(reason_prefix)
    assert backend.empty_cache_calls >= 1
    memory = runner.stats()["memory"]
    assert memory["after"]["allocated_bytes"] == 160
    assert memory["after_rollback"]["allocated_bytes"] == 100


def test_rollback_drops_failed_traceback_objects_before_cache_cleanup() -> None:
    backend = _LiveAllocationCudaBackend(replay_error_at=3)

    runner, backend, _model = _build_runner(backend=backend)

    assert runner.enabled is False
    assert runner.graphs == {}
    assert runner.disable_reason is not None
    assert runner.disable_reason.startswith("capture_failed")
    assert backend.failed_refs
    assert backend.failed_refs_dead_at_empty_cache is True
    assert all(ref() is None for ref in backend.failed_refs)
    memory = runner.stats()["memory"]
    assert memory["after"]["allocated_bytes"] > 100
    assert memory["after_rollback"]["allocated_bytes"] == 100


@pytest.mark.parametrize("fraction", [None, 0.0, -0.1, 1.01, float("nan")])
def test_memory_fraction_must_be_explicit_and_in_range(
    fraction: float | None,
) -> None:
    runner, _backend, _model = _build_runner(
        total_gpu_memory_fraction=fraction,
    )

    assert runner.enabled is False
    assert runner.graphs == {}
    assert runner.disable_reason == "invalid_total_gpu_memory_fraction"


def test_runtime_replay_failure_is_raised_and_disables_all_graphs() -> None:
    runner, backend, _model = _build_runner()
    build_stats = runner.stats()["build"]
    graph = next(
        graph
        for graph in backend.graphs
        if tuple(graph.static_input.shape) == (1, 16, 10)
    )
    graph.fail_replay = RuntimeError("replay exploded")

    with pytest.raises(RuntimeError, match="replay exploded"):
        runner.run(_codes(backend, 1, 10))

    assert runner.enabled is False
    assert runner.graphs == {}
    assert (
        runner.disable_reason == "runtime_replay_failed: RuntimeError: replay exploded"
    )
    stats = runner.stats()
    assert stats["build"] == build_stats
    assert stats["runtime"]["replay_failures"] == 1

    calls_before = len(_model.calls)
    fallback = runner.run(_codes(backend, 1, 10))
    assert fallback.execution_mode == "eager"
    assert fallback.fallback_reason == "disabled"
    assert len(_model.calls) == calls_before + 1


def test_stats_are_strictly_json_safe_after_success_and_failure() -> None:
    successful, successful_backend, _model = _build_runner()
    successful.run(_codes(successful_backend, 3, 10))
    failed, _failed_backend, _model = _build_runner(
        backend=_FakeCudaBackend(corrupt_at=0)
    )

    json.dumps(successful.stats(), allow_nan=False)
    json.dumps(failed.stats(), allow_nan=False)
