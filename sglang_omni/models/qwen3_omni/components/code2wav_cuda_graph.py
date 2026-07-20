# SPDX-License-Identifier: Apache-2.0
"""Exact-shape CUDA graphs for the Qwen3-Omni Code2Wav component."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass
import gc
import logging
import math
import os
from types import MappingProxyType
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GraphKey:
    """One exact Code2Wav input shape, excluding fixed quantizer count."""

    batch_size: int
    frames: int


CODE2WAV_GRAPH_KEYS = tuple(
    GraphKey(batch_size=batch_size, frames=frames)
    for frames in (10, 20, 30, 35)
    for batch_size in (1, 2, 4, 8)
)


@dataclass(frozen=True, slots=True)
class Code2WavRunResult:
    """Result metadata for either an exact graph replay or eager fallback.

    A ``cuda_graph`` output is a borrowed static buffer. The caller must finish
    all reads, including trim and device-to-host transfer, before the next graph
    replay. It must not retain the tensor or use it concurrently. The current
    scheduler's outer ``_state_lock`` is intended to cover that whole lifetime;
    this runner deliberately does not clone the output.
    """

    output: torch.Tensor
    execution_mode: str
    key: GraphKey | None
    fallback_reason: str | None


@dataclass(slots=True)
class _CapturedGraph:
    graph: Any
    static_input: torch.Tensor
    static_output: torch.Tensor


class _BuildFailure(RuntimeError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class _TorchCudaApi:
    """Small injectable boundary around CUDA-only operations."""

    def resolve_device(self, device: torch.device) -> torch.device:
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return device

    def device_context(self, device: torch.device) -> AbstractContextManager[Any]:
        return torch.cuda.device(device)

    def memory_stats(self, device: torch.device) -> dict[str, int]:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def new_static_input(
        self, shape: tuple[int, int, int], *, device: torch.device
    ) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.long, device=device)

    def new_stream(self, device: torch.device) -> torch.cuda.Stream:
        return torch.cuda.Stream(device=device)

    def warmup(
        self,
        model: Any,
        static_input: torch.Tensor,
        *,
        iterations: int,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_stream = torch.cuda.current_stream(device)
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(iterations):
                model(static_input)
        current_stream.wait_stream(stream)

    def graph_pool_handle(self) -> Any:
        return torch.cuda.graph_pool_handle()

    def capture(
        self,
        model: Any,
        static_input: torch.Tensor,
        *,
        pool: Any,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        current_stream = torch.cuda.current_stream(static_input.device)
        stream.wait_stream(current_stream)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.inference_mode():
                with torch.cuda.graph(
                    graph,
                    pool=pool,
                    stream=stream,
                    capture_error_mode="thread_local",
                ):
                    static_output = model(static_input)
        finally:
            # torch.cuda.graph.__exit__ calls capture_end before restoring its
            # stream context. If capture_end raises, restore explicitly using
            # the original stream's device-aware identity.
            torch.cuda.set_stream(current_stream)
        current_stream.wait_stream(stream)
        return graph, static_output

    def synchronize(self, device: torch.device) -> None:
        torch.cuda.synchronize(device)

    def is_cuda_tensor(self, tensor: torch.Tensor) -> bool:
        return tensor.is_cuda

    def tensor_device_matches(self, tensor: torch.Tensor, device: torch.device) -> bool:
        return tensor.device == device

    def is_current_stream_capturing(self) -> bool:
        return bool(torch.cuda.is_current_stream_capturing())


class Code2WavCudaGraphRunner:
    """Atomic exact-shape CUDA graph runner for ``[B, Q, T]`` long codes.

    One instance is permanently bound to one model, CUDA device, quantizer
    count, ``torch.long`` input dtype, and owner process. Build failures disable
    the complete runner and leave no partial graph matrix published.
    """

    _WARMUP_ITERATIONS = 3

    def __init__(
        self,
        model: Any,
        *,
        device: str | torch.device,
        num_quantizers: int,
        cuda_api: Any,
    ) -> None:
        self._model = model
        self._requested_device = str(device)
        try:
            self._device = torch.device(device)
        except (RuntimeError, TypeError, ValueError):
            self._device = torch.device("cpu")
        try:
            self._num_quantizers = int(num_quantizers)
        except (TypeError, ValueError):
            self._num_quantizers = 0
        self._input_dtype = torch.long
        self._owner_pid = os.getpid()
        self._cuda = cuda_api
        self._graphs: dict[GraphKey, _CapturedGraph] = {}
        self._pool: Any | None = None
        self._capture_stream: Any | None = None
        self._enabled = False
        self._disable_reason: str | None = None
        self._build_stats: dict[str, Any] = {
            "attempted_graph_count": 0,
            "published_graph_count": 0,
            "warmup_iterations_per_graph": self._WARMUP_ITERATIONS,
            "static_input_bytes": 0,
            "static_output_bytes": 0,
            "static_tensor_bytes": 0,
        }
        self._memory_stats: dict[str, Any] = {
            "total_gpu_memory_fraction": None,
            "stage_budget_bytes": None,
            "loaded_model_footprint_bytes": None,
            "graph_budget_bytes": None,
            "graph_footprint_bytes": None,
            "before": None,
            "after": None,
            "after_rollback": None,
        }
        self._fallback_counts: Counter[str] = Counter()
        self._graph_replays = 0
        self._replay_failures = 0
        self._runtime_failure_counts: Counter[str] = Counter()

    @classmethod
    def build(
        cls,
        model: Any,
        *,
        device: str | torch.device,
        num_quantizers: int,
        total_gpu_memory_fraction: float | None,
        cuda_api: Any | None = None,
    ) -> Code2WavCudaGraphRunner:
        """Build all sixteen graphs, returning a disabled runner on any failure."""

        runner = cls(
            model,
            device=device,
            num_quantizers=num_quantizers,
            cuda_api=_TorchCudaApi() if cuda_api is None else cuda_api,
        )
        runner._build(total_gpu_memory_fraction)
        return runner

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def disable_reason(self) -> str | None:
        return self._disable_reason

    @property
    def graphs(self) -> Mapping[GraphKey, _CapturedGraph]:
        return MappingProxyType(self._graphs)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_quantizers(self) -> int:
        return self._num_quantizers

    @property
    def owner_pid(self) -> int:
        return self._owner_pid

    def _build(self, total_gpu_memory_fraction: float | None) -> None:
        fraction = self._valid_fraction(total_gpu_memory_fraction)
        if fraction is None:
            self._disable_reason = "invalid_total_gpu_memory_fraction"
            return
        self._memory_stats["total_gpu_memory_fraction"] = fraction
        if self._device.type != "cuda":
            self._disable_reason = "invalid_cuda_device"
            return
        if self._num_quantizers <= 0:
            self._disable_reason = "invalid_num_quantizers"
            return
        try:
            resolved_device = torch.device(self._cuda.resolve_device(self._device))
        except Exception as exc:
            self._disable_reason = (
                f"cuda_device_resolution_failed: {type(exc).__name__}: {exc}"
            )
            return
        if resolved_device.type != "cuda" or resolved_device.index is None:
            self._disable_reason = "invalid_cuda_device"
            return
        self._device = resolved_device

        temporary: dict[GraphKey, _CapturedGraph] = {}
        pool: Any | None = None
        capture_stream: Any | None = None
        static_input: torch.Tensor | None = None
        static_output: torch.Tensor | None = None
        eager_output: torch.Tensor | None = None
        graph: Any | None = None
        captured: _CapturedGraph | None = None
        before: dict[str, int] | None = None
        after: dict[str, int] | None = None
        failure_reason: str | None = None
        try:
            with self._cuda.device_context(self._device):
                before = self._normalized_memory_stats(
                    self._cuda.memory_stats(self._device)
                )
                self._memory_stats["before"] = before
                stage_budget = int(before["total_bytes"] * fraction)
                loaded_model_footprint = before["allocated_bytes"]
                graph_budget = max(0, stage_budget - loaded_model_footprint)
                self._memory_stats.update(
                    {
                        "stage_budget_bytes": stage_budget,
                        "loaded_model_footprint_bytes": loaded_model_footprint,
                        "graph_budget_bytes": graph_budget,
                    }
                )

                pool = self._cuda.graph_pool_handle()
                capture_stream = self._cuda.new_stream(self._device)
                capture_order = sorted(
                    CODE2WAV_GRAPH_KEYS,
                    key=lambda key: (key.batch_size * key.frames, key.frames),
                    reverse=True,
                )
                for key in capture_order:
                    self._build_stats["attempted_graph_count"] += 1
                    static_input = self._cuda.new_static_input(
                        (
                            key.batch_size,
                            self._num_quantizers,
                            key.frames,
                        ),
                        device=self._device,
                    )
                    input_bytes = static_input.numel() * static_input.element_size()
                    self._build_stats["static_input_bytes"] += input_bytes
                    self._build_stats["static_tensor_bytes"] += input_bytes
                    self._cuda.warmup(
                        self._model,
                        static_input,
                        iterations=self._WARMUP_ITERATIONS,
                        device=self._device,
                        stream=capture_stream,
                    )
                    graph, static_output = self._cuda.capture(
                        self._model,
                        static_input,
                        pool=pool,
                        stream=capture_stream,
                    )
                    captured = _CapturedGraph(
                        graph=graph,
                        static_input=static_input,
                        static_output=static_output,
                    )
                    temporary[key] = captured
                    output_bytes = static_output.numel() * static_output.element_size()
                    self._build_stats["static_output_bytes"] += output_bytes
                    self._build_stats["static_tensor_bytes"] += output_bytes

                    with torch.inference_mode():
                        eager_output = self._model(static_input).detach().clone()
                        graph.replay()
                    self._verify_equivalence(
                        key=key,
                        eager_output=eager_output,
                        graph_output=static_output,
                    )
                    eager_output = None

                # Capture, replay and equivalence checks enqueue CUDA work. Do
                # not make the all-or-nothing graph matrix visible until every
                # key has completed on the bound device.
                self._cuda.synchronize(self._device)
                gc.collect()
                self._cuda.empty_cache()
                after = self._normalized_memory_stats(
                    self._cuda.memory_stats(self._device)
                )
                self._memory_stats["after"] = after
                allocated_delta = max(
                    0,
                    after["allocated_bytes"] - before["allocated_bytes"],
                )
                reserved_delta = max(
                    0,
                    after["reserved_bytes"] - before["reserved_bytes"],
                )
                graph_footprint = max(allocated_delta, reserved_delta)
                self._memory_stats["graph_footprint_bytes"] = graph_footprint
                if graph_footprint > graph_budget:
                    raise _BuildFailure(
                        "memory_budget_exceeded",
                        f"graph footprint {graph_footprint} exceeds budget "
                        f"{graph_budget}",
                    )

        except Exception as exc:
            reason = exc.reason if isinstance(exc, _BuildFailure) else "capture_failed"
            detail = (
                exc.detail
                if isinstance(exc, _BuildFailure)
                else f"{type(exc).__name__}: {exc}"
            )
            failure_reason = f"{reason}: {detail}"

        # Leave the ``except`` scope before cleanup: the active exception and
        # its traceback can otherwise retain failed capture frames and their
        # tensors/graphs through empty_cache() and the rollback snapshot.
        static_input = None
        static_output = None
        eager_output = None
        graph = None
        captured = None
        after = None
        if failure_reason is not None:
            pool = None
            capture_stream = None
            self._rollback_build(
                temporary=temporary,
                reason=failure_reason,
            )
            return

        self._pool = pool
        self._capture_stream = capture_stream
        self._graphs = {key: temporary[key] for key in CODE2WAV_GRAPH_KEYS}
        self._build_stats["published_graph_count"] = len(self._graphs)
        self._enabled = True
        logger.info(
            "Code2Wav CUDA graph runner published %d exact graphs on %s",
            len(self._graphs),
            self._device,
        )

    @staticmethod
    def _valid_fraction(value: float | None) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            fraction = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            return None
        return fraction

    @staticmethod
    def _normalized_memory_stats(stats: Mapping[str, Any]) -> dict[str, int]:
        fields = (
            "allocated_bytes",
            "reserved_bytes",
            "max_reserved_bytes",
            "free_bytes",
            "total_bytes",
        )
        return {field: int(stats[field]) for field in fields}

    @staticmethod
    def _verify_equivalence(
        *,
        key: GraphKey,
        eager_output: torch.Tensor,
        graph_output: torch.Tensor,
    ) -> None:
        if tuple(eager_output.shape) != tuple(graph_output.shape):
            raise _BuildFailure(
                "equivalence_failed",
                f"{key}: output shape mismatch "
                f"{tuple(eager_output.shape)} != {tuple(graph_output.shape)}",
            )
        eager_finite = bool(torch.isfinite(eager_output).all().item())
        graph_finite = bool(torch.isfinite(graph_output).all().item())
        if not eager_finite or not graph_finite:
            raise _BuildFailure(
                "equivalence_failed",
                f"{key}: non-finite output "
                f"eager={not eager_finite} graph={not graph_finite}",
            )
        if not torch.equal(eager_output, graph_output):
            raise _BuildFailure(
                "equivalence_failed",
                f"{key}: eager and graph outputs are not exactly equal",
            )

    def _rollback_build(
        self,
        *,
        temporary: dict[GraphKey, _CapturedGraph],
        reason: str,
    ) -> None:
        if self._memory_stats["after"] is None:
            try:
                self._cuda.synchronize(self._device)
            except Exception as synchronize_exc:
                logger.warning(
                    "Code2Wav CUDA graph rollback synchronize failed: %s",
                    synchronize_exc,
                )
            try:
                self._memory_stats["after"] = self._normalized_memory_stats(
                    self._cuda.memory_stats(self._device)
                )
            except Exception as snapshot_exc:
                logger.warning(
                    "Code2Wav CUDA graph rollback snapshot failed: %s",
                    snapshot_exc,
                )
        self._graphs.clear()
        temporary.clear()
        self._pool = None
        self._capture_stream = None
        self._enabled = False
        self._disable_reason = reason
        gc.collect()
        try:
            with self._cuda.device_context(self._device):
                self._cuda.empty_cache()
                self._memory_stats["after_rollback"] = self._normalized_memory_stats(
                    self._cuda.memory_stats(self._device)
                )
        except Exception as cleanup_exc:
            logger.warning(
                "Code2Wav CUDA graph rollback cleanup failed: %s",
                cleanup_exc,
            )
        logger.warning("Code2Wav CUDA graph runner disabled: %s", reason)

    def run(
        self,
        codes: torch.Tensor,
        *,
        eligible: bool = True,
    ) -> Code2WavRunResult:
        """Replay an exact graph or eagerly execute with a stable reason.

        Graph outputs are borrowed and valid only until the next graph replay;
        callers must serialize replay through trim and D2H consumption.
        """

        current_pid = os.getpid()
        if current_pid != self._owner_pid:
            self._runtime_failure_counts["pid_mismatch"] += 1
            raise RuntimeError(
                "Code2Wav CUDA graph runner/model belongs to PID "
                f"{self._owner_pid}, but was used in PID {current_pid}; it must "
                "be rebuilt in a spawned process before inference"
            )
        if not self._enabled:
            return self._eager(codes, key=None, reason="disabled")
        if not eligible:
            return self._eager(codes, key=None, reason="ineligible")
        if not self._cuda.is_cuda_tensor(codes):
            return self._eager(codes, key=None, reason="non_cuda")
        if codes.dtype != self._input_dtype:
            return self._eager(codes, key=None, reason="wrong_dtype")
        if not self._cuda.tensor_device_matches(codes, self._device):
            return self._eager(codes, key=None, reason="wrong_device")
        if codes.ndim != 3:
            return self._eager(codes, key=None, reason="wrong_shape")
        if int(codes.shape[1]) != self._num_quantizers:
            return self._eager(codes, key=None, reason="wrong_num_quantizers")
        if self._cuda.is_current_stream_capturing():
            return self._eager(codes, key=None, reason="outer_cuda_capture")

        key = GraphKey(
            batch_size=int(codes.shape[0]),
            frames=int(codes.shape[2]),
        )
        captured = self._graphs.get(key)
        if captured is None:
            return self._eager(codes, key=key, reason="key_miss")

        try:
            captured.static_input.copy_(codes)
            captured.graph.replay()
        except Exception as exc:
            self._replay_failures += 1
            self._runtime_failure_counts["runtime_replay_failed"] += 1
            reason = f"runtime_replay_failed: {type(exc).__name__}: {exc}"
            captured = None
            self._disable_runtime(reason)
            raise
        self._graph_replays += 1
        return Code2WavRunResult(
            output=captured.static_output,
            execution_mode="cuda_graph",
            key=key,
            fallback_reason=None,
        )

    def _eager(
        self,
        codes: torch.Tensor,
        *,
        key: GraphKey | None,
        reason: str,
    ) -> Code2WavRunResult:
        self._fallback_counts[reason] += 1
        with torch.inference_mode():
            output = self._model(codes)
        return Code2WavRunResult(
            output=output,
            execution_mode="eager",
            key=key,
            fallback_reason=reason,
        )

    def _disable_runtime(self, reason: str) -> None:
        self._graphs.clear()
        self._pool = None
        self._capture_stream = None
        self._enabled = False
        self._disable_reason = reason
        gc.collect()
        try:
            with self._cuda.device_context(self._device):
                self._cuda.empty_cache()
        except Exception as cleanup_exc:
            logger.warning(
                "Code2Wav CUDA graph runtime cleanup failed: %s",
                cleanup_exc,
            )
        logger.exception("Code2Wav CUDA graph replay disabled the runner")

    def stats(self) -> dict[str, Any]:
        """Return a strict JSON-safe snapshot of build and runtime state."""

        return {
            "schema_version": 1,
            "enabled": self._enabled,
            "disable_reason": self._disable_reason,
            "binding": {
                "device": str(self._device),
                "requested_device": self._requested_device,
                "num_quantizers": self._num_quantizers,
                "input_dtype": "torch.long",
                "owner_pid": self._owner_pid,
            },
            "graph_contract": {
                "exact_shapes": True,
                "keys": [
                    {
                        "batch_size": key.batch_size,
                        "frames": key.frames,
                    }
                    for key in CODE2WAV_GRAPH_KEYS
                ],
            },
            "build": deepcopy(self._build_stats),
            "memory": deepcopy(self._memory_stats),
            "runtime": {
                "graph_replays": self._graph_replays,
                "replay_failures": self._replay_failures,
                "fallback_counts": dict(sorted(self._fallback_counts.items())),
                "failure_counts": dict(sorted(self._runtime_failure_counts.items())),
            },
        }


__all__ = [
    "CODE2WAV_GRAPH_KEYS",
    "Code2WavCudaGraphRunner",
    "Code2WavRunResult",
    "GraphKey",
]
