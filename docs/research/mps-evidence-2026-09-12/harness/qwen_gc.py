"""Experiment-only Qwen3-TTS AR Green Context factory.

Select ``qwen_gc.create_sglang_tts_engine_executor`` through factory_path.
``green_context_sms=None`` is the matched ordinary side-stream control.
The intervention covers AR prefill/decode and the code predictor, not the
preprocessing or vocoder stages. Capture-stream receipts are necessary but
not sufficient: verify actual graph-node kernels with Nsight Systems too.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Iterator


def validate_request(sms: int | None, active_thread_percentage: str | None) -> None:
    if sms is not None and (type(sms) is not int or sms <= 0):
        raise ValueError("green_context_sms must be a positive integer or None")
    if active_thread_percentage not in (None, "", "100"):
        raise ValueError("This comparison requires MPS active thread percentage unset or 100")


def _driver_value(result: tuple[Any, ...], operation: str) -> Any:
    if int(result[0]) != 0:
        raise RuntimeError(f"{operation} failed: {result}")
    return result[1]


def describe_stream(stream: Any, driver: Any) -> dict[str, int]:
    """Query the actual resource associated with this stream, not device totals."""
    handle = driver.CUstream(stream.cuda_stream)
    context = _driver_value(driver.cuStreamGetCtx(handle), "cuStreamGetCtx")
    resource = _driver_value(
        driver.cuCtxGetDevResource(
            context, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        ),
        "cuCtxGetDevResource",
    )
    actual_sms = int(resource.sm.smCount)
    if actual_sms <= 0:
        raise RuntimeError(f"Invalid stream SM count: {actual_sms}")
    return {
        "stream_handle": int(stream.cuda_stream),
        "stream_id": int(_driver_value(driver.cuStreamGetId(handle), "cuStreamGetId")),
        "context_handle": int(context),
        "actual_sms": actual_sms,
    }


class ARStream:
    """Own the stream and preserve producer/consumer ordering across each AR step."""

    def __init__(self, cuda: Any, stream: Any, resource: dict[str, int], context: Any = None):
        self.cuda = cuda
        self.stream = stream
        self.resource = resource
        # Keep the Green Context alive at least as long as the runner/graphs.
        self.context = context

    @contextlib.contextmanager
    def forward(self, phase: str, *, trace: bool = False) -> Iterator[None]:
        caller = self.cuda.current_stream(self.stream.device)
        self.stream.wait_stream(caller)
        try:
            with self.cuda.stream(self.stream):
                if trace:
                    self.cuda.nvtx.range_push(
                        f"qwen_gc_ar|phase={phase}|stream_id={self.resource['stream_id']}"
                        f"|actual_sms={self.resource['actual_sms']}"
                    )
                try:
                    yield
                finally:
                    if trace:
                        self.cuda.nvtx.range_pop()
        finally:
            # _finalize and output clones run on the caller's original stream.
            # Keep this fence in GC-off too; it is part of the matched adapter.
            caller.wait_stream(self.stream)


def install_capture_streams(model: Any, runtime: Any, stream: Any) -> None:
    if runtime.get_resources().streams.get("cuda_graph_capture") is not None:
        raise RuntimeError("A generation capture stream already exists; start a fresh process")
    if model._predictor_capture_stream is not None or model._predictor_graphs:
        raise RuntimeError("Predictor capture already started before Green Context injection")
    runtime.set_stream("cuda_graph_capture", stream)
    model._predictor_capture_stream = stream


def check_captures(
    model: Any, model_worker: Any, runtime: Any, stream: Any, graph: bool
) -> dict[str, Any]:
    wanted = int(stream.cuda_stream)
    named = runtime.get_stream("cuda_graph_capture")
    if int(named.cuda_stream) != wanted:
        raise RuntimeError("Generation capture stream was replaced")
    if int(model._predictor_capture_stream.cuda_stream) != wanted:
        raise RuntimeError("Predictor capture stream was replaced")
    predictor_count = len(model._predictor_graphs)
    decode = model_worker.model_runner.decode_cuda_graph_runner
    receipt: dict[str, Any] = {
        "generation_named_stream_handle": wanted,
        "predictor_stream_handle": wanted,
        "predictor_graph_count": predictor_count,
        "decode_runner": type(decode).__name__,
    }
    if graph:
        if not predictor_count:
            raise RuntimeError("Graph arm captured no predictor graphs")
        if int(getattr(getattr(decode, "stream", None), "cuda_stream", -1)) != wanted:
            raise RuntimeError("Decode graphs did not capture on the experiment stream")
        receipt["decode_capture_stream_handle"] = wanted
        receipt["decode_capture_batch_sizes"] = list(decode.capture_bs)
        if not receipt["decode_capture_batch_sizes"]:
            raise RuntimeError("Graph arm captured no decode batch sizes")
    elif predictor_count:
        raise RuntimeError("Eager arm unexpectedly captured predictor graphs")
    return receipt


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    green_context_sms: int | None = None,
    green_context_receipt_dir: str,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
    prefill_coalesce_requests: int = 0,
    prefill_coalesce_wait_ms: float = 60.0,
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    validate_request(green_context_sms, os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"))
    receipt_dir = Path(green_context_receipt_dir)
    if (
        not receipt_dir.is_absolute()
        or receipt_dir == Path("/workspace")
        or Path("/workspace") in receipt_dir.parents
    ):
        raise ValueError(
            "Receipt directory must be an absolute persistent experiment path "
            "outside /workspace"
        )

    import torch
    from cuda.bindings import driver
    from sglang.srt import runtime_context

    from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder
    from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner

    class ExperimentRunner(Qwen3TTSModelRunner):
        def __init__(self, worker: Any, output: Any, ar: ARStream):
            super().__init__(worker, output)
            self._ar = ar
            self._trace_gc = os.environ.get("SGLANG_OMNI_PROFILE_NVTX") == "1"

        @contextlib.contextmanager
        def _execution_context(self, batch: Any, *, isolate_sampling: bool = False):
            phase = "prefill" if batch.forward_mode.is_extend() else "decode"
            with self._ar.forward(phase, trace=self._trace_gc):
                with super()._execution_context(batch, isolate_sampling=isolate_sampling):
                    yield

    class ExperimentBuilder(Qwen3TtsEngineBuilder):
        def validate_before_infrastructure(self, server_args: Any) -> None:
            super().validate_before_infrastructure(server_args)
            if not bool(server_args.disable_overlap_schedule):
                raise ValueError("This adapter requires synchronous AR scheduling")
            if bool(getattr(server_args, "enable_pdmux", False)):
                raise ValueError("PDMux uses other capture streams and is outside this experiment")

        def before_memory_pool(self, **kwargs: Any) -> None:
            self._worker = kwargs["model_worker"]
            self._graph = not bool(kwargs["server_args"].disable_cuda_graph)
            model = self._worker.model_runner.model
            context = None
            if green_context_sms is None:
                stream = torch.cuda.Stream(device=kwargs["gpu_id"])
            else:
                from torch.cuda.green_contexts import GreenContext

                context = GreenContext.create(
                    num_sms=green_context_sms, device_id=kwargs["gpu_id"]
                )
                stream = context.Stream()
            resource = describe_stream(stream, driver)
            if green_context_sms is not None and resource["actual_sms"] < green_context_sms:
                raise RuntimeError("Green Context has fewer SMs than requested")
            self._ar = ARStream(torch.cuda, stream, resource, context)
            install_capture_streams(model, runtime_context, stream)
            super().before_memory_pool(**kwargs)

        def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
            model = model_worker.model_runner.model
            capture_receipt = check_captures(
                model, model_worker, runtime_context, self._ar.stream, self._graph
            )
            # Re-query after graph setup to catch context/stream replacement.
            current_resource = describe_stream(self._ar.stream, driver)
            if current_resource != self._ar.resource:
                raise RuntimeError("AR stream resources changed during graph setup")
            receipt = {
                "status": "stream_and_capture_setup_verified",
                "kernel_trace_verified": False,
                "scope": "AR prefill/decode and code predictor; preprocessing/vocoder excluded",
                "globally_disjoint_replica_sms_proven": False,
                "pid": os.getpid(),
                "requested_sms": green_context_sms,
                "graph": self._graph,
                "resource": current_resource,
                "captures": capture_receipt,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "sglang": importlib.metadata.version("sglang"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
                "cuda_mps_active_thread_percentage": os.environ.get(
                    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
                ),
            }
            receipt_dir.mkdir(parents=True, exist_ok=True)
            with (receipt_dir / f"qwen-gc-{os.getpid()}.json").open("x") as output:
                json.dump(receipt, output, indent=2)
                output.write("\n")
            return ExperimentRunner(model_worker, output_proc, self._ar)

    return ExperimentBuilder(
        attn_implementation=attn_implementation,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )
