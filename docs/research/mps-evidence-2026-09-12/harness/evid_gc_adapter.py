"""Service adapter: ordinary side stream, indexed disjoint or union2 Green Context.

Derived from round-1 sustained_gc_adapter.py. Only resource creation differs
across placements; qwen_gc capture/fence helpers keep normal Qwen generation.
Scope: AR prefill/decode and the code predictor. Preprocessing and vocoder stay
on their own streams.
"""
from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Literal

from gc_resources import create_stream, describe_stream, validate_request
from qwen_gc import ARStream, check_captures, install_capture_streams

Placement = Literal['ordinary', 'indexed', 'union2']


def resource_selection(placement: Placement, rank: int, sms: int | None, group_count: int):
    if placement not in ('ordinary', 'indexed', 'union2'):
        raise ValueError('Require ordinary/indexed/union2')
    if type(rank) is not int or type(group_count) is not int or not 0 <= rank < group_count:
        raise ValueError('rank must index a replica of the topology')
    if placement == 'ordinary':
        return None, 0, group_count, False, 132
    if type(sms) is not int or sms <= 0:
        raise ValueError('indexed/union2 need a positive SM count')
    if placement == 'union2' and group_count < 3:
        raise ValueError('union2 with two groups gives both ranks the same SMs; need at least three groups')
    if sms % 8:
        raise ValueError('H100 smCoscheduledAlignment is 8; sms must be a multiple of 8 (44 would round to 48)')
    union = placement == 'union2'
    return sms, rank, group_count, union, sms * (2 if union else 1)


def create_sglang_tts_engine_executor(
    model_path: str, *, resource_placement: Placement = 'ordinary', resource_sms: int | None = None,
    resource_rank: int = 0, resource_group_count: int = 3, resource_receipt_dir: str,
    device: str | None = None, gpu_id: int | None = None, dtype: str = 'bfloat16',
    attn_implementation: str | None = None, prefill_coalesce_requests: int = 0,
    prefill_coalesce_wait_ms: float = 60.0, server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    sms, index, count, union, expected_sms = resource_selection(
        resource_placement, resource_rank, resource_sms, resource_group_count)
    validate_request(sms, index, count)
    receipt_dir = Path(resource_receipt_dir)
    if not receipt_dir.is_absolute() or Path('/workspace') in (receipt_dir, *receipt_dir.parents):
        raise ValueError('Resource receipt directory must be absolute and outside /workspace')
    import torch
    from cuda.bindings import driver
    from sglang.srt import runtime_context
    from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder
    from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner

    class ServiceRunner(Qwen3TTSModelRunner):
        def __init__(self, worker: Any, output: Any, ar: ARStream):
            super().__init__(worker, output)
            self._ar = ar
            self._trace_gc = os.environ.get('SGLANG_OMNI_PROFILE_NVTX') == '1'

        @contextlib.contextmanager
        def _execution_context(self, batch: Any, *, isolate_sampling: bool = False):
            phase = 'prefill' if batch.forward_mode.is_extend() else 'decode'
            with self._ar.forward(phase, trace=self._trace_gc):
                with super()._execution_context(batch, isolate_sampling=isolate_sampling):
                    yield

    class ServiceBuilder(Qwen3TtsEngineBuilder):
        def validate_before_infrastructure(self, server_args: Any) -> None:
            super().validate_before_infrastructure(server_args)
            if server_args.disable_cuda_graph or not server_args.disable_overlap_schedule:
                raise ValueError('This comparison requires Graph enabled and synchronous AR scheduling')
            if bool(getattr(server_args, 'enable_pdmux', False)):
                raise ValueError('PDMux is outside this resource comparison')

        def before_memory_pool(self, **kwargs: Any) -> None:
            owner, stream, resource = create_stream(kwargs['gpu_id'], sms, index, count, union)
            if resource['actual_sms'] != expected_sms:
                raise RuntimeError(f'Actual stream SM count {resource["actual_sms"]} != {expected_sms}')
            self._ar = ARStream(torch.cuda, stream, resource, owner)
            install_capture_streams(kwargs['model_worker'].model_runner.model, runtime_context, stream)
            super().before_memory_pool(**kwargs)

        def make_model_runner(self, model_worker: Any, output_proc: Any) -> Any:
            captures = check_captures(model_worker.model_runner.model, model_worker,
                                      runtime_context, self._ar.stream, True)
            after = describe_stream(self._ar.stream, driver)
            if any(self._ar.resource.get(key) != value for key, value in after.items()):
                raise RuntimeError('AR stream resources changed during Graph setup')
            receipt = dict(status='stream_and_capture_setup_verified', pid=os.getpid(), rank=resource_rank,
                placement=resource_placement, requested_sms=sms, group_count=count, union_next=union,
                expected_actual_sms=expected_sms, resource=self._ar.resource,
                resource_after_graph_setup=after, captures=captures, graph=True,
                ordinary_side_stream_control=sms is None, matched_side_stream_fences=True,
                actual_model_kernel_ownership='pending external trace audit',
                kernel_trace_verified=False, globally_disjoint_replica_sms_proven=False,
                scope='Real service AR prefill/decode and code predictor only; preprocessing/vocoder unchanged',
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                cuda_mps_pipe_directory=os.environ.get('CUDA_MPS_PIPE_DIRECTORY'),
                cuda_mps_active_thread_percentage=os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'),
                torch=torch.__version__, cuda=torch.version.cuda, sglang=importlib.metadata.version('sglang'))
            receipt_dir.mkdir(parents=True, exist_ok=True)
            with (receipt_dir / f'qwen-service-resource-{os.getpid()}.json').open('x') as output:
                json.dump(receipt, output, indent=2, allow_nan=False)
                output.write('\n')
            return ServiceRunner(model_worker, output_proc, self._ar)

    return ServiceBuilder(attn_implementation=attn_implementation,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms).build(model_path, device=device,
            gpu_id=gpu_id, dtype=dtype, server_args_overrides=server_args_overrides)
