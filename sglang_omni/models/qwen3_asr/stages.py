# SPDX-License-Identifier: Apache-2.0
"""Stage factories for Qwen3-ASR: the SGLang engine and the CPU-only audio frontend."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import partial

import torch
from transformers import AutoFeatureExtractor

from sglang_omni.models.qwen3_asr.engine_builder import Qwen3ASREngineBuilder
from sglang_omni.models.qwen3_asr.request_builders import (
    Qwen3ASRRequestData,
    prepare_audio_for_engine,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.serve.openai_errors import generation_error_status_code

logger = logging.getLogger(__name__)


def create_sglang_qwen3_asr_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "auto",
    max_running_requests: int = 64,
    max_new_tokens: int = 256,
    mem_fraction_static: float | None = None,
    mm_embedding_cache_size_bytes: int = 0,
    enable_torch_compile: bool | None = None,
    torch_compile_max_bs: int = 1,
    enable_async_decode: bool = True,
    enable_plain_fallback: bool = False,
    async_decode_min_batch_size: int | None = None,
    mm_attention_backend: str | None = None,
    request_build_max_workers: int = 8,
    request_build_max_pending: int | None = 32,
    prefill_coalesce_requests: int = 16,
    prefill_coalesce_wait_ms: float = 40.0,
    prefill_coalesce_when_idle: bool = True,
    prefill_coalesce_requires_pending_builds: bool = True,
    prefill_coalesce_after_builds_during_decode: bool | None = None,
    stream_emit_interval_s: float = 0.05,
    enable_pre_lm_encoder: bool = True,
    pre_lm_cache_max_entries: int = 4096,
    pre_lm_cache_size_bytes: int = 2 * 1024**3,
    pre_lm_max_batch_size: int = 8,
    pre_lm_max_batch_wait_ms: int = 0,
    enable_encoder_cuda_graph: bool = True,
    max_audio_clip_s: float | None = None,
    server_args_overrides: Mapping[str, object] | None = None,
) -> OmniScheduler[Qwen3ASRRequestData]:
    builder = Qwen3ASREngineBuilder(
        max_running_requests=max_running_requests,
        max_new_tokens=max_new_tokens,
        enable_async_decode=enable_async_decode,
        enable_plain_fallback=enable_plain_fallback,
        async_decode_min_batch_size=async_decode_min_batch_size,
        mem_fraction_static=mem_fraction_static,
        mm_embedding_cache_size_bytes=mm_embedding_cache_size_bytes,
        enable_torch_compile=enable_torch_compile,
        torch_compile_max_bs=torch_compile_max_bs,
        mm_attention_backend=mm_attention_backend,
        request_build_max_workers=request_build_max_workers,
        request_build_max_pending=request_build_max_pending,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms,
        prefill_coalesce_when_idle=prefill_coalesce_when_idle,
        prefill_coalesce_requires_pending_builds=(
            prefill_coalesce_requires_pending_builds
        ),
        prefill_coalesce_after_builds_during_decode=(
            prefill_coalesce_after_builds_during_decode
        ),
        stream_emit_interval_s=stream_emit_interval_s,
        enable_pre_lm_encoder=enable_pre_lm_encoder,
        pre_lm_cache_max_entries=pre_lm_cache_max_entries,
        pre_lm_cache_size_bytes=pre_lm_cache_size_bytes,
        pre_lm_max_batch_size=pre_lm_max_batch_size,
        pre_lm_max_batch_wait_ms=pre_lm_max_batch_wait_ms,
        enable_encoder_cuda_graph=enable_encoder_cuda_graph,
        max_audio_clip_s=max_audio_clip_s,
    )
    try:
        return builder.build(
            model_path,
            device=device,
            gpu_id=gpu_id,
            dtype=dtype,
            server_args_overrides=server_args_overrides,
        )
    finally:
        if builder.draft_checkpoint is not None:
            builder.draft_checkpoint.cleanup()
        else:
            pass


def create_qwen3_asr_executor(*args, **kwargs):
    return create_sglang_qwen3_asr_executor(*args, **kwargs)


class Qwen3ASRFrontendScheduler(ThreadedSimpleScheduler[StagePayload, StagePayload]):
    def log_failure(self, request_id: str, exc: BaseException) -> None:
        # note (wenyao): the API answers this failure with 400, as when the engine decodes the upload, so the traceback is noise.
        if generation_error_status_code(exc) == 400:
            logger.warning(f"Qwen3-ASR frontend rejected {request_id}: {exc}")
        else:
            super().log_failure(request_id, exc)


def create_qwen3_asr_frontend_executor(
    model_path: str, *, max_concurrency: int
) -> Qwen3ASRFrontendScheduler:
    # note (wenyao): sglang sets one intra-op thread in the engine process at model load, so the log-mel was single-threaded there; the same count here stops every worker thread from fanning out over the cores this process was given.
    torch.set_num_threads(1)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_path, trust_remote_code=True
    )
    return Qwen3ASRFrontendScheduler(
        partial(prepare_audio_for_engine, feature_extractor=feature_extractor),
        max_concurrency=max_concurrency,
    )


__all__ = [
    "create_sglang_qwen3_asr_executor",
    "create_qwen3_asr_executor",
    "create_qwen3_asr_frontend_executor",
]
