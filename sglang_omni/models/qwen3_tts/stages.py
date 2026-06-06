# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Qwen3-TTS Base pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.models.qwen3_tts.request_builders import (
    cleanup_prepared_qwen3_tts_request,
    make_qwen3_tts_scheduler_adapters,
    preprocess_qwen3_tts_payload,
    set_qwen3_tts_preprocessing_context,
)
from sglang_omni.models.qwen3_tts.streaming_vocoder import (
    Qwen3TTSStreamingVocoderScheduler,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

logger = logging.getLogger(__name__)

_QWEN_TTS_INSTALL_HINT = (
    "Qwen3-TTS support requires the official `qwen-tts` package. "
    "Install `qwen-tts==0.1.1` and its Transformers 4.57.3 requirement "
    "in the serving environment before launching Qwen3-TTS."
)


def load_state(payload: StagePayload) -> Qwen3TTSState:
    return Qwen3TTSState.from_dict(payload.data)


def store_state(payload: StagePayload, state: Qwen3TTSState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _load_qwen3_tts_tokenizer(
    model_path: str,
    *,
    device: str,
    dtype: str,
    attn_implementation: str | None,
):
    try:
        from qwen_tts import Qwen3TTSTokenizer
    except ImportError as exc:
        raise RuntimeError(_QWEN_TTS_INSTALL_HINT) from exc

    checkpoint_dir = _resolve_checkpoint(model_path)
    tokenizer_path = os.path.join(checkpoint_dir, "speech_tokenizer")
    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    kwargs: dict[str, Any] = {
        "device_map": device,
        "dtype": torch_dtype,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    logger.info(f"Loading Qwen3-TTS speech tokenizer from {tokenizer_path} on {device}")
    return Qwen3TTSTokenizer.from_pretrained(tokenizer_path, **kwargs)


def _register_qwen3_tts_hf_config() -> None:
    try:
        from qwen_tts.core.models import Qwen3TTSConfig
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError(_QWEN_TTS_INSTALL_HINT) from exc
    if not hasattr(Qwen3TTSConfig, "_sglang_omni_patched"):
        original_init = Qwen3TTSConfig.__init__

        def _patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            talker_config = getattr(self, "talker_config", None)
            if talker_config is not None:
                self.text_config = talker_config

        Qwen3TTSConfig.__init__ = _patched_init
        Qwen3TTSConfig._sglang_omni_patched = True
    try:
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    except ValueError:
        pass


def _load_qwen3_tts_generate_defaults(checkpoint_dir: str) -> dict[str, Any]:
    import json

    path = os.path.join(checkpoint_dir, "generation_config.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _compile_qwen3_tts_backbone(model: Any) -> None:
    """Compile decoder blocks while leaving decode-input staging eager."""

    text_model = model.model
    layers = text_model.layers

    from sglang.srt.model_executor.cuda_graph_runner import set_torch_compile_config

    set_torch_compile_config()
    compile_mode = os.environ.get(
        "SGLANG_TORCH_COMPILE_MODE",
        "max-autotune-no-cudagraphs",
    )
    text_model._compiled_decode_layers = [
        torch.compile(layer, mode=compile_mode) for layer in layers
    ]


def _audio_to_list(audio: Any) -> list[float]:
    if isinstance(audio, torch.Tensor):
        return audio.detach().float().cpu().flatten().tolist()
    try:
        import numpy as np

        array = np.asarray(audio, dtype=np.float32).reshape(-1)
        return array.tolist()
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Unsupported Qwen3-TTS audio output type: {type(audio)}"
        ) from exc


def _build_usage(state: Qwen3TTSState) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    usage = {
        "prompt_tokens": state.prompt_tokens,
        "completion_tokens": state.completion_tokens,
        "total_tokens": state.prompt_tokens + state.completion_tokens,
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    return usage


def create_preprocessing_executor(model_path: str) -> SimpleScheduler:
    del model_path
    return SimpleScheduler(
        preprocess_qwen3_tts_payload,
        abort_callback=cleanup_prepared_qwen3_tts_request,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
) -> Any:
    from qwen_tts import Qwen3TTSModel
    from transformers import AutoProcessor

    import sglang_omni.models.qwen3_tts.model_runner as model_runner_mod
    import sglang_omni.scheduling.bootstrap as bootstrap_mod
    import sglang_omni.scheduling.omni_scheduler as omni_scheduler_mod
    import sglang_omni.scheduling.sglang_backend as sglang_backend_mod

    _register_qwen3_tts_hf_config()
    checkpoint_dir = _resolve_checkpoint(model_path)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    server_args = sglang_backend_mod.build_sglang_server_args(
        checkpoint_dir,
        context_length=8192,
        dtype=dtype,
        disable_cuda_graph=False,
        disable_overlap_schedule=True,
        enable_torch_compile=True,
        mem_fraction_static=0.85,
        max_prefill_tokens=8192,
        max_running_requests=16,
        sampling_backend="pytorch",
        torch_compile_max_bs=16,
        trust_remote_code=True,
    )

    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    if want_cuda_graph:
        server_args.disable_cuda_graph = True

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = bootstrap_mod.create_sglang_infrastructure(
        server_args,
        gpu_id,
        model_arch_override="Qwen3TTSTalker",
    )

    if want_cuda_graph:
        server_args.disable_cuda_graph = False

    model = model_worker.model_runner.model
    speech_tokenizer = _load_qwen3_tts_tokenizer(
        checkpoint_dir,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model.load_speech_tokenizer(speech_tokenizer)
    processor = AutoProcessor.from_pretrained(checkpoint_dir, fix_mistral_regex=True)
    wrapper = Qwen3TTSModel(
        model=model,
        processor=processor,
        generate_defaults=_load_qwen3_tts_generate_defaults(checkpoint_dir),
    )
    set_qwen3_tts_preprocessing_context(model=model, wrapper=wrapper)
    if bool(getattr(server_args, "enable_torch_compile", False)):
        _compile_qwen3_tts_backbone(model)
        server_args.enable_torch_compile = False
    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()

    output_proc = sglang_backend_mod.SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model,
    )
    request_builder, result_adapter = make_qwen3_tts_scheduler_adapters(
        model=model,
        wrapper=wrapper,
    )

    model_runner = model_runner_mod.Qwen3TTSModelRunner(model_worker, output_proc)
    scheduler = omni_scheduler_mod.OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        abort_callback=cleanup_prepared_qwen3_tts_request,
    )
    model_runner.set_stream_outbox(scheduler.outbox)
    return scheduler


def create_tts_engine_executor(*args, **kwargs) -> Any:
    return create_sglang_tts_engine_executor(*args, **kwargs)


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
    stream_chunk_frames: int = 6,
    left_context_frames: int = 6,
) -> Qwen3TTSStreamingVocoderScheduler:
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    tokenizer = _load_qwen3_tts_tokenizer(
        model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    return Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        stream_chunk_frames=stream_chunk_frames,
        left_context_frames=left_context_frames,
    )
