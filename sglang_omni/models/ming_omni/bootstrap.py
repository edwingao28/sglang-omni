# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

_STREAM_STATE_MAX_REQUESTS = 4096
_STREAM_STATE_EVICT_TO = 3072


def create_thinker_scheduler(
    server_args: Any,
    *,
    model_path: str,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    enable_streaming_outputs: bool = False,
):
    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}")
    if getattr(server_args, "tp_size", None) != tp_size:
        server_args.tp_size = tp_size

    from sglang_omni.model_runner.ming_thinker_model_runner import (
        MingThinkerModelRunner,
    )
    from sglang_omni.models.ming_omni.components.common import (
        load_ming_config,
        load_ming_tokenizer,
    )
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import SGLangOutputProcessor

    tokenizer = load_ming_tokenizer(model_path)
    config = load_ming_config(model_path)
    llm_cfg = getattr(config, "llm_config", config)
    vocab_size = getattr(llm_cfg, "vocab_size", None) or getattr(
        tokenizer, "vocab_size", 32000
    )

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        model_arch_override="BailingMoeV2ForCausalLM",
    )

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    model_runner = MingThinkerModelRunner(model_worker, output_proc)

    image_token_id = getattr(llm_cfg, "image_patch_token", None)
    video_token_id = getattr(llm_cfg, "video_patch_token", None)
    _audio_tok = tokenizer.convert_tokens_to_ids("<audioPatch>")
    _unk = getattr(tokenizer, "unk_token_id", None)
    audio_token_id = (
        _audio_tok if isinstance(_audio_tok, int) and _audio_tok != _unk else None
    )

    request_builder, result_adapter = make_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        image_token_id=image_token_id,
        audio_token_id=audio_token_id,
        video_token_id=video_token_id,
    )
    stream_output_builder = None
    if enable_streaming_outputs:
        stream_output_builder = make_ming_thinker_stream_output_builder(
            tokenizer=tokenizer,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )

    return OmniScheduler(
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
        stream_output_builder=stream_output_builder,
    )


def make_ming_thinker_stream_output_builder(
    *,
    tokenizer: Any,
    eos_token_id: int | None = None,
):
    """Build targeted stream chunks from Ming thinker per-token outputs."""
    from sglang_omni.models.ming_omni.components.streaming_text import (
        text_to_uint8_tensor,
    )
    from sglang_omni.models.ming_omni.io import PipelineState
    from sglang_omni.models.ming_omni.pipeline.merge import decode_events
    from sglang_omni.models.ming_omni.pipeline.next_stage import (
        DECODE_STAGE,
        SEGMENTER_STAGE,
    )
    from sglang_omni.scheduling.messages import OutgoingMessage

    import torch

    states: OrderedDict[str, PipelineState] = OrderedDict()

    def _state_for(request_id: str) -> PipelineState:
        state = states.get(request_id)
        if state is None:
            state = PipelineState()
            states[request_id] = state
            if len(states) > _STREAM_STATE_MAX_REQUESTS:
                for _ in range(len(states) - _STREAM_STATE_EVICT_TO):
                    states.popitem(last=False)
        else:
            states.move_to_end(request_id)
        return state

    def _build_stream_output(
        request_id: str, req_data: Any, req_output: Any
    ) -> list[OutgoingMessage]:
        req = getattr(req_data, "req", None)
        if req is not None and int(getattr(req, "is_chunked", 0) or 0) > 0:
            return []
        if req_output.data is None:
            return []

        token_id = int(req_output.data)
        stage_payload = getattr(req_data, "stage_payload", None)
        request = getattr(stage_payload, "request", None)
        params = getattr(request, "params", None)
        is_streaming = bool(isinstance(params, dict) and params.get("stream") is True)
        text_requested = _is_output_modality_requested(request, "text")
        audio_requested = _is_output_modality_requested(request, "audio")

        state = _state_for(request_id)
        step = len(state.stream_state.get("token_ids", [])) + 1
        events = list(
            decode_events(
                thinker_out={
                    "output_ids": [token_id],
                    "step": step,
                    "is_final": False,
                    "extra_model_outputs": {},
                },
                state=state,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
                step=step,
            )
        )

        messages: list[OutgoingMessage] = []
        metadata = {
            "modality": "text",
            "stage_name": "thinker",
            "token_id": token_id,
            "step": step,
        }
        if is_streaming and text_requested:
            messages.append(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    target=DECODE_STAGE,
                    data=torch.tensor([token_id], dtype=torch.long),
                    metadata=dict(metadata),
                )
            )

        if audio_requested:
            delta = "".join(
                str(event.payload.get("text") or "")
                for event in events
                if event.type == "text_delta"
            )
            if delta:
                text_tensor = text_to_uint8_tensor(delta)
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        target=SEGMENTER_STAGE,
                        data=text_tensor,
                        metadata={
                            **metadata,
                            "text_len": int(text_tensor.numel()),
                        },
                    )
                )

        if (
            (eos_token_id is not None and token_id == int(eos_token_id))
            or getattr(req_output, "finished", False)
            or _has_reached_max_new_tokens(req_data)
        ):
            states.pop(request_id, None)

        return messages

    return _build_stream_output


def _output_modalities(request: Any) -> set[str] | None:
    metadata = getattr(request, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    modalities = metadata.get("output_modalities")
    if modalities is None:
        return None
    if isinstance(modalities, str):
        values = (modalities,)
    elif isinstance(modalities, (list, tuple, set)):
        values = modalities
    else:
        return None
    normalized = {str(modality).lower() for modality in values}
    if not normalized.intersection({"text", "audio"}):
        return None
    return normalized


def _is_output_modality_requested(request: Any, modality: str) -> bool:
    modalities = _output_modalities(request)
    return modalities is None or modality.lower() in modalities


def _has_reached_max_new_tokens(req_data: Any) -> bool:
    max_new_tokens = getattr(req_data, "max_new_tokens", None)
    if max_new_tokens is None:
        return False
    try:
        return int(getattr(req_data, "generation_steps", 0)) >= int(max_new_tokens)
    except (TypeError, ValueError):
        return False


def make_thinker_scheduler_adapters(
    *,
    tokenizer: Any,
    vocab_size: int,
    image_token_id: int | None = None,
    audio_token_id: int | None = None,
    video_token_id: int | None = None,
    stage_name: str = "thinker",
):
    """Build StagePayload <-> SGLang request adapters."""

    def request_builder(payload):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        from sglang_omni.models.ming_omni.io import PipelineState
        from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

        state = PipelineState.from_dict(payload.data)
        prompt = state.prompt
        if not isinstance(prompt, dict):
            raise TypeError("prompt missing for thinker request")

        input_ids = prompt.get("input_ids")
        if not hasattr(input_ids, "to"):
            raise TypeError("prompt.input_ids must be a torch.Tensor")

        # Per-content pad_value substitution to defeat SGLang radix prefix-cache
        # aliasing across multimodal requests that share the same generic
        # image/audio/video patch token id.
        thinker_inputs_early = state.thinker_inputs or {}
        media_cache_keys = thinker_inputs_early.get("media_cache_keys") or {}
        pad_values: dict[str, int] = {}
        if media_cache_keys:
            import xxhash

            token_id_map: dict[int, int] = {}
            for _modality, _orig in [
                ("image", image_token_id),
                ("audio", audio_token_id),
                ("video", video_token_id),
            ]:
                if _orig is None:
                    continue
                _ck = media_cache_keys.get(_modality)
                if _ck is None:
                    continue
                _h = xxhash.xxh3_64(_ck.encode()).intdigest()
                _pad = vocab_size + _h % (1 << 62)
                pad_values[_modality] = _pad
                token_id_map[int(_orig)] = _pad
            if token_id_map:
                input_ids = input_ids.clone()
                for _orig_id, _pad in token_id_map.items():
                    input_ids[input_ids == _orig_id] = _pad

        input_ids_list = input_ids.to(dtype=_torch_long()).flatten().tolist()

        params = payload.request.params or {}
        max_new_tokens = params.get("max_new_tokens", 2048)
        temperature = params.get("temperature", 0.0)
        sampling_params = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        sampling_params.normalize(tokenizer)
        sampling_params.verify(vocab_size)

        eos_token_ids = _collect_eos_token_ids(tokenizer)
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids_list,
            sampling_params=sampling_params,
            vocab_size=vocab_size,
            eos_token_ids=eos_token_ids,
        )
        req.tokenizer = tokenizer

        thinker_inputs = thinker_inputs_early
        model_inputs = dict(thinker_inputs.get("model_inputs", {}))
        if not model_inputs:
            model_inputs = {
                key: value
                for key, value in thinker_inputs.items()
                if key not in ("capture_model_output_keys", "media_cache_keys")
            }
        model_inputs.pop("attention_mask", None)
        if pad_values:
            model_inputs["pad_values"] = pad_values
        capture_keys = thinker_inputs.get("capture_model_output_keys", ())

        req.omni_model_inputs = model_inputs if model_inputs else None
        req._omni_consumed = None
        req._codec_suppress_tokens = None

        attention_mask = prompt.get("attention_mask")
        req_data = SGLangARRequestData(
            input_ids=input_ids.to(dtype=_torch_long()).flatten(),
            attention_mask=attention_mask if hasattr(attention_mask, "to") else None,
            model_inputs=model_inputs,
            capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            output_ids=req.output_ids,
            req=req,
        )
        req_data.stage_payload = payload
        return req_data

    def result_adapter(data):
        from sglang_omni.models.ming_omni.io import PipelineState
        from sglang_omni.proto import StagePayload

        payload = data.stage_payload
        state = PipelineState.from_dict(payload.data)
        output_ids = list(data.output_ids)
        if data.finish_reason is not None or not output_ids:
            logger.info(
                "Ming thinker result request_id=%s finish=%s output_len=%d "
                "output_tail=%s stop_hits=%s",
                payload.request_id,
                data.finish_reason,
                len(output_ids),
                output_ids[-8:],
                _stop_hits(output_ids, tokenizer),
            )
        thinker_out: dict[str, Any] = {
            "output_ids": output_ids,
            "step": len(output_ids),
            "is_final": True,
            "extra_model_outputs": dict(data.extra_model_outputs),
        }
        if data.finish_reason is not None:
            thinker_out["finish_reason"] = data.finish_reason
        state.thinker_out = thinker_out
        state.engine_outputs[stage_name] = thinker_out
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter


def _torch_long():
    import torch

    return torch.long


def _collect_eos_token_ids(tokenizer: Any) -> set[int] | None:
    """Match Ming V0: let the SGLang request stop only on tokenizer EOS."""
    eid = getattr(tokenizer, "eos_token_id", None)
    return {int(eid)} if isinstance(eid, int) and eid >= 0 else None


def _stop_hits(output_ids: list[int], tokenizer: Any) -> list[int]:
    stop_ids = _collect_eos_token_ids(tokenizer) or set()
    return [int(token_id) for token_id in output_ids[-8:] if int(token_id) in stop_ids]
