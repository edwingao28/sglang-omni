# SPDX-License-Identifier: Apache-2.0
"""Engine request/response helpers for Ming-Omni stages."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from sglang_omni.engines.omni.runtime import ARRequestData, EncoderRequestData
from sglang_omni.models.ming_omni.io import PipelineState, ThinkerOutput

if TYPE_CHECKING:
    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangARRequestData


logger = logging.getLogger(__name__)
_DEFAULT_THINKER_MAX_NEW_TOKENS = 2048


def _validate_prompt_seq_len(
    input_ids: torch.Tensor,
    *,
    max_seq_len: int | None,
    max_new_tokens: int | None = None,
    request_id: str | None = None,
) -> None:
    if max_seq_len is None:
        return
    prompt_len = int(input_ids.numel())
    if prompt_len >= max_seq_len:
        logger.info(
            f"rejecting request {request_id}: prompt {prompt_len} tokens "
            f">= max_seq_len {max_seq_len}"
        )
        raise ValueError(
            f"The input ({prompt_len} tokens) is longer than the model's "
            f"context length ({max_seq_len} tokens)."
        )
    if max_new_tokens is None:
        return
    total_tokens = prompt_len + int(max_new_tokens)
    if total_tokens >= max_seq_len:
        logger.info(
            f"rejecting request {request_id}: prompt {prompt_len} + "
            f"max_new_tokens {int(max_new_tokens)} = {total_tokens} tokens "
            f">= max_seq_len {max_seq_len}"
        )
        raise ValueError(
            f"Requested token count exceeds the model's maximum context length "
            f"of {max_seq_len} tokens. You requested a total of {total_tokens} "
            f"tokens: {prompt_len} tokens from the input messages and "
            f"{int(max_new_tokens)} tokens for the completion. Please reduce "
            f"the number of tokens in the input messages or the completion to "
            f"fit within the limit."
        )


def build_encoder_request(
    state: PipelineState, *, stage_name: str
) -> EncoderRequestData:
    inputs = state.encoder_inputs.get(stage_name)
    if not isinstance(inputs, dict) or not inputs:
        return EncoderRequestData(input_dict={"_skip": True, "_result": {}})
    if inputs.get("_skip"):
        skip_result = inputs.get("_result")
        return EncoderRequestData(
            input_dict=inputs,
            output_dict=skip_result if isinstance(skip_result, dict) else {},
        )
    cache_key = inputs.get("cache_key")
    return EncoderRequestData(
        input_dict=inputs,
        cache_key=str(cache_key) if cache_key is not None else None,
    )


def apply_encoder_result(
    state: PipelineState,
    *,
    stage_name: str,
    result: Any,
) -> None:
    if isinstance(result, EncoderRequestData):
        if result.output_dict is not None:
            encoder_out = result.output_dict
        elif result.embeddings is not None:
            encoder_out = result.embeddings
        else:
            encoder_out = {}
    else:
        encoder_out = result if isinstance(result, dict) else {"result": result}

    state.encoder_outs[stage_name] = encoder_out
    state.engine_outputs[stage_name] = encoder_out


def build_thinker_request(
    state: PipelineState,
    *,
    params: dict[str, Any],
) -> ARRequestData:
    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for thinker request")

    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("prompt.input_ids must be a torch.Tensor")

    attention_mask = prompt.get("attention_mask")
    thinker_inputs = state.thinker_inputs or {}

    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if not model_inputs:
        model_inputs = {
            k: v for k, v in thinker_inputs.items() if k != "capture_model_output_keys"
        }

    capture_keys = thinker_inputs.get("capture_model_output_keys", ())
    model_inputs.pop("attention_mask", None)

    return ARRequestData(
        input_ids=input_ids.to(dtype=torch.long),
        attention_mask=(
            attention_mask if isinstance(attention_mask, torch.Tensor) else None
        ),
        model_inputs=model_inputs,
        capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
        max_new_tokens=params.get("max_new_tokens"),
        temperature=params.get("temperature", 0.0),
    )


def build_sglang_thinker_request(
    state: PipelineState,
    *,
    params: dict[str, Any],
    tokenizer: Any,
    vocab_size: int,
    max_seq_len: int | None = None,
    request_id: str | None = None,
) -> "SGLangARRequestData":
    """Build SGLangARRequestData for the Ming thinker.

    Ming uses standard 1D RoPE (no M-RoPE), so no special position computation needed.
    """
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    from sglang_omni.engines.omni.runtime.sglang_ar import SGLangARRequestData

    prompt = state.prompt
    if not isinstance(prompt, dict):
        raise TypeError("prompt missing for thinker request")

    input_ids = prompt.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("prompt.input_ids must be a torch.Tensor")

    max_new_tokens = params.get("max_new_tokens", _DEFAULT_THINKER_MAX_NEW_TOKENS)
    _validate_prompt_seq_len(
        input_ids,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        request_id=request_id,
    )

    input_ids_list = input_ids.to(dtype=torch.long).flatten().tolist()

    attention_mask = prompt.get("attention_mask")
    thinker_inputs = state.thinker_inputs or {}

    model_inputs = dict(thinker_inputs.get("model_inputs", {}))
    if not model_inputs:
        model_inputs = {
            k: v for k, v in thinker_inputs.items() if k != "capture_model_output_keys"
        }
    capture_keys = thinker_inputs.get("capture_model_output_keys", ())
    model_inputs.pop("attention_mask", None)

    temperature = params.get("temperature", 0.0)

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(vocab_size)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token_ids = {eos_token_id} if eos_token_id is not None else None

    rid = request_id or "req-0"
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
        eos_token_ids=eos_token_ids,
    )

    # Attach multimodal model inputs (audio embeddings, placeholder locations)
    req.omni_model_inputs = model_inputs if model_inputs else None
    req._omni_consumed = None

    data = SGLangARRequestData(
        input_ids=input_ids.to(dtype=torch.long).flatten(),
        attention_mask=(
            attention_mask if isinstance(attention_mask, torch.Tensor) else None
        ),
        model_inputs=model_inputs,
        capture_model_output_keys=tuple(capture_keys) if capture_keys else (),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        output_ids=req.output_ids,
        req=req,
    )
    return data


def apply_thinker_result(
    state: PipelineState,
    *,
    stage_name: str,
    result: Any,
) -> ThinkerOutput:
    if isinstance(result, ARRequestData):
        output_ids = list(result.output_ids)
        prompt_tokens = (
            int(result.input_ids.shape[0])
            if result.input_ids is not None and hasattr(result.input_ids, "shape")
            else 0
        )
        finish_reason = None
        req_finish_reason = getattr(
            getattr(result, "req", None), "finished_reason", None
        )
        if hasattr(req_finish_reason, "to_json"):
            finish_reason = req_finish_reason.to_json().get("type")
        thinker_out: ThinkerOutput = {
            "output_ids": output_ids,
            "step": len(output_ids),
            "is_final": True,
            "finish_reason": finish_reason,
            "extra_model_outputs": dict(result.extra_model_outputs),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(output_ids),
        }
    else:
        thinker_out = {
            "output_ids": [],
            "step": 0,
            "is_final": True,
            "finish_reason": None,
            "extra_model_outputs": {"result": result},
        }

    state.thinker_out = thinker_out
    state.engine_outputs[stage_name] = thinker_out
    return thinker_out
