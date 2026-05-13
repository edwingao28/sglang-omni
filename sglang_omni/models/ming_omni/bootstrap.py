# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def create_thinker_scheduler(
    server_args: Any,
    *,
    model_path: str,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
):
    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}")
    if getattr(server_args, "tp_size", None) != tp_size:
        server_args.tp_size = tp_size

    from sglang_omni_v1.model_runner.ming_thinker_model_runner import (
        MingThinkerModelRunner,
    )
    from sglang_omni_v1.models.ming_omni.components.common import (
        load_ming_config,
        load_ming_tokenizer,
    )
    from sglang_omni_v1.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni_v1.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni_v1.scheduling.sglang_backend import SGLangOutputProcessor

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

    request_builder, result_adapter = make_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=vocab_size,
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
    )


def make_thinker_scheduler_adapters(
    *,
    tokenizer: Any,
    vocab_size: int,
    stage_name: str = "thinker",
):
    """Build StagePayload <-> SGLang request adapters."""

    def request_builder(payload):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        from sglang_omni_v1.models.ming_omni.io import PipelineState
        from sglang_omni_v1.scheduling.sglang_backend import SGLangARRequestData

        state = PipelineState.from_dict(payload.data)
        prompt = state.prompt
        if not isinstance(prompt, dict):
            raise TypeError("prompt missing for thinker request")

        input_ids = prompt.get("input_ids")
        if not hasattr(input_ids, "to"):
            raise TypeError("prompt.input_ids must be a torch.Tensor")
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

        thinker_inputs = state.thinker_inputs or {}
        model_inputs = dict(thinker_inputs.get("model_inputs", {}))
        if not model_inputs:
            model_inputs = {
                key: value
                for key, value in thinker_inputs.items()
                if key != "capture_model_output_keys"
            }
        model_inputs.pop("attention_mask", None)
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
        from sglang_omni_v1.models.ming_omni.io import PipelineState
        from sglang_omni_v1.proto import StagePayload

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
