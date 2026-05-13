# SPDX-License-Identifier: Apache-2.0
"""Lazy registration helpers."""

from __future__ import annotations

_ming_hf_config_registered = False


def register_ming_hf_config() -> None:
    """Register Ming's composite HF config before SGLang loads ModelConfig."""
    global _ming_hf_config_registered
    if _ming_hf_config_registered:
        return

    from transformers import AutoConfig

    from sglang_omni_v1.models.ming_omni.thinker import BailingMM2Config

    try:
        AutoConfig.register("bailingmm_moe_v2_lite", BailingMM2Config)
    except ValueError:
        pass
    _ming_hf_config_registered = True


def register_ming_model_registry() -> None:
    from sglang.srt.models.registry import ModelRegistry

    from sglang_omni_v1.models.ming_omni.thinker import BailingMoeV2ForCausalLM

    ModelRegistry.models["BailingMoeV2ForCausalLM"] = BailingMoeV2ForCausalLM
