# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR pipeline state carried from the asr_frontend stage to the engine stage."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class Qwen3ASRPreparedAudio(DeclarativeStateBase):
    audio_fingerprint: str = ""
    audio_duration_s: float = wire(0.0, codec="float")
    num_samples: int = wire(0, codec="int")
    mel_features: torch.Tensor | None = wire(None, codec="tensor_cpu")
    feature_attention_mask: torch.Tensor | None = wire(None, codec="tensor_cpu")


__all__ = ["Qwen3ASRPreparedAudio"]
