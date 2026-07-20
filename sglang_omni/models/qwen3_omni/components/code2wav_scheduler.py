# SPDX-License-Identifier: Apache-2.0
"""Code2Wav scheduler — streaming vocoder with inbox/outbox interface.

Receives codec code chunks via inbox (stream_chunk), accumulates them,
runs vocoder incrementally, outputs final audio via outbox.
"""
from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)


def load_code2wav_model(
    model_path: str, *, device: str = "cuda", dtype: str | None = None
):
    """Load Code2Wav model from HF checkpoint."""
    from transformers import AutoConfig

    from sglang_omni.models.weight_loader import load_module, resolve_dtype

    torch_dtype = resolve_dtype(dtype)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    code2wav_config = config.code2wav_config

    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeCode2Wav,
    )

    model = Qwen3OmniMoeCode2Wav._from_config(code2wav_config)
    model = load_module(
        model,
        model_path,
        prefix="code2wav.",
        dtype=torch_dtype,
        device=device,
        strict=False,
    )
    return model


@dataclass
class Code2WavStreamState:
    chunks: list[torch.Tensor] = field(default_factory=list)
    emitted: int = 0
    audio_parts: list[np.ndarray] = field(default_factory=list)
    stream_enabled: bool | None = None
    due_since: float | None = None


class Code2WavScheduler(StreamingVocoderBase[Code2WavStreamState, "list[int]"]):
    """Streaming vocoder scheduler. Same inbox/outbox interface as OmniScheduler."""

    def __init__(
        self,
        model: Any,
        device: str,
        stream_chunk_size: int = 10,
        left_context_size: int = 25,
        sample_rate: int = 24000,
        codec_eos_token_id: int = 2150,
        enable_batching: bool = False,
        max_batch_wait_ms: int = 0,
        batch_floor: int = 2,
        batch_ceiling: int = 8,
        graph_runner: Any | None = None,
    ):
        self._model = model
        self._device = torch.device(device)
        self._stream_chunk_size = max(int(stream_chunk_size), 1)
        self._left_context_size = max(int(left_context_size), 0)
        self._codec_eos_token_id = codec_eos_token_id
        self._total_upsample = int(model.total_upsample)
        self._graph_runner = graph_runner
        super().__init__(
            None,
            sample_rate=sample_rate,
            stream_source_hint="Qwen3-Omni code2wav",
        )
        # Note (wenyao): batching fields set after super().__init__ — the base
        # scheduler assigns its own _max_batch_wait_s and would clobber ours.
        self._enable_batching = bool(enable_batching)
        self._max_batch_wait_s = max(int(max_batch_wait_ms), 0) / 1000.0
        self._batch_floor = max(int(batch_floor), 1)
        self._batch_ceiling = min(max(int(batch_ceiling), 1), 8)
        self._drain_mode = False
        self._last_fire_reason: str | None = None
        self._can_batch_stream_chunks = self._enable_batching
        if self._enable_batching:
            self._stream_chunk_batch_max = self._batch_ceiling

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        del payload
        return True

    def create_stream_state(self, request_id: str) -> Code2WavStreamState:
        del request_id
        return Code2WavStreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: Code2WavStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        del request_id
        if origin != "stream metadata":
            return
        if state.stream_enabled is None:
            state.stream_enabled = bool(source["stream"])

    def validate_chunk(
        self, request_id: str, state: Code2WavStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        del request_id, state
        return codes.to(device=self._device, dtype=torch.long)

    def ingest(
        self, request_id: str, state: Code2WavStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        if codes.ndim >= 1 and codes[0].item() == self._codec_eos_token_id:
            return
        state.chunks.append(codes)

    def should_decode(self, state: Code2WavStreamState, *, is_final: bool) -> bool:
        del is_final
        return self._ready(state) >= self._stream_chunk_size

    def decode_delta(
        self, request_id: str, state: Code2WavStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        start, end = state.emitted, len(state.chunks)
        if start >= end:
            return None
        context = min(self._left_context_size, start)
        window = torch.stack(state.chunks[start - context : end], dim=0)
        codes = window.transpose(0, 1).unsqueeze(0)
        wav, _meta = self._forward_codes(codes, graph_eligible=not is_final)
        trim = context * self._total_upsample
        if trim:
            wav = wav[..., trim:]
        audio = wav.reshape(-1).detach().cpu().float().numpy().copy()
        state.emitted = end
        if audio.size == 0:
            return None
        if not state.audio_parts:
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="code2wav_first_audio",
                metadata={"samples": int(audio.shape[0])},
            )
        state.audio_parts.append(audio)
        if not state.stream_enabled:
            return None
        return torch.from_numpy(audio)

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: Code2WavStreamState
    ) -> dict[str, Any]:
        del payload
        if not state.audio_parts:
            raise RuntimeError(f"code2wav produced no audio for {request_id!r}")
        if state.stream_enabled:
            return {"modality": "audio", "sample_rate": self._sample_rate}
        full = np.concatenate(state.audio_parts).astype(np.float32, copy=False)
        return audio_waveform_payload(
            full,
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="Qwen3-Omni code2wav",
        )

    def _forward_codes(
        self, codes: torch.Tensor, *, graph_eligible: bool
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del graph_eligible
        with torch.no_grad():
            if self._device.type == "cuda":
                torch.cuda.set_device(self._device)
            wav = self._model(codes)
        return wav, {
            "execution_mode": "eager",
            "graph_key": None,
            "fallback_reason": None,
        }

    def _batch_deadline(self) -> float | None:
        return None

    def _collect_stream_chunk_batch(self, first_msg):
        batch = super()._collect_stream_chunk_batch(first_msg)
        deadline = self._batch_deadline()
        if deadline is None:
            return batch
        cap = self._stream_chunk_batch_max or max(self._max_batch_size, 1)
        while len(batch) < cap:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                msg = self.inbox.get(timeout=remaining)
            except queue.Empty:
                break
            if msg.type != "stream_chunk":
                self._pending_messages.appendleft(msg)
                break
            if self._is_aborted(msg.request_id):
                continue
            batch.append(msg)
        return batch

    def _ready(self, state: Code2WavStreamState) -> int:
        return len(state.chunks) - state.emitted

    def _bucket(self, state: Code2WavStreamState) -> tuple[int, int]:
        context = min(self._left_context_size, state.emitted)
        return (context, context + self._ready(state))


def create_code2wav_scheduler(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    gpu_id: int | None = None,
    stream_chunk_size: int = 10,
    left_context_size: int = 25,
):
    """Factory: returns Code2WavScheduler."""
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    model = load_code2wav_model(model_path, device=device, dtype=dtype)
    return Code2WavScheduler(
        model,
        device=device,
        stream_chunk_size=stream_chunk_size,
        left_context_size=left_context_size,
    )
