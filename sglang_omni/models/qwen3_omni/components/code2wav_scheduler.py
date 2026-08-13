# SPDX-License-Identifier: Apache-2.0
"""Code2Wav scheduler — streaming vocoder with inbox/outbox interface.

Receives codec code chunks via inbox (stream_chunk), accumulates them,
runs vocoder incrementally, outputs final audio via outbox.
"""
from __future__ import annotations

import json
import logging
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    Code2WavCudaGraphRunner,
    Code2WavRunResult,
    GraphKey,
)
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.profiler.event_recorder import get_recorder as _get_event_recorder
from sglang_omni.profiler.event_recorder import get_recorder as _get_recorder
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)


def _serial_threshold_graph_keys(
    stream_chunk_size: int,
    left_context_size: int,
    initial_chunk_size: int = 0,
    batch_sizes: tuple[int, ...] = (1,),
) -> tuple[GraphKey, ...]:
    frames: list[int] = []
    start = 0
    new_frames = initial_chunk_size or stream_chunk_size
    while True:
        frames.append(min(left_context_size, start) + new_frames)
        if start >= left_context_size and new_frames == stream_chunk_size:
            break
        start += new_frames
        new_frames = stream_chunk_size
    return tuple(
        GraphKey(batch_size=batch_size, frames=window_frames)
        for batch_size in batch_sizes
        for window_frames in frames
    )


def _step_plan_batch_sizes(batch_ceiling: int) -> tuple[int, ...]:
    # _decompose_batch only ever emits these sub-batch sizes.
    return tuple(size for size in (1, 2, 4, 8) if size <= batch_ceiling)


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
    return model.eval()


@dataclass
class Code2WavStreamState:
    chunks: list[torch.Tensor] = field(default_factory=list)
    emitted: int = 0
    initial_chunk_frames: int = 0
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
        initial_chunk_size: int = 0,
        left_context_size: int = 25,
        sample_rate: int = 24000,
        codec_eos_token_id: int = 2150,
        enable_batching: bool = False,
        max_batch_wait_ms: int = 0,
        batch_floor: int = 2,
        batch_ceiling: int = 8,
        enable_cuda_graph: bool = False,
        _cuda_graph_runner: Code2WavCudaGraphRunner | None = None,
    ):
        self._model = model
        self._device = torch.device(device)
        self._stream_chunk_size = max(int(stream_chunk_size), 1)
        self._default_initial_chunk_size = max(
            0, min(int(initial_chunk_size), self._stream_chunk_size)
        )
        self._left_context_size = max(int(left_context_size), 0)
        self._codec_eos_token_id = codec_eos_token_id
        self._total_upsample = int(model.total_upsample)
        self._cuda_graph_runner = (
            _cuda_graph_runner if bool(enable_cuda_graph) else None
        )
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
        self._last_oldest_wait_ms: float = 0.0
        self._last_due_bucket_count: int = 0
        self._can_batch_stream_chunks = self._enable_batching
        if self._enable_batching:
            self._stream_chunk_batch_max = self._batch_ceiling

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        del payload
        return True

    def create_stream_state(self, request_id: str) -> Code2WavStreamState:
        del request_id
        return Code2WavStreamState(
            initial_chunk_frames=self._default_initial_chunk_size
        )

    def latch_stream_contract(
        self,
        request_id: str,
        state: Code2WavStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        del request_id
        # Note (wenyao): the payload always arrives after every codec frame and
        # after stream_done on this stage, so a per-request first-chunk size
        # could never reach the threshold that consumed it.
        if origin != "stream metadata":
            return
        if state.stream_enabled is None:
            state.stream_enabled = bool(source["stream"])

    def _threshold(self, state: Code2WavStreamState) -> int:
        if state.emitted == 0 and state.initial_chunk_frames > 0:
            return state.initial_chunk_frames
        return self._stream_chunk_size

    def validate_chunk(
        self, request_id: str, state: Code2WavStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        del request_id, state
        assert codes.ndim in (1, 2), f"expected 1-D or 2-D codes, got {codes.ndim}-D"
        return codes.to(device=self._device, dtype=torch.long)

    def ingest(
        self, request_id: str, state: Code2WavStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        if codes.ndim != 2:
            if codes.ndim >= 1 and codes[0].item() == self._codec_eos_token_id:
                return
            state.chunks.append(codes)
            return
        # Multi-frame message: one host sync for the EOS scan, not one per
        # row. Rows before an in-message EOS are kept; the EOS row and
        # anything after it are dropped. Unpacking into per-frame entries
        # keeps every downstream consumer (thresholds, windows, graph keys)
        # untouched.
        eos_positions = (codes[:, 0] == self._codec_eos_token_id).nonzero()
        n_keep = (
            int(eos_positions[0].item()) if eos_positions.numel() else codes.shape[0]
        )
        if n_keep > 0:
            state.chunks.extend(codes[:n_keep].unbind(0))

    def should_decode(self, state: Code2WavStreamState, *, is_final: bool) -> bool:
        del is_final
        return self._ready(state) >= self._threshold(state)

    def decode_delta(
        self, request_id: str, state: Code2WavStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        start, end = state.emitted, len(state.chunks)
        if start >= end:
            return None
        context = min(self._left_context_size, start)
        profile_metadata: dict[str, Any] | None = None
        if _get_event_recorder().is_active():
            profile_metadata = {
                "trigger": "stream_done" if is_final else "threshold",
                "start_frame": start,
                "end_frame": end,
                "new_frames": end - start,
                "context_frames": context,
                "window_frames": end - start + context,
                "active_request_count": len(self._stream_states),
                "threshold_ready_request_count": sum(
                    self._ready(ready_state) >= self._threshold(ready_state)
                    for _, ready_state in self._stream_state_items()
                ),
                "inbox_depth": self.inbox.qsize(),
                "pending_message_depth": len(self._pending_messages),
            }
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="code2wav_decode_start",
                metadata=profile_metadata,
            )
        window = torch.stack(state.chunks[start - context : end], dim=0)
        codes = window.transpose(0, 1).unsqueeze(0)
        wav, execution_metadata = self._forward_codes(
            codes,
            graph_eligible=not is_final,
        )
        wav = wav[..., -(end - start) * self._total_upsample :]
        audio = wav.reshape(-1).detach().cpu().float().numpy().copy()
        if profile_metadata is not None:
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="code2wav_decode_end",
                metadata={
                    **profile_metadata,
                    "audio_samples": int(audio.shape[0]),
                    **execution_metadata,
                },
            )
        state.emitted = end
        state.due_since = None
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
        self,
        codes: torch.Tensor,
        *,
        graph_eligible: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        with torch.no_grad():
            if self._device.type == "cuda":
                torch.cuda.set_device(self._device)
            if self._cuda_graph_runner is None:
                result = Code2WavRunResult(
                    output=self._model(codes),
                    execution_mode="eager",
                    key=None,
                    fallback_reason=None,
                )
            else:
                result = self._cuda_graph_runner.run(
                    codes,
                    eligible=graph_eligible,
                )

        graph_key = None
        if result.key is not None:
            graph_key = {
                "batch_size": int(result.key.batch_size),
                "frames": int(result.key.frames),
            }
        return result.output, {
            "execution_mode": str(result.execution_mode),
            "graph_key": graph_key,
            "fallback_reason": (
                None if result.fallback_reason is None else str(result.fallback_reason)
            ),
        }

    def _batch_deadline(self) -> float | None:
        with self._state_lock:
            due = [
                state.due_since
                for _, state in self._stream_state_items()
                if state.due_since is not None
            ]
        if not due:
            return None
        return min(due) + self._max_batch_wait_s

    def _drain_inbox(self):
        while True:
            try:
                yield self.inbox.get_nowait()
            except queue.Empty:
                return

    def _next_message(self):
        # Drain already-queued stream chunks before handling any other message:
        # at a wave boundary ~2x batch_ceiling stream_done finals hit the inbox
        # together and each runs a serial eager final decode, so a newly
        # arrived first chunk otherwise waits behind 13-17 of them (measured
        # ingest p95 ~780ms at c32). Ingesting queued chunks first is
        # content-invariant: every request's decode window is pinned by its
        # threshold, so only cross-request batch composition changes.
        if self._can_batch_stream_chunks:
            first_chunks: list = []
            for msg in self._drain_inbox():
                if (
                    msg.type == "stream_chunk"
                    and msg.request_id not in self._stream_states
                    and not self._is_aborted(msg.request_id)
                ):
                    # Only a request's FIRST chunk jumps the queue — that is
                    # the TTFA-critical message. Steady chunks and finals keep
                    # FIFO order so final decodes are not deferred (deferring
                    # them measurably lengthens e2e and drops closed-loop rps).
                    first_chunks.append(msg)
                else:
                    self._pending_messages.append(msg)
            if first_chunks:
                self._handle_stream_chunk_batch(first_chunks)
        if self._pending_messages:
            return self._pending_messages.popleft()
        deadline = self._batch_deadline()
        timeout = 0.1
        if deadline is not None:
            timeout = min(timeout, max(deadline - time.monotonic(), 0.0))
        try:
            return self.inbox.get(timeout=timeout)
        except queue.Empty:
            if deadline is not None and time.monotonic() >= deadline:
                self._pump_due_streams()
            return None

    def _pump_due_streams(self) -> None:
        with self._state_lock:
            failed = self._pump_streams()
        for request_id in failed:
            self._cleanup_aborted_request(request_id)

    def _ready(self, state: Code2WavStreamState) -> int:
        return len(state.chunks) - state.emitted

    def _bucket(self, state: Code2WavStreamState) -> tuple[int, int]:
        # Cap each threshold step at one stream chunk of new frames so the
        # window stays on the captured CUDA graph grid even when a stream has
        # accumulated a backlog; the surplus stays pending for the next step.
        context = min(self._left_context_size, state.emitted)
        ready = min(self._ready(state), self._threshold(state))
        return (context, context + ready)

    @staticmethod
    def _decompose_batch(n: int) -> list[int]:
        plan: list[int] = []
        for size in (8, 4, 2, 1):
            while n >= size:
                plan.append(size)
                n -= size
        return plan

    def stop(self) -> None:
        self._drain_mode = True
        super().stop()

    def on_stream_done(self, request_id: str):
        state = self._stream_states.get(request_id)
        prev_drain = self._drain_mode
        if state is not None and state.due_since is not None:
            self._drain_mode = True
        try:
            return super().on_stream_done(request_id)
        finally:
            self._drain_mode = prev_drain

    def select_step_participants(self) -> list[tuple[str, Code2WavStreamState]]:
        now = time.monotonic()
        first_ready: list[tuple[str, Code2WavStreamState]] = []
        due: dict[tuple[int, int], list[tuple[str, Code2WavStreamState]]] = {}
        for rid, state in self._stream_state_items():
            ready = self._ready(state)
            if state.emitted == 0 and ready >= self._threshold(state):
                first_ready.append((rid, state))
                continue
            if state.emitted > 0 and ready >= self._stream_chunk_size:
                if state.due_since is None:
                    state.due_since = now
                due.setdefault(self._bucket(state), []).append((rid, state))
        if first_ready:
            # Note (wenyao): same bucket ⇒ one trim scalar holds for the batch.
            key = self._bucket(first_ready[0][1])
            same_bucket = [p for p in first_ready if self._bucket(p[1]) == key]
            self._last_fire_reason = "first"
            self._last_oldest_wait_ms = 0.0
            self._last_due_bucket_count = len(due)
            return same_bucket[: self._batch_ceiling]
        if not due:
            return []
        anchor_key = min(due, key=lambda k: min(s.due_since for _, s in due[k]))
        anchor = sorted(due[anchor_key], key=lambda p: p[1].due_since)
        oldest_wait = now - anchor[0][1].due_since
        fire = (
            len(anchor) >= self._batch_floor
            or oldest_wait >= self._max_batch_wait_s
            or self._drain_mode
        )
        if not fire:
            return []
        if len(anchor) >= self._batch_floor:
            reason = "floor"
        elif oldest_wait >= self._max_batch_wait_s:
            reason = "deadline"
        else:
            reason = "drain"
        self._last_fire_reason = reason
        self._last_oldest_wait_ms = oldest_wait * 1000.0
        self._last_due_bucket_count = len(due)
        return anchor[: self._batch_ceiling]

    def build_step_plan(
        self, participants: list[tuple[str, Code2WavStreamState]]
    ) -> list[int]:
        if self._cuda_graph_runner is None:
            return [len(participants)]
        return self._decompose_batch(len(participants))

    def run_step(
        self,
        participants: list[tuple[str, Code2WavStreamState]],
        plan: list[int],
    ) -> dict[str, torch.Tensor]:
        decoded: dict[str, torch.Tensor] = {}
        profile_metadata: dict[str, Any] | None = None
        if _get_recorder().is_active():
            first_state = participants[0][1]
            bucket = self._bucket(first_state)
            profile_metadata = {
                "batch_size": len(participants),
                "bucket": list(bucket),
                "new_frames": self._ready(first_state),
                "window_frames": bucket[1],
                "active_request_count": len(self._stream_states),
                "inbox_depth": self.inbox.qsize(),
                "oldest_wait_ms": self._last_oldest_wait_ms,
                "fire_reason": self._last_fire_reason,
                "due_bucket_count": self._last_due_bucket_count,
                "subbatch_decomposition": list(plan),
            }
            _emit_event(
                request_id=participants[0][0],
                stage=None,
                event_name="code2wav_batch_start",
                metadata=profile_metadata,
            )
        execution_metadata = {
            "execution_mode": "eager",
            "graph_key": None,
            "fallback_reason": None,
        }
        audio_samples = 0
        cursor = 0
        for sub in plan:
            group = participants[cursor : cursor + sub]
            cursor += sub
            rows = []
            window_ends: list[int] = []
            for _, state in group:
                start = state.emitted
                end = min(len(state.chunks), start + self._threshold(state))
                window_ends.append(end)
                context = min(self._left_context_size, start)
                rows.append(
                    torch.stack(state.chunks[start - context : end], dim=0).transpose(
                        0, 1
                    )
                )
            window_frames = rows[0].shape[-1]
            for row in rows[1:]:
                if row.shape[-1] != window_frames:
                    raise RuntimeError(
                        f"code2wav bucket mismatch: window {row.shape[-1]} vs "
                        f"{window_frames}"
                    )
            codes = torch.stack(rows, dim=0)
            wav, execution_metadata = self._forward_codes(
                codes,
                graph_eligible=True,
            )
            if wav.shape[0] != len(group):
                raise RuntimeError(
                    f"code2wav step returned {wav.shape[0]} rows for "
                    f"{len(group)} requests"
                )
            context = min(self._left_context_size, group[0][1].emitted)
            wav = wav[..., -(window_frames - context) * self._total_upsample :]
            host = wav.detach().cpu().float()
            for i, (rid, state) in enumerate(group):
                audio = host[i].reshape(-1).numpy().copy()
                state.emitted = window_ends[i]
                state.due_since = None
                if audio.size == 0:
                    continue
                if profile_metadata is not None:
                    audio_samples += int(audio.size)
                if not state.audio_parts:
                    _emit_event(
                        request_id=rid,
                        stage=None,
                        event_name="code2wav_first_audio",
                        metadata={"samples": int(audio.shape[0])},
                    )
                state.audio_parts.append(audio)
                if state.stream_enabled:
                    decoded[rid] = torch.from_numpy(audio)
        if profile_metadata is not None:
            _emit_event(
                request_id=participants[0][0],
                stage=None,
                event_name="code2wav_batch_end",
                metadata={
                    **profile_metadata,
                    "audio_samples": audio_samples,
                    **execution_metadata,
                },
            )
        return decoded


def create_code2wav_scheduler(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    gpu_id: int | None = None,
    stream_chunk_size: int = 10,
    initial_chunk_size: int = 0,
    left_context_size: int = 25,
    enable_batching: bool = False,
    max_batch_wait_ms: int = 0,
    batch_floor: int = 2,
    batch_ceiling: int = 8,
    enable_cuda_graph: bool = False,
    total_gpu_memory_fraction: float | None = None,
):
    """Factory: returns Code2WavScheduler."""
    if enable_cuda_graph and total_gpu_memory_fraction is None:
        raise ValueError(
            "Code2Wav CUDA graph requires "
            "runtime.resources.total_gpu_memory_fraction"
        )
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    concrete_device = torch.device(device)
    if concrete_device.type == "cuda" and concrete_device.index is None:
        concrete_device = torch.device("cuda", torch.cuda.current_device())
    device = str(concrete_device)
    stream_chunk_size = max(int(stream_chunk_size), 1)
    initial_chunk_size = max(0, min(int(initial_chunk_size), stream_chunk_size))
    left_context_size = max(int(left_context_size), 0)
    model = load_code2wav_model(model_path, device=device, dtype=dtype)
    cuda_graph_runner = None
    if enable_cuda_graph:
        batch_sizes = (
            _step_plan_batch_sizes(min(max(int(batch_ceiling), 1), 8))
            if enable_batching
            else (1,)
        )
        cuda_graph_runner = Code2WavCudaGraphRunner.build(
            model,
            device=concrete_device,
            num_quantizers=int(model.config.num_quantizers),
            total_gpu_memory_fraction=total_gpu_memory_fraction,
            graph_keys=_serial_threshold_graph_keys(
                stream_chunk_size,
                left_context_size,
                initial_chunk_size,
                batch_sizes=batch_sizes,
            ),
        )
        logger.info(
            "Code2Wav CUDA graph startup stats=%s",
            json.dumps(
                cuda_graph_runner.stats(),
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    return Code2WavScheduler(
        model,
        device=device,
        stream_chunk_size=stream_chunk_size,
        initial_chunk_size=initial_chunk_size,
        left_context_size=left_context_size,
        enable_batching=enable_batching,
        max_batch_wait_ms=max_batch_wait_ms,
        batch_floor=batch_floor,
        batch_ceiling=batch_ceiling,
        enable_cuda_graph=enable_cuda_graph,
        _cuda_graph_runner=cuda_graph_runner,
    )
