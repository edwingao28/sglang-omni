# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for Qwen3-TTS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.models.tts_streaming import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload


@dataclass
class _Qwen3TTSStreamState:
    rows: list[torch.Tensor] = field(default_factory=list)
    emitted_frame_count: int = 0
    decoded_audio_chunks: list[np.ndarray] = field(default_factory=list)
    num_codebooks: int | None = None
    sample_rate: int = 24000
    initial_codec_chunk_frames: int = 0
    has_emitted: bool = False


class Qwen3TTSStreamingVocoderScheduler(StreamingSimpleScheduler):
    """Decode Qwen3-TTS codec rows incrementally, with batched final decode."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        stream_chunk_frames: int = 6,
        left_context_frames: int = 6,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
    ) -> None:
        if stream_chunk_frames <= 0:
            raise ValueError("stream_chunk_frames must be > 0")
        if left_context_frames < 0:
            raise ValueError("left_context_frames must be >= 0")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if max_batch_wait_ms < 0:
            raise ValueError("max_batch_wait_ms must be >= 0")

        self._tokenizer = tokenizer
        self._stream_chunk_frames = int(stream_chunk_frames)
        self._left_context_frames = int(left_context_frames)
        self._stream_states: dict[str, _Qwen3TTSStreamState] = {}

        super().__init__(
            self._vocode_payload,
            batch_compute_fn=self._vocode_payloads,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        params = payload.request.params
        if not isinstance(params, dict):
            raise TypeError(
                f"Qwen3-TTS request params must be a dict, got {type(params).__name__}"
            )
        state = Qwen3TTSState.from_dict(payload.data)
        return bool(params.get("stream", False)) and not state.non_streaming_mode

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        state = Qwen3TTSState.from_dict(payload.data)
        stream_state = self._stream_states.setdefault(
            request_id, _Qwen3TTSStreamState()
        )
        stream_state.sample_rate = int(state.sample_rate)
        self._latch_initial_codec_chunk_frames_from_mapping(
            request_id,
            stream_state,
            (
                payload.request.params
                if isinstance(payload.request.params, dict)
                else None
            ),
        )

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        state = self._stream_states.setdefault(request_id, _Qwen3TTSStreamState())
        self._latch_stream_metadata(request_id, state, item.metadata)

        row = item.data
        if not isinstance(row, torch.Tensor):
            raise TypeError(
                f"Qwen3-TTS stream chunk for {request_id!r} must carry a torch.Tensor, "
                f"got {type(row).__name__}"
            )
        row = row.to(dtype=torch.long)
        if row.ndim != 1:
            raise ValueError(
                f"Qwen3-TTS stream chunk must be 1-D [codebooks], "
                f"got {tuple(row.shape)}"
            )

        num_codebooks = self._require_num_codebooks(state, request_id)
        if int(row.shape[0]) != num_codebooks:
            raise ValueError(
                f"Qwen3-TTS stream chunk has {int(row.shape[0])} codebooks, "
                f"expected {num_codebooks}"
            )
        state.rows.append(row)

        output = self._decode_delta(state, is_final=False)
        if output is None:
            return []
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=output,
                metadata={"modality": "audio"},
            )
        ]

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        payload = self._stream_payloads[request_id]
        state = self._stream_states.setdefault(request_id, _Qwen3TTSStreamState())
        output = self._decode_delta(state, is_final=True)

        messages: list[OutgoingMessage] = []
        if output is not None:
            messages.append(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=output,
                    metadata={"modality": "audio"},
                )
            )

        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": int(state.sample_rate),
        }
        usage = self._build_usage(Qwen3TTSState.from_dict(payload.data))
        if usage is not None:
            final_data["usage"] = usage
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=final_data,
                ),
            )
        )
        return messages

    def clear_stream_state(self, request_id: str) -> None:
        self._stream_states.pop(request_id, None)

    def _latch_stream_metadata(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        metadata: dict[str, Any] | None,
    ) -> None:
        if not isinstance(metadata, dict):
            raise RuntimeError(
                f"Qwen3-TTS stream chunk for {request_id!r} is missing metadata"
            )
        if metadata.get("modality") != "audio_codes":
            raise ValueError(
                f"Qwen3-TTS stream chunk modality must be audio_codes, "
                f"got {metadata.get('modality')!r}"
            )
        if metadata.get("stream") is not True:
            raise RuntimeError(
                f"Qwen3-TTS stream chunk for {request_id!r} must include "
                "metadata['stream'] == True"
            )
        if metadata.get("codec") != "qwen3_tts":
            raise ValueError(
                f"Qwen3-TTS stream chunk codec must be qwen3_tts, "
                f"got {metadata.get('codec')!r}"
            )
        if "num_codebooks" not in metadata:
            raise RuntimeError(
                f"Qwen3-TTS stream chunk for {request_id!r} is missing num_codebooks"
            )
        self._latch_num_codebooks(
            request_id,
            state,
            metadata["num_codebooks"],
            source="stream metadata",
        )
        if "sample_rate" in metadata:
            self._latch_sample_rate(
                request_id,
                state,
                metadata["sample_rate"],
                source="stream metadata",
            )
        if INITIAL_CODEC_CHUNK_FRAMES_PARAM in metadata:
            self._latch_initial_codec_chunk_frames_from_mapping(
                request_id,
                state,
                metadata,
            )

    @staticmethod
    def _latch_num_codebooks(
        request_id: str,
        state: _Qwen3TTSStreamState,
        value: Any,
        *,
        source: str,
    ) -> None:
        try:
            num_codebooks = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Qwen3-TTS {source} for {request_id!r} must include integer "
                "num_codebooks"
            ) from exc
        if num_codebooks <= 0:
            raise ValueError(
                f"Qwen3-TTS {source} for {request_id!r} has invalid "
                f"num_codebooks={num_codebooks}"
            )
        if state.num_codebooks is not None and state.num_codebooks != num_codebooks:
            raise ValueError(
                f"Qwen3-TTS stream num_codebooks changed for {request_id!r}: "
                f"{state.num_codebooks} -> {num_codebooks}"
            )
        state.num_codebooks = num_codebooks

    @staticmethod
    def _latch_sample_rate(
        request_id: str,
        state: _Qwen3TTSStreamState,
        value: Any,
        *,
        source: str,
    ) -> None:
        try:
            sample_rate = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Qwen3-TTS {source} for {request_id!r} must include integer "
                "sample_rate"
            ) from exc
        if sample_rate <= 0:
            raise ValueError(
                f"Qwen3-TTS {source} for {request_id!r} has invalid "
                f"sample_rate={sample_rate}"
            )
        if state.sample_rate and state.sample_rate != sample_rate:
            raise ValueError(
                f"Qwen3-TTS stream sample_rate changed for {request_id!r}: "
                f"{state.sample_rate} -> {sample_rate}"
            )
        state.sample_rate = sample_rate

    def _latch_initial_codec_chunk_frames_from_mapping(
        self,
        request_id: str,
        state: _Qwen3TTSStreamState,
        params: Mapping[str, Any] | None,
    ) -> None:
        del request_id
        state.initial_codec_chunk_frames = resolve_initial_codec_chunk_frames(
            params,
            steady_chunk_frames=self._stream_chunk_frames,
        )

    @staticmethod
    def _require_num_codebooks(
        state: _Qwen3TTSStreamState,
        request_id: str,
    ) -> int:
        if state.num_codebooks is None:
            raise RuntimeError(
                f"Qwen3-TTS stream contract for {request_id!r} is missing num_codebooks"
            )
        return state.num_codebooks

    def _decode_delta(
        self,
        state: _Qwen3TTSStreamState,
        *,
        is_final: bool,
    ) -> dict[str, Any] | None:
        total_frames = len(state.rows)
        if total_frames == 0 or total_frames <= state.emitted_frame_count:
            return None

        next_chunk_frames = (
            state.initial_codec_chunk_frames
            if state.initial_codec_chunk_frames > 0 and not state.has_emitted
            else self._stream_chunk_frames
        )
        emit_until = total_frames
        if not is_final:
            target = state.emitted_frame_count + next_chunk_frames
            if total_frames < target:
                return None
            emit_until = target

        window_start = max(0, state.emitted_frame_count - self._left_context_frames)
        rows = state.rows[window_start:emit_until]
        audio = self._decode_rows(rows)
        decoded_frames = emit_until - window_start
        samples_per_frame = max(int(audio.shape[0]) // max(decoded_frames, 1), 1)
        trim_frames = state.emitted_frame_count - window_start
        trim_samples = min(int(trim_frames * samples_per_frame), int(audio.shape[0]))
        delta = np.ascontiguousarray(audio[trim_samples:], dtype=np.float32)
        if delta.size == 0:
            return None

        state.emitted_frame_count = emit_until
        state.has_emitted = True
        state.decoded_audio_chunks.append(delta)
        return audio_waveform_payload(
            delta,
            sample_rate=state.sample_rate,
            modality="audio",
            source_hint="Qwen3-TTS streaming",
        )

    def _decode_rows(self, rows: list[torch.Tensor]) -> np.ndarray:
        codes = torch.stack(rows, dim=0).to(dtype=torch.long)
        wavs, _sample_rate = self._tokenizer.decode([{"audio_codes": codes}])
        wav = wavs[0] if wavs else None
        if wav is None:
            raise RuntimeError("Qwen3-TTS speech tokenizer did not return audio")
        return np.asarray(wav, dtype=np.float32).reshape(-1)

    def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        return self._vocode_payloads([payload])[0]

    def _vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        items = [self._prepare_vocoder_item(payload) for payload in payloads]
        wavs, sample_rate = self._tokenizer.decode(
            [{"audio_codes": codes} for _, codes in items]
        )
        if len(wavs) != len(items):
            raise RuntimeError(
                f"Qwen3-TTS speech tokenizer returned {len(wavs)} audios "
                f"for {len(items)} requests"
            )
        return [
            self._store_vocoder_result(payload, state, codes, wav, sample_rate)
            for payload, (state, codes), wav in zip(payloads, items, wavs)
        ]

    @staticmethod
    def _prepare_vocoder_item(
        payload: StagePayload,
    ) -> tuple[Qwen3TTSState, torch.Tensor]:
        state = Qwen3TTSState.from_dict(payload.data)
        if state.audio_codes is None:
            raise RuntimeError("Qwen3-TTS vocoder requires audio_codes from tts_engine")
        codes = torch.as_tensor(state.audio_codes, dtype=torch.long)
        return state, codes

    @classmethod
    def _store_vocoder_result(
        cls,
        payload: StagePayload,
        state: Qwen3TTSState,
        codes: torch.Tensor,
        wav: Any,
        sample_rate: int,
    ) -> StagePayload:
        if wav is None:
            raise RuntimeError("Qwen3-TTS speech tokenizer did not return audio")

        if state.ref_code_len:
            total_len = int(codes.shape[0])
            cut = int(state.ref_code_len / max(total_len, 1) * wav.shape[0])
            wav = wav[cut:]
        audio_payload = audio_waveform_payload(wav, source_hint="Qwen3-TTS")
        state.audio_samples = None
        state.sample_rate = int(sample_rate)
        state.audio_codes = None

        payload.data = state.to_dict()
        payload.data.update(audio_payload)
        payload.data["sample_rate"] = state.sample_rate
        payload.data["modality"] = "audio"
        usage = cls._build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    @staticmethod
    def _build_usage(state: Qwen3TTSState) -> dict[str, Any] | None:
        if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
            return None
        usage: dict[str, Any] = {
            "prompt_tokens": state.prompt_tokens,
            "completion_tokens": state.completion_tokens,
            "total_tokens": state.prompt_tokens + state.completion_tokens,
        }
        if state.engine_time_s:
            usage["engine_time_s"] = round(float(state.engine_time_s), 6)
        return usage


__all__ = ["Qwen3TTSStreamingVocoderScheduler"]
