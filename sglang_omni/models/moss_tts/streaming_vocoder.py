# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for MOSS-TTS Delay."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import torch

from sglang_omni.models.moss_tts.codec import apply_de_delay_pattern
from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.models.moss_tts.request_builders import _resolve_audio_payload_bounds
from sglang_omni.models.tts_streaming import (
    INITIAL_CODEC_CHUNK_FRAMES_PARAM,
    build_tts_usage,
    resolve_initial_codec_chunk_frames,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

MOSS_STREAM_MODALITY = "moss_delayed_audio_row"

_CONTRACT_FIELDS = (
    "n_vq",
    "audio_pad_code",
    "audio_start_token_id",
    "audio_end_token_id",
    "audio_assistant_gen_slot_token_id",
    "audio_assistant_delay_slot_token_id",
)


@dataclass
class _MossSegmentState:
    start_frame: int
    end_frame: int
    emitted_frames: int = 0
    closed: bool = False
    flushed: bool = False


@dataclass
class _MossStreamState:
    rows: list[torch.Tensor] = field(default_factory=list)
    text_tokens: list[int] = field(default_factory=list)
    scanned_rows: int = 0
    start_row: int | None = None
    end_row: int | None = None
    first_audio_start_row: int | None = None
    first_gen_slot_row: int | None = None
    last_slot_row: int = -1
    frame_values: list[torch.Tensor] = field(default_factory=list)
    segments: list[_MossSegmentState] = field(default_factory=list)
    next_decode_rows: int = 0
    emitted_raw_frames: int = 0
    has_emitted: bool = False
    initial_codec_chunk_frames: int = 0
    contract: dict[str, int] | None = None
    assistant_start_length: int = 0
    prefix_attached: bool = False


class MossStreamingVocoderScheduler(StreamingSimpleScheduler):
    """Decode MOSS delayed rows incrementally; non-streaming uses the stage closures."""

    def __init__(
        self,
        processor: Any,
        *,
        compute_fn: Callable[[StagePayload], StagePayload],
        batch_compute_fn: (
            Callable[[list[StagePayload]], list[StagePayload]] | None
        ) = None,
        device: str = "cpu",
        stream_stride: int = 40,
        stream_followup_stride: int = 25,
        stream_overlap_frames: int = 16,
        stream_holdback_frames: int = 4,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
    ) -> None:
        if stream_stride <= 0 or stream_followup_stride <= 0:
            raise ValueError("stream_stride and stream_followup_stride must be > 0")
        if stream_overlap_frames < 0:
            raise ValueError("stream_overlap_frames must be >= 0")
        if stream_holdback_frames < 0:
            raise ValueError("stream_holdback_frames must be >= 0")

        self._processor = processor
        self._device = device
        self._stream_stride = int(stream_stride)
        self._stream_followup_stride = int(stream_followup_stride)
        self._stream_overlap_frames = int(stream_overlap_frames)
        self._stream_holdback_frames = int(stream_holdback_frames)
        self._sample_rate = self._resolve_sample_rate(processor)
        self._samples_per_frame = self._resolve_samples_per_frame(processor)
        self._stream_states: dict[str, _MossStreamState] = {}

        super().__init__(
            compute_fn,
            batch_compute_fn=batch_compute_fn,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
        )

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        params = payload.request.params
        if not isinstance(params, dict):
            raise TypeError(
                f"MOSS-TTS request params must be a dict, got {type(params).__name__}"
            )
        return bool(params.get("stream", False))

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        state = self._stream_states.setdefault(request_id, _MossStreamState())
        if not isinstance(payload.data, dict):
            raise TypeError(
                f"MOSS-TTS streaming payload for {request_id!r} must be a dict, "
                f"got {type(payload.data).__name__}"
            )
        params = payload.request.params
        if state.contract is not None and isinstance(params, dict):
            self._latch_initial_codec_chunk_frames(state, params)

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        state = self._stream_states.setdefault(request_id, _MossStreamState())
        self._latch_stream_metadata(request_id, state, item.metadata)

        row = item.data
        if not isinstance(row, torch.Tensor):
            raise TypeError(
                f"MOSS-TTS stream chunk for {request_id!r} must carry a torch.Tensor, "
                f"got {type(row).__name__}"
            )
        row = row.detach().to(dtype=torch.long, device="cpu")
        if row.ndim != 1:
            raise ValueError(
                f"MOSS-TTS stream chunk must be 1-D [n_vq + 1], got {tuple(row.shape)}"
            )
        n_vq = state.contract["n_vq"]
        if int(row.shape[0]) != n_vq + 1:
            raise ValueError(
                f"MOSS-TTS stream chunk has {int(row.shape[0])} channels, "
                f"expected {n_vq + 1}"
            )
        state.rows.append(row)
        state.text_tokens.append(int(row[0].item()))
        self._advance_scan(state)

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
        state = self._stream_states.setdefault(request_id, _MossStreamState())
        tts_state = MossTTSState.from_dict(payload.data)

        messages: list[OutgoingMessage] = []
        if not state.has_emitted:
            # (wenyao) Nothing streamed: the full non-streaming closure is bit-identical
            # to today's vocoder output (and raises identically on no audio).
            result = self._fn(payload)
            chunk: dict[str, Any] = {}
            for key in (
                "audio_waveform",
                "audio_waveform_shape",
                "audio_waveform_dtype",
                "sample_rate",
                "modality",
            ):
                if key in result.data:
                    chunk[key] = result.data[key]
            messages.append(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=chunk,
                    metadata={"modality": "audio"},
                )
            )
        else:
            self._verify_final_bounds(request_id, state)
            output = self._decode_delta(state, is_final=True)
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
            "sample_rate": self._sample_rate,
        }
        usage = build_tts_usage(
            tts_state.prompt_tokens,
            tts_state.completion_tokens,
            tts_state.engine_time_s,
        )
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
        state: _MossStreamState,
        metadata: dict[str, Any] | None,
    ) -> None:
        if not isinstance(metadata, dict):
            if state.contract is None:
                raise RuntimeError(
                    f"MOSS-TTS stream chunk for {request_id!r} is missing its "
                    "decode metadata"
                )
            return
        if metadata.get("modality") not in (None, MOSS_STREAM_MODALITY):
            raise ValueError(
                f"MOSS-TTS stream chunk modality must be {MOSS_STREAM_MODALITY}, "
                f"got {metadata.get('modality')!r}"
            )
        if metadata.get("stream") is not True:
            raise RuntimeError(
                f"MOSS-TTS stream chunk for {request_id!r} must include "
                "metadata['stream'] == True"
            )
        missing = [key for key in _CONTRACT_FIELDS if key not in metadata]
        if missing:
            if state.contract is None:
                raise RuntimeError(
                    f"MOSS-TTS stream chunk for {request_id!r} is missing metadata "
                    f"fields: {', '.join(missing)}"
                )
            for key in _CONTRACT_FIELDS:
                if key in metadata and int(metadata[key]) != state.contract[key]:
                    raise ValueError(
                        f"MOSS-TTS stream contract changed for {request_id!r}: "
                        f"{key}={state.contract[key]} -> {metadata[key]}"
                    )
        else:
            try:
                contract = {key: int(metadata[key]) for key in _CONTRACT_FIELDS}
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"MOSS-TTS stream metadata for {request_id!r} must carry "
                    "integer contract fields"
                ) from exc
            if contract["n_vq"] <= 0 or contract["audio_pad_code"] <= 0:
                raise ValueError(
                    f"MOSS-TTS stream metadata for {request_id!r} has invalid "
                    f"n_vq={contract['n_vq']}, "
                    f"audio_pad_code={contract['audio_pad_code']}"
                )
            if state.contract is not None and state.contract != contract:
                raise ValueError(
                    f"MOSS-TTS stream contract changed for {request_id!r}: "
                    f"{state.contract} -> {contract}"
                )
            state.contract = contract
        if "assistant_start_length" in metadata:
            state.assistant_start_length = int(metadata["assistant_start_length"])
        self._attach_prefix_rows(request_id, state, metadata)
        if INITIAL_CODEC_CHUNK_FRAMES_PARAM in metadata:
            self._latch_initial_codec_chunk_frames(state, metadata)

    def _attach_prefix_rows(
        self,
        request_id: str,
        state: _MossStreamState,
        metadata: dict[str, Any],
    ) -> None:
        prefix = metadata.get("assistant_prefix_rows")
        if prefix is None or state.prefix_attached:
            return
        if state.rows:
            raise ValueError(
                f"MOSS-TTS assistant_prefix_rows for {request_id!r} arrived after "
                "streamed rows"
            )
        n_vq = state.contract["n_vq"]
        for row in prefix:
            tensor = torch.as_tensor(row, dtype=torch.long)
            if tensor.ndim != 1 or int(tensor.shape[0]) != n_vq + 1:
                raise ValueError(
                    f"MOSS-TTS assistant_prefix_rows for {request_id!r} must be "
                    f"[n_vq + 1] rows, got {tuple(tensor.shape)}"
                )
            state.rows.append(tensor)
            state.text_tokens.append(int(tensor[0].item()))
        state.prefix_attached = True

    def _latch_initial_codec_chunk_frames(
        self,
        state: _MossStreamState,
        mapping: Mapping[str, Any],
    ) -> None:
        n_vq = state.contract["n_vq"]
        steady_codec_frames = max(1, self._stream_stride - n_vq + 1)
        state.initial_codec_chunk_frames = resolve_initial_codec_chunk_frames(
            mapping,
            steady_chunk_frames=steady_codec_frames,
        )

    def _advance_scan(self, state: _MossStreamState) -> None:
        contract = state.contract
        audio_start = contract["audio_start_token_id"]
        audio_end = contract["audio_end_token_id"]
        gen_slot = contract["audio_assistant_gen_slot_token_id"]
        delay_slot = contract["audio_assistant_delay_slot_token_id"]
        total = len(state.rows)
        index = state.scanned_rows
        while index < total:
            token = state.text_tokens[index]
            if state.first_audio_start_row is None and token == audio_start:
                state.first_audio_start_row = index
                if state.start_row is not None:
                    self._relatch_start_row(state, index)
            if state.first_gen_slot_row is None and token == gen_slot:
                state.first_gen_slot_row = index
            if token == gen_slot or token == delay_slot:
                state.last_slot_row = index
            if state.start_row is None:
                if state.first_audio_start_row is not None:
                    state.start_row = state.first_audio_start_row + 1
                elif state.first_gen_slot_row is not None:
                    state.start_row = state.first_gen_slot_row
            if (
                state.start_row is not None
                and state.end_row is None
                and index >= state.start_row
                and token == audio_end
            ):
                state.end_row = index
            index += 1
        state.scanned_rows = total
        self._materialize_frames(state)

    @staticmethod
    def _relatch_start_row(state: _MossStreamState, audio_start_row: int) -> None:
        # (wenyao) EOS bounds anchor at the first audio_start regardless of position
        # (_resolve_audio_payload_bounds); a start latched via the gen-slot fallback
        # must re-anchor before any audio is emitted.
        if state.has_emitted:
            raise RuntimeError(
                f"MOSS-TTS stream saw audio_start at row {audio_start_row} "
                f"after emitting audio anchored at gen-slot row "
                f"{state.start_row}; streamed audio would diverge from the "
                "non-streaming waveform"
            )
        state.start_row = audio_start_row + 1
        state.end_row = None
        state.frame_values.clear()
        state.segments.clear()
        state.next_decode_rows = 0
        state.emitted_raw_frames = 0

    @staticmethod
    def _payload_view_end(state: _MossStreamState) -> int:
        total = len(state.rows)
        return min(total, state.end_row) if state.end_row is not None else total

    def _materialize_frames(self, state: _MossStreamState) -> None:
        if state.start_row is None:
            return
        n_vq = state.contract["n_vq"]
        payload_len = self._payload_view_end(state) - state.start_row
        avail = payload_len - n_vq + 1 if payload_len >= n_vq else 0
        known = len(state.frame_values)
        if avail <= known:
            return

        window_start = state.start_row + known
        window_end = state.start_row + (avail - 1) + n_vq
        window_rows = torch.stack(state.rows[window_start:window_end], dim=0)[:, 1:]
        new_frames = apply_de_delay_pattern(window_rows)

        pad_code = state.contract["audio_pad_code"]
        for offset in range(int(new_frames.shape[0])):
            frame = new_frames[offset]
            frame_index = known + offset
            is_pad = bool((frame == pad_code).all())
            is_complete = bool(((frame >= 0) & (frame < pad_code)).all())
            state.frame_values.append(frame)
            if (not is_pad) and is_complete:
                last = state.segments[-1] if state.segments else None
                if (
                    last is not None
                    and not last.closed
                    and last.end_frame == frame_index
                ):
                    last.end_frame = frame_index + 1
                else:
                    state.segments.append(
                        _MossSegmentState(
                            start_frame=frame_index, end_frame=frame_index + 1
                        )
                    )
            elif state.segments and not state.segments[-1].closed:
                state.segments[-1].closed = True

    def _emission_resolves_bounds(self, state: _MossStreamState) -> bool:
        # (wenyao) Intermediate frames may only flow when the EOS pass is provably going
        # to use the same payload view: either bounds already resolve over the received
        # rows, or the request has no assistant prefix, in which case the bounds-None
        # fallback (full matrix, trim 0) decodes the exact same member frames.
        n_vq = state.contract["n_vq"]
        if state.end_row is not None:
            return state.end_row - state.start_row > n_vq
        if state.last_slot_row >= state.start_row + n_vq:
            return True
        return state.assistant_start_length == 0 and not state.prefix_attached

    def _verify_final_bounds(self, request_id: str, state: _MossStreamState) -> None:
        if not state.rows or state.start_row is None:
            return
        rows = torch.stack(state.rows, dim=0)
        cfg = SimpleNamespace(
            audio_start_token_id=state.contract["audio_start_token_id"],
            audio_end_token_id=state.contract["audio_end_token_id"],
            audio_assistant_gen_slot_token_id=state.contract[
                "audio_assistant_gen_slot_token_id"
            ],
            audio_assistant_delay_slot_token_id=state.contract[
                "audio_assistant_delay_slot_token_id"
            ],
        )
        bounds = _resolve_audio_payload_bounds(rows, cfg)
        online_end = (
            state.end_row if state.end_row is not None else state.last_slot_row + 1
        )
        if bounds is None:
            if state.assistant_start_length or state.prefix_attached:
                logger.warning(
                    "MOSS-TTS streaming bounds for %s did not resolve at EOS "
                    "after intermediate emission",
                    request_id,
                )
        elif bounds != (state.start_row, online_end):
            logger.warning(
                "MOSS-TTS streaming bounds mismatch for %s: online=%s exact=%s",
                request_id,
                (state.start_row, online_end),
                bounds,
            )

    def _decode_delta(
        self, state: _MossStreamState, *, is_final: bool
    ) -> dict[str, Any] | None:
        if state.contract is None or state.start_row is None:
            return None
        n_vq = state.contract["n_vq"]
        delayed_count = self._payload_view_end(state) - state.start_row
        if delayed_count < n_vq:
            return None
        # (wenyao) De-delay: raw frame f needs delayed rows f..f+n_vq-1, so F frames
        # require F + n_vq - 1 rows and the first frame exists at n_vq rows.
        raw_total = delayed_count - n_vq + 1

        use_initial_chunk = False
        if not is_final:
            steady_codec_frames = max(1, self._stream_stride - n_vq + 1)
            use_initial_chunk = (
                0 < state.initial_codec_chunk_frames < steady_codec_frames
                and not state.has_emitted
                and state.emitted_raw_frames < state.initial_codec_chunk_frames
            )
            first_decode_rows = max(n_vq, state.initial_codec_chunk_frames + n_vq - 1)
            next_decode_rows = state.next_decode_rows or (
                first_decode_rows
                if use_initial_chunk
                else max(n_vq, self._stream_stride)
            )
            if delayed_count < next_decode_rows:
                state.next_decode_rows = next_decode_rows
                return None
            if not self._emission_resolves_bounds(state):
                state.next_decode_rows = delayed_count + 1
                return None
            if use_initial_chunk:
                emit_until = min(raw_total, state.initial_codec_chunk_frames)
            else:
                emit_until = max(0, raw_total - self._stream_holdback_frames)
            if emit_until <= state.emitted_raw_frames:
                state.next_decode_rows = delayed_count + self._stream_followup_stride
                return None
        else:
            emit_until = raw_total

        delta = self._emit_segments(state, emit_until, is_final=is_final)
        state.emitted_raw_frames = max(state.emitted_raw_frames, emit_until)
        if delta is None:
            if not is_final:
                state.next_decode_rows = delayed_count + self._stream_followup_stride
            return None
        if not is_final:
            state.next_decode_rows = self._next_decode_rows_after_emit(
                delayed_count,
                n_vq=n_vq,
                emitted_initial_chunk=use_initial_chunk,
            )
        state.has_emitted = True
        return audio_waveform_payload(
            delta,
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="MOSS-TTS streaming",
        )

    def _next_decode_rows_after_emit(
        self,
        delayed_count: int,
        *,
        n_vq: int,
        emitted_initial_chunk: bool,
    ) -> int:
        if emitted_initial_chunk:
            return max(n_vq, self._stream_stride) + self._stream_followup_stride
        return delayed_count + self._stream_followup_stride

    def _emit_segments(
        self, state: _MossStreamState, emit_until: int, *, is_final: bool
    ) -> torch.Tensor | None:
        pieces: list[torch.Tensor] = []
        for segment in state.segments:
            if segment.flushed:
                continue
            segment_next = segment.start_frame + segment.emitted_frames
            take_all = is_final or (segment.closed and segment.end_frame <= emit_until)
            limit = (
                segment.end_frame if take_all else min(segment.end_frame, emit_until)
            )
            new_frames = max(0, limit - segment_next)
            # (wenyao) take_all re-decodes a tail-only window for segments that already
            # emitted, capturing any codec tail dropped by the exact slicing.
            if new_frames == 0 and not (take_all and segment.emitted_frames > 0):
                continue
            window_start = max(
                segment.start_frame, segment_next - self._stream_overlap_frames
            )
            if limit <= window_start:
                continue
            window = torch.stack(state.frame_values[window_start:limit], dim=0)
            waveform = self._decode_codes(window)
            if self._samples_per_frame is None:
                window_frames = limit - window_start
                if window_frames > 0 and int(waveform.shape[0]) > 0:
                    self._samples_per_frame = max(
                        1, int(waveform.shape[0]) // window_frames
                    )
            samples_per_frame = self._samples_per_frame or 1
            trim = min(
                (segment_next - window_start) * samples_per_frame,
                int(waveform.shape[0]),
            )
            if take_all:
                piece = waveform[trim:]
                segment.emitted_frames = segment.end_frame - segment.start_frame
                segment.flushed = True
            else:
                piece = waveform[trim : trim + new_frames * samples_per_frame]
                segment.emitted_frames += new_frames
            if piece.numel():
                pieces.append(piece.contiguous())
        if not pieces:
            return None
        return torch.cat(pieces, dim=0)

    def _decode_codes(self, codes_TN: torch.Tensor) -> torch.Tensor:
        decoded = self._processor.decode_audio_codes(
            [codes_TN.to(device=self._device, dtype=torch.long)]
        )
        waveforms = [
            torch.as_tensor(wav).detach().reshape(-1).to("cpu") for wav in decoded
        ]
        if not waveforms:
            raise RuntimeError("MOSS-TTS streaming vocoder decoded no audio")
        if len(waveforms) == 1:
            return waveforms[0]
        return torch.cat(waveforms, dim=0)

    @staticmethod
    def _resolve_sample_rate(processor: Any) -> int:
        return int(
            getattr(getattr(processor, "model_config", None), "sampling_rate", 0)
            or getattr(
                getattr(getattr(processor, "audio_tokenizer", None), "config", None),
                "sampling_rate",
                0,
            )
            or 24000
        )

    @staticmethod
    def _resolve_samples_per_frame(processor: Any) -> int | None:
        value = getattr(
            getattr(processor, "model_config", None), "downsample_rate", None
        ) or getattr(
            getattr(getattr(processor, "audio_tokenizer", None), "config", None),
            "downsample_rate",
            None,
        )
        if value is None:
            return None
        try:
            samples_per_frame = int(value)
        except (TypeError, ValueError):
            return None
        return samples_per_frame if samples_per_frame > 0 else None


__all__ = ["MOSS_STREAM_MODALITY", "MossStreamingVocoderScheduler"]
