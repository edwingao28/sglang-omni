# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni's model-local adopter for the shared prefill sidecar."""

from __future__ import annotations

import logging
from array import array
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch

from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
    get_omni_prefill_inputs,
)
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner

logger = logging.getLogger(__name__)

_PREFILL_AUDIO_INPUT_KEYS = frozenset(
    {
        "audio_embeds",
        "audio_feature_lengths",
        "feature_attention_mask",
        "pad_values",
    }
)

_SIDECAR = "sidecar"
_UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class _PrefillDisposition:
    kind: str
    has_audio: bool = False
    reason: str | None = None


class Qwen3OmniThinkerModelRunner(ThinkerModelRunner):
    """Adopt the shared prefill sidecar for text/audio-to-text prefills.

    This class only qualifies Qwen request payloads. SGLang continues to own
    graph admission, bucket selection, padding, replay metadata, and eager
    fallback. Unsupported requests deliberately delegate to the inherited
    eager multimodal path.
    """

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self._sidecar_decline_warned: set[str] = set()

    @staticmethod
    def _origin_num_tokens(value: Any) -> int | None:
        if isinstance(value, torch.Tensor):
            return int(value.numel()) if value.ndim == 1 else None
        if isinstance(value, (array, list, tuple)):
            return len(value)
        return None

    @staticmethod
    def _valid_positions(value: Any) -> bool:
        if not (
            isinstance(value, torch.Tensor)
            and value.ndim == 1
            and value.device.type == "cpu"
            and value.dtype != torch.bool
            and not torch.is_floating_point(value)
            and not torch.is_complex(value)
        ):
            return False
        if value.numel() and int(value[0]) < 0:
            return False
        return bool(torch.all(value[1:] > value[:-1]))

    @staticmethod
    def _cpu_int_sequence(value: Any) -> list[int] | None:
        if isinstance(value, torch.Tensor):
            if value.ndim != 1 or value.device.type != "cpu":
                return None
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            return None
        if any(
            not isinstance(item, Integral) or isinstance(item, bool) for item in value
        ):
            return None
        return [int(item) for item in value]

    def _mm_positions(
        self, req: Any, pad_values: dict[str, Any]
    ) -> dict[str, torch.Tensor] | None:
        positions = self._req_mm_token_positions(req, pad_values)
        if not isinstance(positions, dict):
            return None

        origin_num_tokens = self._origin_num_tokens(
            getattr(req, "origin_input_ids", None)
        )
        if origin_num_tokens is None:
            return None

        validated: dict[str, torch.Tensor] = {}
        for modality in ("image", "video", "audio"):
            value = positions.get(modality)
            if not self._valid_positions(value) or (
                value.numel() and int(value[-1]) >= origin_num_tokens
            ):
                return None
            validated[modality] = value.to(dtype=torch.long)
        return validated

    @classmethod
    def _batch_chunk_spans(
        cls, forward_batch: Any, expected_batch_size: int
    ) -> list[tuple[int, int]] | None:
        extend_lens = cls._cpu_int_sequence(
            getattr(forward_batch, "extend_seq_lens_cpu", None)
        )
        if extend_lens is None or len(extend_lens) != expected_batch_size:
            return None

        prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        if prefix_lens is None:
            prefix_values = [0] * expected_batch_size
        else:
            prefix_values = cls._cpu_int_sequence(prefix_lens)
            if prefix_values is None or len(prefix_values) != expected_batch_size:
                return None

        spans: list[tuple[int, int]] = []
        for prefix, length in zip(prefix_values, extend_lens):
            if prefix < 0 or length <= 0:
                return None
            spans.append((prefix, length))
        return spans

    def _audio_inputs_decline_reason(
        self,
        req: Any,
        model_inputs: Any,
        chunk_span: tuple[int, int],
    ) -> str | None:
        if not isinstance(model_inputs, dict) or not model_inputs:
            return "model_inputs"
        if set(model_inputs) - _PREFILL_AUDIO_INPUT_KEYS:
            return "model_inputs"

        audio_embeds = model_inputs.get("audio_embeds")
        if (
            not isinstance(audio_embeds, torch.Tensor)
            or audio_embeds.ndim != 2
            or audio_embeds.shape[0] <= 0
            or audio_embeds.shape[1] <= 0
        ):
            return "audio_embeds"
        embedding_dim = getattr(self._embed_tokens, "embedding_dim", None)
        if embedding_dim is not None and audio_embeds.shape[1] != embedding_dim:
            return "audio_embeds"

        feature_lengths = model_inputs.get("audio_feature_lengths")
        if feature_lengths is not None and (
            not isinstance(feature_lengths, torch.Tensor)
            or feature_lengths.ndim != 1
            or feature_lengths.numel() == 0
            or feature_lengths.dtype == torch.bool
            or torch.is_floating_point(feature_lengths)
            or torch.is_complex(feature_lengths)
        ):
            return "audio_feature_lengths"
        feature_mask = model_inputs.get("feature_attention_mask")
        if feature_mask is not None and (
            not isinstance(feature_mask, torch.Tensor) or feature_mask.ndim != 2
        ):
            return "feature_attention_mask"

        pad_values = model_inputs.get("pad_values", {})
        if not isinstance(pad_values, dict) or set(pad_values) - {"audio"}:
            return "pad_values"
        if "audio" in pad_values and (
            not isinstance(pad_values["audio"], Integral)
            or isinstance(pad_values["audio"], bool)
        ):
            return "pad_values"

        positions = self._mm_positions(req, pad_values)
        if positions is None:
            return "mm_positions"
        if positions["image"].numel() or positions["video"].numel():
            return "visual_inputs"
        if positions["audio"].numel() != audio_embeds.shape[0]:
            return "audio_positions"

        prefix, length = chunk_span
        consumed = getattr(req, "_omni_consumed", None)
        if consumed is None:
            cached_audio = positions["audio"][positions["audio"] < prefix]
            future_audio = positions["audio"][positions["audio"] >= prefix]
            # note(chenye): A fresh radix prefix can hide prior audio rows without
            # advancing the shared multimodal cursor; keep that state outside the
            # Qwen sidecar.
            if cached_audio.numel() and future_audio.numel():
                return "cached_prefix"
            chunk_consumed = {}
        elif isinstance(consumed, dict) and set(consumed) <= {"audio"}:
            chunk_consumed = consumed
        else:
            return "consumed_state"
        _, audio_offset, live_audio_count = self._plan_modality_chunk(
            positions["audio"], chunk_consumed, "audio", prefix, length
        )
        if (
            not isinstance(audio_offset, Integral)
            or isinstance(audio_offset, bool)
            or audio_offset < 0
            or audio_offset + live_audio_count > audio_embeds.shape[0]
        ):
            return "audio_offset"

        middle_chunks = getattr(req, "inflight_middle_chunks", None)
        if (
            not isinstance(middle_chunks, Integral)
            or isinstance(middle_chunks, bool)
            or middle_chunks < 0
        ):
            return "middle_chunks"
        return None

    def _classify_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list[Any]
    ) -> _PrefillDisposition:
        if len(requests) != getattr(forward_batch, "batch_size", None):
            return _PrefillDisposition(_UNSUPPORTED, reason="batch_shape")
        schedule_reqs = getattr(schedule_batch, "reqs", None)
        if schedule_reqs is None or len(schedule_reqs) != len(requests):
            return _PrefillDisposition(_UNSUPPORTED, reason="batch_shape")
        if (
            getattr(forward_batch, "input_embeds", None) is not None
            or getattr(forward_batch, "replace_embeds", None) is not None
            or get_omni_prefill_inputs(forward_batch) is not None
        ):
            return _PrefillDisposition(_UNSUPPORTED, reason="official_embeds")

        model_inputs_by_request = [
            getattr(req, "omni_model_inputs", None) for req in schedule_reqs
        ]
        has_model_inputs = any(
            model_inputs is not None
            and not (isinstance(model_inputs, dict) and not model_inputs)
            for model_inputs in model_inputs_by_request
        )
        if not has_model_inputs:
            return _PrefillDisposition(_SIDECAR)

        chunk_spans = self._batch_chunk_spans(forward_batch, len(schedule_reqs))
        if chunk_spans is None:
            return _PrefillDisposition(_UNSUPPORTED, reason="chunk_spans")
        has_audio = False
        for request_index, (req, model_inputs) in enumerate(
            zip(schedule_reqs, model_inputs_by_request)
        ):
            if model_inputs is None or (
                isinstance(model_inputs, dict) and not model_inputs
            ):
                continue
            reason = self._audio_inputs_decline_reason(
                req, model_inputs, chunk_spans[request_index]
            )
            if reason is not None:
                return _PrefillDisposition(_UNSUPPORTED, reason=reason)
            has_audio = True

        return _PrefillDisposition(_SIDECAR, has_audio=has_audio)

    def _text_input_embeds(self, forward_batch: Any) -> torch.Tensor:
        return self._embed_tokens(forward_batch.input_ids)

    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list[Any]
    ) -> None:
        disposition = self._classify_prefill(forward_batch, schedule_batch, requests)
        if disposition.kind != _SIDECAR:
            return

        if disposition.has_audio:
            omni_result = self._inject_multimodal_embeds(forward_batch, schedule_batch)
            if omni_result is None:
                raise RuntimeError(
                    "Qwen audio prefill was classified as sidecar-compatible, "
                    "but multimodal embedding composition returned no result"
                )
            input_embeds, deepstack_embeds, visual_masks = omni_result
            if input_embeds is None:
                raise RuntimeError(
                    "Qwen audio prefill composition returned no input embeddings"
                )
            if deepstack_embeds is not None or visual_masks is not None:
                raise RuntimeError(
                    "Qwen text-output sidecar cannot carry visual deepstack embeddings"
                )
        else:
            input_embeds = self._text_input_embeds(forward_batch)

        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(input_embeds=input_embeds),
        )

    def _note_sidecar_decline(self, reason: str) -> None:
        self.tp_worker.record_prefill_sidecar_decline(reason)
        if reason in self._sidecar_decline_warned:
            return
        self._sidecar_decline_warned.add(reason)
        logger.warning(
            "Qwen thinker prefill sidecar declined a batch (%s); it runs through "
            "the inherited ThinkerModelRunner prefill instead. Per-reason counts: "
            "prefill_cuda_graph.sidecar_decline_reasons in /model_info",
            reason,
        )

    def custom_prefill_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list[Any]
    ) -> Any | None:
        if get_omni_prefill_inputs(forward_batch) is not None:
            return None

        disposition = self._classify_prefill(forward_batch, schedule_batch, requests)
        if disposition.kind == _SIDECAR:
            raise RuntimeError("Qwen prefill sidecar was not attached before forward")
        self._note_sidecar_decline(disposition.reason or "unspecified")

        result = super().custom_prefill_forward(
            forward_batch,
            schedule_batch,
            requests,
        )
        if result is not None:
            self.tp_worker.record_custom_prefill_eager()
        return result


__all__ = ["Qwen3OmniThinkerModelRunner"]
