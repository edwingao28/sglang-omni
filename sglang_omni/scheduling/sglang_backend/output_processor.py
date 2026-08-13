# SPDX-License-Identifier: Apache-2.0
"""Converts SGLang GenerationBatchResult to per-request RequestOutputs."""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from typing import Any

import torch

from sglang_omni.scheduling.types import RequestOutput, SchedulerOutput

_PREFILL_DEBUG_SNAPSHOT_MAX_ROWS = 512


class SGLangOutputProcessor:
    """Converts GenerationBatchResult to per-request RequestOutputs."""

    def __init__(
        self,
        capture_hidden: bool = False,
        capture_hidden_layers: list[int] | None = None,
        model: Any = None,
        should_emit_hidden: Callable[[Any], bool] | None = None,
        capture_prefill_debug_snapshot: bool = False,
    ):
        self._capture_hidden = capture_hidden
        self._capture_hidden_layers = capture_hidden_layers
        self._model = model
        self._should_emit_hidden = should_emit_hidden
        self._capture_prefill_debug_snapshot = capture_prefill_debug_snapshot
        self._prefill_debug_snapshot: dict[str, Any] | None = None

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
        host_token_ids: torch.Tensor | None = None,
    ) -> dict[str, RequestOutput]:
        ids = host_token_ids
        if ids is None:
            ids = model_output.next_token_ids
        token_list = ids.tolist() if ids is not None else []

        hidden_extras_by_request: dict[int, dict[str, Any] | None] = {}
        if self._capture_hidden:
            should_emit_hidden_by_request = [
                self._should_emit_hidden_for_request(request)
                for request in scheduler_output.requests
            ]
            hidden_extras_by_request = self._build_hidden_extras_by_request(
                model_output,
                scheduler_output=scheduler_output,
                should_emit_hidden_by_request=should_emit_hidden_by_request,
            )

        outputs = {}
        for i, sched_req in enumerate(scheduler_output.requests):
            token_id = token_list[i] if i < len(token_list) else None
            extra = hidden_extras_by_request.get(i)
            outputs[sched_req.request_id] = RequestOutput(
                request_id=sched_req.request_id,
                data=token_id,
                finished=False,
                extra=extra,
            )
        if self._capture_prefill_debug_snapshot:
            self._record_prefill_debug_snapshot(
                model_output,
                scheduler_output=scheduler_output,
                outputs=outputs,
            )
        return outputs

    @staticmethod
    def _serialize_debug_tensor(tensor: torch.Tensor) -> dict[str, Any]:
        value = tensor.detach().contiguous().cpu()
        raw = value.view(torch.uint8).numpy().tobytes()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "data": base64.b64encode(raw).decode("ascii"),
        }

    def _record_prefill_debug_snapshot(
        self,
        model_output: Any,
        *,
        scheduler_output: SchedulerOutput,
        outputs: dict[str, RequestOutput],
    ) -> None:
        """Retain the latest prefill tensors for the opt-in H100 parity gate."""
        batch_data = scheduler_output.batch_data
        if not batch_data.forward_mode.is_extend():
            return
        logical_rows = self._logical_hidden_rows(scheduler_output)
        if logical_rows > _PREFILL_DEBUG_SNAPSHOT_MAX_ROWS:
            self._prefill_debug_snapshot = {
                "requests": [],
                "skipped": "logical_rows_exceed_limit",
                "logical_rows": logical_rows,
            }
            return

        logits_output = getattr(model_output, "logits_output", None)
        logits = getattr(logits_output, "next_token_logits", None)
        next_token_ids = getattr(model_output, "next_token_ids", None)
        requests = []
        for index, scheduler_request in enumerate(scheduler_output.requests):
            request_output = outputs[scheduler_request.request_id]
            hidden = (
                request_output.extra.get("hidden_states")
                if isinstance(request_output.extra, dict)
                else None
            )
            if not isinstance(hidden, dict):
                continue

            encoded_hidden = {
                str(layer): self._serialize_debug_tensor(tensor)
                for layer, tensor in hidden.items()
                if isinstance(tensor, torch.Tensor)
            }
            if not encoded_hidden:
                continue

            request_snapshot: dict[str, Any] = {
                "request_id": scheduler_request.request_id,
                "hidden_states": encoded_hidden,
            }
            batch_requests = getattr(batch_data, "reqs", ())
            if index < len(batch_requests):
                extend_range = getattr(batch_requests[index], "extend_range", None)
                request_rows = getattr(extend_range, "length", None)
                if request_rows is not None:
                    request_snapshot["logical_rows"] = int(request_rows)
            if isinstance(next_token_ids, torch.Tensor) and index < len(next_token_ids):
                request_snapshot["next_token_id"] = int(next_token_ids[index])
            if isinstance(logits, torch.Tensor) and index < len(logits):
                request_snapshot["next_token_logits"] = self._serialize_debug_tensor(
                    logits[index]
                )
            requests.append(request_snapshot)

        self._prefill_debug_snapshot = {
            "requests": requests,
            "logical_rows": logical_rows,
        }

    def prefill_debug_snapshot(self) -> dict[str, Any] | None:
        return self._prefill_debug_snapshot

    def _should_emit_hidden_for_request(self, request: Any) -> bool:
        if self._should_emit_hidden is None:
            return True
        return self._should_emit_hidden(request)

    def _build_hidden_extras_by_request(
        self,
        model_output: Any,
        *,
        scheduler_output: SchedulerOutput,
        should_emit_hidden_by_request: list[bool],
    ) -> dict[int, dict[str, Any] | None]:
        request_indexes = [
            i
            for i, should_emit in enumerate(should_emit_hidden_by_request)
            if should_emit
        ]
        if not request_indexes:
            return {}

        if self._model is not None and self._capture_hidden_layers:
            static_capture = getattr(self._model, "_omni_aux_hidden_capture", None)
            if static_capture is not None:
                logical_rows = self._logical_hidden_rows(scheduler_output)
                return self._build_aux_hidden_extras(
                    static_capture.views(logical_rows),
                    model_output=model_output,
                    scheduler_output=scheduler_output,
                    request_indexes=request_indexes,
                )

        logits_output = model_output.logits_output
        if logits_output is None:
            return {}
        raw_hidden = logits_output.hidden_states
        if raw_hidden is None:
            return {}

        if isinstance(raw_hidden, dict):
            return {
                request_index: self._build_dict_hidden_extra(
                    raw_hidden,
                    request_index=request_index,
                    scheduler_output=scheduler_output,
                )
                for request_index in request_indexes
            }
        elif isinstance(raw_hidden, torch.Tensor):
            return {
                request_index: {
                    "hidden_states": self._slice_per_request_tensor(
                        raw_hidden,
                        request_index=request_index,
                        scheduler_output=scheduler_output,
                    )
                }
                for request_index in request_indexes
            }
        return {}

    def _build_aux_hidden_extras(
        self,
        aux_hidden_states: Sequence[torch.Tensor],
        *,
        model_output: Any,
        scheduler_output: SchedulerOutput,
        request_indexes: list[int],
    ) -> dict[int, dict[str, Any] | None]:
        if not request_indexes:
            return {}
        stream_hidden_states = self._extract_stream_hidden_states(model_output)
        return {
            request_index: self._build_aux_hidden_extra(
                aux_hidden_states,
                request_index=request_index,
                scheduler_output=scheduler_output,
                stream_hidden_states=stream_hidden_states,
            )
            for request_index in request_indexes
        }

    def _build_aux_hidden_extra(
        self,
        aux_hidden_states: Sequence[torch.Tensor],
        *,
        request_index: int,
        scheduler_output: SchedulerOutput,
        stream_hidden_states: torch.Tensor | None,
    ) -> dict[str, Any]:
        per_request_hidden = {}
        for layer_id, tensor in zip(
            self._capture_hidden_layers or [],
            aux_hidden_states,
        ):
            key = "embed" if layer_id == 0 else layer_id
            per_request_hidden[key] = self._slice_per_request_tensor(
                tensor,
                request_index=request_index,
                scheduler_output=scheduler_output,
                prefer_token_axis=True,
            ).clone()

        extra: dict[str, Any] = {"hidden_states": per_request_hidden}
        if stream_hidden_states is not None:
            extra["stream_hidden_states"] = self._slice_per_request_tensor(
                stream_hidden_states,
                request_index=request_index,
                scheduler_output=scheduler_output,
            ).clone()
        return extra

    def _build_dict_hidden_extra(
        self,
        hidden_states: dict[Any, torch.Tensor],
        *,
        request_index: int,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, Any]:
        return {
            "hidden_states": {
                key: self._slice_per_request_tensor(
                    tensor,
                    request_index=request_index,
                    scheduler_output=scheduler_output,
                )
                for key, tensor in hidden_states.items()
            }
        }

    def _extract_stream_hidden_states(self, model_output: Any) -> torch.Tensor | None:
        logits_output = model_output.logits_output
        if logits_output is None:
            return None
        raw_hidden = logits_output.hidden_states
        return raw_hidden if isinstance(raw_hidden, torch.Tensor) else None

    @staticmethod
    def _logical_hidden_rows(scheduler_output: SchedulerOutput) -> int:
        batch_data = scheduler_output.batch_data
        if batch_data.forward_mode.is_extend():
            return sum(req.extend_range.length for req in batch_data.reqs)
        return len(batch_data.reqs)

    @staticmethod
    def _slice_per_request_tensor(
        tensor: torch.Tensor,
        *,
        request_index: int,
        scheduler_output: SchedulerOutput,
        prefer_token_axis: bool = False,
    ) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor

        requests = scheduler_output.requests
        batch_data = scheduler_output.batch_data
        reqs = batch_data.reqs
        num_requests = len(reqs)

        # Preserve the generic output contract: ordinary hidden tensors are
        # request-major and do not require forward-mode metadata. Static aux
        # capture is the one token-major representation and opts into the
        # logical-token check below before these request-major fast paths.
        if not prefer_token_axis:
            if len(requests) == 1:
                return tensor[0] if tensor.ndim >= 2 else tensor
            if tensor.shape[0] == num_requests:
                return tensor[request_index]

        is_extend_fn = getattr(
            getattr(batch_data, "forward_mode", None),
            "is_extend",
            None,
        )
        is_extend = bool(callable(is_extend_fn) and is_extend_fn())
        lengths = None
        if prefer_token_axis and is_extend:
            lengths = [req.extend_range.length for req in reqs]
        if lengths is not None and tensor.shape[0] == sum(lengths):
            start = sum(lengths[:request_index])
            end = start + lengths[request_index]
            return tensor[start:end]

        if prefer_token_axis:
            if len(requests) == 1:
                return tensor[0] if tensor.ndim >= 2 else tensor
            if tensor.shape[0] == num_requests:
                return tensor[request_index]

        if lengths is None and is_extend:
            lengths = [req.extend_range.length for req in reqs]
        if lengths is not None and tensor.shape[0] == sum(lengths):
            start = sum(lengths[:request_index])
            end = start + lengths[request_index]
            return tensor[start:end]

        return tensor
