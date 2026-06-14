# SPDX-License-Identifier: Apache-2.0
"""Image generation terminal executor for Ming-Omni."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from collections.abc import Mapping
from typing import Any

import torch

from sglang_omni.models.ming_omni.components.image_gen_request import (
    extract_image_generation_params,
)
from sglang_omni.models.ming_omni.components.image_gen_request import (
    should_generate_image as request_should_generate_image,
)
from sglang_omni.models.ming_omni.diffusion.backend import (
    DiffusionBackend,
    ImageGenParams,
)
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


def _create_backend(dit_type: str) -> DiffusionBackend:
    """Instantiate a diffusion backend without importing backend modules eagerly."""
    if dit_type == "sd3":
        from sglang_omni.models.ming_omni.diffusion.sd3_backend import SD3Backend

        return SD3Backend()
    if dit_type == "zimage":
        from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

        return ZImageBackend()
    raise ValueError(f"Unknown dit_type: {dit_type!r}. Must be 'sd3' or 'zimage'.")


class MingImageGenExecutor:
    """Terminal executor that generates image payloads via a diffusion backend."""

    def __init__(
        self,
        model_path: str,
        dit_type: str = "zimage",
        dit_model_path: str | None = None,
        device: str = "cuda",
        conditioner: Any | None = None,
        skip_semantic_encoder: bool = False,
        backend: DiffusionBackend | None = None,
    ):
        self._model_path = model_path
        self._dit_type = dit_type
        self._dit_model_path = dit_model_path or model_path
        self._device = device
        self._conditioner = conditioner
        self._skip_semantic_encoder = skip_semantic_encoder

        self._backend = backend
        self._thinker_tokenizer = None
        self._results: asyncio.Queue[StagePayload] = asyncio.Queue()
        self._aborted: set[str] = set()

    async def start(self) -> None:
        """Load diffusion backend and thinker tokenizer."""
        logger.info(
            "[IMG_GEN] Loading %s backend from %s (device=%s)",
            self._dit_type,
            self._dit_model_path,
            self._device,
        )
        await asyncio.to_thread(self._load_models)
        logger.info("[IMG_GEN] Backend loaded and ready")

    def _load_models(self) -> None:
        """Load diffusion backend and tokenizer in a worker thread."""
        t0 = time.time()
        if self._backend is None:
            self._backend = _create_backend(self._dit_type)

        device = torch.device(self._device)
        if self._dit_type == "zimage" and self._skip_semantic_encoder:
            self._backend.load_models(
                self._dit_model_path,
                device,
                skip_semantic_encoder=True,
            )
        else:
            self._backend.load_models(self._dit_model_path, device)
        logger.info("[IMG_GEN] Diffusion backend loaded in %.1fs", time.time() - t0)

        conditioner_load = getattr(self._conditioner, "load", None)
        if callable(conditioner_load):
            conditioner_load(self._model_path, device)
            logger.info("[IMG_GEN] Semantic conditioner loaded")

        try:
            from sglang_omni.models.ming_omni.components.common import (
                load_ming_tokenizer,
            )

            self._thinker_tokenizer = load_ming_tokenizer(self._model_path)
            logger.info(
                "[IMG_GEN] Thinker tokenizer loaded: %s",
                type(self._thinker_tokenizer).__name__,
            )
        except Exception as exc:
            logger.warning("[IMG_GEN] Could not load thinker tokenizer: %s", exc)

    async def add_request(self, payload: StagePayload) -> None:
        """Generate an image result for a requested image-modality payload."""
        request_id = payload.request_id
        if request_id in self._aborted:
            return

        if not self.should_generate_image(payload):
            logger.info(
                "[IMG_GEN] Skipping image generation for request %s", request_id
            )
            await self._results.put(self.build_empty_image_result(payload))
            return

        data = payload.data if isinstance(payload.data, dict) else {}
        condition_embeds = None
        negative_embeds = None

        if self._conditioner is not None:
            condition_embeds, negative_embeds = self._try_condition_from_hidden_states(
                data
            )

        if condition_embeds is not None:
            params = self._extract_params(data, payload.request)
            prompt = self._extract_text_for_byt5(data)
            try:
                image = await asyncio.to_thread(
                    self._generate_with_condition_embeds,
                    prompt,
                    params,
                    condition_embeds,
                    negative_embeds,
                )
            except Exception as exc:
                await self._results.put(self._build_error_image_result(payload, exc))
                return
        else:
            text, params = self._extract_input(payload)
            if not text:
                await self._results.put(
                    self._build_empty_requested_image_result(payload)
                )
                return
            try:
                image = await asyncio.to_thread(self._generate_image, text, params)
            except Exception as exc:
                await self._results.put(self._build_error_image_result(payload, exc))
                return

        await self._results.put(self._build_image_result(payload, image))

    async def get_result(self) -> StagePayload:
        while True:
            result = await self._results.get()
            if result.request_id in self._aborted:
                continue
            return result

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)

    async def stop(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
        conditioner_unload = getattr(self._conditioner, "unload", None)
        if callable(conditioner_unload):
            conditioner_unload()

    def should_generate_image(self, payload: StagePayload) -> bool:
        return request_should_generate_image(payload)

    @staticmethod
    def build_empty_image_result(payload: StagePayload) -> StagePayload:
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "modality": "image",
                "images": [],
                "skipped": True,
                "finish_reason": "stop",
            },
        )

    def _extract_input(self, payload: StagePayload) -> tuple[str, ImageGenParams]:
        data = payload.data if isinstance(payload.data, dict) else {}
        text = self._extract_generated_text(data)
        return text, self._extract_params(data, payload.request)

    def _extract_generated_text(self, data: dict[str, Any]) -> str:
        thinker_out = data.get("thinker_out")
        if isinstance(thinker_out, Mapping):
            output_ids = thinker_out.get("output_ids")
            if output_ids and self._thinker_tokenizer is not None:
                return self._thinker_tokenizer.decode(
                    output_ids, skip_special_tokens=True
                )

        text = data.get("generated_text")
        if text:
            return str(text)

        stream_state = data.get("stream_state")
        if isinstance(stream_state, Mapping):
            text = stream_state.get("text")
            if text:
                return str(text)
            accumulated_text = stream_state.get("accumulated_text")
            if accumulated_text:
                return str(accumulated_text)
        return ""

    def _try_condition_from_hidden_states(
        self, data: dict[str, Any]
    ) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
        thinker_out = data.get("thinker_out")
        if not isinstance(thinker_out, Mapping):
            return None, None

        extra = thinker_out.get("extra_model_outputs")
        if not isinstance(extra, Mapping):
            return None, None

        hidden_states = extra.get("hidden_states")
        if hidden_states is None:
            return None, None

        mm_inputs = data.get("mm_inputs")
        if not isinstance(mm_inputs, Mapping):
            return None, None
        image_gen = mm_inputs.get("image_gen")
        if not isinstance(image_gen, Mapping):
            return None, None
        gen_mask_list = image_gen.get("gen_mask")
        if not isinstance(gen_mask_list, list):
            return None, None

        hs = self._select_hidden_state_tensor(hidden_states)
        if hs is None:
            return None, None

        gen_mask = torch.tensor(gen_mask_list, dtype=torch.bool, device=hs.device)
        if gen_mask.dim() != 1:
            return None, None
        if hs.dim() == 2:
            gen_mask = self._align_gen_mask(gen_mask, hs.shape[0])
            if gen_mask is None:
                return None, None
            query_hidden = hs[gen_mask].unsqueeze(0)
        elif hs.dim() == 3:
            gen_mask = self._align_gen_mask(gen_mask, hs.shape[1])
            if gen_mask is None:
                return None, None
            query_hidden = hs[:, gen_mask, :]
        else:
            logger.warning("[IMG_GEN] Unexpected hidden_states dim=%d", hs.dim())
            return None, None

        if query_hidden.shape[1] == 0:
            return None, None

        condition_embeds = self._conditioner.project(query_hidden)
        negative_embeds = condition_embeds * 0.0
        return list(condition_embeds.unbind(dim=0)), list(negative_embeds.unbind(dim=0))

    @staticmethod
    def _align_gen_mask(
        gen_mask: torch.Tensor, seq_len: int
    ) -> torch.Tensor | None:
        """Align gen_mask to captured hidden states.

        Radix prefix-cache hits and chunked prefill leave only the trailing
        computed segment of the sequence in the captured hidden states, while
        gen_mask covers the full original sequence. The query tokens sit at
        the end of the prompt, so the tail of the mask is the valid view.
        """
        if gen_mask.numel() == seq_len:
            return gen_mask
        if gen_mask.numel() < seq_len:
            return None
        tail_mask = gen_mask[gen_mask.numel() - seq_len :]
        if int(tail_mask.sum()) != int(gen_mask.sum()):
            logger.warning(
                "[IMG_GEN] captured hidden states cover %d trailing positions "
                "but the %d query positions extend beyond them; dropping "
                "semantic conditioning",
                seq_len,
                int(gen_mask.sum()),
            )
            return None
        return tail_mask

    @staticmethod
    def _select_hidden_state_tensor(hidden_states: Any) -> torch.Tensor | None:
        if isinstance(hidden_states, torch.Tensor):
            return hidden_states
        if not isinstance(hidden_states, Mapping):
            return None

        numeric_keys = [
            key
            for key in hidden_states
            if isinstance(key, int) or (isinstance(key, str) and key.isdigit())
        ]
        if not numeric_keys:
            return None
        last_key = max(numeric_keys, key=lambda key: int(key))
        value = hidden_states[last_key]
        return value if isinstance(value, torch.Tensor) else None

    def _extract_params(self, data: dict[str, Any], request: Any) -> ImageGenParams:
        img_params: dict[str, Any] = {}

        raw_inputs = data.get("raw_inputs")
        if isinstance(raw_inputs, Mapping):
            raw_params = raw_inputs.get("image_generation")
            if isinstance(raw_params, Mapping):
                img_params = dict(raw_params)

        if not img_params:
            mm_inputs = data.get("mm_inputs")
            if isinstance(mm_inputs, Mapping):
                image_gen = mm_inputs.get("image_gen")
                if isinstance(image_gen, Mapping):
                    mm_params = image_gen.get("image_gen_params")
                    if isinstance(mm_params, Mapping):
                        img_params = dict(mm_params)

        if not img_params and request is not None:
            helper_payload = StagePayload(
                request_id="",
                request=request,
                data={},
            )
            img_params = extract_image_generation_params(helper_payload)

        defaults = ImageGenParams()
        width = img_params.get("width", defaults.width)
        height = img_params.get("height", defaults.height)
        size = img_params.get("size")
        if isinstance(size, str) and "x" in size:
            raw_width, raw_height = size.lower().split("x", maxsplit=1)
            try:
                width = int(raw_width)
                height = int(raw_height)
            except ValueError:
                width = img_params.get("width", defaults.width)
                height = img_params.get("height", defaults.height)

        return ImageGenParams(
            width=int(width),
            height=int(height),
            num_inference_steps=int(
                img_params.get(
                    "num_inference_steps",
                    defaults.num_inference_steps,
                )
            ),
            guidance_scale=float(
                img_params.get("guidance_scale", defaults.guidance_scale)
            ),
            seed=img_params.get("seed", defaults.seed),
            negative_prompt=str(
                img_params.get("negative_prompt", defaults.negative_prompt)
            ),
        )

    @staticmethod
    def _extract_text_for_byt5(data: dict[str, Any]) -> str:
        prompt = data.get("prompt")
        if isinstance(prompt, Mapping):
            prompt_text = prompt.get("prompt_text")
            return str(prompt_text) if prompt_text else ""
        return ""

    @staticmethod
    def _image_to_payload(image: Any) -> dict[str, Any]:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return {
            "b64_json": base64.b64encode(buf.getvalue()).decode("ascii"),
            "format": "png",
            "width": int(image.width),
            "height": int(image.height),
        }

    @classmethod
    def _build_image_result(cls, payload: StagePayload, image: Any) -> StagePayload:
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "modality": "image",
                "images": [cls._image_to_payload(image)],
                "finish_reason": "stop",
            },
        )

    @staticmethod
    def _build_error_image_result(
        payload: StagePayload, exc: Exception
    ) -> StagePayload:
        logger.error("[IMG_GEN] ERROR: %s", exc, exc_info=True)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "modality": "image",
                "images": [],
                "error": str(exc),
                "finish_reason": "error",
            },
        )

    @staticmethod
    def _build_empty_requested_image_result(payload: StagePayload) -> StagePayload:
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "modality": "image",
                "images": [],
                "finish_reason": "stop",
            },
        )

    @torch.no_grad()
    def _generate_with_condition_embeds(
        self,
        prompt_text: str,
        params: ImageGenParams,
        condition_embeds: list[torch.Tensor],
        negative_embeds: list[torch.Tensor] | None,
    ):
        if self._backend is None:
            raise RuntimeError("Diffusion backend not loaded")
        return self._backend.generate(
            prompt_text or "",
            params,
            condition_embeds=condition_embeds,
            negative_condition_embeds=negative_embeds,
        )

    @torch.no_grad()
    def _generate_image(self, text: str, params: ImageGenParams):
        if self._backend is None:
            raise RuntimeError("Diffusion backend not loaded")
        return self._backend.generate(text, params)
