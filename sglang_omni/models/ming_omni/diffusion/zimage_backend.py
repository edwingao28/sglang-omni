# SPDX-License-Identifier: Apache-2.0
"""Z-Image diffusion backend for thinker-fused semantic conditioning.

The initial image-generation path projects Ming thinker hidden states before
calling this backend.  This class owns Z-Image loading, denoising, and VAE
decode only; standalone semantic encoding and ByT5 text-rendering are deferred.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image

from sglang_omni.models.ming_omni.diffusion.backend import (
    DiffusionBackend,
    ImageGenParams,
)

logger = logging.getLogger(__name__)


class ZImageBackend(DiffusionBackend):
    """Z-Image diffusion backend with precomputed semantic conditioning."""

    def __init__(self) -> None:
        self._pipe = None
        self._device: torch.device | None = None

    def load_models(self, model_path: str, device: torch.device) -> None:
        self._device = device

        from diffusers import (
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler,
            ZImagePipeline,
            ZImageTransformer2DModel,
        )

        logger.info("[ZImage] Loading pipeline components from %s", model_path)

        # 1. Scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        scheduler.config["use_dynamic_shifting"] = True

        # 2. VAE
        vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.bfloat16
        )

        # 3. Transformer (ZImageTransformer2DModel)
        transformer = ZImageTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        logger.info(
            "[ZImage] Transformer loaded (cap_feat_dim=%d)",
            transformer.config.cap_feat_dim,
        )

        # 4. Assemble pipeline (text encoding handled separately)
        self._pipe = ZImagePipeline(
            scheduler=scheduler,
            vae=vae,
            transformer=transformer,
            text_encoder=None,
            tokenizer=None,
        )
        self._pipe = self._pipe.to(device)
        logger.info("[ZImage] Pipeline assembled on %s", device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        params: ImageGenParams,
        *,
        condition_embeds: list[torch.Tensor] | None = None,
        negative_condition_embeds: list[torch.Tensor] | None = None,
    ) -> Image.Image:
        if self._pipe is None:
            raise RuntimeError("ZImage pipeline not loaded")

        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(params.seed)

        if condition_embeds is None:
            raise RuntimeError("ZImageBackend requires semantic condition embeddings")
        prompt_embeds = condition_embeds
        neg_embeds = (
            negative_condition_embeds
            if negative_condition_embeds is not None
            else [e * 0.0 for e in condition_embeds]
        )

        result = self._pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            generator=generator,
            max_sequence_length=512,
        )

        return result.images[0]

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        torch.cuda.empty_cache()
