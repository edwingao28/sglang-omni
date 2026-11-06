# SPDX-License-Identifier: Apache-2.0
"""Tuned fused-MoE Triton configs for the Drama 3 experts on H100.

sglang ships no config for this expert shape (E=256, N=512, fp8_w8a8, 128x128
blocks), so its kernel uses one BLOCK_SIZE_M=64 tile at every token count. At
decode batch sizes each expert sees about one token, so most of that tile is
padding. The configs choose kernel tiling only; outputs are unchanged.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEVICE_NAME = "NVIDIA H100 80GB HBM3"
TRITON_VERSION = "3.8.0"
# (E, N, dtype, block_n, block_k) as sglang's get_moe_configs receives them.
EXPERT_SHAPE = (256, 512, "fp8_w8a8", 128, 128)

# sglang's get_default_config for this shape, which does not depend on M; token
# counts that were not tuned keep it.
DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 3,
}
DEFAULT_TOKEN_COUNTS = (24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 4096)

# Tuned with sglang's benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py
# on one H100 80GB HBM3 under Triton 3.8.0. Keys are token counts M.
# Decode at concurrency 16 runs at most 16 tokens, so only those counts were tuned.
DECODE_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 64,
    "num_warps": 4,
    "num_stages": 3,
}
TUNED_CONFIGS: dict[int, dict[str, int]] = {8: DECODE_CONFIG, 16: DECODE_CONFIG}

CONFIGS = {
    **{m: DEFAULT_CONFIG for m in DEFAULT_TOKEN_COUNTS},
    **TUNED_CONFIGS,
}


def install_tuned_moe_configs() -> None:
    """Serve CONFIGS for the Drama 3 expert shape from sglang's config lookup.

    Other shapes, other devices and other Triton versions keep sglang's lookup.
    """
    import triton
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        fused_moe_triton_config as moe_config,
    )

    lookup = moe_config.get_moe_configs
    if getattr(lookup, "serves_drama3_configs", False):
        return
    elif (
        moe_config.get_device_name() != DEVICE_NAME
        or triton.__version__ != TRITON_VERSION
    ):
        logger.info(
            "Drama 3 fused MoE: no tuned configs for %s under Triton %s",
            moe_config.get_device_name(),
            triton.__version__,
        )
        return
    else:
        pass

    def get_moe_configs(
        E: int,
        N: int,
        dtype: str | None,
        block_n: int | None = 0,
        block_k: int | None = 0,
        per_channel_quant: bool = False,
        down_moe: bool = False,
    ) -> dict[int, dict[str, int]] | None:
        if (
            (E, N, dtype, block_n, block_k) == EXPERT_SHAPE
            and not per_channel_quant
            and not down_moe
            and not moe_config.get_exec().deterministic.enable_deterministic_inference
        ):
            return CONFIGS
        else:
            return lookup(
                E,
                N,
                dtype,
                block_n,
                block_k,
                per_channel_quant=per_channel_quant,
                down_moe=down_moe,
            )

    get_moe_configs.serves_drama3_configs = True
    moe_config.get_moe_configs = get_moe_configs
    logger.info(
        "Drama 3 fused MoE: tuned configs for token counts %s",
        sorted(TUNED_CONFIGS),
    )
