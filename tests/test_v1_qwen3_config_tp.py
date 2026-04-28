# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni_v1.models.qwen3_omni.config import (
    _validate_qwen3_speech_gpu_placement,
)


def test_validate_qwen3_speech_gpu_placement_rejects_talker_collision_tp2():
    placement = {
        "thinker": 0,
        "talker_ar": 1,
        "code_predictor": 4,
        "code2wav": 5,
    }

    with pytest.raises(ValueError, match="GPU placement"):
        _validate_qwen3_speech_gpu_placement(placement, tp_size=2)


def test_validate_qwen3_speech_gpu_placement_rejects_code_predictor_collision_tp2():
    placement = {
        "thinker": 0,
        "talker_ar": 4,
        "code_predictor": 1,
        "code2wav": 5,
    }

    with pytest.raises(ValueError, match="GPU placement"):
        _validate_qwen3_speech_gpu_placement(placement, tp_size=2)


def test_validate_qwen3_speech_gpu_placement_accepts_disjoint_tp2():
    placement = {
        "thinker": 0,
        "talker_ar": 2,
        "code_predictor": 3,
        "code2wav": 4,
    }

    _validate_qwen3_speech_gpu_placement(placement, tp_size=2)


def test_validate_qwen3_speech_gpu_placement_skips_missing_stage_keys():
    placement = {
        "thinker": 0,
        "talker_ar": 2,
        "code2wav": 3,
    }

    _validate_qwen3_speech_gpu_placement(placement, tp_size=2)
