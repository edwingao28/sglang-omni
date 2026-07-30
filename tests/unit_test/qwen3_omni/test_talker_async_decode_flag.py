# SPDX-License-Identifier: Apache-2.0
"""The talker stage defaults async decode off; --talker-async-decode overrides it.

Mirrors test_thinker_async_decode_flag.py. Unlike the thinker, the talker
defaults OFF: the flip to on is a follow-up after an unrelated TTFA fix.
"""
from __future__ import annotations

from sglang_omni.cli.serve import apply_talker_async_decode_cli_overrides
from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig


def _talker_factory_args(cfg):
    return next(s for s in cfg.stages if s.name == "talker_ar").factory_args


def test_talker_async_decode_off_by_default():
    cfg = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    assert _talker_factory_args(cfg)["enable_async_decode"] is False


def test_talker_async_decode_flag_enables_it():
    cfg = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    apply_talker_async_decode_cli_overrides(cfg, talker_async_decode=True)
    assert _talker_factory_args(cfg)["enable_async_decode"] is True


def test_talker_async_decode_flag_can_force_off():
    cfg = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    apply_talker_async_decode_cli_overrides(cfg, talker_async_decode=False)
    assert _talker_factory_args(cfg)["enable_async_decode"] is False


def test_talker_async_decode_flag_omitted_keeps_default():
    cfg = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    apply_talker_async_decode_cli_overrides(cfg, talker_async_decode=None)
    assert _talker_factory_args(cfg)["enable_async_decode"] is False
