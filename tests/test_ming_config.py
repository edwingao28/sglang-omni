# SPDX-License-Identifier: Apache-2.0
"""Tests for pipeline config GPU validation (Ming + Qwen3)."""
from __future__ import annotations

import unittest


class TestMingOmniSpeechGPUValidation(unittest.TestCase):
    """Validate that TP GPU range does not collide with talker GPU."""

    def test_tp2_default_gpus_rejected(self):
        """--tp-size 2 with default gpu_thinker=0, gpu_talker=1 must raise."""
        from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

        with self.assertRaises(ValueError) as ctx:
            MingOmniSpeechPipelineConfig(
                model_path="test/model",
                gpu_placement={"thinker": 0, "talker": 1},
                server_args_overrides={"tp_size": 2},
            )
        self.assertIn("collides", str(ctx.exception).lower())

    def test_tp2_talker_gpu2_accepted(self):
        """--tp-size 2 with talker on GPU 2 should be valid."""
        from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

        config = MingOmniSpeechPipelineConfig(
            model_path="test/model",
            gpu_placement={"thinker": 0, "talker": 2},
            server_args_overrides={"tp_size": 2},
        )
        self.assertEqual(config.gpu_placement["talker"], 2)

    def test_tp1_default_accepted(self):
        """TP=1 with default placement must work."""
        from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig

        config = MingOmniSpeechPipelineConfig(
            model_path="test/model",
        )
        self.assertEqual(config.gpu_placement["thinker"], 0)
        self.assertEqual(config.gpu_placement["talker"], 1)


try:
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

    _qwen3_available = True
except ImportError:
    _qwen3_available = False


@unittest.skipUnless(_qwen3_available, "qwen3_omni config not importable (missing av?)")
class TestQwen3OmniSpeechGPUValidation(unittest.TestCase):
    """Validate Qwen3 speech config TP/speech-stage GPU collision."""

    def test_tp2_default_gpus_rejected(self):
        """Default placement (talker_ar=1) collides with thinker TP rank 1."""
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

        with self.assertRaises(ValueError) as ctx:
            Qwen3OmniSpeechPipelineConfig(
                model_path="test/model",
                server_args_overrides={"tp_size": 2},
            )
        self.assertIn("collides", str(ctx.exception).lower())

    def test_tp2_speech_on_gpu2_accepted(self):
        """All speech stages on GPU 2+ should pass with TP=2."""
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

        config = Qwen3OmniSpeechPipelineConfig(
            model_path="test/model",
            gpu_placement={
                "thinker": 0,
                "talker_ar": 2,
                "code_predictor": 2,
                "code2wav": 2,
            },
            server_args_overrides={"tp_size": 2},
        )
        self.assertEqual(config.gpu_placement["talker_ar"], 2)

    def test_tp1_default_accepted(self):
        """TP=1 with default placement must work."""
        from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

        config = Qwen3OmniSpeechPipelineConfig(
            model_path="test/model",
        )
        self.assertEqual(config.gpu_placement["thinker"], 0)
        self.assertEqual(config.gpu_placement["talker_ar"], 1)


if __name__ == "__main__":
    unittest.main()
