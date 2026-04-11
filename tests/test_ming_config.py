# SPDX-License-Identifier: Apache-2.0
"""Tests for Ming pipeline config validation."""
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


if __name__ == "__main__":
    unittest.main()
