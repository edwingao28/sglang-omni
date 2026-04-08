# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TP follower worker."""
from __future__ import annotations

import pickle
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


def _make_mock_thinker_module():
    """Build a minimal mock for sglang_omni.models.ming_omni.thinker."""
    mod = types.ModuleType("sglang_omni.models.ming_omni.thinker")
    mod.BailingMM2Config = MagicMock(name="BailingMM2Config")
    mod.BailingMoeV2ForCausalLM = MagicMock(name="BailingMoeV2ForCausalLM")
    return mod


class TestFollowerRegistration(unittest.TestCase):
    """Test that register_omni_models works and is picklable."""

    def test_register_adds_to_model_registry(self):
        from sglang_omni.engines.tp.follower import register_omni_models

        mock_registry = MagicMock()
        mock_registry.models = {}

        mock_thinker = _make_mock_thinker_module()
        mock_auto_config = MagicMock()

        with (
            patch(
                "sglang_omni.engines.tp.follower._get_model_registry",
                return_value=mock_registry,
            ),
            patch.dict(
                sys.modules,
                {"sglang_omni.models.ming_omni.thinker": mock_thinker},
            ),
            patch(
                "sglang_omni.engines.tp.follower.AutoConfig",
                mock_auto_config,
                create=True,
            ),
        ):
            # Patch the AutoConfig import inside the function
            with patch("transformers.AutoConfig", mock_auto_config):
                register_omni_models()

        self.assertIn("BailingMoeV2ForCausalLM", mock_registry.models)

    def test_follower_entry_is_picklable(self):
        """follower_worker_loop must be picklable for mp.Process with spawn."""
        from sglang_omni.engines.tp.follower import follower_worker_loop

        pickled = pickle.dumps(follower_worker_loop)
        restored = pickle.loads(pickled)
        self.assertEqual(restored.__name__, "follower_worker_loop")


class TestFollowerBatchFlow(unittest.TestCase):
    """Tests for sanitized batch flow and follower patching."""

    def _make_sanitized_batch(self):
        """Create a batch as it would arrive at a follower (reqs/sampling None)."""
        import torch

        from sglang_omni.engines.tp.serialization import make_follower_batch

        original = types.SimpleNamespace()
        original.reqs = [MagicMock(), MagicMock()]
        original.input_ids = torch.tensor([10, 20, 30])
        original.seq_lens = torch.tensor([3])
        original.sampling_info = MagicMock()
        return make_follower_batch(original)

    def test_sanitized_batch_has_none_reqs(self):
        follower = self._make_sanitized_batch()
        self.assertIsNone(follower.reqs)

    def test_sanitized_batch_has_none_sampling_info(self):
        follower = self._make_sanitized_batch()
        self.assertIsNone(follower.sampling_info)

    def test_stop_signal_survives_pickle(self):
        """Stop signal (None inside list) must survive pickle round-trip."""
        data = pickle.dumps([None])
        result = pickle.loads(data)
        self.assertIsNone(result[0])

    def _call_patch_batch(self, batch, vocab_size=32000):
        """Call patch_batch_for_follower with SamplingBatchInfo mocked."""
        import torch

        mock_sbi_cls = MagicMock()
        mock_sbi_cls.side_effect = lambda **kw: types.SimpleNamespace(**kw)
        with (
            patch(
                "sglang_omni.engines.tp.follower.SamplingBatchInfo",
                mock_sbi_cls,
                create=True,
            ),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.sampling.sampling_batch_info": MagicMock(
                        SamplingBatchInfo=mock_sbi_cls
                    )
                },
            ),
        ):
            from sglang_omni.engines.tp.follower import patch_batch_for_follower

            device = torch.device("cpu")
            patch_batch_for_follower(batch, device, vocab_size=vocab_size)

    def test_patch_batch_fills_reqs(self):
        """patch_batch_for_follower sets reqs to empty list."""
        batch = self._make_sanitized_batch()
        self._call_patch_batch(batch)
        self.assertEqual(batch.reqs, [])

    def test_patch_batch_creates_sampling_stub(self):
        """patch_batch_for_follower creates a SamplingBatchInfo stub."""
        batch = self._make_sanitized_batch()
        self._call_patch_batch(batch, vocab_size=32000)

        si = batch.sampling_info
        self.assertIsNotNone(si)
        self.assertTrue(si.is_all_greedy)
        self.assertEqual(si.vocab_size, 32000)
        self.assertFalse(si.need_top_p_sampling)
        self.assertFalse(si.need_top_k_sampling)
        self.assertFalse(si.need_min_p_sampling)

    def test_patch_batch_preserves_existing_reqs(self):
        """patch_batch_for_follower doesn't overwrite non-None reqs."""
        import torch

        batch = types.SimpleNamespace()
        batch.reqs = ["existing"]
        batch.input_ids = torch.tensor([1])
        batch.seq_lens = torch.tensor([1])
        batch.sampling_info = None
        self._call_patch_batch(batch)
        self.assertEqual(batch.reqs, ["existing"])

    def test_relocate_batch_tensors(self):
        """relocate_batch_tensors moves tensors to target device."""
        import torch

        from sglang_omni.engines.tp.follower import relocate_batch_tensors

        batch = types.SimpleNamespace()
        batch.input_ids = torch.tensor([1, 2, 3], device="cpu")
        batch.seq_lens = torch.tensor([3], device="cpu")

        target = torch.device("cpu")
        relocate_batch_tensors(batch, target)
        self.assertEqual(batch.input_ids.device, target)
        self.assertEqual(batch.seq_lens.device, target)


if __name__ == "__main__":
    unittest.main()
