# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TP follower worker."""
from __future__ import annotations

import pickle
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


class TestFollowerRegistration(unittest.TestCase):
    """Test that register_omni_models delegates to shared registry helper."""

    def test_register_delegates_to_shared_helper(self):
        from sglang_omni.engines.tp.follower import register_omni_models

        with patch(
            "sglang_omni.models.sglang_registry.register_omni_models_in_sglang"
        ) as mock_reg:
            register_omni_models()
            mock_reg.assert_called_once()

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

    def test_relocate_batch_tensors_moves_model_specific_data(self):
        """relocate_batch_tensors traverses model_specific_data tensor entries."""
        import torch

        from sglang_omni.engines.tp.follower import relocate_batch_tensors

        item = types.SimpleNamespace(
            model_specific_data={"patch_pixel_values": torch.tensor([1, 2, 3])}
        )
        mm = types.SimpleNamespace(mm_items=[item])
        batch = types.SimpleNamespace(multimodal_inputs=[mm])

        with patch.object(torch.Tensor, "to", autospec=True) as mock_to:
            mock_to.side_effect = lambda self, *args, **kwargs: self
            relocate_batch_tensors(batch, torch.device("meta"))

        self.assertEqual(mock_to.call_count, 1)


    def test_sync_page_table_writes_to_pool(self):
        """sync_page_table writes snapshot rows into the follower's req_to_token_pool."""
        import torch

        from sglang_omni.engines.tp.follower import sync_page_table

        pool = types.SimpleNamespace()
        pool.req_to_token = torch.zeros((4, 64), dtype=torch.int32)

        batch = types.SimpleNamespace()
        batch.req_pool_indices = torch.tensor([2])
        batch.seq_lens = torch.tensor([3])
        batch.tp_page_table_rows = [
            torch.tensor([10, 11, 12], dtype=torch.int32),
        ]

        sync_page_table(batch, pool)

        expected = torch.tensor([10, 11, 12], dtype=torch.int32)
        assert torch.equal(pool.req_to_token[2, 0:3], expected)

    def test_sync_page_table_noop_without_snapshot(self):
        """sync_page_table is a no-op when tp_page_table_rows is absent."""
        import torch

        from sglang_omni.engines.tp.follower import sync_page_table

        pool = types.SimpleNamespace()
        pool.req_to_token = torch.zeros((4, 64), dtype=torch.int32)
        batch = types.SimpleNamespace()
        batch.req_pool_indices = torch.tensor([0])
        batch.seq_lens = torch.tensor([1])

        sync_page_table(batch, pool)

        assert pool.req_to_token.sum() == 0


if __name__ == "__main__":
    unittest.main()
