# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TP follower worker."""
from __future__ import annotations

import pickle
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


class TestFollowerRegistration(unittest.TestCase):
    def test_register_delegates_to_shared_helper(self):
        from sglang_omni.engines.tp.follower import register_omni_models

        with patch(
            "sglang_omni.models.sglang_registry.register_omni_models_in_sglang"
        ) as mock_reg:
            register_omni_models()
            mock_reg.assert_called_once()

    def test_follower_entry_is_picklable(self):
        from sglang_omni.engines.tp.follower import follower_worker_loop

        pickled = pickle.dumps(follower_worker_loop)
        restored = pickle.loads(pickled)
        self.assertEqual(restored.__name__, "follower_worker_loop")


class TestFollowerBatchFlow(unittest.TestCase):
    def _make_sanitized_batch(self):
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
        data = pickle.dumps([None])
        result = pickle.loads(data)
        self.assertIsNone(result[0])

    def _call_patch_batch(self, batch, vocab_size=32000):
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
        batch = self._make_sanitized_batch()
        self._call_patch_batch(batch)
        self.assertEqual(batch.reqs, [])

    def test_patch_batch_creates_sampling_stub(self):
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
        import torch

        batch = types.SimpleNamespace()
        batch.reqs = ["existing"]
        batch.input_ids = torch.tensor([1])
        batch.seq_lens = torch.tensor([1])
        batch.sampling_info = None
        self._call_patch_batch(batch)
        self.assertEqual(batch.reqs, ["existing"])

    def test_relocate_batch_tensors(self):
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
        import torch

        from sglang_omni.engines.tp.follower import sync_page_table

        pool = types.SimpleNamespace()
        pool.req_to_token = torch.zeros((4, 64), dtype=torch.int32)
        batch = types.SimpleNamespace()
        batch.req_pool_indices = torch.tensor([0])
        batch.seq_lens = torch.tensor([1])

        sync_page_table(batch, pool)

        assert pool.req_to_token.sum() == 0

    def test_page_table_round_trip(self):
        import pickle
        import torch

        from sglang_omni.engines.tp.follower import sync_page_table
        from sglang_omni.engines.tp.serialization import (
            attach_page_table_snapshot,
            make_follower_batch,
        )

        rank0_pool = types.SimpleNamespace()
        rank0_pool.req_to_token = torch.zeros((4, 64), dtype=torch.int32)
        rank0_pool.req_to_token[0, 0:4] = torch.tensor(
            [100, 101, 102, 103], dtype=torch.int32
        )
        rank0_pool.req_to_token[1, 0:2] = torch.tensor([200, 201], dtype=torch.int32)

        batch = types.SimpleNamespace()
        batch.req_pool_indices = torch.tensor([0, 1])
        batch.seq_lens = torch.tensor([4, 2])
        batch.input_ids = torch.tensor([10, 20])
        batch.sampling_info = MagicMock()
        batch.reqs = [MagicMock(), MagicMock()]

        attach_page_table_snapshot(batch, rank0_pool)
        follower_batch = make_follower_batch(batch)

        data = pickle.dumps(follower_batch)
        received = pickle.loads(data)

        follower_pool = types.SimpleNamespace()
        follower_pool.req_to_token = torch.zeros((4, 64), dtype=torch.int32)

        sync_page_table(received, follower_pool)

        assert torch.equal(
            follower_pool.req_to_token[0, 0:4],
            torch.tensor([100, 101, 102, 103], dtype=torch.int32),
        )
        assert torch.equal(
            follower_pool.req_to_token[1, 0:2],
            torch.tensor([200, 201], dtype=torch.int32),
        )
        assert follower_pool.req_to_token[2:].sum() == 0


class TestFollowerInputEmbeds(unittest.TestCase):
    def test_input_embeds_survives_follower_batch_round_trip(self):
        import torch

        from sglang_omni.engines.tp.serialization import make_follower_batch

        batch = types.SimpleNamespace()
        batch.input_ids = torch.tensor([1, 2, 3])
        batch.seq_lens = torch.tensor([3])
        batch.sampling_info = None
        batch.reqs = None
        batch.input_embeds = torch.randn(3, 128)

        follower = make_follower_batch(batch)
        data = pickle.dumps(follower)
        restored = pickle.loads(data)

        self.assertIsNotNone(restored.input_embeds)
        self.assertTrue(torch.equal(restored.input_embeds, batch.input_embeds))

    def test_input_embeds_none_when_not_set(self):
        import torch

        from sglang_omni.engines.tp.serialization import make_follower_batch

        batch = types.SimpleNamespace()
        batch.input_ids = torch.tensor([1, 2, 3])
        batch.seq_lens = torch.tensor([3])
        batch.sampling_info = None
        batch.reqs = None
        follower = make_follower_batch(batch)
        data = pickle.dumps(follower)
        restored = pickle.loads(data)

        embeds = getattr(restored, "input_embeds", None)
        self.assertIsNone(embeds)

    def test_relocate_moves_input_embeds(self):
        import torch

        from sglang_omni.engines.tp.follower import relocate_batch_tensors

        batch = types.SimpleNamespace()
        batch.input_embeds = torch.randn(3, 128)
        target = torch.device("cpu")
        relocate_batch_tensors(batch, target)
        self.assertEqual(batch.input_embeds.device, target)

    def test_deepstack_payload_survives_round_trip(self):
        import torch

        from sglang_omni.engines.tp.serialization import make_follower_batch

        batch = types.SimpleNamespace()
        batch.input_ids = torch.tensor([1, 2, 3])
        batch.seq_lens = torch.tensor([3])
        batch.sampling_info = None
        batch.reqs = None
        batch.input_embeds = torch.randn(3, 128)
        batch.tp_deepstack_visual_embeds = [torch.randn(2, 64), torch.randn(2, 64)]
        batch.tp_visual_pos_masks = torch.tensor([True, False, True])

        follower = make_follower_batch(batch)
        data = pickle.dumps(follower)
        restored = pickle.loads(data)

        self.assertEqual(len(restored.tp_deepstack_visual_embeds), 2)
        self.assertTrue(
            torch.equal(restored.tp_deepstack_visual_embeds[0],
                        batch.tp_deepstack_visual_embeds[0])
        )
        self.assertTrue(
            torch.equal(restored.tp_visual_pos_masks, batch.tp_visual_pos_masks)
        )

    def test_relocate_moves_deepstack_tensors(self):
        import torch

        from sglang_omni.engines.tp.follower import relocate_batch_tensors

        batch = types.SimpleNamespace()
        batch.tp_deepstack_visual_embeds = [torch.randn(2, 64), torch.randn(2, 64)]
        batch.tp_visual_pos_masks = torch.tensor([True, False])
        target = torch.device("cpu")
        relocate_batch_tensors(batch, target)
        for t in batch.tp_deepstack_visual_embeds:
            self.assertEqual(t.device, target)
        self.assertEqual(batch.tp_visual_pos_masks.device, target)


class TestFollowerShapeAttrs(unittest.TestCase):
    """Shape primitives let follower alloc NCCL receive buffers pre-broadcast."""

    def test_shape_attrs_survive_pickle_round_trip(self):
        import torch

        from sglang_omni.engines.tp.serialization import make_follower_batch

        batch = types.SimpleNamespace()
        batch.input_ids = torch.tensor([1, 2, 3])
        batch.seq_lens = torch.tensor([3])
        batch.sampling_info = None
        batch.reqs = None
        batch.tp_input_embeds_shape = (3, 128)
        batch.tp_input_embeds_dtype = torch.bfloat16
        batch.tp_deepstack_shapes = [(2, 128), (2, 128)]
        batch.tp_deepstack_dtype = torch.bfloat16
        batch.tp_visual_pos_mask_shape = (3,)

        follower = make_follower_batch(batch)
        data = pickle.dumps(follower)
        restored = pickle.loads(data)

        self.assertEqual(restored.tp_input_embeds_shape, (3, 128))
        self.assertEqual(restored.tp_input_embeds_dtype, torch.bfloat16)
        self.assertEqual(restored.tp_deepstack_shapes, [(2, 128), (2, 128)])
        self.assertEqual(restored.tp_deepstack_dtype, torch.bfloat16)
        self.assertEqual(restored.tp_visual_pos_mask_shape, (3,))

    def test_shape_attrs_absent_for_text_only(self):
        import torch

        from sglang_omni.engines.tp.serialization import make_follower_batch

        batch = types.SimpleNamespace()
        batch.input_ids = torch.tensor([1, 2, 3])
        batch.seq_lens = torch.tensor([3])
        batch.sampling_info = None
        batch.reqs = None

        follower = make_follower_batch(batch)
        data = pickle.dumps(follower)
        restored = pickle.loads(data)

        self.assertIsNone(getattr(restored, "tp_input_embeds_shape", None))
        self.assertIsNone(getattr(restored, "tp_deepstack_shapes", None))
        self.assertIsNone(getattr(restored, "tp_visual_pos_mask_shape", None))

    def test_shape_attrs_with_only_input_embeds(self):
        import torch

        from sglang_omni.engines.tp.serialization import make_follower_batch

        batch = types.SimpleNamespace()
        batch.input_ids = torch.tensor([1, 2, 3])
        batch.seq_lens = torch.tensor([3])
        batch.sampling_info = None
        batch.reqs = None
        batch.tp_input_embeds_shape = (3, 64)
        batch.tp_input_embeds_dtype = torch.float16
        batch.tp_deepstack_shapes = None
        batch.tp_deepstack_dtype = None
        batch.tp_visual_pos_mask_shape = None

        follower = make_follower_batch(batch)
        restored = pickle.loads(pickle.dumps(follower))

        self.assertEqual(restored.tp_input_embeds_shape, (3, 64))
        self.assertIsNone(restored.tp_deepstack_shapes)
        self.assertIsNone(restored.tp_visual_pos_mask_shape)


class TestFollowerGpuAssignment(unittest.TestCase):
    def test_gpu_id_formula_step_1(self):
        base, step = 0, 1
        self.assertEqual(base + 1 * step, 1)
        self.assertEqual(base + 2 * step, 2)

    def test_gpu_id_formula_step_2(self):
        base, step = 0, 2
        self.assertEqual(base + 1 * step, 2)
        self.assertEqual(base + 2 * step, 4)

    def test_gpu_id_formula_nonzero_base(self):
        base, step = 4, 2
        self.assertEqual(base + 1 * step, 6)

    def test_spawn_followers_passes_model_config(self):
        from sglang_omni.engines.tp.follower import spawn_followers

        started = []

        class DummyProcess:
            def __init__(self, target, args, daemon):
                self.target = target
                self.args = args
                self.daemon = daemon
                self.pid = 12345

            def start(self):
                started.append(self.args)

        class DummyContext:
            def Process(self, target, args, daemon):
                return DummyProcess(target, args, daemon)

        server_args = types.SimpleNamespace(gpu_id_step=1)
        with patch("sglang_omni.engines.tp.follower.mp.get_context", return_value=DummyContext()):
            processes = spawn_followers(
                server_args=server_args,
                nccl_port=23456,
                base_gpu_id=0,
                tp_size=2,
                model_arch_override="BailingMoeV2ForCausalLM",
                weight_prefix="thinker.",
            )

        self.assertEqual(len(processes), 1)
        self.assertEqual(
            started[0],
            (
                1,
                1,
                server_args,
                23456,
                "BailingMoeV2ForCausalLM",
                "thinker.",
            ),
        )


if __name__ == "__main__":
    unittest.main()
