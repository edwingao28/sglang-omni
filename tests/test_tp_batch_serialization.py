# SPDX-License-Identifier: Apache-2.0
"""Regression tests for TP follower batch serialization (pickle safety)."""
import pickle
import types
import unittest
import weakref

import torch

from sglang_omni.engines.tp.serialization import make_follower_batch


def _make_mock_batch():
    """Create a mock ModelWorkerBatch-like object with unpicklable fields."""
    batch = types.SimpleNamespace()
    batch.input_ids = torch.tensor([1, 2, 3])
    batch.seq_lens = torch.tensor([3])

    # sampling_info with weakref (the actual bug trigger)
    # Need a class that supports weakrefs (object/list don't in Python 3.12+)
    class _Referent:
        pass

    target = _Referent()
    penalizer = types.SimpleNamespace(batch_ref=weakref.ref(target))
    batch.sampling_info = types.SimpleNamespace(
        penalizer_orchestrator=penalizer,
        sampling_info_done=None,
    )

    # reqs with callback
    batch.reqs = [lambda: None]
    return batch


class TestMakeFollowerBatch(unittest.TestCase):
    def test_make_follower_batch_is_pickle_safe(self):
        batch = _make_mock_batch()
        follower = make_follower_batch(batch)
        data = pickle.dumps(follower)
        restored = pickle.loads(data)
        # Verify tensors survive round-trip
        self.assertTrue(torch.equal(restored.input_ids, batch.input_ids))
        self.assertTrue(torch.equal(restored.seq_lens, batch.seq_lens))
        # Verify stripped fields stay None
        self.assertIsNone(restored.sampling_info)
        self.assertIsNone(restored.reqs)

    def test_make_follower_batch_preserves_tensors(self):
        batch = _make_mock_batch()
        follower = make_follower_batch(batch)
        self.assertTrue(torch.equal(follower.input_ids, batch.input_ids))
        self.assertTrue(torch.equal(follower.seq_lens, batch.seq_lens))

    def test_make_follower_batch_nulls_unsafe_fields(self):
        batch = _make_mock_batch()
        follower = make_follower_batch(batch)
        self.assertIsNone(follower.sampling_info)
        self.assertIsNone(follower.reqs)

    def test_make_follower_batch_leaves_original_intact(self):
        batch = _make_mock_batch()
        original_sampling_info = batch.sampling_info
        original_reqs = batch.reqs
        make_follower_batch(batch)
        self.assertIs(batch.sampling_info, original_sampling_info)
        self.assertIs(batch.reqs, original_reqs)

    def test_stop_signal_broadcast_still_works(self):
        data = pickle.dumps(None)
        self.assertIsNone(pickle.loads(data))

    def test_new_unpicklable_field_raises_clear_error(self):
        """Verify the runtime safety net catches new unpicklable fields."""
        import sglang_omni.engines.tp.serialization as ser

        # Reset verification cache so it re-runs
        ser._pickle_verified = False

        batch = _make_mock_batch()
        # Add a new unpicklable field that isn't in _FIELDS_TO_STRIP
        batch.bad_callback = lambda: None

        with self.assertRaises(RuntimeError) as ctx:
            make_follower_batch(batch)
        self.assertIn("bad_callback", str(ctx.exception))

        # Reset for other tests
        ser._pickle_verified = False


if __name__ == "__main__":
    unittest.main()
