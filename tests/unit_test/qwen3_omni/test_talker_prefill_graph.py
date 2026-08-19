# SPDX-License-Identifier: Apache-2.0
"""Talker prefill CUDA graph prerequisites.

The thinker lost a full debugging round to this exact shape: the prefill graph
runner picks capture-time positions by probing ``is_mrope_enabled`` on the
model, and when that disagreed with what ``forward()`` feeds, MRotaryEmbedding
fell through to ``forward_native``. That path rebuilds query and key with
``torch.cat``, and those out-of-place tensors do not survive a breakable
graph-segment boundary — replay then ran attention on an all-zero query while
value (a view into the qkv buffer) stayed live, producing fluent but
prompt-independent speech at exactly the right duration.

The talker's backbone is MRoPE too (``mrope_section`` in its text config), so
it would reproduce the same failure. These tests pin the two prerequisites:
the model advertises MRoPE, and the runner therefore hands the captured body
the same positions serving uses.
"""

from __future__ import annotations

from types import SimpleNamespace


def test_talker_advertises_mrope_from_its_backbone() -> None:
    """Derived from the real rotary_emb instances, never hardcoded."""
    import inspect

    from sglang_omni.models.qwen3_omni.components import talker as talker_mod

    src = inspect.getsource(talker_mod)
    assert "self.is_mrope_enabled = self._uses_mrope" in src, (
        "talker must advertise MRoPE to the prefill graph runner, derived from "
        "whether its layers actually use MRotaryEmbedding"
    )


def test_capture_positions_match_serving_positions() -> None:
    """The runner must hand the captured body the positions forward() uses."""
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )

    runner = object.__new__(PrefillCudaGraphRunner)
    model = SimpleNamespace(is_mrope_enabled=True)
    runner.model_runner = SimpleNamespace(model=model)

    plain, mrope = object(), object()
    batch = SimpleNamespace(positions=plain, mrope_positions=mrope)
    assert runner._get_layer_model_positions(batch) is mrope

    # A batch without MRoPE positions still falls back to the 1-D ones.
    no_mrope = SimpleNamespace(positions=plain, mrope_positions=None)
    assert runner._get_layer_model_positions(no_mrope) is plain


def test_a_model_not_advertising_mrope_would_capture_the_wrong_path() -> None:
    """Guards the regression itself: without the flag, capture picks 1-D."""
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )

    runner = object.__new__(PrefillCudaGraphRunner)
    runner.model_runner = SimpleNamespace(model=SimpleNamespace())
    plain, mrope = object(), object()
    batch = SimpleNamespace(positions=plain, mrope_positions=mrope)
    assert runner._get_layer_model_positions(batch) is plain
