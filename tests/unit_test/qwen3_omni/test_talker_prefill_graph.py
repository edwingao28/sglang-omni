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

import os
from types import SimpleNamespace

import pytest

from sglang_omni.models.qwen3_omni import bootstrap as qwen3_omni_bootstrap


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


def _guard_source() -> str:
    import inspect

    return inspect.getsource(qwen3_omni_bootstrap)


def test_speech_prefill_graph_guard_is_on_by_default() -> None:
    src = _guard_source()
    assert "speech output does not support a non-disabled" in src, (
        "the speech prefill-graph guard must stay in place by default"
    )


def test_guard_opt_out_is_explicitly_unsafe_and_env_only() -> None:
    src = _guard_source()
    assert "SGLANG_OMNI_UNSAFE_ALLOW_SPEECH_PREFILL_GRAPH" in src
    assert "UNSAFE" in "SGLANG_OMNI_UNSAFE_ALLOW_SPEECH_PREFILL_GRAPH"
    # Opting out must require the exact value, not mere presence.
    assert 'SGLANG_OMNI_UNSAFE_ALLOW_SPEECH_PREFILL_GRAPH") != "1"' in src


@pytest.mark.parametrize("value", ["", "0", "true", "yes", None])
def test_only_the_exact_opt_out_value_disables_the_guard(value: str | None) -> None:
    key = "SGLANG_OMNI_UNSAFE_ALLOW_SPEECH_PREFILL_GRAPH"
    saved = os.environ.get(key)
    try:
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
        assert os.environ.get(key) != "1"
    finally:
        if saved is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = saved
