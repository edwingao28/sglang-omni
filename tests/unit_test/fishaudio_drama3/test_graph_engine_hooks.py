# SPDX-License-Identifier: Apache-2.0
"""graph_engines.py hooks sglang's eager attention method. Only GPU benchmark runs
start those engines, so this CPU case catches an sglang release that renames the
method or moves its query."""

from __future__ import annotations

import inspect

import torch
from sglang.srt.layers.radix_attention import RadixAttention


def test_sglang_eager_attention_takes_the_query_after_the_layer() -> None:
    """CaptureBreakRecorder stands in for RadixAttention._eager_attention: it reads
    the query from the first argument after the layer and passes it on there."""
    layer = RadixAttention(
        num_heads=2, head_dim=4, scaling=0.5, num_kv_heads=1, layer_id=0
    )
    query = torch.zeros(8, 2, 4)

    eager_attention = inspect.signature(
        RadixAttention._eager_attention  # noqa: leading-underscore  # upstream spelling
    )

    assert list(eager_attention.bind_partial(layer, query).arguments) == ["self", "q"]
