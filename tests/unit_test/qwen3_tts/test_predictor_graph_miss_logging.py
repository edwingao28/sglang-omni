# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS predictor-graph miss logging: the reachable misses must warn,
the by-design exits must stay silent.
"""

from __future__ import annotations

import logging

import pytest

from sglang_omni.models.qwen3_tts.sglang_model import (
    _PREDICTOR_GRAPH_MAX_KEYS,
    Qwen3TTSTalker,
)


@pytest.fixture()
def talker() -> Qwen3TTSTalker:
    obj = object.__new__(Qwen3TTSTalker)
    obj._predictor_graph_logged_misses = set()
    obj._predictor_graphs = {}
    return obj


def _misses(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if "[predictor-graph miss]" in r.getMessage()
    ]


@pytest.mark.parametrize(
    "reason",
    [
        "code_dtype",
        "non_cuda_tensor",
        "batch_size_mismatch",
        "signature_mixed",
        "bucket_overflow",
        "key_table_full",
    ],
)
def test_alarming_reasons_warn_once_per_shape(talker, caplog, reason) -> None:
    with caplog.at_level(logging.WARNING):
        assert talker._log_predictor_graph_miss(reason, 8) is None
        assert talker._log_predictor_graph_miss(reason, 8) is None
        assert talker._log_predictor_graph_miss(reason, 16) is None

    misses = _misses(caplog)
    assert len(misses) == 2
    assert f"reason={reason}" in misses[0]
    assert "batch_size=8" in misses[0]
    assert "batch_size=16" in misses[1]
    assert "model=qwen3-tts" in misses[0]


@pytest.mark.parametrize(
    "reason", ["disabled", "multi_token", "empty_batch", "capturing", "unknown"]
)
def test_by_design_reasons_never_warn(talker, caplog, reason) -> None:
    with caplog.at_level(logging.WARNING):
        assert talker._log_predictor_graph_miss(reason, 8) is None

    assert not _misses(caplog)


def test_reports_key_table_occupancy(talker, caplog) -> None:
    talker._predictor_graphs = {f"k{i}": object() for i in range(3)}

    with caplog.at_level(logging.WARNING):
        talker._log_predictor_graph_miss("key_table_full", 4)

    assert f"captured_keys=3/{_PREDICTOR_GRAPH_MAX_KEYS}" in _misses(caplog)[0]


def test_helper_always_returns_none_so_dispatch_can_return_it(talker) -> None:
    assert talker._log_predictor_graph_miss("bucket_overflow", 1) is None
    assert talker._log_predictor_graph_miss("multi_token", 1) is None
