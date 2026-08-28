# SPDX-License-Identifier: Apache-2.0
"""Predictor-graph miss logging: "capturing" must stay silent (it is the
designed path into the outer whole-Talker graph), the alarming misses must warn.
"""

from __future__ import annotations

import logging

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker


@pytest.fixture()
def talker(monkeypatch) -> Qwen3OmniTalker:
    # Note (wenyao): torch.cuda is stubbed so the reason table is exercised
    # identically on CPU-only hosts.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    obj = object.__new__(Qwen3OmniTalker)
    obj._predictor_decode_graph_batch_sizes = (1, 2, 4, 8)
    obj._predictor_decode_graph_logged_misses = set()
    return obj


def _codes(dtype: torch.dtype = torch.long) -> torch.Tensor:
    return torch.zeros((2, 1), dtype=dtype)


def _hidden() -> torch.Tensor:
    return torch.zeros((2, 1, 8), dtype=torch.float32)


def test_multi_token_is_a_by_design_skip(talker) -> None:
    assert (
        talker._predictor_decode_graph_skip_reason(
            layer0_codes=_codes(), talker_hidden=_hidden(), seq_len=4
        )
        == "multi_token"
    )


def test_capturing_wins_over_tensor_checks(talker, monkeypatch) -> None:
    # Note (wenyao): "capturing" must be checked before the tensor predicates,
    # or a dtype quirk under capture would warn on the designed path.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    assert (
        talker._predictor_decode_graph_skip_reason(
            layer0_codes=_codes(torch.float32), talker_hidden=_hidden(), seq_len=1
        )
        == "capturing"
    )


def test_non_cuda_tensor_is_reported(talker) -> None:
    assert (
        talker._predictor_decode_graph_skip_reason(
            layer0_codes=_codes(), talker_hidden=_hidden(), seq_len=1
        )
        == "non_cuda_tensor"
    )


def test_code_dtype_is_reported(talker) -> None:
    assert (
        talker._predictor_decode_graph_skip_reason(
            layer0_codes=_codes(torch.float32), talker_hidden=_hidden(), seq_len=1
        )
        == "code_dtype"
    )


def test_no_cuda_is_a_by_design_skip(talker, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert (
        talker._predictor_decode_graph_skip_reason(
            layer0_codes=_codes(), talker_hidden=_hidden(), seq_len=1
        )
        == "no_cuda"
    )


@pytest.mark.parametrize("reason", ["capturing", "multi_token", "no_cuda"])
def test_by_design_reasons_never_warn(talker, caplog, reason) -> None:
    with caplog.at_level(logging.WARNING):
        talker._log_predictor_graph_miss(reason, 2)

    assert not [r for r in caplog.records if "[predictor-graph miss]" in r.getMessage()]


@pytest.mark.parametrize("reason", ["bucket_overflow", "code_dtype", "non_cuda_tensor"])
def test_alarming_reasons_warn_once_per_shape(talker, caplog, reason) -> None:
    with caplog.at_level(logging.WARNING):
        talker._log_predictor_graph_miss(reason, 12)
        talker._log_predictor_graph_miss(reason, 12)
        talker._log_predictor_graph_miss(reason, 16)

    misses = [r for r in caplog.records if "[predictor-graph miss]" in r.getMessage()]
    assert len(misses) == 2
    assert f"reason={reason}" in misses[0].getMessage()
    assert "batch_size=12" in misses[0].getMessage()
    assert "batch_size=16" in misses[1].getMessage()
    assert "[1, 2, 4, 8]" in misses[0].getMessage()


def test_bucket_overflow_returns_none_and_warns(talker, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        result = Qwen3OmniTalker._code_predictor_forward_single_token_graph(
            talker,
            layer0_codes=_codes(),
            talker_hidden=_hidden(),
            batch_size=64,
            code_dtype=torch.long,
        )

    assert result is None
    misses = [r for r in caplog.records if "[predictor-graph miss]" in r.getMessage()]
    assert len(misses) == 1
    assert "reason=bucket_overflow" in misses[0].getMessage()
