# SPDX-License-Identifier: Apache-2.0
"""Talker overlap disable is gated by enable_async_decode.

Feedback-enabled talkers force `disable_overlap_schedule` by default;
enable_async_decode=True lets an experiment keep the caller's value.
CUDA-graph handling is unaffected either way.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.models.qwen3_omni.talker_scheduler import configure_talker_server_args


def _server_args(*, disable_overlap_schedule=False, disable_cuda_graph=False):
    return SimpleNamespace(
        disable_overlap_schedule=disable_overlap_schedule,
        disable_cuda_graph=disable_cuda_graph,
        disable_radix_cache=False,
        chunked_prefill_size=8192,
    )


def test_default_forces_overlap_disabled():
    args = _server_args()
    want_cuda_graph = configure_talker_server_args(args, feedback_enabled=True)
    assert args.disable_overlap_schedule is True
    assert want_cuda_graph is True
    assert args.disable_cuda_graph is True


@pytest.mark.parametrize("caller_value", [True, False])
def test_async_decode_keeps_caller_overlap_value(caller_value):
    args = _server_args(disable_overlap_schedule=caller_value)
    want_cuda_graph = configure_talker_server_args(
        args, feedback_enabled=True, enable_async_decode=True
    )
    assert args.disable_overlap_schedule is caller_value
    assert want_cuda_graph is True
    assert args.disable_cuda_graph is True


def test_async_decode_leaves_cuda_graph_disabled_when_not_requested():
    args = _server_args(disable_cuda_graph=True)
    want_cuda_graph = configure_talker_server_args(
        args, feedback_enabled=True, enable_async_decode=True
    )
    assert want_cuda_graph is False
    assert args.disable_cuda_graph is True


def test_async_decode_false_forces_overlap_disabled():
    args = _server_args()
    configure_talker_server_args(args, feedback_enabled=True, enable_async_decode=False)
    assert args.disable_overlap_schedule is True


@pytest.mark.parametrize("enable_async_decode", [False, True])
def test_feedback_disabled_ignores_async_decode(enable_async_decode):
    args = _server_args()
    want_cuda_graph = configure_talker_server_args(
        args, feedback_enabled=False, enable_async_decode=enable_async_decode
    )
    assert args.disable_overlap_schedule is False
    assert args.disable_cuda_graph is False
    assert want_cuda_graph is True
    assert args.disable_radix_cache is True
    assert args.chunked_prefill_size == 0
