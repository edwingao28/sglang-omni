# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure ``_snap_step_t`` staticmethod."""

from sglang_omni.models.moss_tts_local.streaming_vocoder import (
    MossTTSLocalStreamingVocoderScheduler as S,
)

CAPTURED = [4, 5, 8, 9, 10, 11, 12, 13, 20, 22, 24, 25]


def test_snap_keeps_exactly_captured_lengths() -> None:
    for t in CAPTURED:
        assert S._snap_step_t(t, CAPTURED) == t


def test_snap_rounds_non_captured_length_down_to_nearest_graph() -> None:
    assert S._snap_step_t(7, CAPTURED) == 5
    assert S._snap_step_t(19, CAPTURED) == 13
    assert S._snap_step_t(21, CAPTURED) == 20
    assert S._snap_step_t(23, CAPTURED) == 22


def test_snap_caps_long_step_to_largest_captured() -> None:
    assert S._snap_step_t(100, CAPTURED) == 25


def test_snap_never_raises_step_t_above_available_frames() -> None:
    assert S._snap_step_t(3, CAPTURED) == 3
    assert S._snap_step_t(1, CAPTURED) == 1


def test_snap_is_noop_without_captured_graphs() -> None:
    assert S._snap_step_t(25, []) == 25
    assert S._snap_step_t(7, []) == 7
