# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Ming streaming text segmentation."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.ming_omni.components.streaming_text import (
    SegmenterConfig,
    SegmenterState,
    split_whitespace_tokens,
    text_to_uint8_tensor,
    uint8_tensor_to_text,
)


def _count_words(text: str) -> int:
    return len(text.split())


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello streaming tts",
        "你好，世界！",
        "first sentence. second sentence?",
    ],
)
def test_utf8_tensor_roundtrip(text: str) -> None:
    tensor = text_to_uint8_tensor(text)

    assert tensor.dtype == torch.uint8
    assert tensor.device.type == "cpu"
    assert uint8_tensor_to_text(tensor) == text


def test_uint8_tensor_rejects_wrong_dtype() -> None:
    with pytest.raises(TypeError, match="torch.uint8"):
        uint8_tensor_to_text(torch.tensor([1, 2, 3], dtype=torch.int64))


def test_split_whitespace_tokens_caps_window() -> None:
    left, right = split_whitespace_tokens("one two three four five", 3)

    assert left == "one two three "
    assert right == "four five"


def test_split_whitespace_tokens_preserves_original_text() -> None:
    text = "one  two\tthree\nfour five"

    left, right = split_whitespace_tokens(text, 3)

    assert left == "one  two\tthree\n"
    assert right == "four five"
    assert left + right == text


def test_first_segment_timeout_emits_early() -> None:
    state = SegmenterState(
        SegmenterConfig(
            segment_min_tokens=8,
            segment_max_tokens=40,
            first_segment_min_tokens=4,
            first_segment_max_wait_ms=450,
        ),
        _count_words,
    )

    assert state.push("one two three four", now_ms=0) == []
    segments = state.push("", now_ms=451)

    assert len(segments) == 1
    assert segments[0].segment_id == 0
    assert segments[0].text == "one two three four"
    assert segments[0].is_final_segment is False


@pytest.mark.parametrize(
    "text",
    [
        "one two three.",
        "one two three,",
        "one two three，",
        "one two three\n",
    ],
)
def test_punctuation_boundary_emits_after_min_tokens(text: str) -> None:
    state = SegmenterState(SegmenterConfig(segment_min_tokens=3), _count_words)

    segments = state.push(text, now_ms=10)

    assert [(s.segment_id, s.text, s.is_final_segment) for s in segments] == [
        (0, text, False)
    ]


def test_max_tokens_emits_window_and_keeps_remainder() -> None:
    state = SegmenterState(
        SegmenterConfig(segment_min_tokens=2, segment_max_tokens=3),
        _count_words,
    )

    segments = state.push("one two three four five", now_ms=20)

    assert [(s.segment_id, s.text, s.is_final_segment) for s in segments] == [
        (0, "one two three ", False)
    ]
    assert state.flush()[0].text == "four five"


def test_large_push_emits_all_ready_max_windows() -> None:
    state = SegmenterState(
        SegmenterConfig(segment_min_tokens=2, segment_max_tokens=3),
        _count_words,
    )

    segments = state.push("one two three four five six seven", now_ms=25)

    assert [(s.segment_id, s.text, s.is_final_segment) for s in segments] == [
        (0, "one two three ", False),
        (1, "four five six ", False),
    ]
    assert state.flush()[0].text == "seven"


def test_flush_marks_final_segment() -> None:
    state = SegmenterState(SegmenterConfig(), _count_words)

    assert state.push("trailing text without punctuation", now_ms=0) == []
    segments = state.flush()

    assert len(segments) == 1
    assert segments[0].text == "trailing text without punctuation"
    assert segments[0].is_final_segment is True


def test_whitespace_only_flush_is_ignored_by_scheduler_contract() -> None:
    state = SegmenterState(SegmenterConfig(), _count_words)

    assert state.push("   ", now_ms=0) == []
    assert state.buffer_token_count() == 0
    assert state.flush() == []
    assert state.flush() == []
