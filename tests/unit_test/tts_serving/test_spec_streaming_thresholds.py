# SPDX-License-Identifier: Apache-2.0
"""Tests for streaming TTFA threshold fields on BenchmarkParams."""

from benchmarks.tts_serving.spec import BenchmarkParams


def test_streaming_threshold_defaults_are_conservative() -> None:
    params = BenchmarkParams.from_obj({})

    assert params.streaming_ttfa_max_ratio == 1.0
    assert params.streaming_ttfa_target_ratio == 0.8
    assert params.streaming_ttfa_min_high_concurrency == 32


def test_streaming_threshold_values_parse_from_params() -> None:
    params = BenchmarkParams.from_obj(
        {
            "streaming_ttfa_max_ratio": 1.25,
            "streaming_ttfa_target_ratio": 0.75,
            "streaming_ttfa_min_high_concurrency": 64,
        }
    )

    assert params.streaming_ttfa_max_ratio == 1.25
    assert params.streaming_ttfa_target_ratio == 0.75
    assert params.streaming_ttfa_min_high_concurrency == 64
