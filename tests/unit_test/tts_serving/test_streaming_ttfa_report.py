# SPDX-License-Identifier: Apache-2.0
from benchmarks.tts_serving.metrics import ScenarioResult
from benchmarks.tts_serving.report import build_results_report
from benchmarks.tts_serving.spec import BenchmarkSpec


def _spec() -> BenchmarkSpec:
    return BenchmarkSpec.from_obj(
        {
            "base_url": "http://localhost:8000",
            "model_name": "test-model",
            "test_type": "external",
            "params": {
                "streaming_ttfa_max_ratio": 1.0,
                "streaming_ttfa_target_ratio": 0.8,
                "streaming_ttfa_min_high_concurrency": 32,
                "enabled_endpoints": ["speech", "speech_sse", "speech_stream_audio"],
                "load_stages": [
                    {
                        "id": "c64",
                        "mode": "burst",
                        "request_count": 3,
                        "max_concurrency": 64,
                        "enabled_endpoints": [
                            "speech",
                            "speech_sse",
                            "speech_stream_audio",
                        ],
                    }
                ],
            },
        }
    )


def _result(
    scenario_id: str,
    endpoint: str,
    *,
    latency_s: float,
    ttfa_s: float | None,
    concurrency: int = 64,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=scenario_id,
        endpoint=endpoint,
        category=endpoint,
        capability_key=f"{endpoint}.test",
        stage_id="c64",
        load_mode="burst",
        load_concurrency=concurrency,
        configured_max_concurrency=concurrency,
        status="success",
        success=True,
        capability="pass",
        latency_s=latency_s,
        ttfa_s=ttfa_s,
        response_format="pcm",
        audio_duration_s=1.0,
        rtf=latency_s,
    )


def test_streaming_ttfa_analysis_flags_high_concurrency_inversion() -> None:
    report = build_results_report(
        _spec(),
        [
            _result("nonstream", "speech", latency_s=2.0, ttfa_s=None),
            _result("sse", "speech_sse", latency_s=4.0, ttfa_s=2.5),
            _result("audio", "speech_stream_audio", latency_s=4.0, ttfa_s=1.2),
        ],
    )

    analysis = report["streaming_ttfa_analysis"]

    assert analysis["status"] == "fail"
    assert analysis["worst_ratio"] == 1.25
    assert analysis["thresholds"]["max_ratio"] == 1.0
    assert analysis["thresholds"]["target_ratio"] == 0.8
    assert analysis["thresholds"]["min_high_concurrency"] == 32
    assert analysis["rows"][0]["stage_id"] == "c64"
    assert analysis["rows"][0]["stream_endpoint"] == "speech_sse"
    assert analysis["rows"][0]["ratio"] == 1.25
    assert analysis["rows"][0]["status"] == "fail"


def test_streaming_ttfa_analysis_is_not_applicable_without_high_concurrency() -> None:
    low = _result("sse", "speech_sse", latency_s=1.0, ttfa_s=0.4, concurrency=1)

    report = build_results_report(_spec(), [low])

    assert report["streaming_ttfa_analysis"]["status"] == "not_applicable"
    assert report["streaming_ttfa_analysis"]["rows"] == []
