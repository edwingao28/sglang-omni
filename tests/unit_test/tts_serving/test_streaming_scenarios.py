# SPDX-License-Identifier: Apache-2.0
"""Tests for the raw PCM streaming speech benchmark scenario."""

from benchmarks.tts_serving.scenarios import build_scenarios
from benchmarks.tts_serving.spec import BenchmarkSpec


def _spec_with_endpoints(enabled_endpoints: list[str]) -> BenchmarkSpec:
    return BenchmarkSpec.from_obj(
        {
            "base_url": "http://localhost:8000",
            "model_name": "test-model",
            "test_type": "external",
            "params": {
                "enabled_endpoints": enabled_endpoints,
                "load_stages": [
                    {
                        "id": "c1",
                        "mode": "closed_loop",
                        "request_count": 8,
                        "max_concurrency": 1,
                        "enabled_endpoints": enabled_endpoints,
                    }
                ],
            },
        }
    )


def test_speech_stream_audio_scenario_uses_raw_pcm_transport() -> None:
    spec = _spec_with_endpoints(["speech_stream_audio"])

    scenarios = build_scenarios(spec)
    scenario = next(s for s in scenarios if s.endpoint == "speech_stream_audio")

    assert scenario.path == "/v1/audio/speech"
    assert scenario.payload["stream"] is True
    assert scenario.payload["stream_format"] == "audio"
    assert scenario.payload["response_format"] == "pcm"
    assert scenario.planned_metadata["streaming_mode"] == "stream_audio"
    assert scenario.planned_metadata["response_format"] == "pcm"
