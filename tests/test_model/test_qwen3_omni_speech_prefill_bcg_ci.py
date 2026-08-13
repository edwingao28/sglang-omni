# SPDX-License-Identifier: Apache-2.0
"""H100 qualification gate for Qwen3-Omni speech prefill BPCG.

The unpadded eager, shape-matched qualification replay, and captured-graph
servers run sequentially so the test never holds two Qwen3-Omni models on the
same H100. Qualification replay retains graph admission, bucket padding,
static buffers, and live serving metadata while eagerly executing the exact
captured body. Runtime counters prove actual replay admission; the opt-in
debug snapshot compares real thinker logits and logical layer-0/layer-24 rows
without exposing tensors in normal deployments.
"""

from __future__ import annotations

import base64
import json
import shlex
import time
from pathlib import Path
from typing import Any

import pytest
import requests
import torch
import yaml
from PIL import Image

from benchmarks.tts_serving.audio_validation import validate_audio_response
from sglang_omni.scheduling.generation_batch_policy import (
    build_default_prefill_cuda_graph_bs,
)
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    launch_managed_router,
)
from tests.test_model.qwen3_omni_speech_prefill_bcg_assertions import (
    assert_exact_prefill_embeddings,
    assert_prefill_parity,
    decode_tensor,
    prefill_parity_diagnostics,
    snapshot,
    snapshot_logical_rows,
)
from tests.utils import disable_proxy, wait_for_gpu_memory_release

from .conftest import QWEN3_OMNI_MODEL_NAME, QWEN3_OMNI_TEST_MODEL_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_CONFIG = (
    PROJECT_ROOT
    / "examples"
    / "configs"
    / "qwen3_omni_colocated_h100_bf16_speech_prefill_graph.yaml"
)
AUDIO_FIXTURE = PROJECT_ROOT / "tests" / "data" / "cough.wav"
IMAGE_FIXTURE = PROJECT_ROOT / "tests" / "data" / "cars.jpg"
GRAPH_CAP = 2048
THINKER_CONTEXT = 8192
REQUEST_TIMEOUT = 600

pytestmark = [pytest.mark.benchmark, pytest.mark.gpu]


def _materialize_eager_config(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Build an otherwise-identical, radix-disabled eager diagnostic profile."""
    config = yaml.safe_load(GRAPH_CONFIG.read_text(encoding="utf-8"))
    config["name"] = "qwen3-omni-colocated-h100-bf16-speech-prefill-eager"
    overrides = config["runtime_overrides"]["thinker"]["server_args_overrides"]
    overrides["cuda_graph_backend_prefill"] = "disabled"
    overrides.pop("cuda_graph_bs_prefill", None)
    overrides.pop("cuda_graph_max_bs_prefill", None)
    path = tmp_path_factory.mktemp("speech_prefill_bcg_eager_config") / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _materialize_eager_replay_config(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Build the identical graph profile used by qualification replay."""
    config = yaml.safe_load(GRAPH_CONFIG.read_text(encoding="utf-8"))
    config["name"] = "qwen3-omni-speech-prefill-eager-replay-oracle"
    path = (
        tmp_path_factory.mktemp("speech_prefill_bcg_eager_replay_config")
        / "config.yaml"
    )
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _materialize_visual_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Keep the visual fallback probe well below the thinker context limit."""
    path = tmp_path_factory.mktemp("speech_prefill_bcg_visual") / "cars-small.jpg"
    with Image.open(IMAGE_FIXTURE) as image:
        image.thumbnail((448, 448))
        image.convert("RGB").save(path, format="JPEG", quality=90)
    return path


def _worker_args(config_path: Path) -> str:
    return shlex.join(["--config", str(config_path), "--colocate"])


def _graph_info(handle: ManagedRouterHandle) -> dict[str, Any]:
    with disable_proxy():
        response = requests.post(
            f"http://127.0.0.1:{handle.worker_ports[0]}/model_info",
            json={"stages": ["thinker"], "timeout_s": 30},
            timeout=60,
        )
    response.raise_for_status()
    payload = response.json()
    thinker_items = [
        item for item in payload["stages"] if item.get("stage") == "thinker"
    ]
    assert len(thinker_items) == 1, payload
    thinker = thinker_items[0]
    assert thinker["success"], thinker
    return thinker["data"]["prefill_cuda_graph"]


def _base_payload(*, request_id: str, prompt: str) -> dict[str, Any]:
    return {
        "model": QWEN3_OMNI_MODEL_NAME,
        "request_id": request_id,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text", "audio"],
        "audio": {"format": "wav"},
        "max_tokens": 8,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": 123,
        "talker_temperature": 0.0,
        "talker_top_k": 1,
        "talker_top_p": 1.0,
        "talker_repetition_penalty": 1.0,
        "talker_max_new_tokens": 128,
        "stream": False,
    }


def _post_probe(
    handle: ManagedRouterHandle,
    payload: dict[str, Any],
) -> dict[str, Any]:
    request_started = time.perf_counter()
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(
            f"http://127.0.0.1:{handle.worker_ports[0]}/v1/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    request_elapsed_s = time.perf_counter() - request_started
    assert response.status_code == 200, response.text
    body = response.json()
    message = body["choices"][0]["message"]
    encoded_audio = message["audio"]["data"]
    wav_bytes = base64.b64decode(encoded_audio, validate=True)
    audio_validation = validate_audio_response(
        wav_bytes,
        response_format="wav",
        require_content_type=False,
    )
    assert audio_validation.ok, audio_validation.error
    assert audio_validation.duration_s > 0
    return {
        "body": body,
        "audio_duration_s": audio_validation.duration_s,
        "request_elapsed_s": request_elapsed_s,
    }


def _send_probe(
    handle: ManagedRouterHandle,
    payload: dict[str, Any],
) -> dict[str, Any]:
    before = _graph_info(handle)
    with disable_proxy():
        probe = _post_probe(handle, payload)
    probe["before"] = before
    probe["after"] = _graph_info(handle)
    return probe


def _bucket_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    buckets = set(before["replay_buckets"]) | set(after["replay_buckets"])
    return {
        bucket: int(after["replay_buckets"].get(bucket, 0))
        - int(before["replay_buckets"].get(bucket, 0))
        for bucket in buckets
        if int(after["replay_buckets"].get(bucket, 0))
        != int(before["replay_buckets"].get(bucket, 0))
    }


def _counter_delta(probe: dict[str, Any]) -> dict[str, Any]:
    before, after = probe["before"], probe["after"]
    return {
        "replay_count": int(after["replay_count"]) - int(before["replay_count"]),
        "standard_eager_count": int(after["standard_eager_count"])
        - int(before["standard_eager_count"]),
        "custom_eager_count": int(after["custom_eager_count"])
        - int(before["custom_eager_count"]),
        "replay_buckets": _bucket_delta(before, after),
    }


def _assert_single_replay(probe: dict[str, Any]) -> int:
    delta_stats = _counter_delta(probe)
    assert delta_stats["replay_count"] == 1
    assert delta_stats["standard_eager_count"] == 0
    assert delta_stats["custom_eager_count"] == 0
    delta = delta_stats["replay_buckets"]
    assert len(delta) == 1 and next(iter(delta.values())) == 1, delta
    return int(next(iter(delta)))


def _assert_single_eager_fallback(probe: dict[str, Any]) -> None:
    delta = _counter_delta(probe)
    assert delta == {
        "replay_count": 0,
        "standard_eager_count": 1,
        "custom_eager_count": 0,
        "replay_buckets": {},
    }


def _assert_custom_eager_fallback(probe: dict[str, Any]) -> None:
    """Visual/deepstack execution positively identifies the custom eager path."""
    assert _counter_delta(probe) == {
        "replay_count": 0,
        "standard_eager_count": 0,
        "custom_eager_count": 1,
        "replay_buckets": {},
    }


def _stats_without_snapshot(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if key != "debug_snapshot"}


def _safe_prefill_parity_diagnostics(
    eager_probe: dict[str, Any],
    graph_probe: dict[str, Any],
    **request_ids: str,
) -> dict[str, Any]:
    try:
        return prefill_parity_diagnostics(
            eager_probe,
            graph_probe,
            **request_ids,
        )
    except Exception as exc:  # Preserve evidence before the assertion re-raises.
        return {"error": f"{type(exc).__name__}: {exc}"}


def test_qwen3_omni_speech_prefill_bcg_replay_parity_and_fallback(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.fail("speech prefill BPCG qualification requires one CUDA GPU")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name:
        pytest.fail(
            f"speech prefill BPCG qualification requires H100, got {device_name}"
        )
    assert GRAPH_CONFIG.is_file()
    assert AUDIO_FIXTURE.is_file()
    assert IMAGE_FIXTURE.is_file()
    result_dir = tmp_path_factory.mktemp("speech_prefill_bcg")
    result_path = result_dir / "speech_prefill_bcg_results.json"
    result_path.write_text(
        json.dumps({"qualification_status": "started"}, indent=2) + "\n",
        encoding="utf-8",
    )

    process_env = {
        "PYTHONPATH": str(PROJECT_ROOT),
        "SGLANG_OMNI_PREFILL_GRAPH_DEBUG_SNAPSHOTS": "1",
        "SGLANG_OMNI_PREFILL_GRAPH_EAGER_REPLAY": "0",
    }
    eager_replay_process_env = {
        **process_env,
        "SGLANG_OMNI_PREFILL_GRAPH_EAGER_REPLAY": "1",
    }
    eager_config = _materialize_eager_config(tmp_path_factory)
    eager_replay_config = _materialize_eager_replay_config(tmp_path_factory)
    visual_fixture = _materialize_visual_fixture(tmp_path_factory)
    text_request_id = "speech-bpcg-parity-text"
    refresh_request_id = "speech-bpcg-refresh-text"
    audio_request_id = "speech-bpcg-parity-audio"
    audio_repeat_request_id = "speech-bpcg-repeat-audio"
    visual_request_id = "speech-bpcg-visual-eager-fallback"
    text_payload = _base_payload(
        request_id=text_request_id,
        prompt="Please say exactly this word aloud: cat.",
    )
    audio_payload = _base_payload(
        request_id=audio_request_id,
        prompt="Listen to the audio, then briefly say what sound you heard.",
    )
    audio_payload["audios"] = [str(AUDIO_FIXTURE.resolve())]
    audio_repeat_payload = _base_payload(
        request_id=audio_repeat_request_id,
        prompt="Listen to the audio, then briefly say what sound you heard.",
    )
    audio_repeat_payload["audios"] = [str(AUDIO_FIXTURE.resolve())]
    refresh_payload = _base_payload(
        request_id=refresh_request_id,
        prompt="Please say exactly this word aloud: dog.",
    )

    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=QWEN3_OMNI_TEST_MODEL_PATH,
        model_name=QWEN3_OMNI_MODEL_NAME,
        worker_extra_args=_worker_args(eager_config),
        num_workers=1,
        num_gpus_per_worker=1,
        wait_timeout=900,
        log_prefix="server_logs_speech_prefill_bcg_eager",
        force_log=True,
        process_env=process_env,
    ) as eager_server:
        assert eager_server.router_ready_s is not None
        eager_router_ready_s = float(eager_server.router_ready_s)
        eager_start = _graph_info(eager_server)
        assert eager_start["backend"] == "disabled"
        assert eager_start["qualification_eager_replay"] is False
        assert eager_start["upstream_debug_eager"] is False
        eager_text = _send_probe(eager_server, text_payload)
        eager_refresh = _send_probe(eager_server, refresh_payload)
        eager_audio = _send_probe(eager_server, audio_payload)

    wait_for_gpu_memory_release()

    # This is not an ordinary unpadded eager run. The test-only replay hook
    # takes the same bucket/static-buffer/live-metadata path as production,
    # then executes the runner's capture body eagerly instead of launching the
    # captured CUDA graph. It isolates capture from padded-shape model math.
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=QWEN3_OMNI_TEST_MODEL_PATH,
        model_name=QWEN3_OMNI_MODEL_NAME,
        worker_extra_args=_worker_args(eager_replay_config),
        num_workers=1,
        num_gpus_per_worker=1,
        wait_timeout=900,
        log_prefix="server_logs_speech_prefill_bcg_eager_replay",
        force_log=True,
        process_env=eager_replay_process_env,
    ) as eager_replay_server:
        assert eager_replay_server.router_ready_s is not None
        eager_replay_router_ready_s = float(eager_replay_server.router_ready_s)
        eager_replay_start = _graph_info(eager_replay_server)
        assert eager_replay_start["backend"] == "breakable"
        assert eager_replay_start["runner"] == "PrefillCudaGraphRunner"
        assert eager_replay_start["backend_runner"] == "BreakableCudaGraphBackend"
        assert eager_replay_start["qualification_eager_replay"] is True
        assert eager_replay_start["upstream_debug_eager"] is False
        assert eager_replay_start["capture_num_tokens"] == (
            build_default_prefill_cuda_graph_bs(GRAPH_CAP)
        )
        assert eager_replay_start["input_embeds_slot"] is True
        assert eager_replay_start["replay_count"] == 0
        assert eager_replay_start["standard_eager_count"] == 0
        assert eager_replay_start["custom_eager_count"] == 0

        eager_replay_text = _send_probe(eager_replay_server, text_payload)
        eager_replay_text_bucket = _assert_single_replay(eager_replay_text)
        eager_replay_refresh = _send_probe(eager_replay_server, refresh_payload)
        assert _assert_single_replay(eager_replay_refresh) == eager_replay_text_bucket
        eager_replay_audio = _send_probe(eager_replay_server, audio_payload)
        eager_replay_audio_bucket = _assert_single_replay(eager_replay_audio)
        eager_replay_end = eager_replay_audio["after"]

    wait_for_gpu_memory_release()

    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=QWEN3_OMNI_TEST_MODEL_PATH,
        model_name=QWEN3_OMNI_MODEL_NAME,
        worker_extra_args=_worker_args(GRAPH_CONFIG),
        num_workers=1,
        num_gpus_per_worker=1,
        wait_timeout=900,
        log_prefix="server_logs_speech_prefill_bcg_graph",
        force_log=True,
        process_env=process_env,
    ) as graph_server:
        assert graph_server.router_ready_s is not None
        graph_router_ready_s = float(graph_server.router_ready_s)
        graph_start = _graph_info(graph_server)
        assert graph_start["backend"] == "breakable"
        assert graph_start["runner"] == "PrefillCudaGraphRunner"
        assert graph_start["backend_runner"] == "BreakableCudaGraphBackend"
        assert graph_start["qualification_eager_replay"] is False
        assert graph_start["upstream_debug_eager"] is False
        assert graph_start["capture_num_tokens"] == (
            build_default_prefill_cuda_graph_bs(GRAPH_CAP)
        )
        assert graph_start["input_embeds_slot"] is True
        assert graph_start["replay_count"] == 0
        assert graph_start["standard_eager_count"] == 0
        assert graph_start["custom_eager_count"] == 0

        graph_text = _send_probe(graph_server, text_payload)
        text_bucket = _assert_single_replay(graph_text)

        graph_refresh = _send_probe(graph_server, refresh_payload)
        assert _assert_single_replay(graph_refresh) == text_bucket

        graph_audio = _send_probe(graph_server, audio_payload)
        audio_bucket = _assert_single_replay(graph_audio)
        graph_audio_repeat = _send_probe(graph_server, audio_repeat_payload)
        assert _assert_single_replay(graph_audio_repeat) == audio_bucket

        visual_payload = _base_payload(
            request_id=visual_request_id,
            prompt="Look at the image and briefly say what you see.",
        )
        visual_payload["images"] = [str(visual_fixture.resolve())]
        graph_visual = _send_probe(graph_server, visual_payload)
        visual_prompt_tokens = int(graph_visual["body"]["usage"]["prompt_tokens"])
        assert 0 < visual_prompt_tokens <= GRAPH_CAP

        long_payload = _base_payload(
            request_id="speech-bpcg-over-cap",
            prompt=(
                ("cat " * 2400)
                + "Now ignore the preceding filler and reply only with okay."
            ),
        )
        long_payload["max_tokens"] = 4
        long_payload["talker_max_new_tokens"] = 64
        graph_over_cap = _send_probe(graph_server, long_payload)
        long_prompt_tokens = int(graph_over_cap["body"]["usage"]["prompt_tokens"])
        assert GRAPH_CAP < long_prompt_tokens < THINKER_CONTEXT
        _assert_single_eager_fallback(graph_over_cap)
        over_cap_snapshot = graph_over_cap["after"]["debug_snapshot"]
        assert over_cap_snapshot["skipped"] == "logical_rows_exceed_limit"
        assert int(over_cap_snapshot["logical_rows"]) == long_prompt_tokens
        graph_end = graph_over_cap["after"]

    parity_diagnostics = {
        # Unpadded eager comparisons characterize deployment-visible BF16
        # shape drift but are not the graph-capture oracle: replay executes a
        # padded token bucket, which can change MoE/GEMM reduction order.
        "text_unpadded_eager": _safe_prefill_parity_diagnostics(
            eager_text,
            graph_text,
            request_id=text_request_id,
        ),
        "same_bucket_refresh_unpadded_eager": _safe_prefill_parity_diagnostics(
            eager_refresh,
            graph_refresh,
            request_id=refresh_request_id,
        ),
        "audio_unpadded_eager": _safe_prefill_parity_diagnostics(
            eager_audio,
            graph_audio,
            request_id=audio_request_id,
        ),
        # Shape-matched qualification oracle: eager replay uses the same
        # static buffers, live metadata, and 20/64-token buckets as captured
        # replay while executing the captured body eagerly.
        "text_shape_matched_eager_replay": _safe_prefill_parity_diagnostics(
            eager_replay_text,
            graph_text,
            request_id=text_request_id,
        ),
        "same_bucket_refresh_shape_matched_eager_replay": (
            _safe_prefill_parity_diagnostics(
                eager_replay_refresh,
                graph_refresh,
                request_id=refresh_request_id,
            )
        ),
        "audio_shape_matched_eager_replay": _safe_prefill_parity_diagnostics(
            eager_replay_audio,
            graph_audio,
            request_id=audio_request_id,
        ),
        "audio_captured_repeat": _safe_prefill_parity_diagnostics(
            graph_audio,
            graph_audio_repeat,
            eager_request_id=audio_request_id,
            graph_request_id=audio_repeat_request_id,
        ),
    }
    result = {
        "qualification_status": "diagnostics_collected",
        "eager_start": _stats_without_snapshot(eager_start),
        "eager_replay_start": _stats_without_snapshot(eager_replay_start),
        "eager_replay_end": _stats_without_snapshot(eager_replay_end),
        "graph_start": _stats_without_snapshot(graph_start),
        "graph_end": _stats_without_snapshot(graph_end),
        "router_ready_s": {
            "eager": eager_router_ready_s,
            "eager_replay": eager_replay_router_ready_s,
            "graph": graph_router_ready_s,
        },
        "text_bucket": text_bucket,
        "audio_bucket": audio_bucket,
        "eager_replay_text_bucket": eager_replay_text_bucket,
        "eager_replay_audio_bucket": eager_replay_audio_bucket,
        "visual_prompt_tokens": visual_prompt_tokens,
        "long_prompt_tokens": long_prompt_tokens,
        "counter_deltas": {
            "text": _counter_delta(graph_text),
            "same_bucket_refresh": _counter_delta(graph_refresh),
            "audio_input": _counter_delta(graph_audio),
            "audio_repeat": _counter_delta(graph_audio_repeat),
            "visual_custom_eager": _counter_delta(graph_visual),
            "over_cap_standard_eager": _counter_delta(graph_over_cap),
            "eager_replay_text": _counter_delta(eager_replay_text),
            "eager_replay_same_bucket_refresh": _counter_delta(eager_replay_refresh),
            "eager_replay_audio_input": _counter_delta(eager_replay_audio),
        },
        "parity_diagnostics": parity_diagnostics,
        "completion_text": {
            "eager_text": eager_text["body"]["choices"][0]["message"].get("content"),
            "graph_text": graph_text["body"]["choices"][0]["message"].get("content"),
            "eager_refresh": eager_refresh["body"]["choices"][0]["message"].get(
                "content"
            ),
            "graph_refresh": graph_refresh["body"]["choices"][0]["message"].get(
                "content"
            ),
            "eager_audio_input": eager_audio["body"]["choices"][0]["message"].get(
                "content"
            ),
            "eager_replay_text": eager_replay_text["body"]["choices"][0]["message"].get(
                "content"
            ),
            "eager_replay_refresh": eager_replay_refresh["body"]["choices"][0][
                "message"
            ].get("content"),
            "eager_replay_audio_input": eager_replay_audio["body"]["choices"][0][
                "message"
            ].get("content"),
            "graph_audio_input": graph_audio["body"]["choices"][0]["message"].get(
                "content"
            ),
            "graph_audio_repeat": graph_audio_repeat["body"]["choices"][0][
                "message"
            ].get("content"),
            "graph_visual_eager_fallback": graph_visual["body"]["choices"][0][
                "message"
            ].get("content"),
            "graph_over_cap": graph_over_cap["body"]["choices"][0]["message"].get(
                "content"
            ),
        },
        "audio_durations_s": {
            "eager_text": eager_text["audio_duration_s"],
            "graph_text": graph_text["audio_duration_s"],
            "eager_refresh": eager_refresh["audio_duration_s"],
            "eager_audio_input": eager_audio["audio_duration_s"],
            "eager_replay_text": eager_replay_text["audio_duration_s"],
            "eager_replay_refresh": eager_replay_refresh["audio_duration_s"],
            "eager_replay_audio_input": eager_replay_audio["audio_duration_s"],
            "graph_audio_input": graph_audio["audio_duration_s"],
            "graph_audio_repeat": graph_audio_repeat["audio_duration_s"],
            "graph_refresh": graph_refresh["audio_duration_s"],
            "graph_visual_eager_fallback": graph_visual["audio_duration_s"],
            "graph_over_cap": graph_over_cap["audio_duration_s"],
        },
        "request_elapsed_s": {
            "eager_text": eager_text["request_elapsed_s"],
            "graph_text": graph_text["request_elapsed_s"],
            "eager_refresh": eager_refresh["request_elapsed_s"],
            "graph_refresh": graph_refresh["request_elapsed_s"],
            "eager_audio_input": eager_audio["request_elapsed_s"],
            "eager_replay_text": eager_replay_text["request_elapsed_s"],
            "eager_replay_refresh": eager_replay_refresh["request_elapsed_s"],
            "eager_replay_audio_input": eager_replay_audio["request_elapsed_s"],
            "graph_audio_input": graph_audio["request_elapsed_s"],
            "graph_audio_repeat": graph_audio_repeat["request_elapsed_s"],
            "graph_visual_eager_fallback": graph_visual["request_elapsed_s"],
            "graph_over_cap": graph_over_cap["request_elapsed_s"],
        },
    }
    # Write all numerical diagnostics before enforcing parity so a failed gate
    # still uploads evidence instead of server logs alone.
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert eager_replay_text_bucket == text_bucket
    assert eager_replay_audio_bucket == audio_bucket
    assert eager_replay_end["replay_count"] == 3
    assert eager_replay_end["standard_eager_count"] == 0
    assert eager_replay_end["custom_eager_count"] == 0
    assert eager_replay_end["replay_buckets"] == {
        str(eager_replay_text_bucket): 2,
        str(eager_replay_audio_bucket): 1,
    }
    assert graph_end["replay_count"] == 4
    assert graph_end["standard_eager_count"] == 1
    assert graph_end["custom_eager_count"] == 1
    assert graph_end["replay_buckets"] == {
        str(text_bucket): 2,
        str(audio_bucket): 2,
    }

    _assert_custom_eager_fallback(graph_visual)
    assert snapshot(graph_visual, visual_request_id)
    # Padding may legitimately select different BF16/MoE tactics, so the
    # unpadded server is not a tensor-level graph oracle. It must still choose
    # the same greedy prefill token for every supported request.
    for diagnostic_name in (
        "text_unpadded_eager",
        "same_bucket_refresh_unpadded_eager",
        "audio_unpadded_eager",
    ):
        assert parity_diagnostics[diagnostic_name]["next_token_match"] is True

    # The private sidecar/static-slot transport must preserve the natural
    # eager embeddings bit-for-bit before shape-sensitive transformer math.
    natural_embed_errors = {
        "text": assert_exact_prefill_embeddings(
            eager_text,
            eager_replay_text,
            request_id=text_request_id,
        ),
        "same_bucket_refresh": assert_exact_prefill_embeddings(
            eager_refresh,
            eager_replay_refresh,
            request_id=refresh_request_id,
        ),
        "audio": assert_exact_prefill_embeddings(
            eager_audio,
            eager_replay_audio,
            request_id=audio_request_id,
        ),
    }

    text_errors = assert_prefill_parity(
        eager_replay_text,
        graph_text,
        request_id=text_request_id,
        require_exact_embed=True,
    )
    refresh_errors = assert_prefill_parity(
        eager_replay_refresh,
        graph_refresh,
        request_id=refresh_request_id,
        require_exact_embed=True,
    )
    audio_errors = assert_prefill_parity(
        eager_replay_audio,
        graph_audio,
        request_id=audio_request_id,
        require_exact_embed=True,
    )
    audio_repeat_errors = assert_prefill_parity(
        graph_audio,
        graph_audio_repeat,
        eager_request_id=audio_request_id,
        graph_request_id=audio_repeat_request_id,
        require_exact_embed=True,
    )
    first_audio_snapshot = snapshot(graph_audio, audio_request_id)
    repeated_audio_snapshot = snapshot(graph_audio_repeat, audio_repeat_request_id)
    for layer in ("embed", "24"):
        torch.testing.assert_close(
            decode_tensor(repeated_audio_snapshot["hidden_states"][layer]),
            decode_tensor(first_audio_snapshot["hidden_states"][layer]),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        decode_tensor(repeated_audio_snapshot["next_token_logits"]),
        decode_tensor(first_audio_snapshot["next_token_logits"]),
        rtol=0,
        atol=0,
    )

    first_text = snapshot(graph_text, text_request_id)
    refreshed_text = snapshot(graph_refresh, refresh_request_id)
    first_embed = decode_tensor(first_text["hidden_states"]["embed"])
    refreshed_embed = decode_tensor(refreshed_text["hidden_states"]["embed"])
    first_hidden_24 = decode_tensor(first_text["hidden_states"]["24"])
    refreshed_hidden_24 = decode_tensor(refreshed_text["hidden_states"]["24"])
    assert first_embed.shape == refreshed_embed.shape
    assert first_hidden_24.shape == refreshed_hidden_24.shape
    assert not torch.equal(first_embed, refreshed_embed)
    assert not torch.equal(first_hidden_24, refreshed_hidden_24)
    assert first_embed.shape[0] == snapshot_logical_rows(graph_text)
    assert refreshed_embed.shape[0] == snapshot_logical_rows(graph_refresh)
    assert snapshot_logical_rows(graph_refresh) == int(
        graph_refresh["body"]["usage"]["prompt_tokens"]
    )

    result["qualified_max_abs_errors"] = {
        "natural_eager_embeddings": natural_embed_errors,
        "text": text_errors,
        "same_bucket_refresh": refresh_errors,
        "audio_shape_matched_eager_replay": audio_errors,
        "audio_captured_repeat": audio_repeat_errors,
    }
    result["qualification_status"] = "passed"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert result_path.is_file()
