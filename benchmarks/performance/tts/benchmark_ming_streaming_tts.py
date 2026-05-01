from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
import time
import wave
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Respond with exactly this sentence and no extra words: "
    "The sun rises in the east."
)


@dataclass
class StreamingTtsMetrics:
    ttfa_ms: float | None
    total_time_ms: float
    audio_duration_ms: float
    rtf: float | None
    jitter_ms: float | None
    chunk_count: int
    sample_rate_hz: int | None
    queue_depth_max: int | None
    abort_cleanup_ms: float | None
    stage_times_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class MingStreamingTtsThresholds:
    ttfa_p50_ms: float = 800.0
    ttfa_p95_ms: float = 1500.0
    abort_cleanup_ms: float = 500.0
    # Default to the Ming-flash-omni-2.0 VAE rate so GPU validation
    # fails loud on a propagation regression unless callers explicitly
    # opt out by passing required_sample_rate_hz=None.
    required_sample_rate_hz: int | None = 44100
    require_abort_cleanup: bool = False


def summarize_events(
    *,
    request_start_s: float,
    audio_events: list[dict[str, Any]],
    request_end_s: float,
    stage_times_ms: dict[str, float] | None = None,
    queue_depth_max: int | None = None,
    abort_cleanup_ms: float | None = None,
) -> StreamingTtsMetrics:
    first_audio = next(
        (event for event in audio_events if event.get("num_samples", 0) > 0),
        None,
    )
    ttfa_ms = (
        None
        if first_audio is None
        else (first_audio["t_s"] - request_start_s) * 1000.0
    )
    sample_rate = first_audio.get("sample_rate") if first_audio else None

    audio_duration_ms = 0.0
    for event in audio_events:
        event_sample_rate = event.get("sample_rate") or sample_rate
        if event_sample_rate:
            audio_duration_ms += (
                event.get("num_samples", 0) / event_sample_rate
            ) * 1000.0

    total_time_ms = (request_end_s - request_start_s) * 1000.0
    rtf = None if audio_duration_ms <= 0 else total_time_ms / audio_duration_ms
    gaps_ms = [
        (audio_events[index]["t_s"] - audio_events[index - 1]["t_s"]) * 1000.0
        for index in range(1, len(audio_events))
    ]
    jitter_ms = statistics.pstdev(gaps_ms) if len(gaps_ms) > 1 else None

    return StreamingTtsMetrics(
        ttfa_ms=ttfa_ms,
        total_time_ms=total_time_ms,
        audio_duration_ms=audio_duration_ms,
        rtf=rtf,
        jitter_ms=jitter_ms,
        chunk_count=len(audio_events),
        sample_rate_hz=sample_rate,
        queue_depth_max=queue_depth_max,
        abort_cleanup_ms=abort_cleanup_ms,
        stage_times_ms=dict(stage_times_ms or {}),
    )


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AssertionError("No values for percentile")
    index = round((percentile_value / 100.0) * (len(ordered) - 1))
    index = min(len(ordered) - 1, max(0, index))
    return ordered[index]


def load_thresholds(path: str | Path | None) -> MingStreamingTtsThresholds:
    if path is None:
        return MingStreamingTtsThresholds()
    with Path(path).open() as threshold_file:
        data = json.load(threshold_file)
    if "thresholds" in data:
        data = data["thresholds"]
    return MingStreamingTtsThresholds(**data)


def build_chat_payload(
    *,
    model: str,
    prompt: str,
    stream: bool,
    request_id: str,
    max_tokens: int = 64,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text", "audio"],
        "audio": {"format": "wav", "voice": "DB30"},
        "stream": stream,
        "max_tokens": max_tokens,
        "request_id": request_id,
    }


def _metrics_from_json(data: dict[str, Any]) -> list[StreamingTtsMetrics]:
    raw_metrics = data.get("metrics")
    if not isinstance(raw_metrics, list):
        raise ValueError("Metrics JSON must contain a 'metrics' list")

    metric_fields = {metric_field.name for metric_field in fields(StreamingTtsMetrics)}
    metrics: list[StreamingTtsMetrics] = []
    for index, raw_metric in enumerate(raw_metrics):
        if not isinstance(raw_metric, dict):
            raise ValueError(f"metrics[{index}] must be an object")
        values = {
            name: value for name, value in raw_metric.items() if name in metric_fields
        }
        values.setdefault("stage_times_ms", {})
        metrics.append(StreamingTtsMetrics(**values))
    return metrics


def assert_json_thresholds(path: str | Path) -> None:
    with Path(path).open() as metrics_file:
        data = json.load(metrics_file)
    if not isinstance(data, dict):
        raise ValueError("Metrics JSON must be an object")

    thresholds_data = data.get("thresholds", {})
    if not isinstance(thresholds_data, dict):
        raise ValueError("Metrics JSON 'thresholds' must be an object when present")

    metrics = _metrics_from_json(data)
    thresholds = MingStreamingTtsThresholds(**thresholds_data)
    assert_thresholds(metrics, thresholds)


def assert_thresholds(
    metrics: list[StreamingTtsMetrics],
    thresholds: MingStreamingTtsThresholds,
) -> None:
    ttfa_values = [metric.ttfa_ms for metric in metrics if metric.ttfa_ms is not None]
    if not ttfa_values:
        raise AssertionError("No non-empty audio chunks; TTFA unavailable")

    ttfa_p50 = percentile(ttfa_values, 50)
    ttfa_p95 = percentile(ttfa_values, 95)
    assert ttfa_p50 <= thresholds.ttfa_p50_ms, (
        f"TTFA p50 {ttfa_p50:.1f}ms > {thresholds.ttfa_p50_ms:.1f}ms"
    )
    assert ttfa_p95 <= thresholds.ttfa_p95_ms, (
        f"TTFA p95 {ttfa_p95:.1f}ms > {thresholds.ttfa_p95_ms:.1f}ms"
    )

    if thresholds.required_sample_rate_hz is not None:
        bad_sample_rates = [
            metric.sample_rate_hz
            for metric in metrics
            if metric.sample_rate_hz != thresholds.required_sample_rate_hz
        ]
        assert not bad_sample_rates, f"Unexpected sample rates: {bad_sample_rates}"

    abort_cleanup_values = [
        metric.abort_cleanup_ms
        for metric in metrics
        if metric.abort_cleanup_ms is not None
    ]
    if thresholds.require_abort_cleanup and not abort_cleanup_values:
        raise AssertionError("Abort cleanup unavailable")

    slow_abort_cleanup = [
        value
        for value in abort_cleanup_values
        if value > thresholds.abort_cleanup_ms
    ]
    assert not slow_abort_cleanup, (
        f"Abort cleanup exceeded threshold: {slow_abort_cleanup}"
    )


def _summary(metrics: list[StreamingTtsMetrics]) -> dict[str, Any]:
    ttfa_values = [metric.ttfa_ms for metric in metrics if metric.ttfa_ms is not None]
    rtf_values = [metric.rtf for metric in metrics if metric.rtf is not None]
    jitter_values = [
        metric.jitter_ms for metric in metrics if metric.jitter_ms is not None
    ]
    abort_cleanup_values = [
        metric.abort_cleanup_ms
        for metric in metrics
        if metric.abort_cleanup_ms is not None
    ]
    sample_rates = [
        metric.sample_rate_hz
        for metric in metrics
        if metric.sample_rate_hz is not None
    ]
    queue_depths = [
        metric.queue_depth_max
        for metric in metrics
        if metric.queue_depth_max is not None
    ]
    return {
        "count": len(metrics),
        "ttfa_p50_ms": percentile(ttfa_values, 50) if ttfa_values else None,
        "ttfa_p95_ms": percentile(ttfa_values, 95) if ttfa_values else None,
        "rtf_mean": statistics.fmean(rtf_values) if rtf_values else None,
        "chunk_count_total": sum(metric.chunk_count for metric in metrics),
        "ming_streaming_ttfa_ms": percentile(ttfa_values, 50)
        if ttfa_values
        else None,
        "ming_streaming_rtf": statistics.fmean(rtf_values) if rtf_values else None,
        "ming_streaming_chunk_jitter_ms": statistics.fmean(jitter_values)
        if jitter_values
        else None,
        "ming_streaming_abort_cleanup_ms": max(abort_cleanup_values)
        if abort_cleanup_values
        else None,
        "ming_streaming_sample_rate_hz": sample_rates[0] if sample_rates else None,
        "ming_streaming_talker_queue_depth": max(queue_depths)
        if queue_depths
        else None,
    }


def _audio_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("audio"), dict):
                return delta["audio"]
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("audio"), dict):
                return message["audio"]

    audio = payload.get("audio")
    if isinstance(audio, dict):
        return audio
    return None


def update_stage_times_from_payload(
    stage_times_ms: dict[str, float], payload: dict[str, Any]
) -> None:
    for source in (payload, _audio_from_payload(payload)):
        if not isinstance(source, dict):
            continue
        stage_times = source.get("stage_times_ms")
        if not isinstance(stage_times, dict):
            continue
        for name, value in stage_times.items():
            if isinstance(name, str) and isinstance(value, (int, float)):
                stage_times_ms[name] = float(value)


def collect_pcm_from_audio_payloads(
    audio_payloads: list[dict[str, Any]],
) -> tuple[bytes, int | None]:
    pcm_chunks: list[bytes] = []
    collected_sample_rate: int | None = None
    for payload in audio_payloads:
        audio = _audio_from_payload(payload)
        audio_source = audio if isinstance(audio, dict) else payload
        data = audio_source.get("data")
        if not isinstance(data, str):
            continue
        try:
            audio_bytes = base64.b64decode(data, validate=True)
        except Exception:
            continue
        if audio_bytes.startswith(b"RIFF"):
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                    pcm_chunks.append(wav_file.readframes(wav_file.getnframes()))
                    collected_sample_rate = wav_file.getframerate()
                    continue
            except (EOFError, wave.Error):
                continue
        pcm_chunks.append(audio_bytes)
        sample_rate = audio_source.get("sample_rate")
        if isinstance(sample_rate, int):
            collected_sample_rate = sample_rate
    return b"".join(pcm_chunks), collected_sample_rate


def write_wav_from_pcm(
    path: str | Path,
    pcm: bytes,
    sample_rate_hz: int,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm)


def write_wav_from_audio_payloads(
    audio_payloads: list[dict[str, Any]],
    path: str | Path,
) -> None:
    pcm, sample_rate_hz = collect_pcm_from_audio_payloads(audio_payloads)
    if not pcm:
        raise RuntimeError("No valid PCM audio collected; refusing to write empty WAV")
    write_wav_from_pcm(path, pcm, sample_rate_hz or 44100)


def _audio_details_from_payload(payload: dict[str, Any]) -> tuple[int, int | None]:
    audio = _audio_from_payload(payload)
    audio_source = audio if isinstance(audio, dict) else payload
    sample_rate = audio_source.get("sample_rate")
    if not isinstance(sample_rate, int):
        sample_rate = None
    samples = audio_source.get("num_samples")
    if isinstance(samples, int):
        return samples, sample_rate
    data = audio_source.get("data")
    if not isinstance(data, str):
        return 0, sample_rate
    try:
        audio_bytes = base64.b64decode(data)
    except ValueError:
        return 0, sample_rate
    if audio_bytes.startswith(b"RIFF"):
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                return wav_file.getnframes(), wav_file.getframerate()
        except wave.Error:
            pass
    return len(audio_bytes) // 2, sample_rate


def _num_samples_from_audio_payload(payload: dict[str, Any]) -> int:
    return _audio_details_from_payload(payload)[0]


def _has_audio_payload(payload: dict[str, Any]) -> bool:
    audio = _audio_from_payload(payload)
    audio_source = audio if isinstance(audio, dict) else payload
    return (
        isinstance(audio_source.get("data"), str)
        or isinstance(audio_source.get("num_samples"), int)
        or isinstance(audio_source.get("sample_rate"), int)
    )


def _audio_event_from_payload(
    payload: dict[str, Any], t_s: float
) -> dict[str, Any] | None:
    if not _has_audio_payload(payload):
        return None
    num_samples, sample_rate = _audio_details_from_payload(payload)
    return {
        "t_s": t_s,
        "num_samples": num_samples,
        "sample_rate": payload.get("sample_rate") or sample_rate,
    }


def _text_event_from_payload(payload: dict[str, Any], t_s: float) -> dict[str, Any] | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    delta = choice.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
        return {"t_s": t_s, "text": delta["content"]}
    return None


def _queue_depth_from_payload(payload: dict[str, Any]) -> int | None:
    queue_depth = payload.get("talker_queue_depth")
    if isinstance(queue_depth, int):
        return queue_depth
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("talker_queue_depth"), int):
        return metrics["talker_queue_depth"]
    return None


def _health_total_requests(httpx: Any, base_url: str) -> int | None:
    get = getattr(httpx, "get", None)
    if not callable(get):
        return None
    try:
        response = get(f"{base_url.rstrip('/')}/health", timeout=5.0)
        data = response.json()
    except Exception:
        return None
    total_requests = data.get("total_requests")
    return total_requests if isinstance(total_requests, int) else None


def _wait_for_abort_cleanup_ms(
    httpx: Any,
    *,
    base_url: str,
    baseline_total_requests: int | None,
    abort_start_s: float,
    timeout_s: float = 5.0,
) -> float | None:
    if baseline_total_requests is None:
        return None
    deadline_s = abort_start_s + timeout_s
    while time.perf_counter() < deadline_s:
        total_requests = _health_total_requests(httpx, base_url)
        if total_requests is not None and total_requests <= baseline_total_requests:
            return (time.perf_counter() - abort_start_s) * 1000.0
        time.sleep(0.05)
    return (time.perf_counter() - abort_start_s) * 1000.0


def run_chat_completion_once(
    *,
    base_url: str,
    model: str,
    prompt: str,
    stream: bool,
    abort_after_first_audio: bool = False,
    observed_audio_payloads: list[dict[str, Any]] | None = None,
) -> StreamingTtsMetrics:
    import httpx

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    request_id = f"ming-streaming-tts-bench-{time.time_ns()}"
    payload = build_chat_payload(
        model=model,
        prompt=prompt,
        stream=stream,
        request_id=request_id,
    )
    audio_events: list[dict[str, Any]] = []
    text_events: list[dict[str, Any]] = []
    stage_times_ms: dict[str, float] = {}
    queue_depth_max: int | None = None
    abort_cleanup_ms: float | None = None
    baseline_total_requests = (
        _health_total_requests(httpx, base_url) if abort_after_first_audio else None
    )
    request_start_s = time.perf_counter()

    if stream:
        with httpx.stream("POST", url, json=payload, timeout=None) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: ") :].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                update_stage_times_from_payload(stage_times_ms, event)
                event_time_s = time.perf_counter()
                text_event = _text_event_from_payload(event, event_time_s)
                if text_event is not None:
                    text_events.append(text_event)
                queue_depth = _queue_depth_from_payload(event)
                if queue_depth is not None:
                    queue_depth_max = max(queue_depth_max or 0, queue_depth)
                audio_event = _audio_event_from_payload(event, event_time_s)
                if audio_event is not None:
                    audio_events.append(audio_event)
                    if observed_audio_payloads is not None:
                        observed_audio_payloads.append(event)
                    if (
                        abort_after_first_audio
                        and abort_cleanup_ms is None
                        and audio_event.get("num_samples", 0) > 0
                    ):
                        abort_start_s = time.perf_counter()
                        close = getattr(response, "close", None)
                        if callable(close):
                            close()
                        abort_cleanup_ms = _wait_for_abort_cleanup_ms(
                            httpx,
                            base_url=base_url,
                            baseline_total_requests=baseline_total_requests,
                            abort_start_s=abort_start_s,
                        )
                        break
    else:
        response = httpx.post(url, json=payload, timeout=None)
        response.raise_for_status()
        event = response.json()
        update_stage_times_from_payload(stage_times_ms, event)
        audio_event = _audio_event_from_payload(event, time.perf_counter())
        if audio_event is not None:
            audio_events.append(audio_event)
            if observed_audio_payloads is not None:
                observed_audio_payloads.append(event)

    if text_events and "thinker_first_text" not in stage_times_ms:
        stage_times_ms["thinker_first_text"] = (
            text_events[0]["t_s"] - request_start_s
        ) * 1000.0
    first_audio = next(
        (event for event in audio_events if event.get("num_samples", 0) > 0),
        None,
    )
    if first_audio is not None:
        first_audio_ms = (first_audio["t_s"] - request_start_s) * 1000.0
        stage_times_ms.setdefault("talker_first_audio", first_audio_ms)
        # segmenter_first_emit must come from the server's stage_times_ms in
        # the SSE payload. Synthesizing it from talker_first_audio hides the
        # segmenter contribution to TTFA.
    if queue_depth_max is None:
        queue_depth_max = 0

    return summarize_events(
        request_start_s=request_start_s,
        audio_events=audio_events,
        request_end_s=time.perf_counter(),
        stage_times_ms=stage_times_ms,
        queue_depth_max=queue_depth_max,
        abort_cleanup_ms=abort_cleanup_ms,
    )


def write_results(
    *,
    path: str | Path,
    metrics: list[StreamingTtsMetrics],
    thresholds: MingStreamingTtsThresholds,
) -> dict[str, Any]:
    result = {
        "metrics": [asdict(metric) for metric in metrics],
        "summary": _summary(metrics),
        "thresholds": asdict(thresholds),
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Ming-Omni streaming TTS timing metrics."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="ming-omni")
    parser.add_argument("--mode", choices=["stream", "nonstream"], default="stream")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-wav")
    parser.add_argument("--thresholds-json")
    parser.add_argument(
        "--abort-after-first-audio",
        action="store_true",
        help=(
            "Close the streaming response after the first non-empty audio chunk "
            "and record abort cleanup timing."
        ),
    )
    parser.add_argument(
        "--assert-thresholds",
        "--check-thresholds",
        action="store_true",
        dest="check_thresholds",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.abort_after_first_audio and args.mode != "stream":
        parser.error("--abort-after-first-audio requires --mode stream")
    thresholds = load_thresholds(args.thresholds_json)
    if args.abort_after_first_audio:
        thresholds.require_abort_cleanup = True
    observed_audio_payloads: list[dict[str, Any]] = []
    metrics = [
        run_chat_completion_once(
            base_url=args.base_url,
            model=args.model,
            prompt=args.prompt,
            stream=args.mode == "stream",
            abort_after_first_audio=args.abort_after_first_audio,
            observed_audio_payloads=observed_audio_payloads,
        )
        for _ in range(args.runs)
    ]
    write_results(path=args.output_json, metrics=metrics, thresholds=thresholds)
    if args.output_wav:
        write_wav_from_audio_payloads(observed_audio_payloads, args.output_wav)
    if args.check_thresholds:
        assert_json_thresholds(args.output_json)


if __name__ == "__main__":
    main()
