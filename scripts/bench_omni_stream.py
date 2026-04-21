#!/usr/bin/env python3
"""Streaming TTFT/TTFC/TPOT bench for Ming-Omni vs Qwen3-Omni speech pipelines.

Splits SSE `choices[0].delta` into Thinker text vs Talker audio chunks:
- Thinker TTFT = request_start -> first delta.content
- Talker TTFT  = first delta.content -> first delta.audio.data
  (paper-style: Talker handoff-to-first-codec-token delay)
- Talker TTFC  = request_start -> first delta.audio.data
  (absolute wall-time first audio chunk)
- Thinker TPOT = mean gap between consecutive text deltas (fallback:
  wall-span / (completion_tokens-1) when usage is reported)
- Talker TPOT  = mean gap between consecutive audio deltas
- Overall E2E  = request_start -> last audio delta (or last event if no audio)

Also reports:
- Thinker TPS  = completion_tokens / thinker active wall-span
- Talker chunks/s = n_audio_events / talker active wall-span
- Audio max/p99 inter-chunk gap (playback smoothness)
- Generation RTF = wall_time / decoded_audio_duration

Each latency metric reports median/p95 in ms per concurrency level.

Recommended fair-comparison setup on H200 (no CPU offload on either side):
    # Ming speech: thinker TP=2 on GPU 0,1 + talker on GPU 2 (3xH200)
    python examples/run_ming_omni_speech_server.py \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --tp-size 2 --gpu-thinker 0 --gpu-talker 2 \
        --cpu-offload-gb 0 --port 8000

    # Qwen3 speech: thinker TP=1 on GPU 0 + talker_ar/code_predictor/code2wav on GPU 1 (2xH200)
    python -m sglang_omni.cli.cli serve \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8000

    # Bench (restart server between models):
    python scripts/bench_omni_stream.py --model ming-omni  --sweep 1 4 8 \
        --save-json results/ming_speech.json
    python scripts/bench_omni_stream.py --model qwen3-omni --sweep 1 4 8 \
        --save-json results/qwen3_speech.json

Thinker-only mode (pass `--modalities text`) works on 1 GPU for Qwen3
(`--text-only`) but NOT for Ming FP16 (168GB > H200 141GB) unless TP=2 is used.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_PROMPT = "请讲一个关于机器人的短故事，不少于六句话。"
DEFAULT_SYSTEM = "你是一个友好的AI助手，请用自然、温暖的语气说话。"


@dataclass
class RequestTiming:
    ok: bool
    e2e_s: float
    ttft_s: float | None = None
    ttfc_s: float | None = None
    talker_ttft_s: float | None = None
    n_text_events: int = 0
    n_audio_events: int = 0
    text_tpot_s: float | None = None
    audio_tpot_s: float | None = None
    completion_tokens: int | None = None
    thinker_tps: float | None = None
    talker_chunks_per_s: float | None = None
    audio_max_gap_s: float | None = None
    audio_p99_gap_s: float | None = None
    audio_duration_s: float | None = None
    rtf: float | None = None
    error: str | None = None


def _wav_seconds(audio_bytes: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
        return frames / rate if rate > 0 else None
    except (wave.Error, EOFError, ValueError):
        return None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (len(s) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    w = rank - lo
    return s[lo] * (1 - w) + s[hi] * w


def fmt_ms(v: float | None) -> str:
    return "-" if v is None else f"{v * 1000:.1f}"


def fmt_pair(median: float | None, p95: float | None) -> str:
    if median is None and p95 is None:
        return "-"
    return f"{fmt_ms(median)}/{fmt_ms(p95)}ms"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.payload_file:
        with open(args.payload_file) as f:
            payload = json.load(f)
        payload.setdefault("stream", True)
        payload.setdefault("modalities", ["text", "audio"])
        return payload

    content: Any = args.prompt
    if args.audio:
        content = [
            {"type": "audio_url", "audio_url": {"url": args.audio}},
            {"type": "text", "text": args.prompt},
        ]

    payload: dict[str, Any] = {
        "model": args.model,
        "stream": True,
        "modalities": list(args.modalities),
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": content},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.audio:
        payload["audios"] = [args.audio]
    return payload


def _parse_delta(line: str) -> tuple[dict | None, dict | None, bool]:
    """Return (delta, usage, is_done)."""
    if not line.startswith("data: "):
        return None, None, False
    body = line[len("data: ") :]
    if body == "[DONE]":
        return None, None, True
    try:
        evt = json.loads(body)
    except json.JSONDecodeError:
        return None, None, False
    choices = evt.get("choices") or []
    delta = (
        choices[0].get("delta") if choices and isinstance(choices[0], dict) else None
    )
    usage = evt.get("usage") if isinstance(evt.get("usage"), dict) else None
    return delta, usage, False


async def one_request(
    client: httpx.AsyncClient, url: str, payload: dict[str, Any]
) -> RequestTiming:
    t0 = time.perf_counter()
    t_first_text: float | None = None
    t_first_audio: float | None = None
    t_last_text: float | None = None
    t_last_audio: float | None = None
    n_text = 0
    n_audio = 0
    completion_tokens: int | None = None
    t_end = t0
    audio_gaps: list[float] = []
    audio_duration_s = 0.0

    try:
        async with client.stream("POST", url, json=payload, timeout=None) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                now = time.perf_counter()
                delta, usage, done = _parse_delta(line)
                if usage and "completion_tokens" in usage:
                    ct = usage.get("completion_tokens")
                    if isinstance(ct, int):
                        completion_tokens = ct
                if done:
                    t_end = now
                    break
                if not delta:
                    continue
                if delta.get("content"):
                    if t_first_text is None:
                        t_first_text = now
                    t_last_text = now
                    n_text += 1
                audio = delta.get("audio")
                if isinstance(audio, dict) and audio.get("data"):
                    if t_first_audio is None:
                        t_first_audio = now
                    else:
                        audio_gaps.append(now - t_last_audio)
                    t_last_audio = now
                    n_audio += 1
                    try:
                        chunk_bytes = base64.b64decode(audio["data"])
                        wav_s = _wav_seconds(chunk_bytes)
                        if wav_s is not None:
                            audio_duration_s += wav_s
                    except Exception:
                        pass
                t_end = now
    except Exception as exc:
        return RequestTiming(
            ok=False,
            e2e_s=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )

    ttft = (t_first_text - t0) if t_first_text is not None else None
    ttfc = (t_first_audio - t0) if t_first_audio is not None else None
    talker_ttft = (
        (t_first_audio - t_first_text)
        if t_first_audio is not None and t_first_text is not None
        else None
    )

    text_tpot: float | None = None
    if n_text >= 2 and t_last_text is not None and t_first_text is not None:
        text_tpot = (t_last_text - t_first_text) / (n_text - 1)
    elif (
        completion_tokens
        and completion_tokens > 1
        and t_last_text is not None
        and t_first_text is not None
    ):
        text_tpot = (t_last_text - t_first_text) / (completion_tokens - 1)

    audio_tpot: float | None = None
    talker_chunks_per_s: float | None = None
    if n_audio >= 2 and t_last_audio is not None and t_first_audio is not None:
        talker_span = t_last_audio - t_first_audio
        audio_tpot = talker_span / (n_audio - 1)
        if talker_span > 0:
            talker_chunks_per_s = n_audio / talker_span

    thinker_tps: float | None = None
    if (
        completion_tokens
        and completion_tokens > 0
        and t_last_text is not None
        and t_first_text is not None
        and t_last_text > t_first_text
    ):
        thinker_tps = completion_tokens / (t_last_text - t_first_text)

    audio_max_gap = max(audio_gaps) if audio_gaps else None
    audio_p99_gap = percentile(audio_gaps, 0.99) if audio_gaps else None

    e2e_anchor = t_last_audio if t_last_audio is not None else t_end
    e2e_s = e2e_anchor - t0
    rtf = e2e_s / audio_duration_s if audio_duration_s > 0 else None

    return RequestTiming(
        ok=True,
        e2e_s=e2e_s,
        ttft_s=ttft,
        ttfc_s=ttfc,
        talker_ttft_s=talker_ttft,
        n_text_events=n_text,
        n_audio_events=n_audio,
        text_tpot_s=text_tpot,
        audio_tpot_s=audio_tpot,
        completion_tokens=completion_tokens,
        thinker_tps=thinker_tps,
        talker_chunks_per_s=talker_chunks_per_s,
        audio_max_gap_s=audio_max_gap,
        audio_p99_gap_s=audio_p99_gap,
        audio_duration_s=audio_duration_s if audio_duration_s > 0 else None,
        rtf=rtf,
    )


async def run_level(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    concurrency: int,
    total: int,
) -> dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)

    async def bounded():
        async with sem:
            return await one_request(client, url, payload)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(bounded() for _ in range(total)))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r.ok]
    err = [r for r in results if not r.ok]

    def col(attr: str) -> list[float]:
        return [getattr(r, attr) for r in ok if getattr(r, attr) is not None]

    ttft = col("ttft_s")
    ttfc = col("ttfc_s")
    talker_ttft = col("talker_ttft_s")
    text_tpot = col("text_tpot_s")
    audio_tpot = col("audio_tpot_s")
    e2e = col("e2e_s")
    thinker_tps_vals = col("thinker_tps")
    talker_cps_vals = col("talker_chunks_per_s")
    max_gaps = col("audio_max_gap_s")
    p99_gaps = col("audio_p99_gap_s")
    rtfs = col("rtf")

    agg_tokens = sum(r.completion_tokens or 0 for r in ok)

    def _median(v: list[float]) -> float | None:
        return percentile(v, 0.5) if v else None

    return {
        "concurrency": concurrency,
        "n_total": total,
        "n_ok": len(ok),
        "n_err": len(err),
        "wall_s": wall,
        "throughput_req_per_s": len(ok) / wall if wall > 0 else 0.0,
        "thinker_ttft_p50_s": percentile(ttft, 0.5),
        "thinker_ttft_p95_s": percentile(ttft, 0.95),
        "talker_ttfc_p50_s": percentile(ttfc, 0.5),
        "talker_ttfc_p95_s": percentile(ttfc, 0.95),
        "talker_ttft_p50_s": percentile(talker_ttft, 0.5),
        "talker_ttft_p95_s": percentile(talker_ttft, 0.95),
        "thinker_tpot_p50_s": percentile(text_tpot, 0.5),
        "thinker_tpot_p95_s": percentile(text_tpot, 0.95),
        "talker_tpot_p50_s": percentile(audio_tpot, 0.5),
        "talker_tpot_p95_s": percentile(audio_tpot, 0.95),
        "e2e_p50_s": percentile(e2e, 0.5),
        "e2e_p95_s": percentile(e2e, 0.95),
        "thinker_tps_p50": _median(thinker_tps_vals),
        "thinker_tps_agg": agg_tokens / wall if wall > 0 else None,
        "talker_chunks_per_s_p50": _median(talker_cps_vals),
        "audio_max_gap_p95_s": percentile(max_gaps, 0.95) if max_gaps else None,
        "audio_p99_gap_p50_s": _median(p99_gaps),
        "rtf_p50": _median(rtfs),
        "rtf_p95": percentile(rtfs, 0.95) if rtfs else None,
        "errors": [r.error for r in err][:5],
        "per_request": [asdict(r) for r in results],
    }


def print_header() -> None:
    cols = [
        ("conc", 4),
        ("ok/err", 7),
        ("Thinker TTFT", 18),
        ("Talker TTFC", 18),
        ("Thinker TPOT", 18),
        ("Talker TPOT", 18),
        ("E2E", 18),
        ("req/s", 7),
    ]
    print(" ".join(f"{name:>{w}}" for name, w in cols))
    print("-" * (sum(w for _, w in cols) + len(cols) - 1))


def print_row(r: dict[str, Any]) -> None:
    print(
        f"{r['concurrency']:>4} "
        f"{r['n_ok']}/{r['n_err']:<5} "
        f"{fmt_pair(r['thinker_ttft_p50_s'], r['thinker_ttft_p95_s']):>18} "
        f"{fmt_pair(r['talker_ttfc_p50_s'], r['talker_ttfc_p95_s']):>18} "
        f"{fmt_pair(r['thinker_tpot_p50_s'], r['thinker_tpot_p95_s']):>18} "
        f"{fmt_pair(r['talker_tpot_p50_s'], r['talker_tpot_p95_s']):>18} "
        f"{fmt_pair(r['e2e_p50_s'], r['e2e_p95_s']):>18} "
        f"{r['throughput_req_per_s']:>7.2f}"
    )


async def warmup(client, url, payload, n):
    if n <= 0:
        return
    print(f"Warming up with {n} request(s)...", flush=True)
    for _ in range(n):
        await one_request(client, url, payload)


def _fmt_num(v: float | None, precision: int = 2) -> str:
    return "-" if v is None else f"{v:.{precision}f}"


def print_paper_table(model: str, rows: list[dict[str, Any]]) -> None:
    print(f"\nPaper-style summary per concurrency — model={model}")
    header = f"{'metric':>22} | " + " | ".join(
        f"{r['concurrency']:>4} conc" for r in rows
    )
    print(header)
    print("-" * len(header))

    latency_fields = [
        ("Thinker TTFT (ms)", "thinker_ttft_p50_s", "thinker_ttft_p95_s"),
        ("Talker TTFT (ms)", "talker_ttft_p50_s", "talker_ttft_p95_s"),
        ("Talker TTFC (ms)", "talker_ttfc_p50_s", "talker_ttfc_p95_s"),
        ("Thinker TPOT (ms)", "thinker_tpot_p50_s", "thinker_tpot_p95_s"),
        ("Talker TPOT (ms)", "talker_tpot_p50_s", "talker_tpot_p95_s"),
        ("Overall Latency (ms)", "e2e_p50_s", "e2e_p95_s"),
    ]
    for label, k50, k95 in latency_fields:
        cells = [fmt_pair(r[k50], r[k95]) for r in rows]
        print(f"{label:>22} | " + " | ".join(f"{c:>11}" for c in cells))

    print("-" * len(header))
    scalar_fields = [
        ("Thinker TPS (tok/s)", "thinker_tps_p50", 1, False),
        ("Talker chunks/s", "talker_chunks_per_s_p50", 1, False),
        ("Audio p99 gap (ms)", "audio_p99_gap_p50_s", None, True),
        ("Audio max gap p95 (ms)", "audio_max_gap_p95_s", None, True),
        ("Generation RTF", "rtf_p50", 3, False),
        ("req/s", "throughput_req_per_s", 2, False),
    ]
    for label, key, precision, is_ms in scalar_fields:
        if is_ms:
            cells = [fmt_ms(r.get(key)) for r in rows]
        else:
            cells = [_fmt_num(r.get(key), precision) for r in rows]
        print(f"{label:>22} | " + " | ".join(f"{c:>11}" for c in cells))


async def main_async(args: argparse.Namespace) -> None:
    payload = build_payload(args)
    all_results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=None) as client:
        await warmup(client, args.url, payload, args.warmup)
        print_header()
        for conc in args.sweep:
            total = max(args.per_level, conc * 3)
            res = await run_level(client, args.url, payload, conc, total)
            all_results.append(res)
            print_row(res)
            if res["errors"]:
                print(f"    sample errors: {res['errors']}")

    print_paper_table(args.model, all_results)

    if args.save_json:
        out = {
            "url": args.url,
            "payload": payload,
            "sweep": args.sweep,
            "per_level": args.per_level,
            "warmup": args.warmup,
            "results": all_results,
        }
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nSaved raw results to {args.save_json}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument(
        "--model",
        default="ming-omni",
        help="Model name for the API request (e.g. ming-omni, qwen3-omni).",
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    p.add_argument(
        "--audio",
        default=None,
        help="Optional audio input path/URL (enables audio-in column).",
    )
    p.add_argument(
        "--modalities",
        nargs="+",
        default=["text", "audio"],
        choices=["text", "audio"],
        help="Response modalities. Use 'text' only for Thinker-only bench (1 GPU).",
    )
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--sweep", nargs="+", type=int, default=[1, 4, 8])
    p.add_argument(
        "--per-level",
        type=int,
        default=12,
        help="Requests per concurrency level (>=3x conc).",
    )
    p.add_argument(
        "--payload-file",
        default=None,
        help="Use a full JSON payload from file (overrides prompt/system).",
    )
    p.add_argument("--save-json", default=None)
    return p.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
