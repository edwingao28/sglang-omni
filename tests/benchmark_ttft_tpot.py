#!/usr/bin/env python3
"""Measure TTFT and TPOT for Ming-Omni Thinker (via streaming API) and Talker.

Metrics (matching Qwen3.5-Omni benchmark format):
  - Thinker TTFT: Time to first token (streaming)
  - Thinker TPOT: Time per output token (decode phase)
  - Thinker TPS:  Tokens per second
  - Talker TPOT:  Time per audio token (decode step)
  - Talker TPS:   Audio tokens per second

Usage:
    # Thinker only (requires server running)
    python tests/benchmark_ttft_tpot.py --url http://localhost:8000

    # Thinker + Talker (talker loaded locally on separate GPU)
    python tests/benchmark_ttft_tpot.py --url http://localhost:8000 \
        --talker --talker-device cuda:2 \
        --model-path inclusionAI/Ming-flash-omni-2.0

    # Different concurrency levels
    python tests/benchmark_ttft_tpot.py --url http://localhost:8000 --concurrency 1 4 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field

import aiohttp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROMPTS = {
    "short": "What is 2+3?",
    "medium": "请详细解释一下Transformer架构中的自注意力机制是如何工作的。",
    "long": (
        "请写一篇关于深度学习发展历史的文章，从感知机的诞生开始，"
        "经过反向传播算法的提出、卷积神经网络在图像识别中的突破、"
        "循环神经网络和LSTM在序列建模中的应用、Transformer架构的革命性影响、"
        "到大语言模型GPT和BERT的出现。请分析每个阶段的关键技术突破。"
    ),
}


@dataclass
class ThinkerResult:
    ttft_ms: float = 0.0       # time to first token
    total_time_ms: float = 0.0
    num_tokens: int = 0
    tpot_ms: float = 0.0      # time per output token (decode)
    tps: float = 0.0          # tokens per second
    success: bool = True
    error: str = ""


async def measure_thinker_streaming(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int = 256,
) -> ThinkerResult:
    """Measure TTFT and TPOT via streaming API."""
    payload = {
        "model": "ming-omni",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    result = ThinkerResult()
    token_count = 0
    first_token_time = None

    try:
        t0 = time.perf_counter()
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            async for raw_line in resp.content:
                text = raw_line.decode("utf-8").strip()
                # SSE lines look like: data: {"choices":...}
                idx = text.find(":")
                if idx < 0 or text[:idx] != "data":
                    continue
                data = text[idx + 1:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        token_count += 1
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

        t_end = time.perf_counter()
        result.total_time_ms = (t_end - t0) * 1000
        result.num_tokens = token_count

        if first_token_time is not None:
            result.ttft_ms = (first_token_time - t0) * 1000
            decode_time_ms = (t_end - first_token_time) * 1000
            if token_count > 1:
                result.tpot_ms = decode_time_ms / (token_count - 1)
            result.tps = token_count / (t_end - t0) if (t_end - t0) > 0 else 0
        else:
            result.success = False
            result.error = "No tokens received"

    except Exception as e:
        result.success = False
        result.error = str(e)[:200]

    return result


async def run_thinker_benchmark(url: str, concurrency: int, num_runs: int, max_tokens: int):
    """Run thinker benchmark at a given concurrency level."""
    results_by_prompt: dict[str, list[ThinkerResult]] = {}

    async with aiohttp.ClientSession() as session:
        # Warmup
        await measure_thinker_streaming(session, url, "Hello", max_tokens=16)

        for label, prompt in PROMPTS.items():
            logger.info("  [%s] concurrency=%d, %d runs...", label, concurrency, num_runs)
            sem = asyncio.Semaphore(concurrency)
            all_results = []

            async def run_one():
                async with sem:
                    return await measure_thinker_streaming(session, url, prompt, max_tokens)

            tasks = [run_one() for _ in range(num_runs)]
            all_results = await asyncio.gather(*tasks)
            results_by_prompt[label] = list(all_results)

            ok = [r for r in all_results if r.success]
            if ok:
                ttfts = [r.ttft_ms for r in ok]
                tpots = [r.tpot_ms for r in ok]
                logger.info(
                    "    TTFT=%.0f/%.0fms  TPOT=%.1f/%.1fms  TPS=%.0f/%.0f",
                    np.median(ttfts), np.percentile(ttfts, 99),
                    np.median(tpots), np.percentile(tpots, 99),
                    np.median([r.tps for r in ok]), np.percentile([r.tps for r in ok], 99),
                )

    return results_by_prompt


def run_talker_benchmark(model_path: str, device: str, num_iters: int = 50):
    """Measure Talker TPOT using microbenchmarks (local GPU, no server needed)."""
    import torch

    logger.info("Loading talker on %s...", device)

    def load_ming_omni_talker(model_path, device):
        from transformers import AutoTokenizer
        from sglang_omni.models.ming_omni.talker import (
            MingOmniTalker, MingOmniTalkerConfig, SpkembExtractor,
        )
        from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import AudioVAE
        from sglang_omni.models.weight_loader import load_weights_by_prefix

        local_path = model_path
        if not os.path.isdir(model_path):
            from huggingface_hub import snapshot_download
            local_path = snapshot_download(model_path)
        talker_path = os.path.join(local_path, "talker")
        config = MingOmniTalkerConfig.from_pretrained_dir(talker_path)
        t = MingOmniTalker(config)
        t.eval()
        weights = load_weights_by_prefix(talker_path, prefix="")
        t.load_weights(weights.items())
        t.to(device=device, dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(talker_path, "llm"))
        t.set_tokenizer(tokenizer)
        try:
            t.set_spkemb_extractor(SpkembExtractor(os.path.join(talker_path, "campplus.onnx")))
        except Exception:
            pass
        try:
            from talker_tn.talker_tn import TalkerTN
            t.set_normalizer(TalkerTN())
        except ImportError:
            pass
        t.initial_graph()
        vae_path = os.path.join(talker_path, "vae")
        v = AudioVAE.from_pretrained(vae_path, dtype=torch.bfloat16)
        v.to(device).eval()
        return t, v
    talker, vae = load_ming_omni_talker(model_path, device)

    dtype = torch.bfloat16
    hidden_size = talker.model.config.hidden_size
    patch_size = talker.patch_size
    latent_dim = talker.latent_dim
    his_patch_size = talker.his_patch_size

    def bench(fn, warmup=10, iters=num_iters):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        return np.array(times)

    # LLM prefill
    from transformers import StaticCache
    prefill_len = 50
    dummy_embeds = torch.randn(1, prefill_len, hidden_size, device=device, dtype=dtype)
    pos_ids = torch.arange(prefill_len, device=device).unsqueeze(0)
    cache_pos = torch.arange(prefill_len, device=device)
    cache = StaticCache(
        config=talker.model.config, max_batch_size=1,
        max_cache_len=512, device=device, dtype=dtype,
    )

    def prefill_fn():
        cache.reset()
        talker.model(
            position_ids=pos_ids, cache_position=cache_pos,
            past_key_values=cache, inputs_embeds=dummy_embeds,
            use_cache=True, output_hidden_states=True,
        )

    prefill_times = bench(prefill_fn)

    # CFM+Agg graph step
    dummy_hidden = torch.randn(1, 1, hidden_size, device=device, dtype=dtype)
    dummy_his = torch.randn(1, his_patch_size, latent_dim, device=device, dtype=dtype)

    def cfm_fn():
        talker.sampler_pool.execute(dummy_hidden, dummy_his, 2.0, 0.25, 0.0)

    cfm_times = bench(cfm_fn)

    # LLM decode step (no graph, raw)
    decode_embeds = torch.randn(1, patch_size, hidden_size, device=device, dtype=dtype)
    decode_cache_pos = torch.arange(prefill_len, prefill_len + patch_size, device=device)

    def decode_fn():
        cache.reset()
        talker.model(
            position_ids=pos_ids, cache_position=cache_pos,
            past_key_values=cache, inputs_embeds=dummy_embeds,
            use_cache=True, output_hidden_states=True,
        )
        talker.model(
            position_ids=None, cache_position=decode_cache_pos,
            attention_mask=None, past_key_values=cache,
            inputs_embeds=decode_embeds,
            use_cache=True, output_hidden_states=True,
        )

    decode_total = bench(decode_fn)
    decode_step = decode_total - prefill_times  # subtract prefill

    # VAE decode
    n_tokens = 10
    dummy_latent = torch.randn(1, patch_size * n_tokens, latent_dim, device=device, dtype=dtype)

    def vae_fn():
        vae.decode(
            dummy_latent, use_cache=False, past_key_values=None,
            stream_state=(None, None, None), last_chunk=True,
        )

    vae_times = bench(vae_fn) / n_tokens

    return {
        "prefill_ms": prefill_times,
        "decode_step_ms": decode_step,
        "cfm_step_ms": cfm_times,
        "vae_per_token_ms": vae_times,
        "tpot_ms": decode_step + cfm_times,  # total per output token
    }


def print_report(
    thinker_results: dict[int, dict[str, list[ThinkerResult]]],
    talker_data: dict | None,
):
    w = 90
    print(f"\n{'=' * w}")
    print(f"{'MING-OMNI TTFT / TPOT BENCHMARK':^{w}}")
    print(f"{'=' * w}")

    # Header
    conc_levels = sorted(thinker_results.keys())
    header_parts = [f"{'':>20}"]
    for c in conc_levels:
        header_parts.append(f"{'%d Conc.' % c:^20}")
    print("".join(header_parts))
    print(f"  {'-' * (w - 4)}")

    # Thinker rows
    for metric_name, extract_fn, fmt in [
        ("Thinker TTFT", lambda r: r.ttft_ms, "%.0f/%.0fms"),
        ("Thinker TPOT", lambda r: r.tpot_ms, "%.1f/%.1fms"),
        ("Thinker TPS", lambda r: r.tps, "%.0f/%.0f"),
    ]:
        row_parts = [f"  {metric_name:<18}"]
        for c in conc_levels:
            # Aggregate across all prompts
            all_vals = []
            for label, results in thinker_results[c].items():
                for r in results:
                    if r.success:
                        all_vals.append(extract_fn(r))
            if all_vals:
                med = np.median(all_vals)
                p99 = np.percentile(all_vals, 99)
                row_parts.append(f"{fmt % (med, p99):^20}")
            else:
                row_parts.append(f"{'N/A':^20}")
        print("".join(row_parts))

    # Talker rows
    if talker_data:
        tpot = talker_data["tpot_ms"]
        cfm = talker_data["cfm_step_ms"]
        decode = talker_data["decode_step_ms"]
        prefill = talker_data["prefill_ms"]
        vae = talker_data["vae_per_token_ms"]

        ttfc_ms = np.median(prefill) + np.median(cfm)
        tpot_med = np.median(tpot)
        tpot_p99 = np.percentile(tpot, 99)
        tps_med = 1000.0 / tpot_med
        tps_p99 = 1000.0 / tpot_p99

        print(f"  {'Talker TTFC':<18}", end="")
        print(f"{'%.0f/%.0fms' % (np.median(prefill) + np.median(cfm), np.percentile(prefill, 99) + np.percentile(cfm, 99)):^20}" * len(conc_levels))
        print(f"  {'Talker TPOT':<18}", end="")
        print(f"{'%.1f/%.1fms' % (tpot_med, tpot_p99):^20}" * len(conc_levels))
        print(f"  {'Talker TPS':<18}", end="")
        print(f"{'%.0f/%.0f' % (tps_med, tps_p99):^20}" * len(conc_levels))
        print(f"  {'Codec Decode':<18}", end="")
        print(f"{'%.1f/%.1fms' % (np.median(vae), np.percentile(vae, 99)):^20}" * len(conc_levels))

    print(f"  {'-' * (w - 4)}")

    # Per-prompt breakdown
    print(f"\n  Per-prompt Thinker breakdown:")
    for c in conc_levels:
        print(f"\n  [{c} Conc.]")
        print(f"    {'Prompt':<10} {'TTFT med':>10} {'TTFT p99':>10} {'TPOT med':>10} {'TPOT p99':>10} {'TPS med':>10} {'Tokens':>8}")
        for label in PROMPTS:
            results = thinker_results[c].get(label, [])
            ok = [r for r in results if r.success]
            if ok:
                ttfts = [r.ttft_ms for r in ok]
                tpots = [r.tpot_ms for r in ok]
                tps = [r.tps for r in ok]
                toks = [r.num_tokens for r in ok]
                print(
                    f"    {label:<10} {np.median(ttfts):>9.0f}ms {np.percentile(ttfts, 99):>9.0f}ms "
                    f"{np.median(tpots):>9.1f}ms {np.percentile(tpots, 99):>9.1f}ms "
                    f"{np.median(tps):>9.0f} {np.median(toks):>7.0f}"
                )

    if talker_data:
        print(f"\n  Talker component breakdown (median):")
        print(f"    LLM Prefill:     {np.median(talker_data['prefill_ms']):>8.2f} ms")
        print(f"    LLM Decode/step: {np.median(talker_data['decode_step_ms']):>8.2f} ms")
        print(f"    CFM+Agg/step:    {np.median(talker_data['cfm_step_ms']):>8.2f} ms")
        print(f"    VAE Decode/tok:  {np.median(talker_data['vae_per_token_ms']):>8.2f} ms")
        print(f"    Total TPOT:      {np.median(talker_data['tpot_ms']):>8.2f} ms")

    print(f"\n{'=' * w}")


async def main_async(args):
    # Thinker benchmark
    thinker_results: dict[int, dict[str, list[ThinkerResult]]] = {}

    for conc in args.concurrency:
        logger.info("=== Thinker: concurrency=%d ===", conc)
        results = await run_thinker_benchmark(
            args.url, conc, args.num_runs, args.max_tokens,
        )
        thinker_results[conc] = results

    # Talker benchmark (optional)
    talker_data = None
    if args.talker:
        logger.info("=== Talker microbenchmark ===")
        talker_data = run_talker_benchmark(
            args.model_path, args.talker_device, args.talker_iters,
        )

    print_report(thinker_results, talker_data)

    # Save JSON
    save_data = {
        "thinker": {
            str(c): {
                label: [{"ttft_ms": r.ttft_ms, "tpot_ms": r.tpot_ms, "tps": r.tps,
                          "num_tokens": r.num_tokens, "total_time_ms": r.total_time_ms}
                         for r in results if r.success]
                for label, results in prompt_results.items()
            }
            for c, prompt_results in thinker_results.items()
        },
    }
    if talker_data:
        save_data["talker"] = {
            k: {"median": float(np.median(v)), "p99": float(np.percentile(v, 99))}
            for k, v in talker_data.items()
        }
    with open(args.output, "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info("Saved to %s", args.output)


def main():
    parser = argparse.ArgumentParser(description="TTFT/TPOT Benchmark for Ming-Omni")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1])
    parser.add_argument("--talker", action="store_true", help="Also benchmark talker")
    parser.add_argument("--talker-device", default="cuda:2")
    parser.add_argument("--talker-iters", type=int, default=50)
    parser.add_argument("--model-path", default="inclusionAI/Ming-flash-omni-2.0")
    parser.add_argument("--output", default="/tmp/ttft_tpot_benchmark.json")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
