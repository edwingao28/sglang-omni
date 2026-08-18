# Speech content gate

Manual verification tools. Not wired into CI — they need `whisper-large-v3` and
`wavlm-base-plus-sv`, and that dependency decision is deliberately left open.

## Why this exists

The fixed-256 speech contract pins the generated audio duration. Every
contract-level check we run — failure count, `audio_duration_s` signature,
token counts — therefore stays green even if the model produces **fluent
speech that says the wrong thing**.

That is not hypothetical. Thinker hidden states reach the talker through
forward-pre hooks on layers 0/24. If those copies were ever not recorded into a
CUDA graph, replay would feed the talker stale hidden states and the output
would be correct-duration, contract-passing, and wrong. No duration-based gate
can see it. These two scripts can.

**Run them for any change that touches prefill/decode CUDA graphs, hidden-state
capture, or the thinker→talker handoff.**

## Usage

Point each script at run directories produced by
`benchmarks.eval.benchmark_omni_seedtts` (each needs `speed_results.json` and
an `audio/` directory):

```bash
python wer_check.py RUN_EAGER RUN_GRAPH     # do the words match?
python sim_check.py .                        # is it the same voice?
```

## Reading the results — the control group is the point

**Never read these against a perfect score.** The pipeline is not
bit-deterministic across boots, so two *eager* runs already differ. Always
generate an eager-vs-eager pair and compare the eager-vs-graph number against
that, not against 1.0.

Two cheaper approaches were tried and both are useless here, recorded so nobody
repeats them:

- **Audio hashing** — two eager boots share 0/32 identical files.
- **Waveform correlation** — the eager-vs-eager control lands at ~0.008,
  because TTS output is not sample-aligned across runs.

## Reference measurement (2026-08-17, Qwen3-Omni-30B, colocated 1xH200)

Prefill CUDA graphs on vs off, 32 clips each:

| metric | eager | graph |
|---|---|---|
| corpus prefix WER | 0.0570 / 0.0348 | 0.0538 / 0.0570 |
| utterance p50 WER | 0.0 | 0.0 |
| empty transcripts | 0 | 0 |

| speaker similarity pair | median |
|---|---|
| control, eager vs eager | 0.9647 |
| control, graph vs graph | 0.9627 |
| **test, eager vs graph** | **0.9594 / 0.9522** |

Verdict: right words, right voice — the graph path preserves content.

Absolute WER here is **not** comparable to the paper's corpus numbers (32 clips
vs 512, and prefix truncation may differ). Only same-methodology arm-vs-arm
comparisons are load-bearing.

## Un-gating evidence (wave2 baseline)

`sglang_omni/models/qwen3_omni/bootstrap.py` refuses a non-disabled prefill CUDA
graph backend whenever speech output is enabled. This section records what we
measured with that gate absent, so the decision to lift it can rest on data.

**Read the baseline caveat first.** These runs are from the `wave2-stack`
branch, which has no such gate and does *not* carry main's `prefill_inputs.py`
sidecar. They show that speech prefill CUDA graphs *can* be correct and
faster on this model — they are **not** a validation of main's implementation.
Anything landing on main has to be re-verified there, through main's own
qualification path.

### Contract

14 runs (thinker prefill graphs on and off, c1 and c8, multiple boots), fixed-256
contract, colocated 1xH200:

- 0 failures in every run
- `audio_duration_s` exactly 20.4569 in every run, both arms
- no hangs

### Content — the part contract gates cannot see

32 clips per arm, scored with the tools in this directory:

| metric | eager | graph |
|---|---|---|
| corpus prefix WER | 0.0570 / 0.0348 | 0.0538 / 0.0570 |
| utterance p50 WER | 0.0 | 0.0 |
| empty transcripts | 0 | 0 |

| speaker similarity | median |
|---|---|
| control, eager vs eager | 0.9647 |
| control, graph vs graph | 0.9627 |
| test, eager vs graph | 0.9594 / 0.9522 |

Right words, right voice. This is the dimension that matters for the gate:
thinker hidden states reach the talker through forward-pre hooks, and a graph
that failed to refresh them would produce correct-duration, contract-passing,
wrong-content speech.

### Latency

| | eager | graph |
|---|---|---|
| speech c1 TTFT p95 | 0.2258 | 0.1824 (−19%) |
| speech c8 TTFT p95, 3 interleaved boot pairs | — | median −14.2% (3/3 favour graph) |
| text c1 TTFT p95 | 65.87 ms | 17.27 / 17.04 ms |

Speech TTFA and E2E are flat: TTFA is gated on the talker, which has no prefill
graph.

### Method notes

- **Cold-prefix anti-forgery.** Each prompt is prefixed with a deterministic
  unique string so no request can be served from a warm cache; both arms send
  byte-identical prompts and each run emits a `run_digest` over the per-response
  hashes. On the wave2 implementation both arms produced the same digest.
- **Bucket ladder.** Extending the ladder to a 2048 ceiling with intermediate
  rungs was measured and rejected: all deltas inside the noise band, ~20 GB less
  free GPU memory at capture (34.5 → 15.0 GB), no coverage gain.
- **Known difference on main.** With main's sidecar path, graph vs eager text
  output is *not* bit-identical (5/16 exact, coherent but differently worded),
  whereas the wave2 path was 16/16 identical. Worth understanding before relying
  on a digest-equality gate on main.
