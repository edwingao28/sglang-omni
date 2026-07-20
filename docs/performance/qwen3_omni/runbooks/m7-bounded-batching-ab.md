# Qwen3-Omni M7 Code2Wav Bounded-Batching A/B Runbook

This is the launch authority for the **M7 policy selection** of the Code2Wav
bounded wait/floor coalescer (design `designs/code2wav-bounded-batching.md` §9,
roadmap direction B10/M7, issue sgl-project/sglang-omni#1026). It selects the
batching policy `(W, F)` with the CUDA-graph state held constant. It does **not**
open the M8 graph×batching factorial; that runs only after this runbook names a
winner.

Every arm is a profiler-off timing pair under the frozen-comparison contract
(`quality-gate.md`, "Frozen comparison contract"). Profiler-on runs are for
attribution only and never contribute timing numbers (`README.md` §3.4,
"Profiler-on vs profiler-off"). Because batching is bitwise-neutral (design §7
invariant 1), a run is admissible only when its Code2Wav fingerprint A/B shows
zero same-input/different-output windows; a single divergence fails the arm
outright.

## Environment

- **Host / checkout.** novita-h100 lab, repo `/data/wenyao-lab/qwen3-latest`.
  The full launch recipe (container, image digest, CUDA/PyTorch versions, env
  activation) lives in `/data/wenyao-lab/README.md`; read it before any run and
  do not restate its identities from memory.
- **Weights.** `/data/cache/huggingface` (Qwen3-Omni bf16 snapshot). No
  download at launch time — resolve the frozen snapshot only.
- **Serving config.** Colocated bf16 profile `/data/wenyao-lab/colocated.yaml`
  (Thinker + Talker + Code2Wav on one GPU, per `placement.py`). Two overrides
  are **mandatory** on every launch and must not be inherited from a stale
  shell:
  - `endpoints.base_path` — point at this run's isolated socket/base directory
    so arms never collide on a shared endpoint.
  - `TMPDIR` — a per-run scratch directory; the default `/tmp` is shared and
    corrupts concurrent arms.
- **GPU hygiene.** Run `nvidia-smi` immediately before **every** launch and
  require the target GPU idle (no processes, zero memory, zero utilization).
  Never stop, reset, or kill another user's process to free a GPU. If the GPU
  is busy, wait or pick another idle index — do not preempt.

## Arms

Control vs candidate, graph held at its **current opt-in state** and identical
across every arm (M7 varies only the batching policy):

- **Control.** `enable_batching=False` — per-request decode, exact latest-main
  behavior (design §3 control).
- **Candidate.** `enable_batching=True`, swept over the full grid:
  - `W ∈ {0, 5, 10, 20}` ms (`max_batch_wait_ms`)
  - `F ∈ {1, 2, 4}` (`batch_floor`)
  - `C = 8` (`batch_ceiling`, fixed at the largest graph B)
- **Concurrency.** Each cell is run at **c8 and c16** (the workloads with
  existing Code2Wav evidence, `opportunity-matrix.md`).

`W=0, F=1` reproduces the measured-negative no-wait prototype and is retained
as the mechanism-isolation reference: it attributes the *wait/floor* gain apart
from bare coalescing.

**Frozen-comparison contract (identical across all arms):** same tested
generation SHA, checkpoint/revision, placement, prompts, payload bytes,
sampling params, seeds, and serving config. Only the Code2Wav flags differ.
Record the upstream and tested 40-char SHAs in every arm manifest; numbers from
different generations are not an A/B delta.

## Launch commands

The speech A/B is driven by the profiler-off speech benchmark harness
(`benchmarks/qwen3_omni_perf/speech_client.py`). Activate the lab environment
and export the mandatory overrides exactly as recorded in
`/data/wenyao-lab/README.md`; the block below shows the shape, not a substitute
for that recipe.

**Per-run environment (from `/data/wenyao-lab/README.md`):**

```bash
cd /data/wenyao-lab/qwen3-latest
export TMPDIR=/data/wenyao-lab/scratch/m7-<arm-id>          # per-run, never shared
export QWEN3_OMNI_ENDPOINT_BASE=/data/wenyao-lab/sock/m7-<arm-id>   # -> endpoints.base_path
nvidia-smi   # require target GPU idle before proceeding
```

**Control arm (per concurrency c ∈ {8, 16}):**

```bash
python -m benchmarks.qwen3_omni_perf.speech_client \
  --config /data/wenyao-lab/colocated.yaml \
  --set endpoints.base_path=$QWEN3_OMNI_ENDPOINT_BASE \
  --set stage_overrides.code2wav.runtime.enable_batching=false \
  --concurrency <c> \
  --out /data/wenyao-lab/runs/m7/control-c<c>
```

**Candidate arm (one block per `(W, F, c)` cell):**

```bash
python -m benchmarks.qwen3_omni_perf.speech_client \
  --config /data/wenyao-lab/colocated.yaml \
  --set endpoints.base_path=$QWEN3_OMNI_ENDPOINT_BASE \
  --set stage_overrides.code2wav.runtime.enable_batching=true \
  --set stage_overrides.code2wav.runtime.max_batch_wait_ms=<W> \
  --set stage_overrides.code2wav.runtime.batch_floor=<F> \
  --set stage_overrides.code2wav.runtime.batch_ceiling=8 \
  --concurrency <c> \
  --out /data/wenyao-lab/runs/m7/batch-W<W>-F<F>-c<c>
```

Grid: `{control} ∪ {W∈{0,5,10,20} × F∈{1,2,4}}` at each of c8 and c16. If the
YAML `stage_overrides.code2wav.runtime` path does not reach `factory_args`
(design plan Task 6, unresolved question 1), fall back to threading the same
flags as launcher args the way `stream_chunk_size` reaches the factory today,
and record which override path was used in the arm manifest. The graph opt-in
(`enable_cuda_graph`) is left at its current default and is **not** varied in
M7.

## Retained metrics (per arm)

- request/s (throughput)
- TTFA p50 / p95
- inter-chunk gap p50 / p95
- xRT
- RTF p50 / p99
- attained-batch mean
- queue-delay distribution
- playback-stall count
- lifecycle completion (all requests reach terminal result)

## Behavior gate (every arm)

Identical-input Code2Wav **fingerprint A/B** against control — zero
same-input/different-output windows (the B9 method,
`opportunity-matrix.md`). Batching is defined to be bitwise-neutral (design §7
invariant 1); any divergence is a correctness failure and disqualifies the arm
regardless of its timing. Run the fingerprint check before reading any timing
number from an arm.

## Winner criteria and M8 sequencing

A candidate `(W, F)` wins M7 only with **all** of:

1. a **repeated** request/s gain over control (not a single lucky run),
2. bounded TTFA — no regression beyond the pre-registered margin (the
   first-window exemption means `W` costs only steady latency, design §10),
3. zero new playback stalls,
4. fingerprint A/B pass (zero divergences).

`W=0` is expected to lose (reproduces the no-wait negative); the point of the
sweep is to prove the wait/floor mechanism earns the batch.

**Only after** M7 names a winning `(W, F)` does M8 open: the
`{batch off,on} × {graph off,on}` factorial at that fixed `(W, F)`, four arms,
same frozen contract. M8 is out of scope for this runbook and must not be
started before an M7 winner is recorded here.

## Baseline reference (2026-07-18)

Profiler-off speech benchmark, colocated bf16, single H100. Use these as the
sanity anchor for control-arm numbers; a control run far off these values means
the environment drifted, not that batching helped.

| Concurrency | TTFA p50 | E2E p50 |
|---|---|---|
| c1 | 0.87 s | 3.4 s |
| c4 | 2.15 s | 11.3 s |
| c8 | 4.80 s | 16.1 s |

These are control-path (no batching) numbers and predate this feature; they are
a drift anchor, not an M7 comparison arm. The M7 delta is always
candidate-vs-control within the same frozen generation.

## Smoke status

**Deferred to the H100 lab session.** This runbook was authored on the H200 box
(`hyper00`), not the novita-h100 lab, so the live smoke (design plan Task 8
Step 2 — one `enable_batching=True, W=10, F=2` run at c4, confirming no errors,
audio bytes ≈ control, and `code2wav_batch_*` events in a short profiler-on
run) has **not** been executed. It must run once on the H100 lab before the
first formal M7 sweep, and its result recorded in this section. Until then the
M7 sweep status is **not launched**.
