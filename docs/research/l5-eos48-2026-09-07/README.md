# Cumulative L5 exact serial EOS48 graph candidate

Prepared locally on 2026-09-07. No GPU execution, performance result or PR.

## Increment and numerical contract

Baseline `8cc5739cb18d6d5ecf3d65e69f514699706fe9dc` already has 22 Code2Wav
graph keys for the initial-2 / steady-10 / context-25 configuration. Its
`decode_delta(..., is_final=True)` explicitly makes final tails graph-ineligible.
This candidate reuses the historical local EOS48 implementation at
`/Users/wenyaogao/dev/sgl/sglang-omni-eos48-cheap-20260903`, based on `73b9e0d5`.
It preserves L5's newer scheduler behavior and fused SnakeBeta state guard.

- Enable exact serial terminal replay for existing baseline keys.
- Capture 43 missing batch-1 keys for widths 1 through 48 in one separate pool.
  The existing serial widths are 2, 12, 22, 32 and 35; their graphs stay in the
  original pool, along with the existing batched graph tiers.
- Charge baseline retained graph memory before building the optional pool.
  Optional startup failure releases only that pool and preserves the baseline
  graph matrix and batching. Production falls back to eager; such fallback
  does not count as an activated candidate in the benchmark.
- Route only exact final, batch-1 misses to the optional pool. Nonfinal work,
  batched tail keys and widths above 48 gain no new graph coverage.

The cap is a **graph coverage limit**, not an EOS truncation length. Ingest,
EOS filtering, context selection, final `end`, suffix crop, output deficit,
terminal message order and sample count remain unchanged. Final audio is copied
to owned CPU storage before another replay can overwrite borrowed graph output.

EOS35 uses the same opt-in mechanism with a cap of 35 and has 30 optional keys.
The historical factory default remains 35; the EOS48 candidate explicitly sets
48. This implementation does not batch EOS tails or claim a prior EOS48 serving
win. The old EOS35 failed screen is not evidence that EOS48 has passed or failed.

## Control and switches

Primary control: original L5 source plus byte-identical `s5.yaml`.
Candidate: this branch plus `s5-eos48.yaml`. The only semantic config changes
are these Code2Wav factory fields:

```yaml
enable_eos_cuda_graph: true
eos_cuda_graph_max_frames: 48
```

Equivalent candidate invocation, using the existing dotted config surface:

```sh
sgl-omni serve --config s5.yaml \
  --code2wav.factory.enable_eos_cuda_graph true \
  --code2wav.factory.eos_cuda_graph_max_frames 48
```

The same candidate source with unchanged `s5.yaml` is an optional OFF diagnostic.
All L5 settings remain enabled, including batching, initial chunk 2, steady
chunk 10, wait 8 ms, floor 4 / ceiling 16, SnakeBeta fused and the 0.1 Code2Wav
GPU memory fraction. No SGLang dependency patch is required for this candidate.

## CPU evidence

`cpu-tests.md` records **213 passed, 11 accelerator tests deselected**. The
reused tests cover exact keys, reserved-memory accounting, all-or-nothing
optional capture failure, unchanged baseline graphs/batching, input preservation,
output ownership across replays, widths 1–49 and EOS-before-payload termination.
One added test verifies L5's SnakeBeta guard executes with gradients disabled
and rejects changed state before optional replay. CPU fixtures do not establish
checkpoint numerical equivalence or GPU memory fit.

`cpu-config-probe.json` records that resolved configs differ in exactly the two
fields above. Source and config provenance are in `manifest.json`.

## GPU correctness and activation

The parent task owns GPU admission, selected devices, timeout and cleanup.
`probe_eos48.py` is a component-only adaptation of the historical EOS48
checkpoint probe. It starts no serving process and records no timing claim:

```sh
python -B probe_eos48.py --config s5-eos48.yaml \
  --device cuda:0 --output NEW-eos48-correctness.json
```

It uses the supplied L5 snapshot and fused SnakeBeta, preserving the 0.1 stage
budget. Admission requires all 22 baseline plus 43 optional keys and combined
retained memory within that unchanged budget. For every width 1–48 it checks
nonzero A/B/A inputs, then A again in reverse width order: 192 exact PCM checks.
It verifies the selected runner's counter, unchanged inputs, finite output and
the production suffix crops; borrowed output is materialized before reuse.
Partial failures remain failures in its report. `--help` and syntax were checked
locally; the checkpoint probe itself has not run.

This standalone probe proves neither natural tail hit rate nor serving benefit.
Before accepting a serving screen, verify final-window activation with existing
`code2wav_decode_end` metadata (`trigger=stream_done`, `execution_mode=cuda_graph`,
exact `graph_key`) in a separate short diagnostic if needed. Keep profiling off
for the parent's matched C1/C8/C16 timing run. Preserve every audio tail and use
the parent's continuity/quality checks; do not shorten outputs to create a gain.
