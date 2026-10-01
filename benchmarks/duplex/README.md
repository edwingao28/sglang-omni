# Full-duplex benchmarks

This guide covers realtime serving measurements and the Full-Duplex-Bench v1.5
behavioral evaluation workflow for VoiceChat and MiniCPM-o. Both use native
`/v1/realtime` sessions, with 80 ms input transport packets.

| Evaluation | Question | Main measurements | Population |
|---|---|---|---|
| Serving | Can admitted sessions finish native units on schedule and stream each speech response smoothly? | Admission, unit lag, response TTFA, response gap excess and drift | Admitted sessions at each requested concurrency |
| Behavioral v1.5 | How does the model respond to interruption, backchannel, talking-to-other and background speech? | Reference stop/response intervals and judged behavior labels | Selected overlap/clean dataset pairs, with eligibility and pending labels retained |
| Protocol | Did the native session obey its contract? | Input receipts/accounting, response ordering, EOS drain and close | Each attempted recording |

Serving response TTFA starts at `response.created`. Behavioral response intervals
come from speech spans in fixed observation windows. They have different
definitions and should not be combined into one latency score. The retained
behavioral campaigns below have ASR/timing results, but official behavior labels
remain pending. Serving success does not establish conversational quality.

## Contents

- [Profiles and metric scope](#profiles-and-metric-scope)
- [Serving workflow and metrics](#serving-workflow-and-metrics)
- [Behavioral v1.5 workflow](#behavioral-v15-workflow)
- [Retained Nemotron H100 results](#retained-nemotron-h100-results-2026-09-26)
- [MiniCPM-o H100 results](#minicpm-o-45-h100-results-2026-09-27)
- [Custom judge calibration](#custom-judge-calibration-result-2026-09-27)
- [Serving H200 qualification](#serving-h200-qualification-2026-10-02)

## Profiles and metric scope

| Profile | Native unit | Output rate | Continuous output | Default qualification |
|---|---:|---:|---|---|
| `nemotron-voicechat-pr2188` | 80 ms | 22,050 Hz | Yes | C1 serving; C2/C4 admission when capacity is one |
| `minicpmo-native-pr2377` | 1000 ms | 24,000 Hz | No | C1/C2 serving; C4 admission with `max_sessions=2` |

| Metric | VoiceChat | MiniCPM-o |
|---|---|---|
| Admission counts, input send lateness and loop lag | Measured | Measured |
| Full-unit completion lag | Measured; partial EOS unit excluded | Measured; partial EOS unit excluded |
| Response TTFA, within-response gap excess, max drift and required buffer | Measured when audio exists | Measured when audio exists; gaps need at least two packets |
| Session TTFA, output gap/excess, coverage, continuous/final drift, session buffer and underrun | Measured | N/A |
| Behavioral reference intervals and labels | Evaluated separately on dataset pairs | Evaluated separately on dataset pairs |

MiniCPM-o may listen silently or pause between responses. Session-wide continuous
playout measurements would count those pauses as serving failures, so they remain
`None`/N/A. Empty response or interval distributions also stay unavailable rather
than becoming zero.

Use separate output directories for serving sweeps, behavioral recordings and
scoring phases. Record the server SHA, model revision, configuration and input
identity for each campaign. The benchmark and native server launchers may require
separate checkouts; [the protocol guide](../README.md#native-duplex-voicechat-protocol-benchmark)
describes this setup.

## Serving workflow and metrics

Run against a server exposing `/v1/realtime`:

```bash
python -m benchmarks.duplex.serving \
  --url ws://127.0.0.1:8097/v1/realtime \
  --audio input.wav --profile nemotron-voicechat-pr2188 \
  --concurrencies 1,2,4,8 --repeats 3
```

The input is normalized to mono 16 kHz PCM16 using the existing duplex audio
loader. Each concurrency level starts only after its sessions finish negotiation.
By default, all configured sessions use one monotonic start deadline and fixed
80 ms input deadlines. A late sender sends buffered frames as soon as it can;
subsequent deadlines remain anchored to the original start. Use `--stagger-ms 80`
to spread session starts evenly across an 80 ms phase. Each session has its own
JSONL wire trace and input-send-receipts.json; the latter
records scheduled, send-start, and send-completion times. Trace serialization
and file writes run on a worker thread; each level awaits trace flush, receipts,
and the WebSocket close path before starting the next level. Failed and rejected
attempts retain their own artifacts and count toward attempted concurrency.

Both `nemotron-voicechat-pr2188` and `minicpmo-native-pr2377` use this runner.
Transport packets stay at 80 ms for both; native units are respectively 80 ms
and 1000 ms. No profile flags are needed beyond `continuous_output`.

The common table emphasizes model, concurrency, admitted/attempted,
success/admitted, unit lag p95, response TTFA p95 and response gap-excess p99.
All timing distributions include p50/p75/p95/p99/max with linear interpolation.
Missing measurements have no samples and display `-`; they never become zero.

Unit lag uses the client receipt timestamp of `sglang.unit.done unit_<k>` minus
`session_start + (k + 1) * native_unit_ms / 1000`. It includes transport to the
client. The event is emitted after unit output passes through the runtime/output
buffer, so it measures **unit fully emitted relative to its native-unit deadline**,
not pure GPU compute latency. Audio packet indices do not define this metric.
Only full native units within the unpadded input duration contribute. Terminal
partial/padded EOS units are retained as `excluded_terminal_units`, since their
nominal full-unit deadline would make an early EOS flush appear artificially early.

Response TTFA is first `response.output_audio.delta` receipt minus
`response.created` receipt, grouped by `response_id`. Text before audio contributes
to this interval. Responses without audio have no TTFA sample. Within each
response, gap excess is the positive arrival gap minus the preceding packet's
decoded PCM duration. Drift is arrival minus first arrival minus all earlier PCM
durations; maximum drift and required buffer (positive maximum drift) are reported
per response and aggregated across responses. Gaps between response IDs never
contribute. Raw gaps are unsuitable for comparing MiniCPM-o's one-second bursts
with VoiceChat output.

Late-send rate is the fraction of sent input frames starting more than 20 ms
after their deadline. The threshold is a load-generator diagnostic, not a
server SLO. A 10 ms event-loop ticker records p99 loop lag; runs exceeding
`--loop-lag-limit-ms` (20 ms by default) are marked `client_timing_valid=false`.
Additionally, every admitted session must have send receipts and send-lateness
p99 at most 20 ms. Loop lag p99 and send lateness p95/p99 are reported separately;
a run without admitted sessions is not timing-valid. These checks measure the
benchmark process, not the remote server, and prevent late input from being
silently interpreted as a valid server-capacity measurement.

The following session-wide metrics apply only to `continuous_output=True`
(VoiceChat): session TTFA, output gap/excess, coverage, continuous/final drift,
required playout buffer, underrun count/duration/ratio. MiniCPM-o returns `None`
for all of them, even for failed or rejected sessions; aggregates remain empty
or N/A. Listening and turn-taking time are not session serving latency.

For consecutive output packets, gap excess is the positive part of the receive
gap minus the preceding packet's decoded PCM duration. Output drift at packet
i is its receive time minus the first audio receive time and the duration of
all earlier packets. The first drift is zero; negative values mean audio arrived
ahead of that ideal schedule. Final drift is the last packet's drift, not an
input-to-output causal latency. These quantities use actual decoded PCM lengths,
not an assumed model-frame size. No deadline-miss SLO is defined in v0.

The largest positive drift is reported per session as the required playout
buffer for avoiding gaps between received packets. Its p50 and p95 are reported
across sessions with output audio; `n` shows how many contributed.

When output audio carries `sglang.media_time.t_start_ms`, the first packet for
each matching input media time also contributes a receive-minus-scheduled-send
and a receive-minus-actual-send lag. Repeated chunks for one media time are
counted once. These are wire-observed alignment offsets, not proof that an
individual input frame caused that output packet or pure server latency.
Absent or unmatched metadata produces no lag sample and remains visible in `n`.

Playback underrun is a simple client simulation. Playback starts 80 ms after
the first audio receipt by default (`--startup-reserve-ms` changes this), using
all audio received during that reserve. It consumes PCM time until the common
input deadline plus the reserve. Each interval that exhausts the buffer counts
as an underrun. A session without audio, or whose first audio arrives after the
input window, is assigned one full-input-duration underrun. Underrun ratio is
total underrun duration divided by the fixed input observation duration; the
aggregate ratio includes admitted continuous-output sessions, including failures,
and excludes rejected attempts and non-continuous profiles. This does
not model network jitter, device buffering, or the server's internal stages.
These observations alone do not establish a sustainable concurrency threshold.

One warmup session runs before measured levels and is excluded from the measured
summary. `--repeats` runs every level repeatedly, with separate traces and
per-run summaries. Percentile tables display sample counts. The top-level JSON
also combines all repeats at each concurrency in `levels`, so a C1 p95 based on
three repeats has `n=3`.

Optional `--gpu-index` records local GPU utilization through ResourceMonitor;
without it, resource metrics are unavailable. The top-level summary records local
benchmark provenance and the normalized input SHA-256. Supply `--model-id`,
`--model-revision`, `--server-sha`, and `--server-config config.json` to record
remote server details. The local repository SHA is the benchmark client SHA,
not the remote server SHA. Pass repeatable `--gpu-process-pid` values for
process-level GPU memory and CPU attribution; these must be host PIDs visible
to NVML.

An exhausted HTTP 503 handshake is `status=rejected`, with `admitted=false`,
not an admitted failure. Three short 250 ms retries accommodate closing-session
teardown races; the final denial is explicit in the trace. Summaries report
attempted, admitted, rejected and successful admitted sessions. Lifecycle checks
for admitted sessions still require input drain, session close and no protocol
error; VoiceChat also requires audio. Silence is valid for MiniCPM-o.

For VoiceChat #2188 qualify C1, and treat C2/C4 as admission observations when
the server limits connections. For MiniCPM-o #2377 use the default
`examples/full_duplex/minicpmo.yaml` with `max_sessions=2` and run C1/C2/C4.
C1/C2 measure serving performance; C4 measures admission and should report two
admitted, two rejected, and success 2/2 admitted when both admitted sessions pass.
Record the actual server SHA and config using provenance arguments. Do not
increase admission capacity for qualification.

## Behavioral v1.5 workflow

Use this path to evaluate recorded full-duplex speech with the reference ASR,
timing formulas and behavior prompt. This is the v1.5 scoring workflow; dataset
integration, a passing session protocol and correct conversational behavior are
separate results.

The former event-anchored `benchmark_duplex_v15 score` command is removed. For
existing recordings, use the `benchmark_duplex_reference` phases below:
`export`, `asr`, `timing`, optionally `prepare-judge`/`judge`, then `summarize`.
Write fresh scoring directories and preserve old `fdb-v15-event-v1` results.
Reference whole-file intervals use different definitions; old scores must not
be relabeled or silently compared as the same metric.

### Dependencies and source identity

The evaluation scripts are loaded from a separately obtained
[Full-Duplex-Bench checkout](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/3e799c45a045256f47d5f1c9cda90157e2d2ec9e)
at revision `3e799c45a045256f47d5f1c9cda90157e2d2ec9e`.
Their SHA-256 hashes are checked before execution. Those scripts and the behavior
prompt are not vendored here; their
[CC BY-NC 4.0 license](https://github.com/DanielLin94144/Full-Duplex-Bench/blob/3e799c45a045256f47d5f1c9cda90157e2d2ec9e/LICENSE)
applies to their use. Dataset terms are documented separately in that checkout.

Use a separate scoring environment when serving dependencies differ. The retained
reference run used Python 3.12, torch/torchaudio 2.11.0+cu130, NeMo 3.0.0,
Silero VAD 6.2.1, numpy 2.5.3, scipy 1.18.1, soundfile 0.14.0 and openai 3.19.2.
The wrapper also needs pydantic. Each phase records its actual package versions.
No command downloads a model implicitly.

ASR uses a local `nvidia/parakeet-tdt-0.6b-v2` checkpoint. The measured revision was
`ae9ad07059c7c739ffaf932226a8fe64ae2620b0`; its `.nemo` SHA-256 is
`d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c`.
CUDA ASR requires exactly one visible GPU. Stop the measured generation server
before sharing that GPU with ASR or VAD.

### Record and export

Use the existing `benchmark_duplex_v15 record` command with a model profile and
explicit server/model/dataset revisions. Omit sample limits for a full run.
The released v1.5 inventory contains 498 available pairs: 200 interruption,
98 backchannel, 100 talking-to-other and 100 background speech. Each pair runs
overlap and clean in independent sessions, giving 996 selected sessions.

```bash
python -m benchmarks.eval.benchmark_duplex_v15 record \
    --dataset-root "$FDB_DATASET" --dataset-revision "$FDB_DATASET_SHA" \
    --url "$REALTIME_URL" --model "$MODEL_ID" --model-revision "$MODEL_SHA" \
    --server-revision "$SERVER_SHA" --profile minicpmo-native-pr2377 \
    --timeout 90 --output results/recording

python -m benchmarks.eval.benchmark_duplex_reference export \
    --engine sglang --run results/recording \
    --dataset-root "$FDB_DATASET" --out results/reference-audio
```

Both native profiles request audio; server text can be retained when available,
but independent ASR provides scoring transcripts. Session deadlines are enforced
by the client. An optional server limit is not assumed to equal 240 seconds.

`benchmark_duplex_v15 transcribe` remains an optional Whisper diagnostic. Its
output is not used here: reference ASR transcribes all four input/output roles
across the overlap and clean recordings with Parakeet.

Export reads source artifacts without changing them and requires a new output
directory. `--dataset-root` rechecks normalization and includes uncaptured dataset
samples as missing. `--only category/id` selects an explicit diagnostic subset.
Repeated `--run` accepts disjoint shards and records any incomplete entry
superseded by a later run. Duplicate complete captures are rejected.

`--engine vllm` reads the retained native vLLM trace format: serialized float32
input, send-completion receipts and an observation-end marker. This command is
an offline importer, not a vLLM server launcher or live recorder.

### Score and resume

Run phases serially against one output directory. Use a new directory after
changing source manifests, scoring configuration or reference code. A phase can
resume matching receipts; `--retry-failed` explicitly retries failures and
`--limit N` bounds work units. ASR caches identical audio/configuration pairs.

```bash
python -m benchmarks.eval.benchmark_duplex_reference asr \
    --reference-source "$FDB_SOURCE" --tree model=results/reference-audio \
    --out results/reference-scores --nemo "$PARAKEET_CHECKPOINT" \
    --nemo-sha256 d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c \
    --device cuda

python -m benchmarks.eval.benchmark_duplex_reference timing \
    --reference-source "$FDB_SOURCE" --tree model=results/reference-audio \
    --out results/reference-scores --audio-loader soundfile

python -m benchmarks.eval.benchmark_duplex_reference prepare-judge \
    --reference-source "$FDB_SOURCE" --tree model=results/reference-audio \
    --out results/reference-scores

python -m benchmarks.eval.benchmark_duplex_reference judge \
    --reference-source "$FDB_SOURCE" --tree model=results/reference-audio \
    --out results/reference-scores --judge gpt-4o-2024-08-06 \
    --api-key-env OPENAI_API_KEY

python -m benchmarks.eval.benchmark_duplex_reference summarize \
    --reference-source "$FDB_SOURCE" --tree model=results/reference-audio \
    --out results/reference-scores
```

Configure the API key outside manifests and command arguments. `--base-url`
selects a configured compatible endpoint. The judge requests the exact reference
model and prompt, starting with seed 1; at most three attempts are retained.
A different returned model, malformed label or failed request is not a valid
behavior result. No fallback judge is selected. Missing credentials leave
behavior pending; summarization remains available without calling the API.

Use the same `--tree` mapping for each phase; repeat it to process multiple
engines. Each input/output ASR receipt, timing interval list and judge response
remains attached to its sample. Nonzero phase exits identify execution or
integrity failures. A successful export or summary means the report was written,
not that every selected sample qualified or the model behaved correctly.

### Optional custom behavior judge

`custom-judge` applies the exact reference messages to a separately pinned,
self-hosted model. Its results are **non-official** and do not replace pending
GPT-4o labels. `--source-scores` reads an existing reference scoring root;
`--out` must be an independent new directory. Existing transcripts, audio,
official requests, labels and summaries are never modified.

Save a JSON configuration with `model_id`, `model_revision`, `tokenizer_id`,
`tokenizer_revision` (both revisions must be 40-character commit hashes),
`served_model`, `precision: "bf16"`, `enable_thinking: false`, `seeds: [1, 2, 3]`,
and these explicit decoding settings:

```json
"decoding": {
  "temperature": 0.0, "top_p": 1.0, "top_k": -1,
  "min_p": 0.0, "repetition_penalty": 1.0, "max_tokens": 512
}
```

Also require `server_launch_receipt` (a JSON file path relative to the config,
or absolute) and `server_launch_receipt_sha256`. Retain the server command,
runtime version, checkpoint/tokenizer revisions and precision in that receipt.
The client verifies its hash; the returned model name alone does not establish
which checkpoint the server loaded. Configure endpoint credentials separately
in `CUSTOM_JUDGE_API_KEY`, including the configured placeholder for an endpoint
without authentication.

```bash
python -m benchmarks.eval.benchmark_duplex_reference custom-judge \
    --reference-source "$FDB_SOURCE" --tree model=results/reference-audio \
    --source-scores results/reference-scores --out results/custom-qwen3 \
    --judge-config results/qwen3-judge-config.json --base-url "$JUDGE_URL"

python -m benchmarks.eval.benchmark_duplex_reference custom-summarize \
    --reference-source "$FDB_SOURCE" --tree model=results/reference-audio \
    --source-scores results/reference-scores --out results/custom-qwen3 \
    --judge-config results/qwen3-judge-config.json
```

Matching invocations resume without resending completed samples. Changing
configuration or selection requires a new output directory; changed transcript,
audio or ASR-receipt bytes prevent result reuse. `--limit` bounds new work while
the summary retains the full selected population. `--retry-failed` retries only
failed transport/parsing outcomes, retaining earlier attempts. Wrong models,
invalid labels, and non-`stop` completion reasons remain invalid. Rates are over
valid labels only and remain null when none are valid. Per-category Wilson 95%
intervals describe dataset sampling uncertainty, not human agreement or repeated
generation variance. The custom summary keeps eligibility reasons, attempted
counts, failures and pending samples separately. Parsing retains the reference
policy, including JSON embedded in surrounding text; raw responses are retained.

### Measurement definitions

Reference export uses `derived-analysis-v2-bounded-input-jitter`: input starts
must stay within 80 ms of their 80 ms source cadence. Append `i` must finish in
order by `t0 + min((i+1)*80 ms, T_input) + 80 ms`. Legacy traces without completion
evidence retain the stricter requirement that all starts precede `T_input`.
This bounds transport jitter; it does not prove server-side processing before T.

The output window is `[0,T_input]`, anchored at the first input send. Received
audio plays FIFO with no initial buffer: initial delay, gaps and queueing remain;
late audio is never moved backward. Queued output is cropped at T, remaining
positions are silence, and the result is mono 16 kHz PCM16 with exactly the input
sample count. Healthy silence is an observation. Missing `response.done` and
post-window errors remain lifecycle diagnostics; errors inside the observation
window invalidate that window. The native protocol report remains separate.

The external timing code owns its formulas and constants: the pinned user/model
VAD merge gaps are 0.6/0.5 seconds. Stop intervals are intersections of merged
speech spans and can end when the user stops. Response intervals end at the next
strictly later model speech start. These are whole-file interval measurements,
not event-local interruption-stop or relevant-answer latency.

`summary.json` reports pooled interval means/medians and sample-cluster bootstrap
confidence intervals. This repository defines that aggregation; the reference
script produces per-sample intervals rather than one overall score. Empty
interval sets and missing behavior labels remain null, never zero latency or
0% accuracy. Counts retain every selected, eligible, invalid and unscored sample.

Simulated client playout is not acoustic latency. Fixed windows can censor output.
One generation per sample does not estimate generation variance. These commands
do not cover prosody/MOS or reproduce every model adapter from the paper.

### Verification

The repository tests exercise export, missing samples, valid silence, timing,
ASR deduplication, bounded judging and resumption. To also execute pinned external
formulas against synthetic fixtures without GPU or API calls:

```bash
FDB_REFERENCE_SOURCE="$FDB_SOURCE" python -m pytest -q \
    tests/unit_test/benchmarks/test_duplex_reference*.py
```

The public exporter reproduced all 3,970 eligible WAV files byte-for-byte from
the retained two-engine Nemotron capture. Eligibility was unchanged at 991/996
SGLang and 994/996 vLLM sessions. Reusing retained transcripts and intervals through
the public summary path reproduced both complete engine summaries, including
coverage, null behavior values and 2,000-replicate confidence intervals.


## Retained Nemotron H100 results (2026-09-26)

The historical two-engine run selected all 498 available pairs per engine,
with four additional warmup pairs excluded. Each engine used one H100 and
32 logical CPUs. All 996 corresponding cross-engine input PCM hashes matched.
The public exporter and summarizer reproduce this retained capture; it predates
the current recorder changes and does not validate their live execution.

| Identity | Value |
|---|---|
| Capture harness | `fbc5f4e43483913026678bb9d9f0bb78ac729d7d` plus sealed capture adapters |
| SGLang-Omni server / SGLang | `7f8db0aa0dae56f9fd43caadd3935993994d2791` / 0.5.19 |
| vLLM-Omni / vLLM | `a8576ccb725c4e21cd13c3eb5f9a546b21149d2b` / 0.30 |
| VoiceChat checkpoint | `a4c40ca5b4fe77db13e9840ca4a2b91becf030c8` |
| Nano-9B tokenizer | `6533e8de2c68e4536bf7c411d7a3ce5734111476` |
| Raw capture archive SHA-256 | `9eb018490ec7c15827cba1a60692e9c3244e6f708656b37513b2d18d716ba3a9` |
| Derived v2 archive SHA-256 | `e07490a12533a3a90e11ef01aea98217adb2fa87fc8fcecab0ccdca9c1ab0978` |

Both use bf16 thinkers. SGLang uses fp32 perception/codec and a bf16 talker
backbone with fp32 heads; the vLLM FAST deployment uses bf16 perception,
fp32 talker/codec and disabled talker classifier-free guidance. This compares
engine plus configuration, not an isolated serving implementation.

| Coverage | SGLang | vLLM-Omni |
|---|---:|---:|
| Captured / selected sessions | 996/996 | 996/996 |
| Strict-v1 eligible sessions (historical policy) | 974/996 | 981/996 |
| v2 eligible sessions | 991/996 | 994/996 |
| v2 eligible complete pairs | 493/498 | 496/498 |
| Successful input/output ASR roles / selected roles | 1982/1992 | 1988/1992 |
| Successful timing receipts / selected sessions | 991/996 | 994/996 |
| Technical input-timing exclusions | 5 | 2 |
| Official behavior labels / selected pairs | 0/498 | 0/498 |

These overlap timing values are pooled whole-file intervals in seconds, with
mean / median [95% sample-cluster bootstrap CI]. Interval counts are separate
from sample counts. Each engine has its own eligible population; these rows
are descriptive distributions and are not a paired ranking.

| Category | Engine | Eligible / selected | Stop intervals | Stop seconds | Response intervals | Response seconds |
|---|---|---:|---:|---|---:|---|
| user_interruption | sglang | 200/200 | 232 | 1.666 / 1.804 [1.549, 1.778] | 388 | 2.348 / 2.404 [2.299, 2.399] |
| user_interruption | vllm | 199/200 | 208 | 1.661 / 1.660 [1.547, 1.789] | 379 | 2.220 / 1.988 [2.132, 2.303] |
| user_backchannel | sglang | 98/98 | 97 | 0.719 / 0.732 [0.687, 0.754] | 149 | 2.096 / 1.956 [1.980, 2.216] |
| user_backchannel | vllm | 98/98 | 93 | 0.719 / 0.732 [0.675, 0.758] | 145 | 1.731 / 1.572 [1.629, 1.835] |
| talking_to_other | sglang | 99/100 | 114 | 1.468 / 1.692 [1.343, 1.602] | 193 | 2.119 / 1.988 [2.027, 2.205] |
| talking_to_other | vllm | 100/100 | 101 | 1.421 / 1.596 [1.297, 1.554] | 194 | 1.914 / 1.668 [1.839, 1.985] |
| background_speech | sglang | 100/100 | 112 | 1.075 / 1.148 [0.967, 1.189] | 199 | 2.080 / 2.020 [1.989, 2.171] |
| background_speech | vllm | 99/100 | 94 | 1.197 / 1.276 [1.091, 1.306] | 188 | 1.834 / 1.668 [1.756, 1.912] |

The v2 policy was declared from input metadata before full-run outputs were
inspected. It admits short final frames within the bounded completion grace;
strict-v1 results remain retained. Technical exclusions are not model behavior
failures. Empty stop sets occur in 10/497 SGLang and 46/496 vLLM overlap samples;
empty response sets occur in 0/497 and 3/496. They contribute no invented zeroes.

Every session closed, but 995/996 vLLM sessions lacked `response.done`, including
993 eligible v2 sessions. Fixed-window eligibility does not establish natural
response completion. The final VAD speech span reaches within 50 ms of the
observation boundary in 393/497 SGLang and 380/496 vLLM overlap samples, so
censoring remains material. Official behavior judging is pending; these results
support timing analysis only. No conversational-quality winner is claimed.

Archives are identified by hash for audit; no public raw-artifact download is
provided here. The commands above reproduce new captures and scoring when the
pinned dependencies, models, dataset and endpoint are available.

## MiniCPM-o-4.5 H100 results (2026-09-27)

The native MiniCPM run covered the full 498-pair population, with independent
clean/overlap sessions. The model/tokenizer revision was
`503e754207c94da6bb26850b4469f367c9ea3582`, served by SGLang-Omni
`6eed7bdac4a5fedd27dfbe89b202a92cdc01a8e5` (PR #2377), SGLang 0.5.20,
on one H100 with 32 logical CPUs. Native defaults used the checkpoint reference
voice, greedy decoding and `force_listen_count=3`. Perception/thinker/talker
resolve to BF16, with FP32 waveform generation; no quantization or CUDA graphs.
This is a model/configuration result, not an isolated engine comparison with
Nemotron.

Capture used an exported harness at `8040e30` plus the sealed working changes;
the eight recorded capture-source fingerprints match commit `e872d6c` exactly.
A sealed `sglang_record.py` wrapper invoked that path and added cleanup evidence;
the wrapper is retained campaign code outside the PR. This supports the recorder
path, not every later reference/custom-judge change.

| Coverage | MiniCPM-o |
|---|---:|
| Captured / selected sessions | 996/996 |
| Native protocol pass / matching offline replay | 996/996 each |
| v2 eligible sessions / complete pairs | 993/996; 495/498 |
| Successful ASR roles / selected roles | 1986/1992 |
| Successful timing receipts / selected sessions | 993/996 |
| Technical input-pacing exclusions | 3 |
| Official behavior labels / selected pairs | 0/498 |

All eligible ASR and timing work succeeded. The three exclusions remain in the
selected denominator and are not model behavior failures. The reference overlap
intervals below use the definitions above: pooled mean / median, seconds,
with 95% sample-cluster bootstrap intervals for the mean.

| Category | Eligible / selected | Stop intervals | Stop seconds | Response intervals | Response seconds |
|---|---:|---:|---|---:|---|
| user_interruption | 200/200 | 204 | 2.370 / 2.364 [2.248, 2.498] | 369 | 2.829 / 2.852 [2.764, 2.897] |
| user_backchannel | 98/98 | 98 | 0.728 / 0.732 [0.697, 0.760] | 104 | 3.098 / 3.060 [2.992, 3.207] |
| talking_to_other | 100/100 | 99 | 1.854 / 1.948 [1.746, 1.950] | 143 | 3.094 / 3.108 [3.010, 3.187] |
| background_speech | 99/100 | 97 | 1.416 / 1.436 [1.331, 1.499] | 116 | 3.351 / 3.364 [3.261, 3.444] |

Overlap stop/response sets are empty in 9/497 and 4/497 eligible samples;
all 496 eligible clean samples have empty stop sets. Their values remain
undefined. The last model VAD end is within 50 ms of T in 248/497 overlap samples,
a potential-censoring diagnostic rather than proof of truncated natural EOS.
Fixed-window censoring and one-generation uncertainty remain material.
Natural response completion and relevant-response correctness are not established
by these timing distributions.

Canonical strict-v1 analysis of all 996 raw traces retains 982 variants / 488
pairs, compared with v2's 993 / 495. All 982 both-valid outputs have identical
PCM, so strict timing reuses their verified intervals. Strict/v2 overlap means
are 1.758/1.759 seconds for stop and 3.001/3.002 for response. Eleven additional
strict tail exclusions are technical input-window exclusions, not model failures.
All 996 actual normalized inputs match both historical Nemotron cohorts.

The three v2 pacing exclusions are `talking_to_other/85` clean,
`background_speech/20` overlap and `background_speech/23` clean. Their native
protocol checks passed. Full archive and all 26,586 per-file hashes were verified;
ASR/timing audio and configuration receipts have no missing eligible entries.

The stopped capture/analysis archive SHA-256 is
`3e2e668a276790ca76bf8fe30086dc9c862068c2c5c905b1a2bd48fda4ecfa0c`.

An adjacent Nemotron v1.0 integration covered only five fixed subset samples:
5/5 capture/replay, 4/5 ASR/scoring. Whisper emitted a 20 ms backward word-start
sequence for the backchannel sample; validation retained that failure and its
missing score. This is not full v1.0 qualification.

## Custom judge calibration result (2026-09-27)

Qwen3-32B revision `9216db5781bf21249d130ec9da846c4624c16137` served successfully
in BF16, non-thinking mode on one H100. With the exact reference prompt and the
explicit decoding configuration above, all eight authored semantic fixtures
returned complete valid JSON, but only six matched their instruction-derived
labels. Both expected `C_UNKNOWN` cases returned `C_RESUME`: one had only a
segment begun before overlap; the other introduced an unrelated topic.

The frozen configuration was not accepted for full judging. The 16 selected
real calibration units and all 1,484 eligible pairs across the three retained
model/engine cohorts remain unjudged. All 1,494 selected pairs, including ten
technical pair exclusions, remain in the population ledgers. Class proportions
and confidence intervals are null. Six of eight authored checks is not human
agreement, real-dataset accuracy, or GPT-4o equivalence, and does not establish
that all open models or Qwen configurations are unsuitable.

The stopped custom-judge archive SHA-256 is
`7001925b86232c8ee04bf8bb2b4b7a40f79250675082895b3f222dcd03d80dfc`.
A follow-up configuration needs its own identity and fresh validation; changing
observed fixture labels or silently replacing the judge would not resolve this
failure.

## Serving H200 qualification (2026-10-02)

Benchmark base revision: `4dbac033141d252e024b6d24543f96b3ac8eb180`, with
the native metrics changes on `feat/duplex-serving-native-metrics`.
Tests and captures ran in `docker.io/hongccc/sglang-omni:dev`, image ID
`2bac261779c813672df92a8ce1cab74a4eaa18681b5c645ee89e4dc8765c6119`.
The image's SGLang 0.5.19 was upgraded to the checkouts' pinned 0.5.20;
MiniCPM-o's optional `onnx` dependency was installed (1.23.1).
PyTorch was 2.13.0+cu130 and Transformers 5.12.1. Each server used one H200.

VoiceChat server SHA: `c3c33cfa91111217b7a88bb416b1259fbe5b2af8` (#2188).
Model revision: `443794ea956ef0065f001967ffd00e77f519cb39`.
Server command, from that checkout:

```bash
python examples/run_nemotron_voicechat_duplex.py \
  --model-path nvidia/NVIDIA-NemotronLabs-VoiceChat-11B --serve --port 8097
```

MiniCPM-o server SHA: `167f16580acada0e6e144f06ba4cef007a747171` (#2377).
Model revision: `503e754207c94da6bb26850b4469f367c9ea3582`.
The unmodified `examples/full_duplex/minicpmo.yaml` supplies `max_sessions=2`,
sampled decoding, `force_listen_count=3`, and stage memory fractions
0.12/0.52/0.15/0.15. Server command, from that checkout:

```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang_omni.cli serve \
  --config examples/full_duplex/minicpmo.yaml \
  --model-path openbmb/MiniCPM-o-4_5 --enable-realtime
```

The input was the VoiceChat checkpoint's `turn_taking.wav`, normalized by the
benchmark to mono 16 kHz PCM16, duration 41.08 s. Captures used the following
command with the corresponding profile, endpoint, revisions and config JSON:

```bash
python -m benchmarks.duplex.serving --url "$URL" --audio "$INPUT" \
  --profile "$PROFILE" --concurrencies "$LEVELS" --output-dir "$OUTPUT" \
  --model-id "$MODEL" --model-revision "$MODEL_SHA" \
  --server-sha "$SERVER_SHA" --server-config "$CONFIG_JSON" --timeout-s 180
```

VoiceChat used levels `1` and a separate `2,4` sweep; MiniCPM-o used `1,2,4`.
Each sweep included one excluded warmup and one measured repeat per level.
Raw traces, input receipts, config JSON and provenance summaries are retained
in the workspace's `duplex-qualification/` directory, outside the repository.

| Model | C | Admitted/attempted | Success/admitted | Unit lag p95 (ms) | Response TTFA p95 (ms) | Response gap-excess p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| VoiceChat | 1 | 1/1 | 1/1 | 3239.8 | 0.1 | 21.0 |
| VoiceChat | 2 | 1/2 | 1/1 | 3596.6 | 0.1 | 22.4 |
| VoiceChat | 4 | 1/4 | 1/1 | 3434.3 | 0.1 | 23.9 |
| MiniCPM-o | 1 | 1/1 | 1/1 | 3.4 | 0.3 | N/A |
| MiniCPM-o | 2 | 2/2 | 2/2 | 544.5 | 0.3 | 256.0 |
| MiniCPM-o | 4 | 2/4 | 2/2 | 515.6 | 1076.1 | 426.1 |

Additional serving metrics are below. Timing values are milliseconds and use
p95 unless the row specifies p99; counts, ratios and durations retain their
stated units. Output samples and duration are summed across admitted sessions.
The raw summaries retain p50/p75/p95/p99/max and sample counts.

| Metric | VoiceChat C1 | VoiceChat C2 | VoiceChat C4 | MiniCPM-o C1 | MiniCPM-o C2 | MiniCPM-o C4 |
|---|---:|---:|---:|---:|---:|---:|
| Client timing valid | Yes | Yes | Yes | Yes | Yes | Yes |
| Full-unit samples | 513 | 513 | 513 | 41 | 82 | 82 |
| Audio responses | 1 | 1 | 1 | 1 | 4 | 6 |
| Response gap samples | 513 | 513 | 513 | 0 | 20 | 7 |
| Response max drift | 3352.2 | 3709.1 | 3507.8 | 0.0 | 262.7 | 415.2 |
| Response required buffer | 3352.2 | 3709.1 | 3507.8 | 0.0 | 262.7 | 415.2 |
| Send lateness p95 | 1.7 | 1.6 | 1.7 | 1.7 | 1.8 | 1.8 |
| Send lateness p99 | 2.0 | 2.1 | 2.1 | 2.0 | 2.1 | 2.1 |
| Loop lag p99 | 1.5 | 1.5 | 1.5 | 1.4 | 1.5 | 1.5 |
| Sends >20 ms late | 0 | 0 | 0 | 0 | 0 | 0 |
| Late-send rate | 0% | 0% | 0% | 0% | 0% | 0% |
| Session TTFA | 102.3 | 107.3 | 109.6 | N/A | N/A | N/A |
| Session output gap | 98.9 | 99.6 | 100.6 | N/A | N/A | N/A |
| Session gap excess | 18.9 | 19.6 | 20.6 | N/A | N/A | N/A |
| Output coverage | 100.1% | 100.1% | 100.1% | N/A | N/A | N/A |
| Continuous output drift | 3234.9 | 3584.9 | 3419.6 | N/A | N/A | N/A |
| Final output drift | 3352.2 | 3709.1 | 3507.8 | N/A | N/A | N/A |
| Session required buffer | 3352.2 | 3709.1 | 3507.8 | N/A | N/A | N/A |
| Media schedule lag | 3325.7 | 3680.6 | 3517.6 | N/A | 1722.3 | 1831.2 |
| Media actual-send lag | 3324.9 | 3680.2 | 3516.8 | N/A | 1722.0 | 1830.4 |
| Sessions with underrun | 1/1 | 1/1 | 1/1 | N/A | N/A | N/A |
| Underrun count | 470 | 466 | 469 | N/A | N/A | N/A |
| Total underrun duration | 3080.7 | 3404.6 | 3282.6 | N/A | N/A | N/A |
| Worst underrun duration | 78.8 | 76.7 | 88.3 | N/A | N/A | N/A |
| Underrun ratio | 7.5% | 8.3% | 8.0% | N/A | N/A | N/A |
| Total output duration (s) | 41.12 | 41.12 | 41.12 | 1.24 | 23.92 | 13.24 |
| Total output samples | 906696 | 906696 | 906696 | 29760 | 574080 | 317760 |

All measured runs had valid client timing. Loop lag p99 was 1.4-1.5 ms;
send lateness p95 was 1.6-1.8 ms and p99 was 2.0-2.1 ms, below the 20 ms
limit. VoiceChat contributed 513 full-unit samples per run; MiniCPM-o
contributed 41 per admitted session. The terminal partial units were excluded.
MiniCPM-o C1 produced one audio packet, so no within-response gap sample exists;
C2 and C4 contributed 20 and 7 gaps, and 4 and 6 audio responses respectively.
These are single-repeat observations, not a stable capacity estimate.

VoiceChat C1 additionally measured session TTFA 102.3 ms, coverage 100.1%
(terminal padding included), final/required-buffer drift 3352.2 ms and underrun
duration 3080.7 ms (7.5%). This drift shows the default serving configuration
did not keep pace throughout the recording despite valid client pacing.
MiniCPM-o continuous-only fields were N/A in every session and aggregate.
C4 rejected exactly two MiniCPM-o attempts; VoiceChat C2/C4 rejected one/three.
All admitted sessions drained input and closed without protocol errors.

Validation in the same container:

```bash
python -m pytest tests/unit_test/benchmarks/test_duplex_serving.py \
  tests/unit_test/benchmarks/test_duplex_client.py \
  tests/unit_test/benchmarks/test_duplex_oracle.py \
  tests/unit_test/benchmarks/test_duplex_profiles.py -q
python -m pre_commit run --all-files
```

Outcome: 125 tests passed; repository-wide pre-commit checks passed.
The final N/A fallback test was additionally rerun with the serving suite:
11 passed. Container startup alone was not counted as qualification.
