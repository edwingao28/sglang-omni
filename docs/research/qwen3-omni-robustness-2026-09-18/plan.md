# Qwen3-Omni robustness follow-up

This proposal collects Richard's feedback on Qwen3-Omni serving. The goal is
to make performance more consistent when users change machines, launch settings,
or inputs. The observations below are reported feedback, not new benchmark
results. This document does not change runtime defaults.

## 1. Make the default launch work well

Richard reported up to 8x worse p95 time to first text token (TTFT) on a
192-core machine without CPU binding. Setting `OMP_NUM_THREADS=8` or binding
processes to CPUs reduced latency substantially. The same changes had a much
smaller effect on vLLM-Omni in his comparison of the two public main branches.

First reproduce that case with exact commits, launch commands, CPU topology,
and absolute latency values. Compare the default launch, a thread limit, CPU
binding, and both changes together, on the same inputs and hardware.

Too many threads, delays in the scheduler, and cross-socket memory access are
possible causes. Low GPU utilization alone does not tell us which one is
responsible. Also check whether requests are waiting for KV-cache capacity.

Use the result to choose sensible thread and preprocessing-concurrency defaults.
Respect the CPUs available to the process or container and preserve explicit
user settings. Log the settings that actually take effect. Check the candidate
on a smaller CPU allocation too, so a fix for a large host does not become a
problem on a smaller one.

**Done when:** the reported slowdown is explained by a repeatable comparison,
and the chosen default improves that case without harming the smaller setup.
Document any CPU binding that still needs to be configured by the user.

## 2. Check smooth playback as well as the first packet

Richard reported stalls with a two-frame first audio chunk at concurrency 32
and above, and suggested four frames as the default.

Compare two and four frames at concurrency 32 and 64, with a low-concurrency
control. Keep the player's initial buffering policy fixed. Measure how soon
audio arrives, how soon playback starts, and how often playback runs out of
audio. Include the duration of those stalls, throughput, and complete output.

Changing the server's first chunk and adding client buffering are separate
changes. Four frames should remain a candidate until this comparison shows the
tradeoff between starting later and playing more smoothly.

**Done when:** the default choice is supported by both startup latency and
playback results, with any tradeoff stated plainly.

## 3. Test more than one kind of input

Keep the existing English SeedTTS workload with reference audio as an anchor.
Add long-text requests, requests without reference audio, and Chinese requests.
Change one input dimension at a time where possible, and record actual output
length and audio duration; a long input does not guarantee a long output.

Test each optimization on at least two input types. For example, check how
EOS-tail graph benefits change with output length, and how projection-cache
benefits change with language, token repetition, and different prompt templates.
The projection cache stores per-token projections, so template reuse alone is
not a cache-hit measurement. Long utterances can also use EOS-tail graphs.

For each comparison, hold sampling settings, request order, warmup, and cache
policy fixed and record seeds where supported. Repeat baseline and candidate
runs in both orders, keeping the rounds separate. Report throughput, tail
latency, failures, and speech quality together. Include graph memory and
startup cost when evaluating extra graph coverage.

Use the [existing benchmark suite](../../../benchmarks/README.md). Start with
the comparisons needed for each change instead of running every possible
combination. Include a mixed short/long workload before calling the final
combination robust.

**Done when:** each optimization says where it helps, where it does not, and
what it costs. Passing two input types is a useful first check, not proof that
an optimization helps every workload.

## 4. Make logs answer the first debugging questions

Build on the [existing profiler](../../developer_reference/profiler.md) and
runtime counters. Periodically report:

| Question | What to record |
| --- | --- |
| Is each stage getting enough work? | Actual batch-size distribution, separately for prefill, decode, and Code2Wav. |
| Where are requests waiting? | Queue length and wait time, separating admission/KV waits from ready work. |
| Are the optimizations being used? | Graph replay versus eager execution, fallback reasons, and projection-cache hits and misses. |
| Is the scheduler computing or waiting? | Scheduler-thread CPU time and explicit GPU-synchronization wait time, with clear time windows and denominators. |
| Why does this host behave differently? | Effective thread limits, preprocessing concurrency, CPU placement, and available CPU resources. |

Count batches per actual forward, not once per request in that batch. Count
graph use during the measurement window, not graphs captured during startup.
Keep CPU time separate from elapsed wait time; a synchronization call can also
include time when the host thread was not scheduled.

Aggregate these metrics without logging every token or adding GPU synchronization
to the request path. Compare metrics enabled versus disabled to check their
overhead. Keep detailed profiling available for problems the summaries cannot
explain.

**Done when:** a slow run shows which stage is waiting and whether the relevant
optimization is active, without requiring a new full profile for those questions.

## Follow-up order

- [ ] Add effective-setting logs and the smallest useful stage metrics.
- [ ] Reproduce the host sensitivity and choose defaults from the results.
- [ ] Compare two- and four-frame first chunks with the same player.
- [ ] Check individual optimizations on the additional inputs, then recheck
      the final combination for latency, throughput, playback, and quality.

Implementation PRs should link their measurements here as this work progresses.
