# Tensor Parallelism (TP) Gap Analysis

> Research for [#254](https://github.com/sgl-project/sglang-omni/issues/254) and [#255](https://github.com/sgl-project/sglang-omni/issues/255)
> Based on PR [#270](https://github.com/sgl-project/sglang-omni/pull/270) (`feat/tp-driver-worker`)

## Executive Summary

**TP=2 (text-only pipeline)**: ~95% complete, validated on Qwen3-Omni 2xH100.

**TP=2 (full speech pipeline)**: ~80% complete. `gpu_placement` conflict is a **blocking issue**.

**TP=4**: ~75% complete. Same `gpu_placement` conflict + needs end-to-end validation.

## What's Already Done (PR #270)

| Component | Status | Files |
|-----------|--------|-------|
| TP driver-worker (follower loop, spawn, NCCL) | Done | `engines/tp/follower.py` |
| Batch serialization & broadcast | Done | `engines/tp/serialization.py` |
| NCCL group init in OmniEngine factory | Done | `engines/omni/factory.py:241-277` |
| Follower lifecycle (stop signal, terminate) | Done | `engines/omni/engine.py:150-167` |
| mp_runner daemon=False for TP stages | Done | `pipeline/mp_runner.py:359-373` |
| Qwen3-Omni Thinker model TP layers | Done | `models/qwen3_omni/thinker.py` |
| Qwen3-Omni Talker model TP layers | Done | `models/qwen3_omni/talker.py` |
| Ming-Omni Thinker model TP layers | Done | `models/ming_omni/thinker.py` |
| Ming pipeline config (json_model_override) | Done | `models/ming_omni/pipeline/stages.py:280-298` |
| Unit tests (14 tests) | Done | Serialization, follower registration |
| CLI --tp-size flags | Done | `examples/run_*_server.py` |

## Remaining Gaps

### P0: Blocking for TP>1 with Speech Pipeline

#### 1. `gpu_placement` Conflict (CRITICAL)

**Problem**: When `tp_size > 1`, the thinker occupies GPUs `[base_gpu_id, base_gpu_id+1, ..., base_gpu_id+tp_size-1]` (`follower.py:217`). But `gpu_placement` maps other stages to fixed GPU IDs that overlap.

**Current defaults**:
```python
# Ming (config.py:119)
gpu_placement = {"thinker": 0, "talker": 1}

# Qwen3 Speech (config.py:125)
gpu_placement = {"thinker": 0, "talker_ar": 1, "code_predictor": 1, "code2wav": 1}
```

With TP=2: thinker uses GPU 0,1 but talker is also on GPU 1 -> **memory conflict**.
With TP=4: thinker uses GPU 0,1,2,3 but talker is on GPU 1 -> **crash**.

**Why TP=2 validation passed**: It used `Qwen3OmniPipelineConfig` (text-only), which has no `gpu_placement` at all (all stages default to GPU 0, and only thinker is a GPU stage).

**Fix needed**: Auto-adjust `gpu_placement` based on `tp_size`. When `tp_size=N`, non-thinker stages should be placed on GPU N or later.

```python
# Example fix in PipelineConfig.__init__ or server launch:
if tp_size > 1:
    for stage_name, gpu_id in gpu_placement.items():
        if stage_name != "thinker":
            gpu_placement[stage_name] = max(gpu_id, tp_size)
```

**Hardware implication**: TP=4 + talker needs at minimum 5 GPUs (4 for thinker + 1 for talker).

#### 2. Ming `json_model_override` Missing MoE Fields

**File**: `models/ming_omni/pipeline/stages.py:287-295`

The model override JSON sent to SGLang is missing MoE-specific fields:
```python
# Currently:
{
    "architectures": ["BailingMoeV2ForCausalLM"],
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "vocab_size": 157184,
}
# Missing: num_experts, moe_intermediate_size, intermediate_size
```

SGLang's FusedMoE setup may fail or use incorrect defaults without these fields.

### P1: Important but Not Blocking

#### 3. No `tp_size > num_experts` Validation for Ming

Qwen3 has this check (`thinker.py:389`), Ming doesn't. With 256 experts this won't trigger in practice (nobody runs TP=256), but it's a safety gap.

#### 4. Silent Exception in Follower Shutdown

`engine.py:157`: `except Exception: pass` swallows errors. Should log at WARNING level.

#### 5. Follower Health Monitoring

If a follower process dies mid-inference, rank 0 won't detect it until the next `broadcast_pyobj` hangs. No watchdog exists.

### P2: Nice to Have

#### 6. FishAudio S2Pro Hardcoded `tp_size=1`

`models/fishaudio_s2_pro/pipeline/stages.py:318` hardcodes TP=1 despite model layers being TP-capable.

#### 7. End-to-End Validation Matrix

| Config | TP=1 | TP=2 | TP=4 |
|--------|------|------|------|
| Qwen3 text-only | Validated | **Validated (2xH100)** | Not tested |
| Qwen3 speech | Validated | **Not tested (gpu_placement conflict)** | Not tested |
| Ming text | Validated | Not tested | Not tested |
| Ming speech | Validated | **Not tested (gpu_placement conflict)** | Not tested |

## Model Architecture TP Compatibility

### Ming-flash-omni-2.0 (208GB MoE)
- `num_attention_heads = 32` -> TP=4: 32/4=8 per rank
- `num_kv_heads = 4` -> TP=4: 4/4=1 per rank (minimum, but supported via `max(1, ...)`)
- `num_experts = 256` -> TP=4: 256/4=64 per rank
- **Memory**: 208GB / 4 = 52GB per GPU (fits in H100 80GB without CPU offload)

### Qwen3-Omni-30B-A3B
- `num_attention_heads = 32`, `num_kv_heads = 4`, `num_experts = 128`
- All divisible by 4.
- **Memory**: ~60GB / 4 = ~15GB per GPU

## Recommended PR Strategy

### Option A: Single PR (Recommended)

Keep everything in PR #270, add:
1. Fix `gpu_placement` auto-adjustment (~30 lines in `PipelineConfig.__init__` or each config)
2. Add missing MoE fields to Ming model override (~5 lines)
3. Add `tp_size` validation for Ming (~5 lines)
4. Improve shutdown logging (~1 line)
5. Update validation matrix documentation

**Total: ~50-80 lines of new code**

**Rationale**: All TP components are tightly coupled. Splitting into multiple PRs increases integration risk and review overhead for marginal benefit. The remaining work is small relative to what's already done.

### Option B: Two PRs (If Preferred)

**PR A** (PR #270 as-is): Core TP infra + text-only TP support
- Merge what's there now
- Closes #255 partially

**PR B**: Speech pipeline TP + TP=4
- `gpu_placement` auto-adjustment
- Ming MoE fields fix
- Validation for speech + TP>1
- Closes #255 fully, addresses #254 Phase 1.1

### Recommendation

**Go with Option A** (single PR #270). The `gpu_placement` fix is essential for production use and should not be deferred. The total remaining work is small enough to add to the existing PR without making it unwieldy.

**Do NOT create a new tracking issue** - Issue #255 already tracks this precisely, and Issue #254 serves as the roadmap umbrella.
