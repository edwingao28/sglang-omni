"""Exact Code2Wav EOS48 checkpoint probe; the parent owns GPU admission/lifetime.

Adapted from the historical eos48-probe/checkpoint_warm_timing.py correctness
checks. This uses the supplied cumulative L5 configuration, including fused
SnakeBeta, and produces no timing or serving claim. It does not start a server.
"""

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import yaml

BASELINE_FRAMES = (2, 12, 22, 32, 35)
BASELINE_KEYS = {
    (batch, frames) for batch in (1, 2, 4, 8) for frames in BASELINE_FRAMES
} | {(16, 2), (16, 12)}
OPTIONAL_FRAMES = tuple(t for t in range(1, 49) if t not in BASELINE_FRAMES)


def pattern(frames, variant):
    rows = [
        [1 + (17 * (t * 16 + q) + 97 * frames + 31) % 2047 for q in range(16)]
        for t in range(frames)
    ]
    rows[0][0], rows[-1][-1] = 1, 2047
    return rows if variant == "A" else [[2048 - code for code in row] for row in rows]


def validate_startup(baseline, optional, report):
    report["baseline_startup"] = base = baseline.stats()
    report["optional_startup"] = extra = optional.stats() if optional else None
    assert base["enabled"] and base["build"]["published_graph_count"] == 22
    assert {
        (k["batch_size"], k["frames"]) for k in base["graph_contract"]["keys"]
    } == BASELINE_KEYS
    assert extra and extra["enabled"] and extra["build"]["published_graph_count"] == 43
    assert {
        (k["batch_size"], k["frames"]) for k in extra["graph_contract"]["keys"]
    } == {(1, t) for t in OPTIONAL_FRAMES}
    memory, added = base["memory"], extra["memory"]
    remaining = (
        memory["stage_budget_bytes"]
        - memory["loaded_model_footprint_bytes"]
        - memory["graph_footprint_bytes"]
    )
    assert added["graph_memory_budget_cap_bytes"] == remaining >= 0
    assert added["graph_footprint_bytes"] <= remaining
    report["combined_retained_bytes"] = (
        memory["loaded_model_footprint_bytes"]
        + memory["graph_footprint_bytes"]
        + added["graph_footprint_bytes"]
    )
    assert report["combined_retained_bytes"] <= memory["stage_budget_bytes"]


def run(args, report):
    import torch

    from sglang_omni.models.qwen3_omni.components import code2wav_scheduler as component

    config = yaml.safe_load(args.config.read_text())
    stage = config["stages"]["code2wav"]
    factory = stage["factory"]
    assert factory["enable_eos_cuda_graph"] is True
    assert factory["eos_cuda_graph_max_frames"] == 48
    assert factory["snake_beta_implementation"] == "fused"
    assert (
        factory["initial_codec_chunk_frames"] == 2
        and factory["stream_chunk_size"] == 10
    )
    assert factory["enable_batching"] is True and factory["batch_ceiling"] == 16
    assert stage["gpu_memory_fraction"] == 0.1
    assert torch.cuda.is_available()
    device = torch.device(args.device)
    assert device.type == "cuda"
    torch.cuda.set_device(device)
    report.update(
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        component_path=component.__file__,
        component_sha256=hashlib.sha256(
            Path(component.__file__).read_bytes()
        ).hexdigest(),
        torch=str(torch.__version__),
        device=str(device),
        gpu_uuid=str(torch.cuda.get_device_properties(device).uuid),
        factory=factory,
    )
    with torch.no_grad():
        scheduler = component.create_code2wav_scheduler(
            config["model_path"],
            device=str(device),
            enable_cuda_graph=True,
            total_gpu_memory_fraction=stage["gpu_memory_fraction"],
            **factory,
        )
        baseline, optional = (
            scheduler._cuda_graph_runner,
            scheduler._eos_cuda_graph_runner,
        )
        validate_startup(baseline, optional, report)
        model = scheduler._model
        assert not model.training and int(model.config.num_quantizers) == 16
        assert all(
            p.dtype == torch.bfloat16 and p.device == device for p in model.parameters()
        )
        assert scheduler._snake_beta_guard is not None
        saved = {}

        def gpu_input(frames, variant):
            return torch.tensor(
                pattern(frames, variant), dtype=torch.long, device=device
            ).T.unsqueeze(0)

        def replay(frames, codes, expected, label):
            original = codes.cpu().clone()
            selected = 0 if frames in BASELINE_FRAMES else 1
            before = [
                r.stats()["runtime"]["graph_replays"] for r in (baseline, optional)
            ]
            row = {
                "frames": frames,
                "label": label,
                "status": "started",
                "runner": selected,
            }
            report["checks"].append(row)
            output, meta = scheduler._forward_codes(
                codes, graph_eligible=True, is_final=True
            )
            actual = output.cpu().clone()
            after = [
                r.stats()["runtime"]["graph_replays"] for r in (baseline, optional)
            ]
            assert (
                after[selected] == before[selected] + 1
                and after[1 - selected] == before[1 - selected]
            )
            assert meta["execution_mode"] == "cuda_graph" and meta["graph_key"] == {
                "batch_size": 1,
                "frames": frames,
            }
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert bool(torch.isfinite(actual).all()) and torch.equal(
                actual, expected
            ), (frames, label, "PCM mismatch")
            assert torch.equal(codes.cpu(), original), "Input mutated"
            for new_frames in sorted({1, min(10, frames), frames}):
                length = new_frames * model.total_upsample
                assert torch.equal(
                    actual[..., -length:].reshape(-1).float(),
                    expected[..., -length:].reshape(-1).float(),
                )
            row.update(status="passed", execution=meta, shape=list(actual.shape))

        for frames in range(1, 49):
            a, b = gpu_input(frames, "A"), gpu_input(frames, "B")
            expected_a, expected_b = model(a).cpu().clone(), model(b).cpu().clone()
            assert not torch.equal(
                expected_a, expected_b
            ), "A/B input distinction unobservable"
            replay(frames, a, expected_a, "A_first")
            replay(frames, b, expected_b, "B")
            replay(frames, a, expected_a, "A_again")
            saved[frames] = expected_a
        for frames in reversed(range(1, 49)):
            replay(frames, gpu_input(frames, "A"), saved[frames], "A_after_other_keys")
        torch.cuda.synchronize(device)
        report.update(baseline_final=baseline.stats(), optional_final=optional.stats())
        assert len(report["checks"]) == 192
        report["passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "passed": False,
        "checks": [],
        "scope": "Component correctness only; no timing or serving qualification",
    }
    with args.output.open("x") as output:
        try:
            run(args, report)
        except BaseException:
            report["error"] = traceback.format_exc()
            raise
        finally:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")


if __name__ == "__main__":
    main()
