from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (PROJECT_ROOT / rel).read_text()


def _normalize_needs(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return value


def test_server_launcher_thinker_max_seq_len_is_opt_in():
    src = _read("examples/run_ming_omni_server.py")
    option_start = src.index('"--thinker-max-seq-len"')
    option = src[option_start : src.index(")", option_start)]
    assert "default=None" in option
    assert (
        "If set, override thinker stage max_seq_len; otherwise inherit pipeline default."
        in option
    )
    assert "if args.thinker_max_seq_len is not None" in src
    assert '"thinker_max_seq_len": int(args.thinker_max_seq_len)' in src


def test_bootstrap_validates_prompt_length():
    src = _read("sglang_omni/models/ming_omni/bootstrap.py")
    assert "longer than the model's " in src
    assert "context length" in src
    assert "Requested token count exceeds the model's maximum context" in src


def test_decode_executor_emits_finish_reason_and_usage():
    src = _read("sglang_omni/models/ming_omni/stages.py")
    assert 'result["finish_reason"] = finish_reason' in src
    assert '"usage"' in src
    assert '"prompt_tokens"' in src
    assert '"completion_tokens"' in src


def test_ci_workflow_job_layout():
    import yaml

    wf = yaml.safe_load(_read(".github/workflows/test-ming-omni-ci.yaml"))
    jobs = wf["jobs"]
    expected = {
        "docs": (
            "docs",
            60,
            [],
            "tests/docs/ming_omni/test_docs_ming_omni.py",
        ),
        "stage-1-thinker": (
            "stage 1 - thinker length integration",
            80,
            ["docs"],
            "tests/test_model/test_ming_omni_thinker_length.py",
        ),
        "stage-2-tts": (
            "stage 2 - TTS (RTF perf + WER)",
            120,
            ["docs", "stage-1-thinker"],
            "tests/test_model/test_ming_omni_tts_ci.py",
        ),
        "stage-3-mmmu": (
            "stage 3 - MMMU accuracy + speed",
            180,
            ["docs", "stage-1-thinker"],
            "tests/test_model/test_ming_omni_mmmu_ci.py",
        ),
        "stage-4-mmsu": (
            "stage 4 - MMSU audio-in understanding",
            90,
            ["docs", "stage-1-thinker"],
            "tests/test_model/test_ming_omni_mmsu_ci.py",
        ),
    }
    assert set(jobs) == set(expected)
    for job_id, (name, timeout, needs, pytest_target) in expected.items():
        job = jobs[job_id]
        commands = "\n".join(
            str(step.get("run", "")) for step in job["steps"] if "run" in step
        )
        assert job["name"] == name
        assert job["timeout-minutes"] == timeout
        assert _normalize_needs(job.get("needs")) == needs
        assert pytest_target in commands


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_model/test_ming_omni_mmmu_ci.py",
        "tests/test_model/test_ming_omni_mmsu_ci.py",
        "tests/test_model/test_ming_omni_thinker_length.py",
        "tests/test_model/test_ming_omni_tts_ci.py",
    ],
)
def test_stage_files_share_tp_and_startup_constants(path: str):
    src = _read(path)
    assert "--tp-size" in src
    assert "THINKER_TP_SIZE = 2" in src
    assert "str(THINKER_TP_SIZE)" in src
    assert "STARTUP_TIMEOUT = 2400" in src


def test_docs_smoke_test_pins_tp_size_in_command():
    src = _read("tests/docs/ming_omni/test_docs_ming_omni.py")
    command_start = src.index('"--tp-size"')
    command = src[command_start : src.index("]", command_start)]
    assert "--tp-size" in command
    assert "THINKER_TP_SIZE = 2" in src
    assert "str(THINKER_TP_SIZE)" in command
    assert "STARTUP_TIMEOUT = 2400" in src


def test_mmmu_file_caps_max_tokens():
    src = _read("tests/test_model/test_ming_omni_mmmu_ci.py")
    assert "MAX_TOKENS = 64" in src
    assert "max_tokens=MAX_TOKENS" in src


def test_mmsu_file_keeps_text_only_modality():
    src = _read("tests/test_model/test_ming_omni_mmsu_ci.py")
    assert 'modalities="text"' in src
