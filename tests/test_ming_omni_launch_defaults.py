# SPDX-License-Identifier: Apache-2.0
"""Static launch-configuration guards for Ming-Omni CI."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ming_speech_server_accepts_tp_size(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples import run_ming_omni_speech_server

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_ming_omni_speech_server.py", "--tp-size", "2"],
    )

    args = run_ming_omni_speech_server.parse_args()

    assert args.tp_size == 2


def test_ming_text_server_forwards_thinker_max_seq_len() -> None:
    source = (PROJECT_ROOT / "examples/run_ming_omni_server.py").read_text()

    assert "if args.thinker_max_seq_len is not None" in source
    assert 'overrides["thinker_max_seq_len"] = args.thinker_max_seq_len' in source


def test_ming_thinker_context_validation_uses_openai_bad_request_markers() -> None:
    source = (
        PROJECT_ROOT / "sglang_omni/models/ming_omni/pipeline/engine_io.py"
    ).read_text()

    assert "longer than the model's " in source
    assert "context length" in source
    assert "Requested token count exceeds the model's maximum context length" in source


def test_ming_decode_preserves_finish_reason() -> None:
    engine_io = (
        PROJECT_ROOT / "sglang_omni/models/ming_omni/pipeline/engine_io.py"
    ).read_text()
    stages = (
        PROJECT_ROOT / "sglang_omni/models/ming_omni/pipeline/stages.py"
    ).read_text()

    assert "finished_reason" in engine_io
    assert 'result["finish_reason"] = finish_reason' in stages


@pytest.mark.parametrize(
    "relative_path",
    [
        "tests/docs/ming_omni/test_docs_ming_omni.py",
        "tests/test_model/test_ming_omni_thinker_length.py",
        "tests/test_model/test_ming_omni_tts_ci.py",
        "tests/test_model/test_ming_omni_mmmu_ci.py",
        "tests/test_model/test_ming_omni_mmsu_ci.py",
    ],
)
def test_ming_ci_server_fixtures_pin_tp_size(relative_path: str) -> None:
    source = (PROJECT_ROOT / relative_path).read_text()

    assert '"--tp-size"' in source
