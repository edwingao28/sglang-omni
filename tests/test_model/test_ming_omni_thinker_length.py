# SPDX-License-Identifier: Apache-2.0
"""Integration tests for Ming-Omni thinker length validation.

Starts a real text-only Ming-Omni server and verifies that:
1. overlong prompts return HTTP 400 with SGLang-aligned wording;
2. prompt + max_tokens overflow returns HTTP 400 with SGLang-aligned wording;
3. decode hitting max_tokens returns HTTP 200 with finish_reason="length".
"""

from __future__ import annotations

import subprocess
import sys

import httpx
import pytest

from sglang_omni.utils import find_available_port
from tests.utils import start_server_from_cmd, stop_server

MODEL_PATH = "inclusionAI/Ming-flash-omni-2.0"
MODEL_NAME = "ming-omni"
THINKER_MAX_SEQ_LEN = 128
STARTUP_TIMEOUT = 2400
REQUEST_TIMEOUT = 180
THINKER_TP_SIZE = 2


def _post_chat(
    port: int, payload: dict, timeout: int = REQUEST_TIMEOUT
) -> httpx.Response:
    with httpx.Client(trust_env=False, timeout=timeout) as client:
        return client.post(
            f"http://localhost:{port}/v1/chat/completions",
            json=payload,
        )


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    port = find_available_port()
    log_file = tmp_path_factory.mktemp("ming_thinker_length_logs") / "server.log"
    cmd = [
        sys.executable,
        "examples/run_ming_omni_server.py",
        "--model-path",
        MODEL_PATH,
        "--model-name",
        MODEL_NAME,
        "--thinker-max-seq-len",
        str(THINKER_MAX_SEQ_LEN),
        "--port",
        str(port),
        "--tp-size",
        str(THINKER_TP_SIZE),
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
    proc.port = port
    yield proc
    stop_server(proc)


def test_overlong_prompt_returns_400(server_process: subprocess.Popen) -> None:
    resp = _post_chat(
        server_process.port,
        {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": "a " * 10000,
                }
            ],
            "max_tokens": 16,
            "stream": False,
        },
    )

    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert "The input (" in body["detail"]
    assert "is longer than the model's context length" in body["detail"]


def test_total_token_overflow_returns_400(server_process: subprocess.Popen) -> None:
    resp = _post_chat(
        server_process.port,
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 200,
            "stream": False,
        },
    )

    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert (
        "Requested token count exceeds the model's maximum context length"
        in body["detail"]
    )


def test_length_finish_reason_is_preserved(server_process: subprocess.Popen) -> None:
    resp = _post_chat(
        server_process.port,
        {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": "Count from 1 to 20, separated by commas.",
                }
            ],
            "max_tokens": 2,
            "stream": False,
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["choices"][0]["finish_reason"] == "length"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
