# SPDX-License-Identifier: Apache-2.0
"""Text-generation integration smoke tests for the Ming-Omni thinker.

This borrows the Qwen3-Omni server fixture style, but falls back to
short/medium/long text-generation smoke checks because Ming's thinker length
override is not wired through the launcher yet.
"""

from __future__ import annotations

import subprocess
import sys

import httpx
import pytest

from sglang_omni.utils import find_available_port
from tests.utils import start_server_from_cmd, stop_server

MODEL_PATH = "inclusionAI/Ming-flash-omni-2.0"
STARTUP_TIMEOUT = 1200
_TEST_PROMPTS = [
    ("short", "Count from one to five."),
    ("medium", " ".join(["Tell me a story about a robot."] * 5)),
    ("long", " ".join(["Describe the solar system in detail."] * 20)),
]


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    port = find_available_port()
    log_file = tmp_path_factory.mktemp("ming_thinker_length_logs") / "server.log"
    cmd = [
        sys.executable,
        "examples/run_ming_omni_server.py",
        "--model-path",
        MODEL_PATH,
        "--port",
        str(port),
        "--model-name",
        "ming-omni",
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
    proc.port = port
    yield proc
    stop_server(proc)


@pytest.mark.benchmark
@pytest.mark.parametrize("label,prompt", _TEST_PROMPTS)
def test_thinker_length(
    server_process: subprocess.Popen, label: str, prompt: str
) -> None:
    payload = {
        "model": "ming-omni",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
    }

    with httpx.Client(trust_env=False, timeout=180) as client:
        resp = client.post(
            f"http://127.0.0.1:{server_process.port}/v1/chat/completions",
            json=payload,
        )

    assert resp.status_code == 200, f"{label}: {resp.text}"
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content.strip(), f"{label}: {body}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
