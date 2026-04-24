# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the Ming-Omni README quick-start chat completion example.

This keeps the documented quick-start path live in CI without asserting output
language or exact wording. Gate D / English output stability is deferred to the
first GPU CI run, so this test only checks API liveness and response shape.

Usage:
    pytest tests/docs/ming_omni/test_docs_ming_omni.py -s -x
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


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the Ming-Omni server and wait until healthy."""
    port = find_available_port()
    log_file = tmp_path_factory.mktemp("server_logs") / "server.log"
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


@pytest.mark.docs
@pytest.mark.benchmark
def test_readme_chat_completion_smoke(server_process: subprocess.Popen) -> None:
    """POST the README quick-start payload and assert liveness-only response shape."""
    payload = {
        "model": "ming-omni",
        "messages": [
            {"role": "user", "content": "Say 'hello world' and nothing else."}
        ],
        "max_tokens": 32,
        "temperature": 0.0,
    }
    url = f"http://127.0.0.1:{server_process.port}/v1/chat/completions"

    with httpx.Client(trust_env=False, timeout=120) as client:
        response = client.post(url, json=payload)

    assert response.status_code == 200, (
        f"Expected HTTP 200 from {url}; got {response.status_code}: {response.text}"
    )
    result = response.json()
    choices = result.get("choices")
    assert choices, f"Expected non-empty choices in response: {result}"
    content = choices[0]["message"]["content"]
    assert isinstance(content, str) and content.strip(), (
        f"Expected non-empty string content in first choice: {choices[0]}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
