#!/usr/bin/env python3
"""Compare Qwen3-Omni v1 text-only outputs across thinker TP sizes.

This harness is intended for remote GPU validation. It starts a baseline
OpenAI-compatible server, collects deterministic text-only completions, then
starts a TP test server and requires exact stripped output matches.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

MODEL_NAME = "qwen3-omni"
PROMPTS = [
    "What is 2 + 2?",
    "Capital of France?",
    "Reverse the string 'hello'.",
    "Name three primary colors.",
    "What is the speed of light in m/s?",
]


@dataclass(frozen=True)
class CompletionResult:
    prompt: str
    output: str


def http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ensure_port_available(host: str, port: int) -> None:
    try:
        addrinfos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise RuntimeError(f"Unable to resolve requested server host {host!r}: {exc}") from exc

    sockets: list[socket.socket] = []
    seen_addresses = set()
    try:
        for family, socktype, proto, _, sockaddr in addrinfos:
            address_key = (family, sockaddr)
            if address_key in seen_addresses:
                continue
            seen_addresses.add(address_key)
            sock = socket.socket(family, socktype, proto)
            try:
                sock.bind(sockaddr)
            except OSError as exc:
                sock.close()
                raise RuntimeError(
                    f"Requested server address {host}:{port} is not available for binding: {exc}"
                ) from exc
            sockets.append(sock)
    finally:
        for sock in sockets:
            sock.close()


def send_server_signal(proc: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    # The server is started with start_new_session=True, so the original
    # parent PID is also the process-group ID even if the parent exits first.
    process_group_id = proc.pid
    try:
        os.killpg(process_group_id, sig)
    except (ProcessLookupError, PermissionError):
        pass


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_server_group_exit(
    proc: subprocess.Popen[bytes],
    process_group_id: int,
    *,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while process_group_exists(process_group_id):
        proc.poll()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, timeout)
        time.sleep(min(0.2, remaining))


def wait_until_ready(
    proc: subprocess.Popen[bytes],
    command: list[str],
    host: str,
    port: int,
    *,
    startup_timeout: float,
    request_timeout: float,
) -> None:
    url = f"http://{host}:{port}/v1/models"
    deadline = time.monotonic() + startup_timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        exit_code = proc.poll()
        if exit_code is not None:
            raise RuntimeError(
                "Server process exited before readiness "
                f"(exit code {exit_code}): {' '.join(command)}"
            )
        try:
            http_json(url, timeout=request_timeout)
            return
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(2.0)

    message = f"Timed out waiting for server readiness at {url}"
    if last_error is not None:
        message = f"{message}; last error: {last_error}"
    raise TimeoutError(message)


def chat_completion(
    host: str,
    port: int,
    prompt: str,
    *,
    max_tokens: int,
    request_timeout: float,
) -> str:
    body = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "stream": False,
    }
    response = http_json(
        f"http://{host}:{port}/v1/chat/completions",
        method="POST",
        payload=body,
        timeout=request_timeout,
    )
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected chat completion response: {response!r}") from exc
    if not isinstance(content, str):
        raise ValueError(f"Unexpected non-string chat completion content: {content!r}")
    return content


def start_server(
    model_path: str,
    host: str,
    port: int,
    tp_size: int,
) -> tuple[subprocess.Popen[bytes], list[str]]:
    ensure_port_available(host, port)
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni_v1.cli.cli",
        "serve",
        "--model-path",
        model_path,
        "--text-only",
        "--host",
        host,
        "--port",
        str(port),
        "--thinker-tp-size",
        str(tp_size),
    ]
    print(f"Starting TP={tp_size} server on {host}:{port}: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, start_new_session=True), cmd


def stop_server(proc: subprocess.Popen[bytes], *, shutdown_timeout: float = 20.0) -> None:
    process_group_id = proc.pid
    send_server_signal(proc, signal.SIGTERM)
    try:
        wait_for_server_group_exit(proc, process_group_id, timeout=shutdown_timeout)
    except subprocess.TimeoutExpired:
        print("Server did not terminate promptly; killing it.", file=sys.stderr, flush=True)
        send_server_signal(proc, signal.SIGKILL)
        wait_for_server_group_exit(proc, process_group_id, timeout=shutdown_timeout)
    finally:
        try:
            proc.wait(timeout=0)
        except subprocess.TimeoutExpired:
            pass


def collect_results(
    *,
    model_path: str,
    host: str,
    port: int,
    tp_size: int,
    prompts: list[str],
    max_tokens: int,
    startup_timeout: float,
    request_timeout: float,
) -> list[CompletionResult]:
    proc, command = start_server(model_path, host, port, tp_size)
    try:
        wait_until_ready(
            proc,
            command,
            host,
            port,
            startup_timeout=startup_timeout,
            request_timeout=request_timeout,
        )
        print(f"TP={tp_size} server is ready.", flush=True)

        results = []
        for index, prompt in enumerate(prompts, start=1):
            print(f"TP={tp_size} prompt {index}/{len(prompts)}: {prompt}", flush=True)
            output = chat_completion(
                host,
                port,
                prompt,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
            )
            results.append(CompletionResult(prompt=prompt, output=output))
        return results
    finally:
        stop_server(proc)


def compare_results(
    baseline: list[CompletionResult],
    test: list[CompletionResult],
    *,
    tp_baseline: int,
    tp_test: int,
) -> bool:
    for base_result, test_result in zip(baseline, test, strict=True):
        if base_result.prompt != test_result.prompt:
            raise ValueError(
                f"Prompt ordering mismatch: {base_result.prompt!r} != {test_result.prompt!r}"
            )
        if base_result.output.strip() != test_result.output.strip():
            print("Mismatch detected.", file=sys.stderr)
            print(f"Prompt: {base_result.prompt}", file=sys.stderr)
            print(f"Baseline TP={tp_baseline}: {base_result.output}", file=sys.stderr)
            print(f"Test TP={tp_test}: {test_result.output}", file=sys.stderr)
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp-baseline", type=int, default=1)
    parser.add_argument("--tp-test", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port-baseline", type=int, default=18001)
    parser.add_argument("--port-test", type=int, default=18002)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP request timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    baseline = collect_results(
        model_path=args.model_path,
        host=args.host,
        port=args.port_baseline,
        tp_size=args.tp_baseline,
        prompts=PROMPTS,
        max_tokens=args.max_tokens,
        startup_timeout=args.startup_timeout,
        request_timeout=args.timeout,
    )
    test = collect_results(
        model_path=args.model_path,
        host=args.host,
        port=args.port_test,
        tp_size=args.tp_test,
        prompts=PROMPTS,
        max_tokens=args.max_tokens,
        startup_timeout=args.startup_timeout,
        request_timeout=args.timeout,
    )

    if not compare_results(
        baseline,
        test,
        tp_baseline=args.tp_baseline,
        tp_test=args.tp_test,
    ):
        return 1

    print(
        f"All {len(PROMPTS)} prompts matched exactly after strip(): "
        f"TP={args.tp_baseline} vs TP={args.tp_test}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
