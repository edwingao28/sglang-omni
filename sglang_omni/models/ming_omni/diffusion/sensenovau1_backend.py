# SPDX-License-Identifier: Apache-2.0
"""SenseNova-U1 text-to-image backend: out-of-process worker bridge.

The official ``sensenova_u1`` package pins torch==2.8 / transformers==4.57 and
CANNOT be imported in the sglang environment (transformers 5.x breaks it). So
instead of importing it in-process, this backend spawns a long-lived subprocess
running ``sensenovau1_worker.py`` under a separate, pinned interpreter
(``/data/u1-venv/bin/python`` by default). We talk to that worker over a simple
line-oriented JSON protocol on stdin/stdout.

Importing ``sensenova_u1`` floods stdout/stderr with auto_docstring noise, so
the worker prefixes every protocol message with a ``SENTINEL`` token and this
backend ignores any stdout line that lacks it.

This module imports ONLY the stdlib + torch + PIL (for the interface types). It
never imports ``sensenova_u1`` or any torch-heavy reference code, so importing
``sglang_omni`` stays cheap and the unit tests mock the subprocess, not a GPU.

v1 is self-contained: the worker re-encodes the prompt text internally, so the
``condition_embeds`` / ``negative_condition_embeds`` are accepted for interface
parity but unused. The v2 decoupling step will route the precomputed thinker
conditioning into the worker instead.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import threading
from collections import deque
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

from sglang_omni.models.ming_omni.diffusion.backend import (
    DiffusionBackend,
    ImageGenParams,
)

logger = logging.getLogger(__name__)

SENTINEL = "@@U1@@"

_DEFAULT_WORKER_PYTHON = "/data/u1-venv/bin/python"
_DEFAULT_LOAD_TIMEOUT_S = 600.0
_DEFAULT_UNLOAD_TIMEOUT_S = 30.0

_WORKER_SCRIPT = str(Path(__file__).with_name("sensenovau1_worker.py"))


class SenseNovaU1Backend(DiffusionBackend):
    """SenseNova-U1 unified-model T2I backend backed by an out-of-process worker."""

    def __init__(self, **kwargs: object) -> None:
        self._worker_python = str(
            kwargs.get("worker_python")
            or os.environ.get("SENSENOVA_U1_VENV_PYTHON")
            or _DEFAULT_WORKER_PYTHON
        )
        self._worker_script = str(kwargs.get("worker_script") or _WORKER_SCRIPT)
        self._device: torch.device | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=400)
        self._stderr_thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def load_models(
        self,
        model_path: str,
        device: torch.device,
        **kwargs: object,
    ) -> None:
        self._device = device

        load_timeout = float(
            kwargs.get("load_timeout")
            or os.environ.get("SENSENOVA_U1_LOAD_TIMEOUT")
            or _DEFAULT_LOAD_TIMEOUT_S
        )

        cmd = [
            self._worker_python,
            self._worker_script,
            "--model_path",
            model_path,
            "--device",
            str(device),
        ]
        logger.info("[SenseNovaU1] Spawning worker: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Continuously drain stderr for the worker's whole lifetime so its PIPE
        # can never fill and deadlock the stdout handshake (importing
        # sensenova_u1 floods stderr with auto_docstring noise).
        self._stderr_thread = threading.Thread(
            target=self._pump_stderr, args=(proc,), daemon=True
        )
        self._stderr_thread.start()

        ready = self._await_ready(proc, load_timeout)
        if not ready.get("ready"):
            stderr = self._stderr_snapshot()
            raise RuntimeError(
                "SenseNovaU1 worker failed to load: "
                f"{ready.get('error')!r}\n--- worker stderr ---\n{stderr}"
            )

        self._proc = proc
        logger.info("[SenseNovaU1] Worker ready on %s", device)

    def _await_ready(self, proc: subprocess.Popen, timeout: float) -> dict:
        """Wait for the first sentinel line (the readiness handshake).

        Reads stdout in a watchdog thread so we can enforce a hard timeout and
        surface worker stderr if it dies or hangs during model load.
        """
        result: dict[str, object] = {}

        def _reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                message = self._parse_sentinel(line)
                if message is not None:
                    result["msg"] = message
                    return

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            stderr = self._terminate_and_drain(proc)
            raise RuntimeError(
                f"SenseNovaU1 worker load timed out after {timeout:.0f}s"
                f"\n--- worker stderr ---\n{stderr}"
            )

        if "msg" not in result:
            # Reader returned without a sentinel line => stdout hit EOF (death).
            proc.wait()
            stderr = self._stderr_snapshot()
            raise RuntimeError(
                "SenseNovaU1 worker exited before signaling readiness "
                f"(returncode={proc.returncode})\n--- worker stderr ---\n{stderr}"
            )

        return result["msg"]  # type: ignore[return-value]

    # -- inference ---------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        params: ImageGenParams,
        *,
        condition_embeds: list[torch.Tensor] | None = None,
        negative_condition_embeds: list[torch.Tensor] | None = None,
    ) -> Image.Image:
        if self._proc is None:
            raise RuntimeError("SenseNovaU1 worker not loaded")

        # condition_embeds / negative_condition_embeds are accepted for interface
        # parity but unused in v1 (self-contained T2I). The v2 decoupling step
        # will route thinker conditioning into the worker request.

        request = {
            "prompt": prompt,
            "width": params.width,
            "height": params.height,
            "num_steps": params.num_inference_steps,
            "cfg_scale": params.guidance_scale,
            "cfg_norm": "none",
            "timestep_shift": 3.0,
            "cfg_interval": [0.0, 1.0],
            "seed": params.seed if params.seed is not None else 0,
            "think_mode": False,
        }

        with self._lock:
            response = self._exchange(request)

        if not response.get("ok"):
            raise RuntimeError(
                f"SenseNovaU1 worker generation failed: {response.get('error')!r}"
            )

        raw = base64.b64decode(response["b64"])
        return Image.open(BytesIO(raw)).convert("RGB")

    def _exchange(self, request: dict) -> dict:
        """Send one request line and read the next sentinel response line.

        Caller holds ``self._lock`` so only one request is in flight per worker.
        """
        proc = self._proc
        assert proc is not None and proc.stdin is not None and proc.stdout is not None

        if proc.poll() is not None:
            stderr = self._stderr_snapshot()
            raise RuntimeError(
                "SenseNovaU1 worker is not running "
                f"(returncode={proc.returncode})\n--- worker stderr ---\n{stderr}"
            )

        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()

        for line in proc.stdout:
            message = self._parse_sentinel(line)
            if message is not None:
                return message

        # stdout hit EOF without a sentinel line => the worker died mid-request.
        proc.wait()
        stderr = self._stderr_snapshot()
        raise RuntimeError(
            "SenseNovaU1 worker died during generation "
            f"(returncode={proc.returncode})\n--- worker stderr ---\n{stderr}"
        )

    # -- teardown ----------------------------------------------------------

    def unload(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:  # noqa: BLE001 - best-effort shutdown
                pass
            proc.terminate()
            try:
                proc.wait(timeout=_DEFAULT_UNLOAD_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        thread = self._stderr_thread
        self._stderr_thread = None
        if thread is not None:
            thread.join(timeout=2.0)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _parse_sentinel(line: str) -> dict | None:
        """Return the parsed JSON for a sentinel-prefixed line, else None (junk)."""
        if not line.startswith(SENTINEL):
            return None
        return json.loads(line[len(SENTINEL):].strip())

    def _pump_stderr(self, proc: subprocess.Popen) -> None:
        """Continuously drain the worker's stderr so its PIPE never fills.

        Importing sensenova_u1 (and tqdm denoise bars) writes a lot to stderr;
        if it were not drained, the OS pipe buffer (~64KB) would fill and block
        the worker on stderr.write() -- deadlocking the stdout handshake. We keep
        only a bounded tail for diagnostics.
        """
        stderr = proc.stderr
        if stderr is None:
            return
        try:
            for line in stderr:
                self._stderr_tail.append(line)
                logger.debug("[SenseNovaU1 worker] %s", line.rstrip())
        except Exception:  # noqa: BLE001 - drain is best-effort
            pass

    def _stderr_snapshot(self) -> str:
        """Best-effort recent worker stderr for error messages."""
        thread = self._stderr_thread
        if thread is not None:
            thread.join(timeout=2.0)
        return "".join(self._stderr_tail)

    def _terminate_and_drain(self, proc: subprocess.Popen) -> str:
        proc.kill()
        try:
            proc.wait(timeout=_DEFAULT_UNLOAD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            pass
        return self._stderr_snapshot()
