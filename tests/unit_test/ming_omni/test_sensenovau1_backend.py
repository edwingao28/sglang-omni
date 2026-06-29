# SPDX-License-Identifier: Apache-2.0
"""SenseNova-U1 backend tests (registry dispatch + worker-bridge protocol).

The Stage-3 backend talks to an out-of-process worker over a line-oriented JSON
protocol; it never imports ``sensenova_u1``. These tests therefore mock the
SUBPROCESS (``subprocess.Popen``) rather than the GPU model, and assert the
ImageGenParams -> request-JSON mapping plus the sentinel-filtering behavior.
"""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys

import pytest

from sglang_omni.models.ming_omni.diffusion.backend import ImageGenParams
from sglang_omni.models.ming_omni.diffusion.sensenovau1_backend import SENTINEL


def _png_b64(width: int, height: int) -> str:
    from PIL import Image

    img = Image.new("RGB", (width, height), (123, 45, 67))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _FakeStdin:
    """Captures everything written by the backend to the worker stdin."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.closed = False

    def write(self, data: str) -> int:
        self.lines.append(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def requests(self) -> list[dict]:
        return [json.loads(s) for s in self.lines if s.strip()]


class _FakeStdout:
    """A single persistent line iterator (shared across read phases)."""

    def __init__(self, lines: list[str]) -> None:
        self._it = iter(lines)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        return next(self._it)


class _FakeProc:
    """A minimal stand-in for ``subprocess.Popen`` in text/line mode."""

    def __init__(self, stdout_lines: list[str], stderr: str = "") -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(stdout_lines)
        self.stderr = io.StringIO(stderr)
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None) -> int:
        self.waited = True
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _patch_popen(monkeypatch, proc: _FakeProc) -> dict:
    captured: dict = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return captured


# --------------------------------------------------------------------------
# Registry dispatch
# --------------------------------------------------------------------------


def test_create_backend_returns_sensenovau1_without_importing_package(monkeypatch):
    """Registry dispatch must NOT import sensenova_u1 (it can't in this env)."""
    monkeypatch.delitem(sys.modules, "sensenova_u1", raising=False)

    from sglang_omni.models.ming_omni.components.image_gen_executor import (
        _create_backend,
    )
    from sglang_omni.models.ming_omni.diffusion.sensenovau1_backend import (
        SenseNovaU1Backend,
    )

    backend = _create_backend("sensenovau1")

    assert isinstance(backend, SenseNovaU1Backend)
    assert "sensenova_u1" not in sys.modules


def test_registry_lists_sensenovau1():
    from sglang_omni.models.ming_omni.diffusion import registry

    assert registry.available() == ["sensenovau1", "zimage"]


# --------------------------------------------------------------------------
# Worker-bridge protocol
# --------------------------------------------------------------------------


def test_load_models_spawns_worker_and_handshakes(monkeypatch):
    import torch

    from sglang_omni.models.ming_omni.diffusion.sensenovau1_backend import (
        SenseNovaU1Backend,
    )

    proc = _FakeProc([
        "noise: import sensenova_u1 ...\n",
        SENTINEL + json.dumps({"ready": True}) + "\n",
    ])
    captured = _patch_popen(monkeypatch, proc)

    backend = SenseNovaU1Backend(worker_python="/fake/py")
    backend.load_models("/fake/weights", torch.device("cuda:0"))

    cmd = captured["cmd"]
    assert cmd[0] == "/fake/py"
    assert cmd[1].endswith("sensenovau1_worker.py")
    assert "--model_path" in cmd and "/fake/weights" in cmd
    assert "--device" in cmd and "cuda:0" in cmd
    assert backend._proc is proc


def test_load_models_raises_with_stderr_on_failed_ready(monkeypatch):
    import torch

    from sglang_omni.models.ming_omni.diffusion.sensenovau1_backend import (
        SenseNovaU1Backend,
    )

    proc = _FakeProc(
        [SENTINEL + json.dumps({"ready": False, "error": "OOM"}) + "\n"],
        stderr="CUDA out of memory traceback",
    )
    _patch_popen(monkeypatch, proc)

    backend = SenseNovaU1Backend()
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        backend.load_models("/fake/weights", torch.device("cuda:0"))


def test_generate_maps_params_and_returns_pil_image(monkeypatch):
    torch = pytest.importorskip("torch")
    pil_image = pytest.importorskip("PIL.Image")

    from sglang_omni.models.ming_omni.diffusion.sensenovau1_backend import (
        SenseNovaU1Backend,
    )

    proc = _FakeProc([
        # Junk lines BEFORE the ready sentinel must be ignored.
        "\U0001f6a8 auto_docstring noise\n",
        SENTINEL + json.dumps({"ready": True}) + "\n",
        # Junk lines BEFORE the generate response must also be ignored.
        "[ERROR] something is part of the signature\n",
        SENTINEL
        + json.dumps({"ok": True, "b64": _png_b64(64, 48), "width": 64, "height": 48})
        + "\n",
    ])
    _patch_popen(monkeypatch, proc)

    backend = SenseNovaU1Backend()
    backend.load_models("/fake/weights", torch.device("cpu"))

    params = ImageGenParams(
        width=64,
        height=48,
        num_inference_steps=37,
        guidance_scale=4.0,
        seed=123,
    )
    image = backend.generate("a red cube", params)

    assert isinstance(image, pil_image.Image)
    assert image.size == (64, 48)  # PIL size is (width, height)

    requests = proc.stdin.requests()
    assert len(requests) == 1
    req = requests[0]
    assert req["prompt"] == "a red cube"
    assert req["width"] == 64
    assert req["height"] == 48
    assert req["num_steps"] == 37
    assert req["cfg_scale"] == 4.0
    assert req["seed"] == 123
    assert req["cfg_norm"] == "none"
    assert req["timestep_shift"] == 3.0
    assert req["cfg_interval"] == [0.0, 1.0]
    assert req["think_mode"] is False


def test_generate_defaults_missing_seed_to_zero(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL.Image")

    from sglang_omni.models.ming_omni.diffusion.sensenovau1_backend import (
        SenseNovaU1Backend,
    )

    proc = _FakeProc([
        SENTINEL + json.dumps({"ready": True}) + "\n",
        SENTINEL
        + json.dumps({"ok": True, "b64": _png_b64(16, 16), "width": 16, "height": 16})
        + "\n",
    ])
    _patch_popen(monkeypatch, proc)

    backend = SenseNovaU1Backend()
    backend.load_models("/fake/weights", torch.device("cpu"))
    backend.generate("prompt", ImageGenParams(width=16, height=16, seed=None))

    assert proc.stdin.requests()[0]["seed"] == 0


def test_generate_raises_on_worker_error_response(monkeypatch):
    torch = pytest.importorskip("torch")

    from sglang_omni.models.ming_omni.diffusion.sensenovau1_backend import (
        SenseNovaU1Backend,
    )

    proc = _FakeProc([
        SENTINEL + json.dumps({"ready": True}) + "\n",
        SENTINEL + json.dumps({"ok": False, "error": "bad prompt"}) + "\n",
    ])
    _patch_popen(monkeypatch, proc)

    backend = SenseNovaU1Backend()
    backend.load_models("/fake/weights", torch.device("cpu"))

    with pytest.raises(RuntimeError, match="bad prompt"):
        backend.generate("prompt", ImageGenParams(width=16, height=16))


def test_generate_before_load_raises():
    from sglang_omni.models.ming_omni.diffusion.sensenovau1_backend import (
        SenseNovaU1Backend,
    )

    with pytest.raises(RuntimeError, match="worker not loaded"):
        SenseNovaU1Backend().generate("prompt", ImageGenParams())


def test_unload_closes_and_terminates_proc(monkeypatch):
    import torch

    from sglang_omni.models.ming_omni.diffusion.sensenovau1_backend import (
        SenseNovaU1Backend,
    )

    proc = _FakeProc([SENTINEL + json.dumps({"ready": True}) + "\n"])
    _patch_popen(monkeypatch, proc)

    backend = SenseNovaU1Backend()
    backend.load_models("/fake/weights", torch.device("cpu"))
    backend.unload()

    assert proc.stdin.closed
    assert proc.terminated
    assert proc.waited
    assert backend._proc is None
    # Calling again on an already-unloaded backend must not raise.
    backend.unload()
