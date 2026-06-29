# SPDX-License-Identifier: Apache-2.0
"""Standalone SenseNova-U1 T2I worker (runs INSIDE the pinned U1 venv).

This script is the GPU-side half of the two-env bridge. It is launched as a
subprocess by ``SenseNovaU1Backend`` under ``/data/u1-venv/bin/python`` (torch
2.8 / transformers 4.57), because the official ``sensenova_u1`` package cannot
be imported in the sglang environment (transformers 5.x breaks it).

It deliberately depends ONLY on the stdlib + torch + PIL + sensenova_u1 and
MUST NOT import ``sglang_omni`` -- it runs under a different interpreter where
sglang_omni is not installed.

Wire protocol (line-oriented JSON over stdin/stdout):
  * Importing ``sensenova_u1`` floods stdout/stderr with auto_docstring noise
    (lines containing the rocket emoji / "[ERROR]" / "is part of ... signature").
    To stay robust to that junk, EVERY protocol message we emit on stdout is a
    single line prefixed with the ``SENTINEL`` token. The backend ignores any
    stdout line that does not start with the sentinel.
  * Startup: emit ``{"ready": true}`` once the model is loaded, or
    ``{"ready": false, "error": ...}`` then exit(1) on load failure.
  * Per request: read one JSON line from stdin, run ``t2i_generate``, and emit
    ``{"ok": true, "b64": <png-base64>, "width": w, "height": h}``. On a
    per-request error emit ``{"ok": false, "error": ...}`` WITHOUT crashing the
    loop. EOF on stdin terminates the worker (exit 0).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from io import BytesIO

import torch
from PIL import Image

SENTINEL = "@@U1@@"

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _emit(obj: dict) -> None:
    """Write a single sentinel-prefixed JSON protocol line to stdout and flush."""
    sys.stdout.write(SENTINEL + json.dumps(obj) + "\n")
    sys.stdout.flush()


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    """Denormalize a [B,3,H,W] tensor (mean=std=0.5) and return the first image."""
    image = tensor[0].float()
    image = (image * 0.5 + 0.5).clamp(0.0, 1.0)
    image = (image * 255.0).round().to(torch.uint8)
    array = image.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def _handle_request(model, tokenizer, req: dict) -> dict:
    """Run a single T2I generation and return the protocol response object."""
    prompt = req["prompt"]
    width = int(req["width"])
    height = int(req["height"])
    cfg_interval = req.get("cfg_interval", [0.0, 1.0])

    tensor = model.t2i_generate(
        tokenizer,
        prompt,
        image_size=(width, height),
        cfg_scale=float(req.get("cfg_scale", 4.0)),
        cfg_norm=req.get("cfg_norm", "none"),
        timestep_shift=float(req.get("timestep_shift", 3.0)),
        cfg_interval=tuple(cfg_interval),
        num_steps=int(req.get("num_steps", 50)),
        batch_size=1,
        seed=int(req.get("seed", 0)),
        think_mode=bool(req.get("think_mode", False)),
    )

    image = _to_pil(tensor)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"ok": True, "b64": b64, "width": image.width, "height": image.height}


def main() -> None:
    parser = argparse.ArgumentParser(description="SenseNova-U1 T2I worker bridge.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=list(_DTYPES))
    args = parser.parse_args()

    try:
        import sensenova_u1  # noqa: F401
        from sensenova_u1.utils import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(
            args.model_path, dtype=_DTYPES[args.dtype], device=args.device
        )
    except Exception as exc:  # noqa: BLE001 - report any load failure upstream
        _emit({"ready": False, "error": repr(exc)})
        sys.exit(1)

    _emit({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            response = _handle_request(model, tokenizer, req)
        except Exception as exc:  # noqa: BLE001 - never crash the loop
            response = {"ok": False, "error": repr(exc)}
        _emit(response)

    sys.exit(0)


if __name__ == "__main__":
    main()
