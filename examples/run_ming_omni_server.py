# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible server for Ming-Omni (text output).

Usage::

    python examples/run_ming_omni_server.py \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --port 8000

Then test with::

    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "ming-omni",
            "messages": [{"role": "user", "content": "你好！"}],
            "max_tokens": 256,
            "stream": true
        }'
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os

from sglang_omni.models.ming_omni.config import (
    MingOmniPipelineConfig,
    MingOmniSpeechPipelineConfig,
    MingOmniStreamingSpeechPipelineConfig,
)
from sglang_omni.models.ming_omni.pipeline.next_stage import (
    TALKER_STAGE,
    TALKER_STREAM_STAGE,
)

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def resolve_ming_pipeline_config(
    *, enable_speech: bool, enable_streaming_tts: bool
) -> type[
    MingOmniPipelineConfig
    | MingOmniSpeechPipelineConfig
    | MingOmniStreamingSpeechPipelineConfig
]:
    if enable_speech and enable_streaming_tts:
        return MingOmniStreamingSpeechPipelineConfig
    if enable_speech:
        return MingOmniSpeechPipelineConfig
    return MingOmniPipelineConfig


def resolve_ming_gpu_placement(
    *,
    enable_speech: bool,
    enable_streaming_tts: bool,
    gpu_thinker: int,
    gpu_talker: int,
) -> dict[str, int] | None:
    if not enable_speech:
        return None
    talker_stage = TALKER_STREAM_STAGE if enable_streaming_tts else TALKER_STAGE
    return {"thinker": gpu_thinker, talker_stage: gpu_talker}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        default="inclusionAI/Ming-flash-omni-2.0",
        help="Hugging Face model id or local path",
    )

    # Pipeline options
    parser.add_argument("--thinker-max-seq-len", type=int, default=8192)
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for thinker",
    )
    parser.add_argument(
        "--gpu-thinker",
        type=int,
        default=0,
        help="GPU index for the thinker stage",
    )
    parser.add_argument(
        "--gpu-talker",
        type=int,
        default=1,
        help=(
            "GPU index for the speech talker stage. "
            "When using --tp-size 2, set --gpu-talker 2 to avoid "
            "the thinker TP range."
        ),
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Quantization method (e.g., fp8) for thinker model",
    )
    parser.add_argument(
        "--cpu-offload-gb",
        type=int,
        default=80,
        help="GB of model weights to offload to CPU (default: 80 for Ming-flash-omni-2.0)",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help=(
            "Set SGLang mem_fraction_static for the thinker stage. "
            "If omitted, SGLang chooses automatically."
        ),
    )
    parser.add_argument(
        "--relay-backend",
        type=str,
        default="shm",
        choices=["shm", "nixl"],
        help="Relay backend for inter-stage data transfer",
    )
    parser.add_argument(
        "--enable-speech",
        action="store_true",
        help="Enable non-streaming speech output pipeline",
    )
    parser.add_argument(
        "--enable-streaming-tts",
        action="store_true",
        help="Enable streaming TTS speech pipeline; requires --enable-speech",
    )

    # Server
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model-name",
        type=str,
        default="ming-omni",
        help="Model name for /v1/models (default: ming-omni)",
    )

    args = parser.parse_args(argv)
    if args.enable_streaming_tts and not args.enable_speech:
        parser.error("--enable-streaming-tts requires --enable-speech")
    return args


def main() -> None:
    args = parse_args()
    from sglang_omni.serve import launch_server

    overrides = {}
    if args.tp_size and args.tp_size > 1:
        overrides["tp_size"] = args.tp_size
        overrides["disable_custom_all_reduce"] = True
    if args.quantization:
        overrides["quantization"] = args.quantization
    if args.cpu_offload_gb:
        overrides["cpu_offload_gb"] = args.cpu_offload_gb

    config_cls = resolve_ming_pipeline_config(
        enable_speech=args.enable_speech,
        enable_streaming_tts=args.enable_streaming_tts,
    )
    config_kwargs = {
        "model_path": args.model_path,
        "relay_backend": args.relay_backend,
    }
    gpu_placement = resolve_ming_gpu_placement(
        enable_speech=args.enable_speech,
        enable_streaming_tts=args.enable_streaming_tts,
        gpu_thinker=args.gpu_thinker,
        gpu_talker=args.gpu_talker,
    )
    if gpu_placement is not None:
        config_kwargs["gpu_placement"] = gpu_placement
    config = config_cls(
        **config_kwargs,
    )
    if overrides:
        config.apply_server_args_overrides(stage_name="thinker", overrides=overrides)
    if args.mem_fraction_static is not None:
        if not 0.0 < args.mem_fraction_static < 1.0:
            raise ValueError(
                f"--mem-fraction-static must be > 0 and < 1, got {args.mem_fraction_static}"
            )
        config.apply_server_args_overrides(
            stage_name="thinker",
            overrides={"mem_fraction_static": args.mem_fraction_static},
        )

    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
