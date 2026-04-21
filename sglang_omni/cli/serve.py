from __future__ import annotations

import logging
from typing import Annotated, Literal

import typer
import yaml

from sglang_omni.config.manager import ConfigManager
from sglang_omni.serve.launcher import launch_server

logger = logging.getLogger(__name__)


def serve(
    ctx: typer.Context,
    model_path: Annotated[
        str,
        typer.Option(
            help="The Hugging Face model ID or the path to the model directory."
        ),
    ],
    config: Annotated[
        str, typer.Option(help="Path to a pipeline config JSON file.")
    ] = None,
    text_only: Annotated[
        bool,
        typer.Option(
            "--text-only",
            help="Use thinker-only pipeline (1 GPU, no talker/speech output).",
        ),
    ] = False,
    tp_size: Annotated[
        int,
        typer.Option(help="Tensor parallel size for the thinker stage (default: 1)."),
    ] = 1,
    cpu_offload_gb: Annotated[
        int,
        typer.Option(help="GB of thinker weights to offload to CPU (0 disables)."),
    ] = 0,
    mem_fraction_static: Annotated[
        float,
        typer.Option(
            help="Fraction of GPU memory for weights + KV cache on the thinker."
        ),
    ] = None,
    disable_cuda_graph: Annotated[
        bool,
        typer.Option(
            "--disable-cuda-graph",
            help="Disable CUDA graph on the thinker SGLang engine.",
        ),
    ] = False,
    host: Annotated[
        str, typer.Option(help="Server bind address (default: 0.0.0.0).")
    ] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Server bind port (default: 8000).")] = 8000,
    model_name: Annotated[
        str, typer.Option(help="Model name for /v1/models (default: pipeline name).")
    ] = None,
    log_level: Annotated[
        Literal["debug", "info", "warning", "error", "critical"],
        typer.Option(help="Log level (default: info)."),
    ] = "info",
) -> None:
    """Serve the pipeline."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Resolve config ---
    if config:
        config_manager = ConfigManager.from_file(config)
    elif text_only:
        config_manager = ConfigManager.from_model_path(model_path, variant="text")
    else:
        config_manager = ConfigManager.from_model_path(model_path)

    # we use ctx to capture the arguments that are used to modify the configuration on the fly
    # we do expect the extra arguments to be pairs of names and values
    extra_args = config_manager.parse_extra_args(ctx.args)
    merged_config = config_manager.merge_config(extra_args)
    merged_config = merged_config.model_copy(update={"model_path": model_path})

    # Inject thinker server_args overrides from explicit CLI flags so benches
    # can compare models under matched engine settings (CUDA graph, KV budget,
    # CPU offload). This mirrors examples/run_ming_omni_server.py.
    overrides: dict = {}
    if tp_size and tp_size > 1:
        overrides["tp_size"] = tp_size
        overrides["disable_custom_all_reduce"] = True
    if cpu_offload_gb:
        overrides["cpu_offload_gb"] = cpu_offload_gb
    if mem_fraction_static is not None:
        overrides["mem_fraction_static"] = mem_fraction_static
    overrides["disable_cuda_graph"] = disable_cuda_graph

    for stage in merged_config.stages:
        if stage.name != "thinker":
            continue
        if stage.executor.args is None:
            stage.executor.args = {}
        existing = stage.executor.args.setdefault("server_args_overrides", {})
        existing.update(overrides)

    # print merged configuration
    print("=" * 20, "Merged Configuration", "=" * 20)
    print(
        yaml.dump(
            merged_config.model_dump(mode="json"),
            sort_keys=False,
            default_flow_style=False,
            indent=2,
        )
    )
    print("=" * 50)

    launch_server(
        merged_config,
        host=host,
        port=port,
        model_name=model_name,
        log_level=log_level,
    )
