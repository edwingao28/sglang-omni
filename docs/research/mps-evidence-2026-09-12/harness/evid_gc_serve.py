"""Supported typed-config registration followed by the original serve CLI."""


def register_experiment_config() -> None:
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
    PIPELINE_CONFIG_REGISTRY.register_config('evid_gc_configs', overwrite=True, strict=True)


def main() -> None:
    register_experiment_config()
    from sglang_omni.cli import app
    app()


if __name__ == '__main__':
    main()
