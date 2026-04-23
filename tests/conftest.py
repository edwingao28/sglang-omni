# SPDX-License-Identifier: Apache-2.0
"""Shared test collection controls for CI matrix lanes."""

from __future__ import annotations

import os


def _build_collect_ignore_glob() -> list[str]:
    if os.getenv("SGLANG_OMNI_CI_PIPELINE_VERSION") != "v1":
        return []

    # The v1 lane aliases `sglang_omni` to `sglang_omni_v1` so black-box tests
    # can stay unchanged. These files still depend on legacy-only internals or
    # on model families that have not been migrated to the v1 config/runtime.
    return [
        "test_cache_key.py",
        "test_cli_version_dispatch.py",
        "test_direct_model_executor.py",
        "test_factory.py",
        "test_ipc_runtime_dir.py",
        "test_mem_fraction_static.py",
        "test_ming*.py",
        "test_model_worker_ports.py",
        "test_omni_engine.py",
        "test_scheduler.py",
        "test_scheduler_streaming.py",
        "test_talker_error_propagation.py",
        "test_tp_batch_serialization.py",
        "test_tp_follower.py",
    ]


collect_ignore_glob = _build_collect_ignore_glob()
