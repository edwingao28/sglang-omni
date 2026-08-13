# SPDX-License-Identifier: Apache-2.0
"""Test-only qualification hooks for breakable prefill CUDA graphs."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any


def _same_callable(left: Any, right: Any) -> bool:
    """Compare functions and separately-created bound method objects."""
    if left is right:
        return True
    return (
        getattr(left, "__self__", None) is getattr(right, "__self__", None)
        and getattr(left, "__func__", None) is getattr(right, "__func__", None)
        and getattr(left, "__func__", None) is not None
    )


def _cuda_graph_dp_padding_mode() -> Any:
    from sglang.srt.layers.dp_attention import DpPaddingMode

    return DpPaddingMode.get_default_mode_in_cuda_graph()


def enable_prefill_qualification_eager_replay(model_runner: Any) -> None:
    """Replace captured BCG body replay with its live, padded eager body.

    This is a qualification oracle, not a serving mode. The upstream prefill
    runner still performs ordinary graph admission, bucket selection,
    ``load_batch`` static-buffer population, and live attention-metadata
    preparation. Only the final backend replay is replaced with the same
    ``_run_forward`` body used during capture.

    During ``_execute_body_capture``, upstream temporarily replaces
    ``layer_model.forward`` with a closure that calls ``backend.replay``. The
    hook must therefore restore the original layer forward around
    ``_run_forward`` or it would recurse back into itself. It also suppresses
    ``_run_forward``'s inner context manager: SGLang 0.5.16's context is not
    stack-restoring, and the already-active outer context carries the required
    live ``num_tokens``/``raw_num_tokens`` pair into the eager tail. TP=1 live
    batches may leave ``dp_padding_mode`` unset, so the oracle temporarily uses
    the same default mode as graph capture before entering ``_run_forward``.
    """
    runner = getattr(model_runner, "prefill_cuda_graph_runner", None)
    if runner is None or type(runner).__name__ != "PrefillCudaGraphRunner":
        raise RuntimeError("qualification eager replay requires PrefillCudaGraphRunner")

    backend = getattr(runner, "backend", None)
    if backend is None or type(backend).__name__ != "BreakableCudaGraphBackend":
        raise RuntimeError(
            "qualification eager replay requires BreakableCudaGraphBackend"
        )
    if bool(getattr(backend, "_debug_eager", False)):
        raise RuntimeError(
            "qualification eager replay cannot be combined with SGLang "
            "debug_cuda_graph"
        )
    if bool(getattr(backend, "_omni_qualification_eager_replay", False)):
        return

    layer_model = getattr(runner, "layer_model", None)
    if layer_model is None or not callable(getattr(layer_model, "forward", None)):
        raise RuntimeError(
            "qualification eager replay requires a callable layer_model.forward"
        )
    run_forward = getattr(runner, "_run_forward", None)
    if not callable(run_forward):
        raise RuntimeError(
            "qualification eager replay requires PrefillCudaGraphRunner._run_forward"
        )
    original_prefill_forward_context = getattr(runner, "_prefill_forward_context", None)
    if not callable(original_prefill_forward_context):
        raise RuntimeError(
            "qualification eager replay requires "
            "PrefillCudaGraphRunner._prefill_forward_context"
        )

    original_replay = backend.replay
    original_layer_forward = layer_model.forward

    def eager_replay(shape_key: Any, static_forward_batch: Any, **_: Any) -> Any:
        size = getattr(shape_key, "size", None)
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise RuntimeError(
                "qualification eager replay requires a positive integer shape size"
            )

        replay_layer_forward = layer_model.forward
        if _same_callable(replay_layer_forward, original_layer_forward):
            raise RuntimeError(
                "qualification eager replay must run inside the prefill body "
                "replay closure"
            )
        original_dp_padding_mode = static_forward_batch.dp_padding_mode
        replace_dp_padding_mode = original_dp_padding_mode is None
        try:
            if replace_dp_padding_mode:
                static_forward_batch.dp_padding_mode = _cuda_graph_dp_padding_mode()
            layer_model.forward = original_layer_forward
            runner._prefill_forward_context = lambda *args, **kwargs: nullcontext()
            return run_forward(static_forward_batch, size)
        finally:
            if replace_dp_padding_mode:
                static_forward_batch.dp_padding_mode = original_dp_padding_mode
            runner._prefill_forward_context = original_prefill_forward_context
            layer_model.forward = replay_layer_forward

    backend._omni_original_replay = original_replay
    backend._omni_original_layer_forward = original_layer_forward
    backend._omni_original_prefill_forward_context = original_prefill_forward_context
    backend.replay = eager_replay
    backend._omni_qualification_eager_replay = True


__all__ = ["enable_prefill_qualification_eager_replay"]
