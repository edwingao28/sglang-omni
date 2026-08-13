# SPDX-License-Identifier: Apache-2.0
"""Lightweight parity assertions shared by CPU and H100 qualification tests."""

from __future__ import annotations

import base64
from typing import Any

import torch

PARITY_RTOL = 2e-2
PARITY_ATOL = 2e-2


def snapshot(probe: dict[str, Any], request_id: str) -> dict[str, Any]:
    debug_snapshot = probe["after"]["debug_snapshot"]
    assert debug_snapshot, "missing prefill debug snapshot"
    assert not debug_snapshot.get("skipped"), {
        "skipped": debug_snapshot.get("skipped"),
        "logical_rows": debug_snapshot.get("logical_rows"),
    }
    requests = debug_snapshot["requests"]
    matches = [item for item in requests if item.get("request_id") == request_id]
    assert len(matches) == 1, {
        "expected_request_id": request_id,
        "captured_request_ids": [item.get("request_id") for item in requests],
    }
    return matches[0]


def snapshot_logical_rows(probe: dict[str, Any]) -> int:
    debug_snapshot = probe["after"]["debug_snapshot"]
    assert debug_snapshot, "missing prefill debug snapshot"
    assert not debug_snapshot.get("skipped"), {
        "skipped": debug_snapshot.get("skipped"),
        "logical_rows": debug_snapshot.get("logical_rows"),
    }
    return int(debug_snapshot["logical_rows"])


def decode_tensor(payload: dict[str, Any]) -> torch.Tensor:
    dtype = getattr(torch, payload["dtype"])
    raw = base64.b64decode(payload["data"], validate=True)
    values = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return values.view(dtype).reshape(payload["shape"])


def _request_ids(
    *,
    request_id: str | None,
    eager_request_id: str | None,
    graph_request_id: str | None,
) -> tuple[str, str]:
    eager_id = eager_request_id or request_id
    graph_id = graph_request_id or request_id
    if eager_id is None or graph_id is None:
        raise TypeError(
            "provide request_id, or both eager_request_id and graph_request_id"
        )
    return eager_id, graph_id


def _request_logical_rows(
    probe: dict[str, Any], request_snapshot: dict[str, Any]
) -> int:
    if "logical_rows" in request_snapshot:
        return int(request_snapshot["logical_rows"])

    debug_snapshot = probe["after"]["debug_snapshot"]
    if len(debug_snapshot["requests"]) == 1:
        return int(debug_snapshot["logical_rows"])

    row_counts = {
        int(payload["shape"][0])
        for payload in request_snapshot["hidden_states"].values()
        if payload.get("shape")
    }
    assert len(row_counts) == 1, request_snapshot
    return row_counts.pop()


def _assert_finite(tensor: torch.Tensor, *, label: str) -> None:
    nonfinite = int((~torch.isfinite(tensor.float())).sum())
    assert nonfinite == 0, f"{label} contains {nonfinite} non-finite value(s)"


def assert_exact_prefill_embeddings(
    eager_probe: dict[str, Any],
    graph_probe: dict[str, Any],
    *,
    request_id: str | None = None,
    eager_request_id: str | None = None,
    graph_request_id: str | None = None,
) -> float:
    """Gate the input-embedding transport without gating padded model math."""
    eager_id, graph_id = _request_ids(
        request_id=request_id,
        eager_request_id=eager_request_id,
        graph_request_id=graph_request_id,
    )
    eager = snapshot(eager_probe, eager_id)
    graph = snapshot(graph_probe, graph_id)
    eager_prompt_tokens = int(eager_probe["body"]["usage"]["prompt_tokens"])
    graph_prompt_tokens = int(graph_probe["body"]["usage"]["prompt_tokens"])
    eager_logical_rows = _request_logical_rows(eager_probe, eager)
    graph_logical_rows = _request_logical_rows(graph_probe, graph)
    assert eager_prompt_tokens == graph_prompt_tokens
    assert eager_logical_rows == eager_prompt_tokens
    assert graph_logical_rows == graph_prompt_tokens

    eager_embed = decode_tensor(eager["hidden_states"]["embed"])
    graph_embed = decode_tensor(graph["hidden_states"]["embed"])
    assert eager_embed.shape == graph_embed.shape
    assert eager_embed.shape[0] == eager_logical_rows
    _assert_finite(eager_embed, label=f"{eager_id} embedding")
    _assert_finite(graph_embed, label=f"{graph_id} embedding")
    torch.testing.assert_close(graph_embed, eager_embed, rtol=0, atol=0)
    return float((graph_embed.float() - eager_embed.float()).abs().max())


def _tensor_diagnostics(
    eager_tensor: torch.Tensor, graph_tensor: torch.Tensor
) -> dict[str, Any]:
    eager_shape = list(eager_tensor.shape)
    graph_shape = list(graph_tensor.shape)
    diagnostics: dict[str, Any] = {
        "eager_shape": eager_shape,
        "graph_shape": graph_shape,
        "shape_match": eager_shape == graph_shape,
    }
    if eager_shape != graph_shape:
        return diagnostics

    eager_float = eager_tensor.float()
    graph_float = graph_tensor.float()
    delta = (graph_float - eager_float).abs()
    eager_finite = torch.isfinite(eager_float)
    graph_finite = torch.isfinite(graph_float)
    close = torch.isclose(
        graph_float,
        eager_float,
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
        equal_nan=False,
    )
    finite_delta = delta[torch.isfinite(delta)]
    mean_abs_error = float(finite_delta.mean()) if finite_delta.numel() else 0.0
    root_mean_square_error = (
        float(torch.sqrt(torch.mean(finite_delta.square())))
        if finite_delta.numel()
        else 0.0
    )
    diagnostics.update(
        {
            "numel": int(delta.numel()),
            "mismatch_count": int((~close).sum()),
            "eager_nonfinite_count": int((~eager_finite).sum()),
            "graph_nonfinite_count": int((~graph_finite).sum()),
            "max_abs_error": (
                float(finite_delta.max()) if finite_delta.numel() else 0.0
            ),
            "mean_abs_error": mean_abs_error,
            "root_mean_square_error": root_mean_square_error,
        }
    )
    if delta.ndim >= 2:
        mismatched_rows = (~close).reshape(delta.shape[0], -1).any(dim=1)
        row_indexes = mismatched_rows.nonzero(as_tuple=False).flatten().tolist()
        diagnostics.update(
            {
                "mismatched_row_count": len(row_indexes),
                "mismatched_rows": row_indexes[:256],
                "mismatched_rows_truncated": len(row_indexes) > 256,
            }
        )
    return diagnostics


def prefill_parity_diagnostics(
    eager_probe: dict[str, Any],
    graph_probe: dict[str, Any],
    *,
    request_id: str | None = None,
    eager_request_id: str | None = None,
    graph_request_id: str | None = None,
) -> dict[str, Any]:
    """Measure parity without raising on numerical or shape mismatches."""
    eager_id, graph_id = _request_ids(
        request_id=request_id,
        eager_request_id=eager_request_id,
        graph_request_id=graph_request_id,
    )
    eager = snapshot(eager_probe, eager_id)
    graph = snapshot(graph_probe, graph_id)
    eager_prompt_tokens = int(eager_probe["body"]["usage"]["prompt_tokens"])
    graph_prompt_tokens = int(graph_probe["body"]["usage"]["prompt_tokens"])

    hidden_diagnostics = {}
    for layer in ("embed", "24"):
        hidden_diagnostics[layer] = _tensor_diagnostics(
            decode_tensor(eager["hidden_states"][layer]),
            decode_tensor(graph["hidden_states"][layer]),
        )

    return {
        "eager_request_id": eager_id,
        "graph_request_id": graph_id,
        "eager_prompt_tokens": eager_prompt_tokens,
        "graph_prompt_tokens": graph_prompt_tokens,
        "eager_logical_rows": _request_logical_rows(eager_probe, eager),
        "graph_logical_rows": _request_logical_rows(graph_probe, graph),
        "eager_next_token_id": eager["next_token_id"],
        "graph_next_token_id": graph["next_token_id"],
        "next_token_match": eager["next_token_id"] == graph["next_token_id"],
        "hidden_states": hidden_diagnostics,
        "next_token_logits": _tensor_diagnostics(
            decode_tensor(eager["next_token_logits"]),
            decode_tensor(graph["next_token_logits"]),
        ),
    }


def assert_prefill_parity(
    eager_probe: dict[str, Any],
    graph_probe: dict[str, Any],
    *,
    request_id: str | None = None,
    eager_request_id: str | None = None,
    graph_request_id: str | None = None,
    require_exact_embed: bool = False,
) -> dict[str, float]:
    eager_id, graph_id = _request_ids(
        request_id=request_id,
        eager_request_id=eager_request_id,
        graph_request_id=graph_request_id,
    )
    eager = snapshot(eager_probe, eager_id)
    graph = snapshot(graph_probe, graph_id)
    # This gate qualifies the prefill boundary directly. Full completion text
    # is recorded separately, but later decode can amplify BF16 differences
    # that are already within the accepted prefill tolerance.
    assert eager["next_token_id"] == graph["next_token_id"]

    graph_prompt_tokens = int(graph_probe["body"]["usage"]["prompt_tokens"])
    eager_prompt_tokens = int(eager_probe["body"]["usage"]["prompt_tokens"])
    graph_logical_rows = _request_logical_rows(graph_probe, graph)
    eager_logical_rows = _request_logical_rows(eager_probe, eager)
    assert graph_prompt_tokens == eager_prompt_tokens
    # Usage is produced from the original thinker prompt, independently of
    # the hidden-capture slicer. With radix disabled, each fresh request must
    # prefill every logical prompt row on both paths.
    assert graph_logical_rows == graph_prompt_tokens
    assert eager_logical_rows == eager_prompt_tokens
    maximum_errors: dict[str, float] = {}
    for layer in ("embed", "24"):
        eager_tensor = decode_tensor(eager["hidden_states"][layer])
        graph_tensor = decode_tensor(graph["hidden_states"][layer])
        assert eager_tensor.shape == graph_tensor.shape
        assert graph_tensor.shape[0] == graph_logical_rows
        _assert_finite(eager_tensor, label=f"{eager_id} hidden layer {layer}")
        _assert_finite(graph_tensor, label=f"{graph_id} hidden layer {layer}")
        if layer == "embed" and require_exact_embed:
            torch.testing.assert_close(
                graph_tensor,
                eager_tensor,
                rtol=0,
                atol=0,
            )
        else:
            torch.testing.assert_close(
                graph_tensor.float(),
                eager_tensor.float(),
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
            )
        maximum_errors[f"hidden_{layer}"] = float(
            (graph_tensor.float() - eager_tensor.float()).abs().max()
        )

    eager_logits = decode_tensor(eager["next_token_logits"])
    graph_logits = decode_tensor(graph["next_token_logits"])
    assert eager_logits.shape == graph_logits.shape
    _assert_finite(eager_logits, label=f"{eager_id} next-token logits")
    _assert_finite(graph_logits, label=f"{graph_id} next-token logits")
    torch.testing.assert_close(
        graph_logits.float(),
        eager_logits.float(),
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
    )
    maximum_errors["next_token_logits"] = float(
        (graph_logits.float() - eager_logits.float()).abs().max()
    )
    return maximum_errors
