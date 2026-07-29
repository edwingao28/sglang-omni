# SPDX-License-Identifier: Apache-2.0
"""Bounds coverage for the req_pool_idx-keyed talker feedback table.

``ReqToTokenPool`` reserves row 0 and allocates ``req_pool_idx`` from ``[1, size]``
inclusive, so the top allocatable index equals the pool size. Every test here sizes
the table through the production helper, so under-sizing it by the reserved row makes
these fail on CPU instead of only under sustained slot churn on a GPU.
"""

from __future__ import annotations

import logging
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_omni import talker_model_runner
from sglang_omni.models.qwen3_omni.components.feedback_slots import feedback_slot_rows
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner

MAX_RUNNING_REQUESTS = 4
TOP_POOL_IDX = MAX_RUNNING_REQUESTS
HIDDEN = 3
VOCAB = 8


def _model(bs: int) -> SimpleNamespace:
    return SimpleNamespace(
        _feedback_slots=torch.zeros(
            feedback_slot_rows(MAX_RUNNING_REQUESTS), HIDDEN, dtype=torch.float32
        ),
        _feedback_buffer=torch.zeros(bs, HIDDEN, dtype=torch.float32),
        _feedback_mask=torch.zeros(bs, dtype=torch.bool),
        _output_embeds=torch.stack(
            [torch.full((HIDDEN,), float(i + 1)) for i in range(bs)]
        ),
        _output_codes=torch.stack(
            [torch.tensor([i, i + 100], dtype=torch.long) for i in range(bs)]
        ),
    )


def _runner(model: SimpleNamespace) -> QwenTalkerModelRunner:
    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = model
    runner._feedback_enabled = True
    runner._code2wav_target = "code2wav"
    runner._outbox = SimpleNamespace(sent=[])
    runner._outbox.put = runner._outbox.sent.append
    return runner


def _emit_requests(pool_ids: list[int]) -> list:
    return [
        SimpleNamespace(
            data=SimpleNamespace(
                pending_feedback_count=0,
                stage_payload=None,
                req=SimpleNamespace(rid=f"r{i}", req_pool_idx=pool_idx),
            )
        )
        for i, pool_idx in enumerate(pool_ids)
    ]


def _pool_indices(requests: list) -> torch.Tensor:
    """The device-side pool index row the runner slices off the forward batch."""
    return torch.tensor(
        [
            0 if r.data.req.req_pool_idx is None else int(r.data.req.req_pool_idx)
            for r in requests
        ],
        dtype=torch.long,
    )


def _emit_step(runner, requests: list) -> torch.Tensor:
    """Snapshot + scatter, then ship — what the sync post_decode path does."""
    codes_snap = runner._emit_code_chunks_and_feedback(
        requests=requests, pool_indices=_pool_indices(requests)
    )
    runner._put_code_chunks(requests, codes_snap)
    return codes_snap


def _consume_data(
    pool_idx: int | None,
    text: torch.Tensor,
    *,
    override: torch.Tensor | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        pending_feedback_count=1,
        retracted_feedback_embed=override,
        pending_text_queue=deque([text]),
        thinker_chunks_done=False,
        tts_pad_embed=None,
        decode_input_embeds=[],
        req=SimpleNamespace(req_pool_idx=pool_idx),
        feedback_slot_idx=pool_idx,
    )


def test_emit_scatter_lands_at_top_pool_index() -> None:
    model = _model(bs=1)
    runner = _runner(model)
    requests = _emit_requests([TOP_POOL_IDX])

    _emit_step(runner, requests)

    assert torch.equal(model._feedback_slots[TOP_POOL_IDX], model._output_embeds[0])
    assert requests[0].data.feedback_slot_idx == TOP_POOL_IDX


def test_emit_scatter_covers_every_allocatable_index() -> None:
    pool_ids = list(range(1, MAX_RUNNING_REQUESTS + 1))
    model = _model(bs=len(pool_ids))
    runner = _runner(model)

    _emit_step(runner, _emit_requests(pool_ids))

    for i, pool_idx in enumerate(pool_ids):
        assert torch.equal(model._feedback_slots[pool_idx], model._output_embeds[i])
    # Row 0 is the pool's reserved pad row and is never a real request's row.
    assert torch.equal(model._feedback_slots[0], torch.zeros(HIDDEN))


def test_emit_ignores_cuda_graph_padded_rows() -> None:
    # A CUDA-graph padded batch leaves the model's fixed row buffers longer than the
    # real batch. Emit is driven by len(requests), so the pad rows must stay out of
    # the scatter entirely.
    real_pool_ids = [1, TOP_POOL_IDX]
    model = _model(bs=len(real_pool_ids) + 2)
    runner = _runner(model)

    _emit_step(runner, _emit_requests(real_pool_ids))

    for i, pool_idx in enumerate(real_pool_ids):
        assert torch.equal(model._feedback_slots[pool_idx], model._output_embeds[i])
    assert torch.equal(model._feedback_slots[0], torch.zeros(HIDDEN))
    assert len(runner._outbox.sent) == len(real_pool_ids)


def test_consume_gather_reads_top_pool_index() -> None:
    model = _model(bs=1)
    runner = _runner(model)
    feedback = torch.full((HIDDEN,), 5.0)
    model._feedback_slots[TOP_POOL_IDX] = feedback
    text = torch.full((HIDDEN,), 10.0)
    data = _consume_data(TOP_POOL_IDX, text)

    runner._write_feedback_buffers(
        [SimpleNamespace(data=data)],
        _pool_indices([SimpleNamespace(data=data)]),
    )

    assert torch.equal(model._feedback_buffer[0], feedback + text)
    assert model._feedback_mask.tolist() == [True]
    assert data.pending_feedback_count == 0


def test_retract_snapshot_reads_top_pool_index() -> None:
    model = _model(bs=1)
    runner = _runner(model)
    feedback = torch.full((HIDDEN,), 7.0)
    model._feedback_slots[TOP_POOL_IDX] = feedback
    data = _consume_data(TOP_POOL_IDX, torch.zeros(HIDDEN))
    data.retracted_feedback_embed = None

    runner.snapshot_feedback_for_retract(SimpleNamespace(_omni_data=data))

    assert torch.equal(data.retracted_feedback_embed, feedback)
    # The snapshot must be a fresh allocation, not a view of the reusable slot.
    model._feedback_slots[TOP_POOL_IDX] = torch.zeros(HIDDEN)
    assert torch.equal(data.retracted_feedback_embed, feedback)


def test_consume_gather_row_zero_fallback_is_discarded() -> None:
    # A snapshotted request has no pool index; the gather reads the reserved pad row
    # as a placeholder and the override replaces it, so row 0 content cannot leak.
    model = _model(bs=1)
    runner = _runner(model)
    model._feedback_slots[0] = torch.full((HIDDEN,), 99.0)
    override = torch.full((HIDDEN,), 1.0)
    text = torch.full((HIDDEN,), 10.0)
    data = _consume_data(None, text, override=override)

    runner._write_feedback_buffers(
        [SimpleNamespace(data=data)],
        _pool_indices([SimpleNamespace(data=data)]),
    )

    assert torch.equal(model._feedback_buffer[0], override + text)


def _runner_through_init(
    slot_rows: int,
    pool_size: int,
    *,
    mask_rows: int | None = None,
    feedback_enabled: bool = True,
    expose_alloc_size: bool = True,
    expose_pool: bool = True,
) -> QwenTalkerModelRunner:
    mask_rows = slot_rows if mask_rows is None else mask_rows
    model = SimpleNamespace(
        _feedback_slots=torch.zeros(slot_rows, HIDDEN),
        _repetition_mask=torch.zeros(mask_rows, VOCAB, dtype=torch.bool),
        _suppress_mask=torch.zeros(mask_rows, VOCAB, dtype=torch.bool),
    )
    # Mirrors ReqToTokenPool: size rows allocatable from [1, size], _alloc_size total.
    pool = SimpleNamespace(size=pool_size)
    if expose_alloc_size:
        pool._alloc_size = pool_size + 1
    inner_runner = SimpleNamespace(model=model)
    if expose_pool:
        inner_runner.req_to_token_pool = pool
    tp_worker = SimpleNamespace(gpu_id=0, model_runner=inner_runner)
    return QwenTalkerModelRunner(
        tp_worker,
        output_processor=None,
        outbox=None,
        feedback_enabled=feedback_enabled,
    )


def test_startup_guard_accepts_pool_sized_slots() -> None:
    runner = _runner_through_init(
        feedback_slot_rows(MAX_RUNNING_REQUESTS), MAX_RUNNING_REQUESTS
    )

    assert runner.model._feedback_slots.shape[0] == MAX_RUNNING_REQUESTS + 1


def test_startup_guard_rejects_sampling_masks_missing_the_reserved_row() -> None:
    # The repetition/suppress masks are addressed by the same index, so a table that
    # only covers max_running_requests rows must be rejected the same way.
    with pytest.raises(RuntimeError, match="_repetition_mask is too small"):
        _runner_through_init(
            feedback_slot_rows(MAX_RUNNING_REQUESTS),
            MAX_RUNNING_REQUESTS,
            mask_rows=MAX_RUNNING_REQUESTS,
        )


def test_startup_guard_rejects_slots_missing_the_reserved_row() -> None:
    with pytest.raises(RuntimeError, match="too small for the request pool"):
        _runner_through_init(MAX_RUNNING_REQUESTS, MAX_RUNNING_REQUESTS)


def test_startup_guard_rejects_tables_undersized_by_many_rows() -> None:
    # A bound check, not a "+1" check: a table far below the pool must be rejected too.
    with pytest.raises(RuntimeError, match="too small for the request pool"):
        _runner_through_init(4, 16)


def test_startup_guard_logs_instead_of_silently_skipping(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # An upstream rename of req_to_token_pool leaves every index site working and only
    # this check quiet, so the skip has to announce itself.
    with caplog.at_level(logging.WARNING, logger=talker_model_runner.__name__):
        _runner_through_init(
            MAX_RUNNING_REQUESTS, MAX_RUNNING_REQUESTS, expose_pool=False
        )

    assert "bound check skipped" in caplog.text
    assert "req_to_token_pool" in caplog.text


def test_startup_guard_falls_back_to_size_when_alloc_size_absent() -> None:
    with pytest.raises(RuntimeError, match="too small for the request pool"):
        _runner_through_init(
            MAX_RUNNING_REQUESTS, MAX_RUNNING_REQUESTS, expose_alloc_size=False
        )


def test_startup_guard_skipped_when_feedback_disabled() -> None:
    runner = _runner_through_init(
        MAX_RUNNING_REQUESTS, MAX_RUNNING_REQUESTS, feedback_enabled=False
    )

    assert runner._feedback_enabled is False
