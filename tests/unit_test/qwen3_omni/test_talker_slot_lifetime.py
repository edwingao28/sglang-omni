# SPDX-License-Identifier: Apache-2.0
"""Lifetime of ``feedback_slot_idx`` across retract, and the depth-1 replay limit.

``feedback_slot_idx`` records the ``req_pool_idx`` a feedback row was emitted into.
Retract frees that pool index, so the retract snapshot is the last legal read of the
recorded slot: any later read can land on a row another request now owns. These tests
drive a second retract against a recycled slot on CPU, without a server or a GPU.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from sglang.srt.managers.scheduler import Scheduler as _Upstream

from sglang_omni.models.qwen3_omni.components.talker import feedback_slot_rows
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler

MAX_RUNNING_REQUESTS = 4
POOL_IDX = 2
HIDDEN = 3
CPU = torch.device("cpu")


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        _feedback_slots=torch.zeros(
            feedback_slot_rows(MAX_RUNNING_REQUESTS), HIDDEN, dtype=torch.float32
        ),
        _feedback_buffer=torch.zeros(1, HIDDEN, dtype=torch.float32),
        _feedback_mask=torch.zeros(1, dtype=torch.bool),
        _output_embeds=torch.zeros(1, HIDDEN, dtype=torch.float32),
        _output_codes=torch.tensor([[0, 100]], dtype=torch.long),
    )


def _runner(model: SimpleNamespace) -> QwenTalkerModelRunner:
    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = model
    runner._feedback_enabled = True
    runner._code2wav_target = "code2wav"
    runner._outbox = SimpleNamespace(sent=[])
    runner._outbox.put = runner._outbox.sent.append
    return runner


def _req(rid: str, pool_idx: int | None) -> SimpleNamespace:
    return SimpleNamespace(rid=rid, req_pool_idx=pool_idx)


def _data(req: SimpleNamespace) -> SimpleNamespace:
    data = SimpleNamespace(
        pending_feedback_count=0,
        feedback_slot_idx=None,
        retracted_feedback_embed=None,
        pending_text_queue=deque(),
        thinker_chunks_done=False,
        tts_pad_embed=None,
        stage_payload=None,
        decode_input_embeds=[],
        req=req,
    )
    req._omni_data = data
    return data


def _emit(
    runner: QwenTalkerModelRunner,
    req: SimpleNamespace,
    data: SimpleNamespace,
    value: float,
) -> None:
    runner.model._output_embeds[0] = torch.full((HIDDEN,), value)
    # Note (wenyao): batch and data.req must be the same object so the two sources
    # of req_pool_idx cannot diverge.
    runner._emit_code_chunks_and_feedback(
        schedule_batch=SimpleNamespace(reqs=[req]),
        requests=[SimpleNamespace(data=data)],
    )


def _retract_scheduler(
    runner: QwenTalkerModelRunner, monkeypatch: Any
) -> QwenTalkerScheduler:
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler._model_runner = runner
    scheduler.queued = []
    monkeypatch.setattr(
        _Upstream,
        "_add_request_to_queue",
        lambda self, req, is_retracted=False: self.queued.append(
            (req.rid, is_retracted)
        ),
    )
    return scheduler


def _retract(scheduler: QwenTalkerScheduler, req: SimpleNamespace) -> None:
    req.is_retracted = True
    req.req_pool_idx = None
    scheduler._add_request_to_queue(req, is_retracted=True)


def _replay_one_row(data: SimpleNamespace) -> torch.Tensor | None:
    data.pending_text_queue.append(torch.full((HIDDEN,), 10.0))
    return QwenTalkerModelRunner._take_next_decode_input_embed(
        sched_req=SimpleNamespace(data=data),
        device=CPU,
        dtype=torch.float32,
    )


def test_second_retract_cannot_read_a_recycled_slot(monkeypatch: Any) -> None:
    model = _model()
    runner = _runner(model)
    req = _req("rA", POOL_IDX)
    data = _data(req)

    # Two emits with no consume between them: a retract requeue re-prefills and emits
    # again while the previous frame is still owed, so one snapshot cannot cover the
    # count and the request survives its own replay with feedback still pending.
    _emit(runner, req, data, 5.0)
    _emit(runner, req, data, 6.0)
    assert data.pending_feedback_count == 2
    assert data.feedback_slot_idx == POOL_IDX

    scheduler = _retract_scheduler(runner, monkeypatch)
    _retract(scheduler, req)
    assert torch.equal(data.retracted_feedback_embed, torch.full((HIDDEN,), 6.0))

    # Partial replay: the snapshot comes back and is dropped, one frame still owed.
    assert _replay_one_row(data) is not None
    assert data.pending_feedback_count == 1
    assert data.retracted_feedback_embed is None

    # POOL_IDX has been reallocated and now holds another request's row.
    other = _req("rB", POOL_IDX)
    _emit(runner, other, _data(other), 99.0)
    assert torch.equal(model._feedback_slots[POOL_IDX], torch.full((HIDDEN,), 99.0))

    with pytest.raises(RuntimeError, match="no recorded slot"):
        _retract(scheduler, req)

    # The stranger's row must not have been adopted as rA's feedback.
    assert data.retracted_feedback_embed is None


def test_snapshot_clears_the_recorded_slot_index(monkeypatch: Any) -> None:
    model = _model()
    runner = _runner(model)
    req = _req("rA", POOL_IDX)
    data = _data(req)
    _emit(runner, req, data, 5.0)

    _retract(_retract_scheduler(runner, monkeypatch), req)

    assert torch.equal(data.retracted_feedback_embed, torch.full((HIDDEN,), 5.0))
    assert data.feedback_slot_idx is None


def test_retract_clears_the_slot_index_when_a_snapshot_is_already_held(
    monkeypatch: Any,
) -> None:
    # The early return for an unconsumed snapshot must still retire the index: the
    # pool slot is freed on this retract too.
    model = _model()
    runner = _runner(model)
    req = _req("rA", POOL_IDX)
    data = _data(req)
    _emit(runner, req, data, 5.0)
    held = torch.full((HIDDEN,), 42.0)
    data.retracted_feedback_embed = held

    _retract(_retract_scheduler(runner, monkeypatch), req)

    assert data.retracted_feedback_embed is held
    assert data.feedback_slot_idx is None


def test_retract_without_pending_feedback_clears_the_slot_index(
    monkeypatch: Any,
) -> None:
    model = _model()
    runner = _runner(model)
    req = _req("rA", POOL_IDX)
    data = _data(req)
    _emit(runner, req, data, 5.0)
    data.pending_feedback_count = 0

    _retract(_retract_scheduler(runner, monkeypatch), req)

    assert data.retracted_feedback_embed is None
    assert data.feedback_slot_idx is None


def test_emit_after_retract_rearms_the_slot_index(monkeypatch: Any) -> None:
    # Invalidation must not outlive the next emit: the requeued request prefills into
    # a fresh pool index and its snapshot has to work again.
    model = _model()
    runner = _runner(model)
    scheduler = _retract_scheduler(runner, monkeypatch)
    req = _req("rA", POOL_IDX)
    data = _data(req)
    _emit(runner, req, data, 5.0)
    _retract(scheduler, req)
    data.retracted_feedback_embed = None
    data.pending_feedback_count = 0

    new_pool_idx = POOL_IDX + 1
    req.req_pool_idx = new_pool_idx
    _emit(runner, req, data, 7.0)
    assert data.feedback_slot_idx == new_pool_idx

    _retract(scheduler, req)

    assert torch.equal(data.retracted_feedback_embed, torch.full((HIDDEN,), 7.0))
    assert data.feedback_slot_idx is None


def test_replay_past_the_single_snapshot_raises() -> None:
    # Depth is 1 by construction: the slot table keeps only the newest emit per
    # request, so a retract can recover exactly one row however many are owed.
    model = _model()
    runner = _runner(model)
    req = _req("rA", POOL_IDX)
    data = _data(req)
    _emit(runner, req, data, 5.0)
    _emit(runner, req, data, 6.0)
    data.retracted_feedback_embed = model._feedback_slots[POOL_IDX].clone()
    data.pending_text_queue.extend(
        [torch.full((HIDDEN,), 10.0), torch.full((HIDDEN,), 20.0)]
    )

    with pytest.raises(RuntimeError, match="at most one feedback row"):
        QwenTalkerModelRunner._generated_prefill_slice(
            sched_req=SimpleNamespace(data=data),
            gen_start=0,
            gen_end=2,
            device=CPU,
            dtype=torch.float32,
        )

    assert data.pending_feedback_count == 1
