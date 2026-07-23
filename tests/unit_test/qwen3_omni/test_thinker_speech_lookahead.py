# SPDX-License-Identifier: Apache-2.0
"""Aux hidden-state alignment across two in-flight thinker decode steps (item 1.4).

Speech batches are now async-eligible; the guard against launch(N) clobbering
resolve(N-1)'s talker hidden is a launch-time device snapshot that resolve restores
onto the single-slot side channel the output processor reads. These tests drive the
real post_decode_launch / post_decode_resolve / execute_resolve helpers and the real
SGLangOutputProcessor read path on CPU tensors.
"""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import torch

from sglang_omni.model_runner.base import _PendingStep
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.scheduling.types import SchedulerOutput

# Note:(Wenyao Gao) load the output processor straight from its file: the package
# __init__ pulls heavy sglang.srt cache modules the unit shim does not fake
_OP_PATH = (
    Path(__file__).resolve().parents[3]
    / "sglang_omni/scheduling/sglang_backend/output_processor.py"
)
_spec = importlib.util.spec_from_file_location("_op_under_test", _OP_PATH)
_op = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_op)
SGLangOutputProcessor = _op.SGLangOutputProcessor

CAPTURE_LAYERS = [0, 24]


def _aux(step: int, rows: int) -> tuple[torch.Tensor, ...]:
    layer = torch.tensor([[float(step * 100 + row)] for row in range(rows)])
    return tuple(layer.clone() for _ in CAPTURE_LAYERS)


def _expected(step: int, row: int) -> torch.Tensor:
    return torch.tensor([float(step * 100 + row)])


def _runner(model, rows: int) -> ThinkerModelRunner:
    r = object.__new__(ThinkerModelRunner)
    r.model = model
    # Note:(Wenyao Gao) pre-seed non-pinned host buffers: pin_memory needs CUDA,
    # absent on the unit host
    r._th_host_bufs = [torch.zeros(rows, dtype=torch.long) for _ in range(2)]
    r._th_slot = 0
    r._async_query_hit = 0
    r._async_query_miss = 0
    return r


def _model(aux) -> types.SimpleNamespace:
    return types.SimpleNamespace(_captured_aux_hidden_states=aux)


def _result(rows: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        next_token_ids=torch.arange(rows), logits_output=None, can_run_cuda_graph=False
    )


def _sched_req(rid: str, *, retracted: bool = False, finished: bool = False):
    req = types.SimpleNamespace(is_retracted=retracted, finished=lambda f=finished: f)
    data = types.SimpleNamespace(req=req, generation_steps=0, extra_model_outputs={})
    return types.SimpleNamespace(request_id=rid, data=data)


def _sched_output(sched_reqs: list) -> SchedulerOutput:
    batch_data = types.SimpleNamespace(
        reqs=[types.SimpleNamespace(extend_input_len=1) for _ in sched_reqs]
    )
    return SchedulerOutput(requests=sched_reqs, batch_data=batch_data)


def _output_processor(model) -> SGLangOutputProcessor:
    return SGLangOutputProcessor(
        capture_hidden=True, capture_hidden_layers=CAPTURE_LAYERS, model=model
    )


def test_launch_snapshots_and_clears_live_capture():
    rows = 2
    model = _model(_aux(step=1, rows=rows))
    runner = _runner(model, rows)
    launch_buf = runner.post_decode_launch(_result(rows), None, [None] * rows)
    assert model._captured_aux_hidden_states is None
    for layer in launch_buf.aux_hidden:
        assert torch.equal(layer, _aux(1, rows)[0])


def test_resolve_reads_launch_snapshot_not_next_launch():
    rows = 2
    model = _model(_aux(step=1, rows=rows))
    runner = _runner(model, rows)
    result = _result(rows)
    launch_buf = runner.post_decode_launch(result, None, [None] * rows)

    model._captured_aux_hidden_states = _aux(step=2, rows=rows)
    runner.post_decode_launch(_result(rows), None, [None] * rows)
    runner.post_decode_resolve(launch_buf, result, None, None, [None] * rows)

    rids = ["r0", "r1"]
    outputs = _output_processor(model).process(
        result, _sched_output([_sched_req(r) for r in rids])
    )
    for row, rid in enumerate(rids):
        hidden = outputs[rid].extra["hidden_states"]
        assert torch.equal(hidden["embed"], _expected(1, row))
        assert torch.equal(hidden[24], _expected(1, row))
    assert model._captured_aux_hidden_states is None


def test_retract_between_launch_and_resolve_keeps_neighbors_aligned():
    rows = 3
    model = _model(_aux(step=1, rows=rows))
    runner = _runner(model, rows)
    runner.output_processor = _output_processor(model)
    result = _result(rows)
    launch_buf = runner.post_decode_launch(result, None, [None] * rows)

    model._captured_aux_hidden_states = _aux(step=2, rows=rows)
    runner.post_decode_launch(_result(rows), None, [None] * rows)

    sched_reqs = [
        _sched_req("r0"),
        _sched_req("r1", retracted=True),  # dropped downstream by the overrun filter
        _sched_req("r2"),
    ]
    sched_output = _sched_output(sched_reqs)
    pending = _PendingStep(
        event=types.SimpleNamespace(query=lambda: True),
        launch_buf=launch_buf,
        scheduler_output=sched_output,
        forward_batch=None,
        schedule_batch=types.SimpleNamespace(is_prefill_only=False),
        model_worker_batch=None,
        batch_result=result,
        n_real=rows,
    )
    mr_output = runner.execute_resolve(pending)

    outputs = mr_output.outputs
    assert torch.equal(outputs["r0"].extra["hidden_states"]["embed"], _expected(1, 0))
    assert torch.equal(outputs["r2"].extra["hidden_states"]["embed"], _expected(1, 2))
    assert sched_reqs[1].data.generation_steps == 0
    assert sched_reqs[0].data.generation_steps == 1
    assert sched_reqs[2].data.generation_steps == 1


def test_snapshot_clones_against_in_place_capture_mutation():
    # Note:(Wenyao Gao) in-place mutation (not reassignment) is what a graph replay
    # does to the aliased buffers — only this catches a missing .clone()
    rows = 2
    step1 = _aux(step=1, rows=rows)
    model = _model(step1)
    runner = _runner(model, rows)
    result = _result(rows)
    launch_buf = runner.post_decode_launch(result, None, [None] * rows)

    for t in step1:
        t.mul_(0)

    for layer in launch_buf.aux_hidden:
        assert torch.equal(layer, _aux(1, rows)[0])

    runner.post_decode_resolve(launch_buf, result, None, None, [None] * rows)
    rids = ["r0", "r1"]
    outputs = _output_processor(model).process(
        result, _sched_output([_sched_req(r) for r in rids])
    )
    for row, rid in enumerate(rids):
        hidden = outputs[rid].extra["hidden_states"]
        assert torch.equal(hidden["embed"], _expected(1, row))
        assert torch.equal(hidden[24], _expected(1, row))


def test_text_only_none_capture_launch_resolve_finalize():
    rows = 2
    model = _model(None)
    runner = _runner(model, rows)
    result = _result(rows)
    launch_buf = runner.post_decode_launch(result, None, [None] * rows)
    assert launch_buf.aux_hidden is None
    assert model._captured_aux_hidden_states is None

    runner.post_decode_resolve(launch_buf, result, None, None, [None] * rows)
    assert model._captured_aux_hidden_states is None

    rids = ["r0", "r1"]
    outputs = _output_processor(model).process(
        result, _sched_output([_sched_req(r) for r in rids])
    )
    for rid in rids:
        assert rid in outputs
        assert outputs[rid].extra is None
