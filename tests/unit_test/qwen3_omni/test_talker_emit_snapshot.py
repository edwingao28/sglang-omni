# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner

POOL_SIZE = 8
POOL_IDS = [5, 2, 7, 0, 3]


def _fake_model(n: int, hidden: int, code_groups: int) -> SimpleNamespace:
    return SimpleNamespace(
        _feedback_slots=torch.zeros(POOL_SIZE, hidden),
        _output_codes=torch.stack(
            [torch.tensor([i, i + 100], dtype=torch.long) for i in range(n)]
        )[:, :code_groups],
        _output_embeds=torch.stack(
            [torch.full((hidden,), float(i * 7 + 1)) for i in range(n)]
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


def _data(i: int) -> SimpleNamespace:
    return SimpleNamespace(
        pending_feedback_count=0,
        stage_payload=None,
        req=SimpleNamespace(rid=f"r{i}", req_pool_idx=POOL_IDS[i]),
    )


def _requests(n: int) -> list:
    return [SimpleNamespace(data=_data(i)) for i in range(n)]


def _pool_indices(requests: list) -> torch.Tensor:
    return torch.tensor(
        [
            0 if r.data.req.req_pool_idx is None else int(r.data.req.req_pool_idx)
            for r in requests
        ],
        dtype=torch.long,
    )


def _emit_step(runner, requests: list) -> torch.Tensor:
    codes_snap = runner._emit_code_chunks_and_feedback(
        requests=requests, pool_indices=_pool_indices(requests)
    )
    runner._put_code_chunks(requests, codes_snap)
    return codes_snap


def test_emitted_rows_survive_next_step_inplace_write() -> None:
    n, hidden, code_groups = 4, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    codes_before = model._output_codes.clone()
    embeds_before = model._output_embeds.clone()

    requests = _requests(n)
    _emit_step(runner, requests)

    model._output_codes.copy_(model._output_codes + 999)
    model._output_embeds.copy_(model._output_embeds + 999.0)

    for i, msg in enumerate(runner._outbox.sent):
        assert torch.equal(msg.data, codes_before[i])
        assert torch.equal(model._feedback_slots[POOL_IDS[i]], embeds_before[i])


def test_emit_writes_feedback_to_pool_indexed_slots() -> None:
    n, hidden, code_groups = 3, 4, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    requests = _requests(n)
    _emit_step(runner, requests)

    for i in range(n):
        assert torch.equal(model._feedback_slots[POOL_IDS[i]], model._output_embeds[i])
        assert requests[i].data.pending_feedback_count == 1
        assert not hasattr(requests[i].data, "pending_feedback_queue")

    for row in set(range(POOL_SIZE)) - set(POOL_IDS[:n]):
        assert torch.equal(model._feedback_slots[row], torch.zeros(hidden))


def test_emit_keeps_one_batched_clone_for_codes() -> None:
    n, hidden, code_groups = 5, 4, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    clones: list = []
    orig_clone = torch.Tensor.clone

    def _counting_clone(self, *args, **kwargs):
        out = orig_clone(self, *args, **kwargs)
        clones.append(out)
        return out

    requests = _requests(n)
    torch.Tensor.clone = _counting_clone
    try:
        _emit_step(runner, requests)
    finally:
        torch.Tensor.clone = orig_clone

    assert len(clones) == 1

    code_ptrs = {msg.data.untyped_storage().data_ptr() for msg in runner._outbox.sent}
    assert len(code_ptrs) == 1


def test_emit_counts_accumulate_across_steps() -> None:
    n, hidden, code_groups = 2, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)

    requests = _requests(n)

    _emit_step(runner, requests)
    model._output_embeds.copy_(model._output_embeds + 1.0)
    _emit_step(runner, requests)

    for i in range(n):
        assert requests[i].data.pending_feedback_count == 2
        assert torch.equal(model._feedback_slots[POOL_IDS[i]], model._output_embeds[i])


def _step(
    runner: QwenTalkerModelRunner,
    model: SimpleNamespace,
    requests: list,
    first_codes: list[int],
) -> int:
    """Run one decode step with the given per-row layer-0 codes; return #msgs."""
    for row, code in enumerate(first_codes):
        model._output_codes[row] = torch.tensor([code, code + 1000], dtype=torch.long)
    before = len(runner._outbox.sent)
    _emit_step(runner, requests)
    return len(runner._outbox.sent) - before


def test_flush_cadence_and_force_flush() -> None:
    n, hidden, code_groups = 1, 3, 2
    model = _fake_model(n, hidden, code_groups)
    runner = _runner(model)
    requests = _requests(n)

    emitted_at: dict[int, torch.Tensor] = {}
    for frame in range(1, 26):
        if _step(runner, model, requests, [frame]):
            emitted_at[frame] = runner._outbox.sent[-1].data

    # Boundaries 1 (immediate first frame), 10, 20; frames 2-9 stay buffered.
    assert sorted(emitted_at) == [1, 10, 20]
    assert emitted_at[1].ndim == 1
    assert emitted_at[1].tolist() == [1, 1001]
    assert emitted_at[10].shape == (9, code_groups)
    assert emitted_at[10][:, 0].tolist() == list(range(2, 11))
    assert emitted_at[20].shape == (10, code_groups)
    assert emitted_at[20][:, 0].tolist() == list(range(11, 21))

    # Finish force-flushes the 5-frame remainder ahead of the result payload.
    runner.on_request_finished("r0", requests[0].data)
    tail = runner._outbox.sent[-1]
    assert tail.type == "stream"
    assert tail.data.shape == (5, code_groups)
    assert tail.data[:, 0].tolist() == list(range(21, 26))
    assert requests[0].data.code_frames_sent == 25
    assert requests[0].data.pending_code_rows == []
    assert all(m.metadata == {"stream": False} for m in runner._outbox.sent)
    # Feedback stays strictly per-step regardless of IPC batching.
    assert requests[0].data.pending_feedback_count == 25

    # Second finish is a no-op: nothing left to flush.
    runner.on_request_finished("r0", requests[0].data)
    assert len(runner._outbox.sent) == 4


def test_eos_step_flushes_immediately_with_eos_row() -> None:
    model = _fake_model(1, 3, 2)
    runner = _runner(model)
    requests = _requests(1)

    for frame in (1, 2, 3):
        _step(runner, model, requests, [frame])
    assert len(runner._outbox.sent) == 1  # only the first frame flushed

    assert _step(runner, model, requests, [2150]) == 1
    msg = runner._outbox.sent[-1]
    assert msg.data.shape == (3, 2)
    assert msg.data[:, 0].tolist() == [2, 3, 2150]  # EOS row rides inside
    assert msg.metadata == {"stream": False}

    runner.on_request_finished("r0", requests[0].data)
    assert len(runner._outbox.sent) == 2  # nothing left to force-flush


def test_aborted_request_buffer_never_emits() -> None:
    model = _fake_model(1, 3, 2)
    runner = _runner(model)
    requests = _requests(1)

    for frame in (1, 2, 3):
        _step(runner, model, requests, [frame])

    # Abort path: no finish hook, no flush -- the buffer dies with the data.
    assert len(runner._outbox.sent) == 1
    assert len(requests[0].data.pending_code_rows) == 2


def test_buffered_rows_survive_next_step_inplace_write() -> None:
    model = _fake_model(1, 3, 2)
    runner = _runner(model)
    requests = _requests(1)

    for frame in (1, 2, 3):
        _step(runner, model, requests, [frame])
    # Clobber the fixed-address output buffer after the last step.
    model._output_codes.copy_(model._output_codes + 999)
    runner.on_request_finished("r0", requests[0].data)

    tail = runner._outbox.sent[-1]
    assert tail.data[:, 0].tolist() == [2, 3]
