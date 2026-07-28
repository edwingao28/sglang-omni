# SPDX-License-Identifier: Apache-2.0
"""Talker post_decode split into the async-decode launch/resolve pair.

Under the one-step-lookahead loop the launch half runs right after the forward and
the resolve half runs a step later. Everything that reads the model's fixed row
buffers (``_output_codes`` / ``_output_embeds``) has to happen in the launch half,
before the next forward overwrites them, while the launch must not sample: the
talker already sampled inside the forward.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner

POOL_SIZE = 8
POOL_IDS = [5, 2, 7]
HIDDEN = 3


def _model(n: int) -> SimpleNamespace:
    return SimpleNamespace(
        _feedback_slots=torch.zeros(POOL_SIZE, HIDDEN),
        _sampled_token_ids=torch.arange(n, dtype=torch.long) + 40,
        _output_codes=torch.stack(
            [torch.tensor([i, i + 100], dtype=torch.long) for i in range(n)]
        ),
        _output_embeds=torch.stack(
            [torch.full((HIDDEN,), float(i + 1)) for i in range(n)]
        ),
    )


def _runner(model: SimpleNamespace, *, feedback_enabled: bool = True):
    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = model
    runner._feedback_enabled = feedback_enabled
    runner._code2wav_target = "code2wav"
    runner._outbox = SimpleNamespace(sent=[])
    runner._outbox.put = runner._outbox.sent.append

    def _never(*args, **kwargs):
        raise AssertionError(
            "the async launch must not sample: the forward already did"
        )

    runner._sample_next_token_ids = _never
    return runner


def _req(i: int, *, finished: bool = False, retracted: bool = False):
    return SimpleNamespace(
        rid=f"r{i}",
        req_pool_idx=POOL_IDS[i],
        is_retracted=retracted,
        finished=lambda: finished,
    )


def _requests(n: int, **kwargs) -> list:
    return [
        SimpleNamespace(
            data=SimpleNamespace(
                pending_feedback_count=0,
                feedback_slot_idx=None,
                stage_payload=None,
                req=_req(i, **kwargs),
            )
        )
        for i in range(n)
    ]


def test_launch_publishes_tokens_and_emits_without_sampling() -> None:
    n = 3
    model = _model(n)
    runner = _runner(model)
    requests = _requests(n)
    result = SimpleNamespace(next_token_ids=None)

    launch_buf = runner.post_decode_launch(
        result, forward_batch=None, requests=requests
    )

    assert torch.equal(result.next_token_ids, torch.tensor([40, 41, 42]))
    assert launch_buf is result.next_token_ids
    assert [msg.request_id for msg in runner._outbox.sent] == ["r0", "r1", "r2"]
    for i in range(n):
        assert requests[i].data.pending_feedback_count == 1
        assert requests[i].data.feedback_slot_idx == POOL_IDS[i]
        assert torch.equal(model._feedback_slots[POOL_IDS[i]], model._output_embeds[i])


def test_launch_ids_survive_the_next_forward() -> None:
    # The next step's forward writes _sampled_token_ids in place before this step
    # resolves, so the launch payload has to be a private copy.
    model = _model(2)
    runner = _runner(model)
    result = SimpleNamespace(next_token_ids=None)

    launch_buf = runner.post_decode_launch(
        result, forward_batch=None, requests=_requests(2)
    )
    model._sampled_token_ids.copy_(torch.tensor([99, 99]))

    runner.post_decode_resolve(launch_buf, result, None, None, [])
    assert torch.equal(result.next_token_ids, torch.tensor([40, 41]))


def test_resolve_neither_emits_nor_counts_again() -> None:
    model = _model(2)
    runner = _runner(model)
    requests = _requests(2)
    result = SimpleNamespace(next_token_ids=None)

    launch_buf = runner.post_decode_launch(
        result, forward_batch=None, requests=requests
    )
    emitted_at_launch = len(runner._outbox.sent)
    runner.post_decode_resolve(launch_buf, result, None, None, requests)

    assert len(runner._outbox.sent) == emitted_at_launch
    assert [req.data.pending_feedback_count for req in requests] == [1, 1]


def test_launch_matches_the_sync_post_decode() -> None:
    requests_sync = _requests(2)
    runner_sync = _runner(_model(2))
    sync_result = SimpleNamespace(next_token_ids=None)
    runner_sync.post_decode(sync_result, None, None, requests_sync)

    requests_async = _requests(2)
    runner_async = _runner(_model(2))
    async_result = SimpleNamespace(next_token_ids=None)
    runner_async.post_decode_launch(async_result, None, requests_async)

    assert torch.equal(sync_result.next_token_ids, async_result.next_token_ids)
    assert torch.equal(
        runner_sync.model._feedback_slots, runner_async.model._feedback_slots
    )
    assert [msg.request_id for msg in runner_sync._outbox.sent] == [
        msg.request_id for msg in runner_async._outbox.sent
    ]
    assert [req.data.pending_feedback_count for req in requests_sync] == [
        req.data.pending_feedback_count for req in requests_async
    ]


@pytest.mark.parametrize("state", ["finished", "retracted"])
def test_done_rows_keep_their_slot_but_ship_no_frame(state: str) -> None:
    n = 2
    model = _model(n)
    runner = _runner(model)
    requests = _requests(n)
    done = requests[0].data.req
    if state == "finished":
        done.finished = lambda: True
    else:
        done.is_retracted = True
    result = SimpleNamespace(next_token_ids=None)

    runner.post_decode_launch(result, forward_batch=None, requests=requests)

    assert [msg.request_id for msg in runner._outbox.sent] == ["r1"]
    # The slot and the counter stay live: the retract snapshot reads both.
    assert torch.equal(model._feedback_slots[POOL_IDS[0]], model._output_embeds[0])
    assert requests[0].data.pending_feedback_count == 1
    assert requests[0].data.feedback_slot_idx == POOL_IDS[0]


def test_launch_is_inert_without_feedback() -> None:
    runner = _runner(_model(1), feedback_enabled=False)
    result = SimpleNamespace(next_token_ids=None)

    assert runner.post_decode_launch(result, None, _requests(1)) is None
    assert result.next_token_ids is None
    assert runner._outbox.sent == []


def test_feedback_talker_is_lookahead_eligible_despite_penalties() -> None:
    # The base gate rejects repetition penalties because its launch samples on the
    # host against a lagged output history; the talker samples in the forward.
    runner = _runner(_model(1))
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(
                sampling_params=SimpleNamespace(
                    repetition_penalty=1.05,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    min_new_tokens=0,
                )
            )
        ]
    )

    assert runner.lookahead_eligible(batch) is True

    runner._feedback_enabled = False
    assert runner.lookahead_eligible(batch) is False
