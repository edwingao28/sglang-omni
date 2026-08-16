# SPDX-License-Identifier: Apache-2.0
"""Parking the talker prompt KV between the two prefill phases."""

from __future__ import annotations

from array import array
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_omni import talker_scheduler as talker_scheduler_mod
from sglang_omni.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler
from sglang_omni.models.qwen3_omni.two_phase_kv import (
    ASSISTANT_TAIL_ROWS,
    adopt_parked_prompt_kv,
    has_parked_prefix_shadow,
    install_parked_prefix_shadow,
    snapshot_prompt_kv,
)

PROMPT_ROWS = 6


def _chunk_cache():
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.chunk_cache import ChunkCache

    return ChunkCache(
        CacheInitParams(
            disable=True,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
        )
    )


def _real_req(input_ids: list[int]):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    return Req(
        rid="req-kv",
        origin_input_text="",
        origin_input_ids=array("q", input_ids),
        sampling_params=SamplingParams(max_new_tokens=8),
    )


def test_a_plain_requeue_drops_the_parked_prompt_prefix() -> None:
    """The blocker: the talker's ChunkCache answers every match with nothing."""
    req = _real_req(list(range(PROMPT_ROWS + ASSISTANT_TAIL_ROWS)))
    req.prefix_indices = torch.arange(PROMPT_ROWS, dtype=torch.int64)

    req.init_next_round_input(_chunk_cache())

    assert req.prefix_indices.numel() == 0


def test_the_instance_shadow_keeps_the_parked_prefix_across_a_requeue() -> None:
    parked = torch.arange(100, 100 + PROMPT_ROWS, dtype=torch.int64)
    req = _real_req(list(range(PROMPT_ROWS + ASSISTANT_TAIL_ROWS)))
    install_parked_prefix_shadow(req, parked)

    req.init_next_round_input(_chunk_cache())

    assert torch.equal(req.prefix_indices, parked)
    # cache_finished_req frees [cache_protected_len, kv_committed_len); the
    # parked range is this request's own, so it must stay inside that window.
    assert req.cache_protected_len == 0
    assert len(req.full_untruncated_fill_ids) == PROMPT_ROWS + ASSISTANT_TAIL_ROWS


def test_snapshot_prompt_kv_takes_the_rows_the_prompt_extend_wrote() -> None:
    req_to_token = torch.arange(60, dtype=torch.int32).reshape(4, 15)
    req = SimpleNamespace(req_pool_idx=2, extend_range=SimpleNamespace(end=PROMPT_ROWS))

    parked = snapshot_prompt_kv(req, SimpleNamespace(req_to_token=req_to_token))

    assert parked.dtype is torch.int64
    assert parked.tolist() == list(range(30, 30 + PROMPT_ROWS))
    req_to_token.zero_()
    assert parked.tolist() == list(range(30, 30 + PROMPT_ROWS))


def test_adopting_a_parked_request_moves_the_pool_row_and_the_prefix() -> None:
    parked_prefix = torch.arange(PROMPT_ROWS, dtype=torch.int64)
    parked = SimpleNamespace(
        req_pool_idx=3,
        kv=SimpleNamespace(kv_allocated_len=PROMPT_ROWS),
        kv_committed_len=PROMPT_ROWS,
        extend_batch_idx=1,
        prefix_indices=parked_prefix,
    )
    fresh = _real_req(list(range(PROMPT_ROWS + ASSISTANT_TAIL_ROWS)))

    adopt_parked_prompt_kv(fresh, parked)

    assert fresh.req_pool_idx == 3
    assert fresh.kv_committed_len == PROMPT_ROWS
    assert has_parked_prefix_shadow(fresh)
    assert parked.req_pool_idx is None and parked.kv is None
    fresh.init_next_round_input(_chunk_cache())
    assert torch.equal(fresh.prefix_indices, parked_prefix)


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------


def _scheduler(*, max_parked: int = 8) -> QwenTalkerScheduler:
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler._two_phase_prefill = True
    scheduler._two_phase_kv = True
    scheduler._two_phase_max_parked = max_parked
    scheduler._phase_one_queue = deque()
    scheduler._phase_one_data = {}
    scheduler._parked_reqs = {}
    scheduler._prompt_segment_futures = {}
    scheduler._deferred_request_payloads = {}
    scheduler.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(60, dtype=torch.int32).reshape(4, 15),
        available_size=lambda: 4,
    )
    return scheduler


def _phase_one_req(rid: str, *, pool_idx: int = 1):
    req = _real_req(list(range(PROMPT_ROWS)))
    req.rid = rid
    req.req_pool_idx = pool_idx
    req.kv = SimpleNamespace(kv_allocated_len=PROMPT_ROWS)
    req.kv_committed_len = PROMPT_ROWS
    req.set_extend_range(0, PROMPT_ROWS)
    req._omni_data = SimpleNamespace(req=req, tail_pending=True)
    return req


def _batch(reqs: list) -> SimpleNamespace:
    return SimpleNamespace(
        reqs=reqs, forward_mode=SimpleNamespace(is_extend=lambda: True)
    )


def test_parking_empties_the_batch_and_keeps_the_prompt_kv() -> None:
    """An emptied batch is what stops the merge into the running decode batch."""
    scheduler = _scheduler()
    req = _phase_one_req("rid-park", pool_idx=2)
    batch = _batch([req])

    scheduler._park_phase_one_batch(batch)

    assert batch.reqs == []
    assert scheduler._parked_reqs == {"rid-park": req}
    assert req.req_pool_idx == 2
    assert req.cache_protected_len == 0
    assert req.prefix_indices.tolist() == list(range(30, 30 + PROMPT_ROWS))


def test_a_batch_is_phase_one_only_when_every_row_is_tail_pending() -> None:
    pending = _phase_one_req("rid-a")
    ordinary = _phase_one_req("rid-b")
    ordinary._omni_data.tail_pending = False

    assert QwenTalkerScheduler._is_phase_one_batch(_batch([pending]))
    assert not QwenTalkerScheduler._is_phase_one_batch(_batch([pending, ordinary]))
    assert not QwenTalkerScheduler._is_phase_one_batch(_batch([]))


def test_a_parked_batch_never_reaches_the_upstream_result_processor(
    monkeypatch,
) -> None:
    scheduler = _scheduler()
    seen: list = []
    monkeypatch.setattr(
        talker_scheduler_mod._Upstream,
        "process_batch_result",
        lambda self, batch, result: seen.append(batch),
    )

    scheduler.process_batch_result(_batch([_phase_one_req("rid-park")]), object())

    assert seen == []
    assert "rid-park" in scheduler._parked_reqs


def test_abort_between_the_phases_reclaims_the_parked_kv() -> None:
    scheduler = _scheduler()
    req = _phase_one_req("rid-abort")
    scheduler._park_phase_one_batch(_batch([req]))
    released: list = []
    scheduler._release_request_kv_cache = released.append

    scheduler._release_prebuilt_payload("rid-abort")

    assert released == [req]
    assert scheduler._parked_reqs == {}
    assert not has_parked_prefix_shadow(req)
    assert req._omni_data is None


def test_abort_before_the_prompt_extend_drops_the_queued_request() -> None:
    scheduler = _scheduler()
    req = _phase_one_req("rid-queued")
    scheduler._phase_one_queue.append(req)
    scheduler._phase_one_data["rid-queued"] = req._omni_data

    scheduler._release_prebuilt_payload("rid-queued")

    assert list(scheduler._phase_one_queue) == []
    assert scheduler._phase_one_data == {}


def test_adopting_hands_the_parked_kv_to_the_whole_build() -> None:
    scheduler = _scheduler()
    parked = _phase_one_req("rid-adopt", pool_idx=3)
    scheduler._phase_one_data["rid-adopt"] = parked._omni_data
    scheduler._park_phase_one_batch(_batch([parked]))
    fresh = _real_req(list(range(PROMPT_ROWS + ASSISTANT_TAIL_ROWS)))
    req_data = SimpleNamespace(req=fresh, two_phase_composed=True)

    scheduler._adopt_built_request(SimpleNamespace(request_id="rid-adopt"), req_data)

    assert fresh.req_pool_idx == 3
    assert has_parked_prefix_shadow(fresh)
    assert scheduler._parked_reqs == {}
    assert parked.req_pool_idx is None


def test_a_whole_build_reclaims_the_parked_kv_instead_of_adopting_it() -> None:
    """Fallback rebuilt the prompt rows, so the parked KV is not its prefix."""
    scheduler = _scheduler()
    parked = _phase_one_req("rid-fallback")
    scheduler._park_phase_one_batch(_batch([parked]))
    released: list = []
    scheduler._release_request_kv_cache = released.append
    fresh = _real_req(list(range(PROMPT_ROWS + ASSISTANT_TAIL_ROWS)))

    scheduler._adopt_built_request(
        SimpleNamespace(request_id="rid-fallback"),
        SimpleNamespace(req=fresh, two_phase_composed=False),
    )

    assert released == [parked]
    assert fresh.req_pool_idx is None


def test_adoption_rejects_a_build_that_is_not_the_prompt_plus_the_tail() -> None:
    scheduler = _scheduler()
    parked = _phase_one_req("rid-short")
    scheduler._park_phase_one_batch(_batch([parked]))
    released: list = []
    scheduler._release_request_kv_cache = released.append
    fresh = _real_req(list(range(PROMPT_ROWS + ASSISTANT_TAIL_ROWS + 1)))

    scheduler._adopt_built_request(
        SimpleNamespace(request_id="rid-short"),
        SimpleNamespace(req=fresh, two_phase_composed=True),
    )

    assert released == [parked]


def test_adoption_replays_whatever_the_phase_one_request_buffered() -> None:
    scheduler = _scheduler()
    parked = _phase_one_req("rid-replay")
    parked._omni_data.tail_pending_chunks = ["chunk-a", "chunk-b"]
    parked._omni_data.tail_pending_stream_done = True
    scheduler._phase_one_data["rid-replay"] = parked._omni_data
    scheduler._park_phase_one_batch(_batch([parked]))
    appended: list = []
    done: list = []
    scheduler._append_stream_chunk = lambda data, chunk: appended.append(chunk)
    scheduler._mark_stream_done = done.append
    fresh = _real_req(list(range(PROMPT_ROWS + ASSISTANT_TAIL_ROWS)))
    req_data = SimpleNamespace(req=fresh, two_phase_composed=True)

    scheduler._adopt_built_request(SimpleNamespace(request_id="rid-replay"), req_data)

    assert appended == ["chunk-a", "chunk-b"]
    assert done == [req_data]


def test_tail_pending_rows_are_kept_out_of_the_stream_output() -> None:
    scheduler = _scheduler()
    sched_output = SimpleNamespace(
        requests=[
            SimpleNamespace(
                request_id="rid-a", data=SimpleNamespace(tail_pending=True)
            ),
            SimpleNamespace(
                request_id="rid-b", data=SimpleNamespace(tail_pending=False)
            ),
        ]
    )

    assert scheduler._stream_skip_rids(sched_output) == ("rid-a",)


def test_phase_one_admission_stops_at_the_park_cap() -> None:
    scheduler = _scheduler(max_parked=2)
    segments = {}
    scheduler._phase_one_builder = lambda payload, segment: SimpleNamespace(
        req=_phase_one_req(payload.request_id)
    )
    for index in range(4):
        request_id = f"rid-{index}"
        payload = SimpleNamespace(request_id=request_id)
        scheduler._deferred_request_payloads[request_id] = payload
        scheduler._prompt_segment_futures[request_id] = SimpleNamespace(
            done=lambda: True, result=lambda: segments
        )

    scheduler._admit_phase_one_requests()

    assert len(scheduler._phase_one_queue) == 2
    assert len(scheduler._phase_one_data) == 2


def test_phase_one_admission_skips_a_payload_whose_gate_already_opened() -> None:
    scheduler = _scheduler()
    scheduler._phase_one_builder = lambda payload, segment: pytest.fail(
        "an admitted payload must not be pre-admitted again"
    )
    scheduler._prompt_segment_futures["rid-gone"] = SimpleNamespace(
        done=lambda: True, result=lambda: None
    )

    scheduler._admit_phase_one_requests()

    assert list(scheduler._phase_one_queue) == []


def test_a_tail_pending_request_keeps_taking_chunks_through_ingress() -> None:
    """The readiness gate watches the ingress buffer, not the queued request."""
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler

    scheduler = object.__new__(OmniScheduler)
    scheduler.running_batch = None
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler._async_pending = None
    pending = _phase_one_req("rid-pending")
    ordinary = _phase_one_req("rid-ordinary")
    ordinary._omni_data.tail_pending = False
    scheduler.waiting_queue = [pending, ordinary]

    assert scheduler._find_request_data("rid-pending") is None
    assert scheduler._find_request_data("rid-ordinary") is ordinary._omni_data


# ---------------------------------------------------------------------------
# model runner + builders
# ---------------------------------------------------------------------------


def _runner():
    from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner

    runner = object.__new__(QwenTalkerModelRunner)
    runner._feedback_enabled = True
    return runner


def _runner_request(tail_pending: bool) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(tail_pending=tail_pending))


def test_post_prefill_ships_no_codec_frame_for_a_prompt_only_batch() -> None:
    result = SimpleNamespace(next_token_ids=torch.zeros(1, dtype=torch.long))

    _runner().post_prefill(result, None, None, [_runner_request(True)])


def test_post_prefill_refuses_a_batch_mixing_both_phases() -> None:
    with pytest.raises(RuntimeError, match="cannot be batched together"):
        _runner().post_prefill(
            SimpleNamespace(next_token_ids=None),
            None,
            None,
            [_runner_request(True), _runner_request(False)],
        )


def test_append_text_chunk_buffers_while_the_tail_is_pending() -> None:
    from sglang_omni.models.qwen3_omni.components.talker_prefill import (
        TalkerPrefillBuilder,
    )

    builder = object.__new__(TalkerPrefillBuilder)
    req_data = SimpleNamespace(
        tail_pending=True, thinker_chunks_done=False, pending_text_queue=None
    )

    builder.append_text_chunk(req_data, "chunk-a")
    builder.mark_thinker_done(req_data)

    assert req_data.tail_pending_chunks == ["chunk-a"]
    assert req_data.pending_text_queue is None
    assert req_data.thinker_chunks_done is False
    assert req_data.tail_pending_stream_done is True


def test_phase_one_request_carries_mrope_over_the_prompt_plus_the_tail(
    monkeypatch,
) -> None:
    """origin_input_ids stays at the prompt, but the M-RoPE delta is full length."""
    from sglang_omni.models.qwen3_omni import request_builders

    captured: dict = {}
    monkeypatch.setattr(
        request_builders,
        "build_sglang_talker_request",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )
    segment = SimpleNamespace(
        input_embeds=torch.zeros(PROMPT_ROWS, 4),
        input_ids=torch.arange(PROMPT_ROWS, dtype=torch.long),
        tail_input_ids=torch.full((ASSISTANT_TAIL_ROWS,), 7, dtype=torch.long),
        tts_pad_embed=torch.zeros(1, 4),
        tts_eos_embed=torch.zeros(1, 4),
        prompt_model_inputs={},
    )
    payload = SimpleNamespace(
        request_id="rid-mrope", request=SimpleNamespace(params={})
    )

    req_data = request_builders.build_talker_phase_one_request(
        payload,
        segment,
        tokenizer=None,
        codec_vocab_size=32,
        codec_bos_id=20,
        audio_token_id=None,
        image_token_id=None,
        video_token_id=None,
        thinker_config=object(),
        resolve_sampling_config=lambda params: {
            "max_new_tokens": 8,
            "temperature": 0.0,
            "top_k": -1,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "codec_eos_id": 2,
            "suppress_tokens": [],
        },
    )

    assert captured["talker_input_ids"].shape[0] == PROMPT_ROWS
    assert captured["mrope_input_ids"].shape[0] == PROMPT_ROWS + ASSISTANT_TAIL_ROWS
    assert captured["thinker_chunks_done"] is False
    assert req_data.tail_pending is True
