# SPDX-License-Identifier: Apache-2.0
"""B5 regressions: exercise OmniScheduler.__init__, not __new__."""
from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang_omni_v1.scheduling.messages import IncomingMessage
from sglang_omni_v1.scheduling.omni_scheduler import OmniScheduler


def _make_scheduler(tp_rank: int, tp_size: int) -> OmniScheduler:
    fake_tp_group = SimpleNamespace(cpu_group=MagicMock(name="tp_cpu_group"))
    fake_model_runner = SimpleNamespace(
        max_total_num_tokens=1024,
        tp_group=fake_tp_group,
        device=f"cuda:{tp_rank}",
        model=MagicMock(),
    )
    fake_worker = SimpleNamespace(
        tp_rank=tp_rank,
        gpu_id=tp_rank,
        device=f"cuda:{tp_rank}",
        random_seed=0,
        model_runner=fake_model_runner,
        tp_cpu_group=fake_tp_group.cpu_group,
        tp_size=tp_size,
        get_worker_info=lambda: SimpleNamespace(),
    )
    server_args = SimpleNamespace(
        tp_size=tp_size,
        pp_size=1,
        page_size=1,
        max_prefill_tokens=512,
        max_running_requests=4,
        context_length=1024,
        random_seed=0,
        disable_overlap_schedule=True,
        chunked_prefill_size=128,
        enable_mixed_chunk=False,
        schedule_policy="fcfs",
        enable_hierarchical_cache=False,
        enable_priority_scheduling=False,
        schedule_low_priority_values_first=False,
        priority_scheduling_preemption_threshold=0,
        schedule_conservativeness=1.0,
    )
    model_config = SimpleNamespace(model_path="dummy", hf_config=SimpleNamespace())

    return OmniScheduler(
        tp_worker=fake_worker,
        tree_cache=MagicMock(),
        req_to_token_pool=MagicMock(),
        token_to_kv_pool_allocator=MagicMock(),
        server_args=server_args,
        model_config=model_config,
        prefill_manager=MagicMock(),
        decode_manager=MagicMock(),
        request_builder=MagicMock(),
        result_adapter=MagicMock(),
        model_runner=MagicMock(),
        stream_output_builder=MagicMock(),
        enable_overlap=False,
    )


def test_self_tp_rank_threads_from_worker():
    sched = _make_scheduler(tp_rank=1, tp_size=2)
    assert sched.tp_rank == 1


def test_attn_tp_mirrors_self_tp():
    sched = _make_scheduler(tp_rank=1, tp_size=2)
    assert sched.attn_tp_rank == 1
    assert sched.attn_tp_size == 2


def test_recv_requests_tp1_drains_local_inbox():
    sched = _make_scheduler(tp_rank=0, tp_size=1)
    sched.inbox.put(
        IncomingMessage(request_id="r0", type="new_request", data="payload")
    )
    assert sched.recv_requests() == ["payload"]


def test_recv_requests_rank0_broadcasts_full_envelope(monkeypatch):
    sched = _make_scheduler(tp_rank=0, tp_size=2)
    sched.inbox.put(
        IncomingMessage(request_id="r0", type="new_request", data="payload-0")
    )
    sched.inbox.put(
        IncomingMessage(request_id="r0", type="stream_chunk", data="chunk-0")
    )
    sched.inbox.put(IncomingMessage(request_id="r0", type="stream_done"))
    sched.inbox.put(IncomingMessage(request_id="r1", type="abort"))
    sched.inbox.put(IncomingMessage(request_id="__tp__", type="shutdown"))

    sent = {}

    def fake_broadcast(data, rank, dist_group=None, src=0, **kwargs):
        sent["data"] = data
        sent["rank"] = rank
        sent["dist_group"] = dist_group
        sent["src"] = src
        return data

    monkeypatch.setattr(
        "sglang_omni_v1.scheduling.omni_scheduler.broadcast_pyobj",
        fake_broadcast,
    )
    sched.recv_requests()

    assert [m.type for m in sent["data"]] == [
        "new_request",
        "stream_chunk",
        "stream_done",
        "abort",
        "shutdown",
    ]
    assert sent["rank"] == 0
    assert sent["src"] == 0


def test_recv_requests_rank0_sanitizes_new_request_before_broadcast(monkeypatch):
    sched = _make_scheduler(tp_rank=0, tp_size=2)
    sched.inbox.put(
        IncomingMessage(request_id="r0", type="new_request", data="payload-0")
    )

    sent = {}
    monkeypatch.setattr(
        "sglang_omni_v1.scheduling.omni_scheduler.sanitize_request_payload",
        lambda payload: f"sanitized:{payload}",
    )

    def fake_broadcast(data, *args, **kwargs):
        sent["data"] = data
        return data

    monkeypatch.setattr(
        "sglang_omni_v1.scheduling.omni_scheduler.broadcast_pyobj",
        fake_broadcast,
    )

    sched.recv_requests()

    assert sent["data"][0].data == "sanitized:payload-0"


def test_recv_requests_rank0_sanitizes_stream_chunk_before_broadcast(monkeypatch):
    sched = _make_scheduler(tp_rank=0, tp_size=2)
    sched.inbox.put(
        IncomingMessage(request_id="r0", type="stream_chunk", data="chunk-0")
    )

    sent = {}
    monkeypatch.setattr(
        "sglang_omni_v1.scheduling.omni_scheduler.sanitize_request_payload",
        lambda payload: f"sanitized:{payload}",
    )

    def fake_broadcast(data, *args, **kwargs):
        sent["data"] = data
        return data

    monkeypatch.setattr(
        "sglang_omni_v1.scheduling.omni_scheduler.broadcast_pyobj",
        fake_broadcast,
    )

    sched.recv_requests()

    assert sent["data"][0].data == "sanitized:chunk-0"
    assert sched._pending_stream_chunks["r0"] == ["chunk-0"]


def test_recv_requests_rank0_applies_original_payload_after_broadcast(monkeypatch):
    sched = _make_scheduler(tp_rank=0, tp_size=2)
    original_payload = SimpleNamespace(request_id="r0")
    sanitized_payload = SimpleNamespace(request_id="r0", sanitized=True)
    sched.inbox.put(
        IncomingMessage(request_id="r0", type="new_request", data=original_payload)
    )

    sent = {}
    monkeypatch.setattr(
        "sglang_omni_v1.scheduling.omni_scheduler.sanitize_request_payload",
        lambda payload: sanitized_payload,
    )

    def fake_broadcast(data, *args, **kwargs):
        sent["data"] = data
        return data

    monkeypatch.setattr(
        "sglang_omni_v1.scheduling.omni_scheduler.broadcast_pyobj",
        fake_broadcast,
    )

    assert sched.recv_requests() == [original_payload]
    assert sent["data"][0].data is sanitized_payload


def test_recv_requests_rank1_replays_received_envelope(monkeypatch):
    sched = _make_scheduler(tp_rank=1, tp_size=2)
    incoming = [
        IncomingMessage(request_id="r0", type="new_request", data="payload-0"),
        IncomingMessage(request_id="r1", type="abort"),
    ]
    monkeypatch.setattr(
        "sglang_omni_v1.scheduling.omni_scheduler.broadcast_pyobj",
        lambda *args, **kwargs: incoming,
    )

    new_reqs = sched.recv_requests()
    assert new_reqs == ["payload-0"]
    assert "r1" in sched._aborted_request_ids


def test_recv_requests_rank1_relocates_new_request_before_replay(monkeypatch):
    sched = _make_scheduler(tp_rank=1, tp_size=2)
    payload = SimpleNamespace(request_id="r0")
    incoming = [IncomingMessage(request_id="r0", type="new_request", data=payload)]
    relocated = {}
    monkeypatch.setattr(
        "sglang_omni_v1.scheduling.omni_scheduler.broadcast_pyobj",
        lambda *args, **kwargs: incoming,
    )

    def fake_relocate(data, device):
        relocated["data"] = data
        relocated["device"] = device

    monkeypatch.setattr(
        "sglang_omni_v1.scheduling.omni_scheduler.relocate_request_tensors",
        fake_relocate,
    )

    assert sched.recv_requests() == [payload]
    assert relocated == {"data": payload, "device": "cuda:1"}


def test_recv_requests_rank1_relocates_stream_chunk_before_replay(monkeypatch):
    sched = _make_scheduler(tp_rank=1, tp_size=2)
    chunk = SimpleNamespace()
    incoming = [IncomingMessage(request_id="r0", type="stream_chunk", data=chunk)]
    relocated = {}
    monkeypatch.setattr(
        "sglang_omni_v1.scheduling.omni_scheduler.broadcast_pyobj",
        lambda *args, **kwargs: incoming,
    )

    def fake_relocate(data, device):
        relocated["data"] = data
        relocated["device"] = device

    monkeypatch.setattr(
        "sglang_omni_v1.scheduling.omni_scheduler.relocate_request_tensors",
        fake_relocate,
    )

    assert sched.recv_requests() == []
    assert relocated == {"data": chunk, "device": "cuda:1"}
    assert sched._pending_stream_chunks["r0"] == [chunk]


def test_apply_envelope_shutdown_sets_flag():
    sched = _make_scheduler(tp_rank=0, tp_size=1)
    sched._apply_envelope([IncomingMessage(request_id="__tp__", type="shutdown")])
    assert sched._tp_shutdown_requested is True


def test_init_does_not_hardcode_tp_rank_source():
    src = inspect.getsource(OmniScheduler.__init__)
    assert "self.tp_rank = 0" not in src
