# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import sglang.srt.runtime_context as runtime_context

from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.types import KV_CAPACITY_ERROR_PREFIX
from sglang_omni.serve.openai_errors import is_bad_request_error


class _StubScheduler:
    _request_kv_capacity_error = OmniScheduler._request_kv_capacity_error

    def __init__(self, max_req_len: int, kv_cache_bytes: int | None = None) -> None:
        self.max_req_len = max_req_len
        self.tp_worker = SimpleNamespace(kv_cache_bytes=kv_cache_bytes)


def _req(input_len: int, max_new_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        origin_input_ids=[0] * input_len,
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
    )


def test_kv_capacity_prefix_is_a_bad_request() -> None:
    error = ValueError(f"{KV_CAPACITY_ERROR_PREFIX} (kv_capacity=1600).")

    assert is_bad_request_error(error)


@pytest.mark.parametrize(
    ("kv_cache_bytes", "mem_fraction_static"),
    [(None, None), (None, 0.55), (1 << 30, None)],
)
def test_scheduler_rejection_is_a_bad_request(
    monkeypatch, kv_cache_bytes: int | None, mem_fraction_static: float | None
) -> None:
    monkeypatch.setattr(
        runtime_context,
        "get_schedule",
        lambda: SimpleNamespace(mem_fraction_static=mem_fraction_static),
    )
    scheduler = _StubScheduler(max_req_len=10, kv_cache_bytes=kv_cache_bytes)

    message = scheduler._request_kv_capacity_error(_req(8, 4))

    assert message is not None
    assert message.startswith(KV_CAPACITY_ERROR_PREFIX)
    assert "thinker" not in message
    assert is_bad_request_error(ValueError(message))


def test_request_within_capacity_is_not_rejected() -> None:
    assert _StubScheduler(max_req_len=12)._request_kv_capacity_error(_req(8, 4)) is None
