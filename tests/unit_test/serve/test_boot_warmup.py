# SPDX-License-Identifier: Apache-2.0
"""Boot warmup must release requests and temporary inputs on every exit path."""

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.serve import boot_warmup


@pytest.mark.asyncio
async def test_boot_warmup_disabled_does_not_build_inputs(monkeypatch):
    def unexpected(*args):
        pytest.fail("Disabled warmup must not build or submit requests")

    monkeypatch.setattr(boot_warmup, "_write_reference_wav", unexpected)
    await boot_warmup.run_boot_warmup(
        SimpleNamespace(generate=unexpected), model_name="test", num_requests=0
    )


@pytest.mark.asyncio
async def test_boot_warmup_input_failure_is_nonfatal(monkeypatch, caplog):
    paths = []

    def fail(path, tone_hz):
        paths.append(path)
        raise OSError("Cannot write warmup audio")

    monkeypatch.setattr(boot_warmup, "_write_reference_wav", fail)
    await boot_warmup.run_boot_warmup(None, model_name="test", num_requests=1)
    assert "could not build the synthetic inputs" in caplog.text
    assert not paths[0].parent.exists()


@pytest.mark.asyncio
async def test_boot_warmup_drains_other_requests_after_one_failure(caplog):
    paths, closed, requests = [], [], []
    all_started = asyncio.Event()

    async def generate(request, *, request_id):
        requests.append(request)
        path = Path(request.metadata["audios"][0])
        paths.append(path)
        assert path.is_file()
        if len(requests) == 2:
            all_started.set()
        try:
            await asyncio.wait_for(all_started.wait(), timeout=2)
            if request_id == "boot-warmup-0":
                raise RuntimeError("Request failed")
            yield object()
        finally:
            closed.append(request_id)

    with caplog.at_level(logging.INFO, logger=boot_warmup.__name__):
        await boot_warmup.run_boot_warmup(
            SimpleNamespace(generate=generate), model_name="test", num_requests=2
        )
    assert sorted(closed) == ["boot-warmup-0", "boot-warmup-1"]
    assert "1/2 request(s)" in caplog.text
    assert all(not path.exists() for path in paths)
    assert requests[0].messages != requests[1].messages
    assert all(
        request.extra_params["talker_min_new_tokens"]
        == request.extra_params["talker_max_new_tokens"]
        == 32
        for request in requests
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancellation"])
async def test_boot_warmup_closes_streams_before_removing_inputs(
    monkeypatch, caplog, cancel
):
    paths, closed = [], []
    all_started = asyncio.Event()
    never_finishes = asyncio.Event()

    async def generate(request, *, request_id):
        path = Path(request.metadata["audios"][0])
        paths.append(path)
        if len(paths) == 2:
            all_started.set()
        try:
            yield object()
            await never_finishes.wait()
        finally:
            assert path.is_file()
            closed.append(request_id)

    monkeypatch.setattr(boot_warmup, "_TIMEOUT_S", 2 if cancel else 0.05)
    task = asyncio.create_task(
        boot_warmup.run_boot_warmup(
            SimpleNamespace(generate=generate), model_name="test", num_requests=2
        )
    )
    await asyncio.wait_for(all_started.wait(), timeout=2)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
        assert "serving anyway" in caplog.text
    assert sorted(closed) == ["boot-warmup-0", "boot-warmup-1"]
    assert all(not path.exists() for path in paths)
