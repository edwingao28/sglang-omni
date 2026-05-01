from __future__ import annotations

import asyncio
import threading
from typing import Any

import numpy as np
import torch

from sglang_omni.models.ming_omni.components.streaming_text import text_to_uint8_tensor
from sglang_omni.models.ming_omni.pipeline.next_stage import TALKER_STREAM_STAGE
from sglang_omni.pipeline.stage.stream_queue import StreamItem, StreamQueue
from sglang_omni.proto import OmniRequest, StagePayload


def _payload(request_id: str) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={}, params={}),
        data={},
    )


def _stream_item(
    text: str,
    *,
    segment_id: int = 0,
    is_final_segment: bool = False,
) -> StreamItem:
    return StreamItem(
        chunk_id=segment_id,
        data=text_to_uint8_tensor(text),
        from_stage="segmenter",
        metadata={
            "segment_id": segment_id,
            "is_final_segment": is_final_segment,
        },
    )


def _open_queue(executor: Any, request_id: str) -> StreamQueue:
    queue = StreamQueue()
    queue.open(request_id)
    executor._stream_queue = queue
    return queue


class FakeTalker:
    def __init__(
        self,
        *,
        waveforms: list[torch.Tensor] | None = None,
        started: threading.Event | None = None,
        release: threading.Event | None = None,
        sample_rate: int = 44_100,
    ) -> None:
        self.waveforms = waveforms or [torch.tensor([1.0, 2.0], dtype=torch.float32)]
        self.started = started
        self.release = release
        self.calls: list[dict[str, Any]] = []
        self.config = type("Config", (), {"sample_rate": sample_rate})()

    def omni_audio_generation(self, **kwargs):
        self.calls.append(kwargs)
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            self.release.wait()
        for waveform in self.waveforms:
            yield waveform, None, None, None


def _executor(talker: Any, *, sample_rate: int | None = None):
    from sglang_omni.models.ming_omni.components.streaming_talker_executor import (
        MingTalkerStreamExecutor,
    )

    return MingTalkerStreamExecutor(
        model_path="/fake/model",
        talker=talker,
        audio_detokenizer=object(),
        sample_rate=sample_rate,
    )


def test_talker_stream_yields_audio_chunk_before_completion() -> None:
    async def run() -> None:
        started = threading.Event()
        release = threading.Event()
        executor = _executor(FakeTalker(started=started, release=release))
        queue = _open_queue(executor, "req-1")

        await executor.add_request(_payload("req-1"))
        stream_iter = executor.stream("req-1")
        next_chunk = asyncio.create_task(stream_iter.__anext__())
        result_task = asyncio.create_task(executor.get_result())

        queue.put("req-1", _stream_item("hello", is_final_segment=True))
        await asyncio.to_thread(started.wait, 0.5)
        assert not result_task.done()

        release.set()
        chunk = await asyncio.wait_for(next_chunk, timeout=0.5)

        assert chunk["modality"] == "audio"
        assert chunk["stage_name"] == TALKER_STREAM_STAGE
        assert chunk["segment_id"] == 0
        assert chunk["sample_rate"] == 44_100
        result = await asyncio.wait_for(result_task, timeout=0.5)
        assert result.request_id == "req-1"

    asyncio.run(run())


def test_talker_stream_skips_empty_waveform_chunks() -> None:
    async def run() -> None:
        executor = _executor(
            FakeTalker(
                waveforms=[
                    torch.empty((0,), dtype=torch.float32),
                    torch.tensor([3.0, 4.0], dtype=torch.float32),
                ],
            )
        )
        queue = _open_queue(executor, "req-1")

        await executor.add_request(_payload("req-1"))
        stream_iter = executor.stream("req-1")
        queue.put("req-1", _stream_item("hello", is_final_segment=True))

        chunk = await asyncio.wait_for(stream_iter.__anext__(), timeout=0.5)
        audio = np.frombuffer(chunk["audio_waveform"], dtype=np.float32)

        assert audio.tolist() == [3.0, 4.0]
        assert chunk["audio_waveform_shape"] == [2]
        assert chunk["audio_waveform_dtype"] == "float32"

    asyncio.run(run())


def test_talker_stream_audio_chunk_carries_stage_timing_metadata() -> None:
    async def run() -> None:
        executor = _executor(FakeTalker())
        queue = _open_queue(executor, "req-emit")

        await executor.add_request(_payload("req-emit"))
        await asyncio.sleep(0.05)
        queue.put("req-emit", _stream_item("hi", is_final_segment=True))
        stream_iter = executor.stream("req-emit")

        chunk = await asyncio.wait_for(stream_iter.__anext__(), timeout=0.5)

        emit_ms = chunk.get("segmenter_first_emit_ms")
        assert isinstance(emit_ms, float)
        assert emit_ms >= 40.0
        stage_times = chunk["stage_times_ms"]
        assert stage_times["segmenter_first_emit"] == emit_ms
        assert isinstance(stage_times["talker_first_audio"], float)
        assert stage_times["talker_first_audio"] >= emit_ms

    asyncio.run(run())
