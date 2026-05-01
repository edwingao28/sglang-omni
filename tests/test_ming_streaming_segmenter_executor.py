# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import torch

from sglang_omni.models.ming_omni.components.streaming_text import (
    SegmenterConfig,
    text_to_uint8_tensor,
    uint8_tensor_to_text,
)
from sglang_omni.models.ming_omni.pipeline.next_stage import TALKER_STREAM_STAGE
from sglang_omni.pipeline.stage.stream_queue import StreamItem, StreamQueue
from sglang_omni.proto import OmniRequest, StagePayload


def _payload(request_id: str) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={}, params={}),
        data={},
    )


def _executor(**config_kwargs):
    from sglang_omni.models.ming_omni.components.streaming_segmenter_executor import (
        MingStreamingSegmenterExecutor,
    )

    config = SegmenterConfig(
        segment_min_tokens=config_kwargs.pop("segment_min_tokens", 2),
        segment_max_tokens=config_kwargs.pop("segment_max_tokens", 10),
        first_segment_min_tokens=config_kwargs.pop("first_segment_min_tokens", 4),
        first_segment_max_wait_ms=config_kwargs.pop(
            "first_segment_max_wait_ms", 9999
        ),
    )
    return MingStreamingSegmenterExecutor(
        config=config,
        token_count_fn=lambda text: len(text.split()),
        **config_kwargs,
    )


def _open_queue(executor, request_id: str) -> StreamQueue:
    queue = StreamQueue()
    queue.open(request_id)
    executor._stream_queue = queue
    return queue


def _stream_item(text: str, chunk_id: int = 0) -> StreamItem:
    return StreamItem(
        chunk_id=chunk_id,
        data=text_to_uint8_tensor(text),
        from_stage="thinker",
    )


def test_segmenter_emits_uint8_segment_to_talker_stream_with_metadata() -> None:
    async def run() -> None:
        executor = _executor(segment_min_tokens=10)
        queue = _open_queue(executor, "req-1")
        captured = []

        def capture(request_id, data, target_stage, metadata=None):
            captured.append((request_id, data, target_stage, metadata))

        executor.set_stream_fn(capture)
        await executor.add_request(_payload("req-1"))

        queue.put("req-1", _stream_item("partial text"))
        queue.put_done("req-1", from_stage="thinker")

        await asyncio.wait_for(executor.get_result(), timeout=0.5)

        assert len(captured) == 1
        request_id, data, target_stage, metadata = captured[0]
        assert request_id == "req-1"
        assert target_stage == TALKER_STREAM_STAGE
        assert isinstance(data, torch.Tensor)
        assert data.dtype == torch.uint8
        assert uint8_tensor_to_text(data) == "partial text"
        assert metadata == {
            "segment_id": 0,
            "is_final_segment": True,
            "text_len": len("partial text".encode("utf-8")),
        }

    asyncio.run(run())


def test_segmenter_sends_empty_final_marker_after_pre_eos_segment() -> None:
    async def run() -> None:
        executor = _executor()
        queue = _open_queue(executor, "req-1")
        captured = []
        executor.set_stream_fn(
            lambda request_id, data, target_stage, metadata=None: captured.append(
                (request_id, data, target_stage, metadata)
            )
        )

        await executor.add_request(_payload("req-1"))
        queue.put("req-1", _stream_item("hello world."))
        queue.put_done("req-1", from_stage="thinker")

        await asyncio.wait_for(executor.get_result(), timeout=0.5)

        assert len(captured) == 2
        assert uint8_tensor_to_text(captured[0][1]) == "hello world."
        assert captured[0][3]["is_final_segment"] is False
        assert captured[1][1].dtype == torch.uint8
        assert captured[1][1].numel() == 0
        assert captured[1][3] == {
            "segment_id": 1,
            "is_final_segment": True,
            "text_len": 0,
        }

    asyncio.run(run())


def test_segmenter_first_segment_timeout_emits_without_more_input() -> None:
    async def run() -> None:
        executor = _executor(
            segment_min_tokens=10,
            segment_max_tokens=40,
            first_segment_min_tokens=3,
            first_segment_max_wait_ms=20,
        )
        queue = _open_queue(executor, "req-1")
        captured = []
        executor.set_stream_fn(
            lambda request_id, data, target_stage, metadata=None: captured.append(
                (request_id, uint8_tensor_to_text(data), target_stage, metadata)
            )
        )

        await executor.add_request(_payload("req-1"))
        queue.put("req-1", _stream_item("passive first segment"))

        await asyncio.sleep(0.08)
        assert captured == [
            (
                "req-1",
                "passive first segment",
                TALKER_STREAM_STAGE,
                {
                    "segment_id": 0,
                    "is_final_segment": False,
                    "text_len": len("passive first segment".encode("utf-8")),
                },
            )
        ]

        queue.put_done("req-1", from_stage="thinker")
        result = await asyncio.wait_for(executor.get_result(), timeout=0.5)
        assert result.data["segment_count"] == 2

    asyncio.run(run())
