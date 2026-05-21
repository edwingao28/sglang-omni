# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Ming thinker targeted stream output chunks."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.ming_omni.components.streaming_text import uint8_tensor_to_text
from sglang_omni.models.ming_omni.pipeline.next_stage import (
    DECODE_STAGE,
    SEGMENTER_STAGE,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self._tokens = {
            1: "h",
            2: "i",
            3: "!",
            7: "\ufffd",
            8: "é",
        }

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = [
            int(token_id)
            for token_id in ids
            if not skip_special_tokens or int(token_id) != self.eos_token_id
        ]
        if tokens == [7, 8]:
            return "é"
        return "".join(
            self._tokens.get(token_id, f"<{token_id}>") for token_id in tokens
        )


def _payload(
    *,
    stream: bool = True,
    modalities: list[str] | None = None,
) -> StagePayload:
    metadata = {}
    if modalities is not None:
        metadata["output_modalities"] = modalities
    return StagePayload(
        request_id="req",
        request=OmniRequest(
            inputs={},
            params={"stream": stream},
            metadata=metadata,
        ),
        data={},
    )


def _req_data(
    *,
    payload: StagePayload | None = None,
    is_chunked: int = 0,
    generation_steps: int = 0,
    max_new_tokens: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        req=SimpleNamespace(is_chunked=is_chunked),
        stage_payload=payload if payload is not None else _payload(),
        generation_steps=generation_steps,
        max_new_tokens=max_new_tokens,
    )


def _output(token_id: int | None, *, finished: bool = False) -> SimpleNamespace:
    return SimpleNamespace(data=token_id, extra={}, finished=finished)


def _builder():
    from sglang_omni.models.ming_omni.bootstrap import (
        make_ming_thinker_stream_output_builder,
    )

    return make_ming_thinker_stream_output_builder(
        tokenizer=FakeTokenizer(),
        eos_token_id=FakeTokenizer.eos_token_id,
    )


def test_no_messages_for_chunked_prefill_outputs() -> None:
    build = _builder()

    assert build("req", _req_data(is_chunked=1), _output(1)) == []


def test_no_messages_when_req_output_data_is_none() -> None:
    build = _builder()

    assert build("req", _req_data(), _output(None)) == []


def test_streaming_text_only_request_emits_decode_token_chunk() -> None:
    build = _builder()

    messages = build(
        "req",
        _req_data(payload=_payload(stream=True, modalities=["text"])),
        _output(1),
    )

    assert messages == [
        OutgoingMessage(
            request_id="req",
            type="stream",
            target=DECODE_STAGE,
            data=torch.tensor([1], dtype=torch.long),
            metadata={
                "modality": "text",
                "stage_name": "thinker",
                "token_id": 1,
                "step": 1,
            },
        )
    ]


def test_non_streaming_audio_request_emits_segmenter_text_chunk_only() -> None:
    build = _builder()

    messages = build(
        "req",
        _req_data(payload=_payload(stream=False, modalities=["audio"])),
        _output(1),
    )

    assert len(messages) == 1
    msg = messages[0]
    assert msg.request_id == "req"
    assert msg.type == "stream"
    assert msg.target == SEGMENTER_STAGE
    assert uint8_tensor_to_text(msg.data) == "h"
    assert msg.metadata == {
        "modality": "text",
        "stage_name": "thinker",
        "token_id": 1,
        "step": 1,
        "text_len": int(msg.data.numel()),
    }


def test_streaming_text_and_audio_request_emits_decode_and_segmenter_chunks() -> None:
    build = _builder()

    messages = build(
        "req",
        _req_data(payload=_payload(stream=True, modalities=["text", "audio"])),
        _output(1),
    )

    assert [msg.target for msg in messages] == [DECODE_STAGE, SEGMENTER_STAGE]
    assert torch.equal(messages[0].data, torch.tensor([1], dtype=torch.long))
    assert uint8_tensor_to_text(messages[1].data) == "h"
    assert messages[0].metadata == {
        "modality": "text",
        "stage_name": "thinker",
        "token_id": 1,
        "step": 1,
    }
    assert messages[1].metadata == {
        "modality": "text",
        "stage_name": "thinker",
        "token_id": 1,
        "step": 1,
        "text_len": int(messages[1].data.numel()),
    }


def test_multi_token_text_does_not_duplicate_already_emitted_prefixes() -> None:
    build = _builder()
    req_data = _req_data(payload=_payload(stream=True, modalities=["audio"]))

    first = build("req", req_data, _output(1))
    second = build("req", req_data, _output(2))

    assert uint8_tensor_to_text(first[0].data) == "h"
    assert uint8_tensor_to_text(second[0].data) == "i"
    assert second[0].metadata["step"] == 2


def test_replacement_character_decode_waits_for_valid_delta() -> None:
    build = _builder()
    req_data = _req_data(payload=_payload(stream=False, modalities=["audio"]))

    assert build("req", req_data, _output(7)) == []

    messages = build("req", req_data, _output(8))

    assert len(messages) == 1
    assert messages[0].target == SEGMENTER_STAGE
    assert uint8_tensor_to_text(messages[0].data) == "é"
    assert messages[0].metadata["step"] == 2


def test_missing_modalities_are_permissive_for_text_and_audio_outputs() -> None:
    build = _builder()

    messages = build(
        "req",
        _req_data(payload=_payload(stream=True, modalities=None)),
        _output(1),
    )

    assert [msg.target for msg in messages] == [DECODE_STAGE, SEGMENTER_STAGE]


def test_unknown_modalities_are_permissive_for_text_and_audio_outputs() -> None:
    build = _builder()

    messages = build(
        "req",
        _req_data(payload=_payload(stream=True, modalities=["something-new"])),
        _output(1),
    )

    assert [msg.target for msg in messages] == [DECODE_STAGE, SEGMENTER_STAGE]


def test_finished_output_evicts_request_stream_state() -> None:
    build = _builder()
    req_data = _req_data(payload=_payload(stream=True, modalities=["audio"]))

    first = build("req", req_data, _output(1, finished=True))
    second = build("req", req_data, _output(2))

    assert uint8_tensor_to_text(first[0].data) == "h"
    assert uint8_tensor_to_text(second[0].data) == "i"
    assert second[0].metadata["step"] == 1


def test_max_token_completion_evicts_request_stream_state_when_finished_false() -> None:
    build = _builder()
    completing_req_data = _req_data(
        payload=_payload(stream=True, modalities=["audio"]),
        generation_steps=1,
        max_new_tokens=1,
    )

    first = build("req", completing_req_data, _output(1, finished=False))
    second = build(
        "req",
        _req_data(payload=_payload(stream=True, modalities=["audio"])),
        _output(2, finished=False),
    )

    assert uint8_tensor_to_text(first[0].data) == "h"
    assert uint8_tensor_to_text(second[0].data) == "i"
    assert second[0].metadata["step"] == 1
