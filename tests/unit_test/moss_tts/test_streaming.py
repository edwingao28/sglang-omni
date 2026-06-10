# SPDX-License-Identifier: Apache-2.0
"""Engine-side streaming tests for MOSS-TTS: config wiring, stream metadata,
and per-step row emission from the model runner."""

from __future__ import annotations

import queue
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_tts.config import MossTTSPipelineConfig
from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.models.moss_tts.request_builders import build_moss_stream_metadata
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.types import RequestOutput


def _make_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        audio_pad_code=1024,
        audio_start_token_id=151652,
        audio_end_token_id=151653,
        audio_assistant_gen_slot_token_id=151656,
        audio_assistant_delay_slot_token_id=151662,
    )


def _make_payload(params) -> StagePayload:
    return StagePayload(
        request_id="req-stream",
        request=OmniRequest(inputs="hello", params=params),
        data={},
    )


def _make_data(*, prefix_rows: torch.Tensor | None = None) -> SimpleNamespace:
    state = MossTTSState()
    state.assistant_start_length = (
        int(prefix_rows.shape[0]) if prefix_rows is not None else 0
    )
    return SimpleNamespace(
        prompt_rows=torch.zeros((4, 33), dtype=torch.long),
        state=state,
        assistant_prefix_rows=prefix_rows,
    )


def test_moss_streaming_pipeline_routes_chunks_to_vocoder() -> None:
    config = MossTTSPipelineConfig(model_path="model")
    stages_by_name = {stage.name: stage for stage in config.stages}

    assert stages_by_name["tts_engine"].stream_to == ["vocoder"]
    assert stages_by_name["vocoder"].can_accept_stream_before_payload is True


def test_moss_stream_metadata_off_when_stream_falsy() -> None:
    cfg = _make_cfg()
    for params in ({}, {"stream": False}, {"stream": 0}):
        assert (
            build_moss_stream_metadata(_make_payload(params), _make_data(), cfg) is None
        )


def test_moss_stream_metadata_rejects_non_dict_params() -> None:
    with pytest.raises(TypeError):
        build_moss_stream_metadata(_make_payload(None), _make_data(), _make_cfg())


def test_moss_stream_metadata_off_when_prompt_rows_missing() -> None:
    cfg = _make_cfg()
    payload = _make_payload({"stream": True})
    for prompt_rows in (None, torch.zeros((0, 33), dtype=torch.long)):
        data = _make_data()
        data.prompt_rows = prompt_rows
        assert build_moss_stream_metadata(payload, data, cfg) is None


def test_moss_stream_metadata_carries_decode_contract() -> None:
    metadata = build_moss_stream_metadata(
        _make_payload({"stream": True}), _make_data(), _make_cfg()
    )

    assert metadata == {
        "modality": "moss_delayed_audio_row",
        "stream": True,
        "n_vq": 32,
        "audio_pad_code": 1024,
        "audio_start_token_id": 151652,
        "audio_end_token_id": 151653,
        "audio_assistant_gen_slot_token_id": 151656,
        "audio_assistant_delay_slot_token_id": 151662,
        "assistant_start_length": 0,
    }


def test_moss_stream_metadata_includes_prefix_and_initial_chunk_frames() -> None:
    prefix = torch.ones((2, 33), dtype=torch.long)
    metadata = build_moss_stream_metadata(
        _make_payload({"stream": True, "initial_codec_chunk_frames": 1}),
        _make_data(prefix_rows=prefix),
        _make_cfg(),
    )

    assert metadata["assistant_prefix_rows"] == prefix.tolist()
    assert metadata["assistant_start_length"] == 2
    assert metadata["initial_codec_chunk_frames"] == 1

    empty = build_moss_stream_metadata(
        _make_payload({"stream": True}),
        _make_data(prefix_rows=torch.empty((0, 33), dtype=torch.long)),
        _make_cfg(),
    )
    assert "assistant_prefix_rows" not in empty
    assert "initial_codec_chunk_frames" not in empty


def _ensure_sampler_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a stub sglang sampler module only when sglang is not installed,
    so the model_runner import works in this sglang-free unit environment."""
    try:
        import sglang.srt.layers.sampler  # noqa: F401

        return
    except ImportError:
        pass

    names = ("sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.sampler")
    modules = {name: types.ModuleType(name) for name in names}
    for name in names[:-1]:
        modules[name].__path__ = []
    modules["sglang"].srt = modules["sglang.srt"]
    modules["sglang.srt"].layers = modules["sglang.srt.layers"]
    modules["sglang.srt.layers"].sampler = modules["sglang.srt.layers.sampler"]
    modules["sglang.srt.layers.sampler"].multinomial_with_seed = (
        lambda probs, seeds, positions: torch.argmax(probs, dim=-1, keepdim=True)
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _make_runner(outbox):
    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner

    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(config=SimpleNamespace(im_end_token_id=14))
    runner._outbox = outbox
    runner._vocoder_target = "vocoder"
    runner._pending_rows = torch.tensor(
        [[12, 2, 4], [14, 4, 4], [13, 1, 2]], dtype=torch.long
    )
    runner._pending_embeds = torch.ones((3, 3))
    return runner


def _make_requests(metadata_by_rid):
    return [
        SimpleNamespace(
            request_id=rid,
            data=SimpleNamespace(
                output_rows=[],
                pending_feedback_queue=[],
                stream_metadata=metadata,
            ),
        )
        for rid, metadata in metadata_by_rid
    ]


_OUTPUTS = {
    "active": RequestOutput("active", data=12),
    "eos": RequestOutput("eos", data=14),
    "nostream": RequestOutput("nostream", data=13),
}


def test_moss_runner_streams_exactly_the_appended_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_sampler_importable(monkeypatch)
    outbox: queue.Queue = queue.Queue()
    runner = _make_runner(outbox)
    metadata = {"modality": "moss_delayed_audio_row", "stream": True}
    requests = _make_requests(
        [("active", metadata), ("eos", metadata), ("nostream", None)]
    )

    runner.post_process_outputs(object(), SimpleNamespace(requests=requests), _OUTPUTS)

    # (wenyao) Rows appended: active + nostream; eos skipped everywhere.
    assert [row.tolist() for row in requests[0].data.output_rows] == [[12, 2, 4]]
    assert [row.tolist() for row in requests[2].data.output_rows] == [[13, 1, 2]]
    assert requests[1].data.output_rows == []

    message = outbox.get_nowait()
    assert outbox.empty()
    assert message.request_id == "active"
    assert message.type == "stream"
    assert message.target == "vocoder"
    assert message.metadata is metadata
    assert isinstance(message.data, torch.Tensor)
    assert message.data.dtype == torch.long
    assert message.data.device.type == "cpu"
    assert message.data.tolist() == [12, 2, 4]
    assert torch.equal(message.data, requests[0].data.output_rows[0].cpu())


def test_moss_runner_emits_nothing_when_streaming_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_sampler_importable(monkeypatch)
    # (wenyao) No outbox wired: rows still append, nothing else happens.
    runner = _make_runner(None)
    metadata = {"modality": "moss_delayed_audio_row", "stream": True}
    requests = _make_requests([("active", metadata)])
    runner.post_process_outputs(
        object(),
        SimpleNamespace(requests=requests),
        {"active": RequestOutput("active", data=12)},
    )
    assert len(requests[0].data.output_rows) == 1

    # (wenyao) Outbox wired but no request opted in: zero messages.
    outbox: queue.Queue = queue.Queue()
    runner = _make_runner(outbox)
    requests = _make_requests([("active", None), ("nostream", None)])
    runner.post_process_outputs(
        object(),
        SimpleNamespace(requests=requests),
        {
            "active": RequestOutput("active", data=12),
            "nostream": RequestOutput("nostream", data=13),
        },
    )
    assert outbox.empty()
    assert len(requests[0].data.output_rows) == 1
    assert len(requests[1].data.output_rows) == 1
