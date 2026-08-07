# SPDX-License-Identifier: Apache-2.0
import logging
from types import SimpleNamespace

import torch
from sglang.srt.model_executor.forward_context import (
    get_forward_context,
    has_forward_context,
)

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner


def _probe_runner(monkeypatch, method: str):
    """Runner whose base-class execute path records live layers_to_capture."""
    runner = object.__new__(ThinkerModelRunner)
    runner._text_model = SimpleNamespace(layers_to_capture=[0, 24])
    seen: list[list[int]] = []
    monkeypatch.setattr(
        ModelRunner,
        method,
        lambda self, _sched: seen.append(list(self._text_model.layers_to_capture)),
    )
    return runner, seen


def test_execute_keeps_layers_to_capture_stable(monkeypatch) -> None:
    runner, seen = _probe_runner(monkeypatch, "execute")

    runner.execute(SimpleNamespace(requests=[object()]))

    assert seen == [[0, 24]]
    assert runner._text_model.layers_to_capture == [0, 24]


def test_execute_launch_keeps_layers_to_capture_stable(monkeypatch) -> None:
    runner, seen = _probe_runner(monkeypatch, "execute_launch")

    runner.execute_launch(SimpleNamespace(requests=[object()]))

    assert seen == [[0, 24]]
    assert runner._text_model.layers_to_capture == [0, 24]


def test_audio_prefill_publishes_embeds_to_sglang_runner() -> None:
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    input_embeds = torch.ones(3, 4)
    runner._inject_multimodal_embeds = lambda *_args: (input_embeds, None, None)
    forward_batch = SimpleNamespace(input_embeds=None)

    result = runner.custom_prefill_forward(
        forward_batch,
        SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True)),
        [],
    )

    assert result is None
    assert forward_batch.input_embeds is input_embeds


def test_visual_deepstack_prefill_keeps_model_specific_forward() -> None:
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    input_embeds = torch.ones(3, 4)
    deepstack_embeds = [torch.ones(1, 4)]
    visual_mask = torch.tensor([False, True, False])
    runner._inject_multimodal_embeds = lambda *_args: (
        input_embeds,
        deepstack_embeds,
        visual_mask,
    )
    seen = []
    runner._forward_with_omni_embeds = lambda *args: seen.append(args) or "result"
    forward_batch = SimpleNamespace(input_embeds=None)

    result = runner.custom_prefill_forward(
        forward_batch,
        SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True)),
        [],
    )

    assert result == "result"
    assert forward_batch.input_embeds is None
    assert seen == [
        (forward_batch, input_embeds, deepstack_embeds, visual_mask),
    ]


def _probe_runner_with_counters(tp_rank: int) -> ThinkerModelRunner:
    runner = object.__new__(ThinkerModelRunner)
    runner._prefill_graph_replay_count = 0
    runner._prefill_graph_fallback_count = 0
    runner.tp_worker = SimpleNamespace(tp_rank=tp_rank)
    return runner


def test_prefill_counters_increment_and_log(caplog):
    runner = _probe_runner_with_counters(tp_rank=0)
    out_replay = SimpleNamespace(can_run_cuda_graph=True)
    out_fallback = SimpleNamespace(can_run_cuda_graph=False)
    extend_fb = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True))
    decode_fb = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: False))

    with caplog.at_level(logging.INFO):
        runner._count_prefill_graph_outcome(out_replay, extend_fb)
        runner._count_prefill_graph_outcome(out_fallback, extend_fb)
        runner._count_prefill_graph_outcome(out_replay, decode_fb)  # ignored

    assert runner._prefill_graph_replay_count == 1
    assert runner._prefill_graph_fallback_count == 1
    assert any(
        "prefill_graph_counters tp_rank=0 replay=1 fallback=1" in r.message
        for r in caplog.records
    )  # first fallback logs immediately


def test_custom_omni_forward_publishes_sglang_forward_context():
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    attn_backend = SimpleNamespace(init_forward_metadata=lambda _batch: None)
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=attn_backend)
    )

    seen = []

    def model(**kwargs):
        seen.append(get_forward_context().attn_backend)
        assert kwargs["input_deepstack_embeds"] is None
        return torch.ones(1, 2)

    def logits_processor(input_ids, hidden_states, lm_head, forward_batch):
        seen.append(get_forward_context().attn_backend)
        assert input_ids is forward_batch.input_ids
        assert lm_head == "lm_head"
        return "logits"

    runner._outer_model = SimpleNamespace(
        model=model,
        logits_processor=logits_processor,
        lm_head="lm_head",
    )
    forward_batch = SimpleNamespace(
        positions=torch.tensor([0]),
        mrope_positions=None,
        input_ids=torch.tensor([1]),
    )

    assert not has_forward_context()
    result = runner._forward_with_omni_embeds(
        forward_batch,
        torch.ones(1, 2),
    )

    assert result.logits_output == "logits"
