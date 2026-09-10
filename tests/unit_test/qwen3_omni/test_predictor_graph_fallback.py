# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Qwen3-Omni predictor decode-graph fallback counters."""

from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

import sglang_omni.models.qwen3_omni.components.talker as talker_module
from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def _talker() -> Qwen3OmniTalker:
    talker = object.__new__(Qwen3OmniTalker)
    talker._predictor_decode_graph_batch_sizes = (1, 2)
    talker._predictor_decode_graphs = {}
    talker._predictor_decode_graph_disabled = set()
    talker._predictor_decode_graph_replay_count = 0
    talker._predictor_decode_graph_fallback_counts = Counter()
    return talker


def _graph_call(talker: Qwen3OmniTalker, batch_size: int):
    return talker._code_predictor_forward_single_token_graph(
        layer0_codes=torch.zeros(batch_size, 1, dtype=torch.long),
        talker_hidden=torch.zeros(batch_size, 1, 8),
        batch_size=batch_size,
        code_dtype=torch.long,
    )


def test_batch_above_the_largest_bucket_counts_as_no_bucket() -> None:
    talker = _talker()

    assert _graph_call(talker, 4) is None
    assert talker._predictor_decode_graph_fallback_counts == {"no_bucket": 1}


def test_disabled_bucket_counts_every_eager_step_separately() -> None:
    talker = _talker()
    talker._predictor_decode_graph_disabled.add((2, torch.long))

    assert _graph_call(talker, 2) is None
    assert _graph_call(talker, 2) is None
    assert talker._predictor_decode_graph_fallback_counts == {"disabled": 2}


def test_capture_failure_disables_the_bucket_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    talker = _talker()

    def _fail(*_args, **_kwargs):
        raise RuntimeError("capture exploded")

    monkeypatch.setattr(talker_module, "_PredictorDecodeGraph", _fail)

    with caplog.at_level("WARNING", logger=talker_module.__name__):
        assert _graph_call(talker, 2) is None
        assert _graph_call(talker, 2) is None

    assert (2, torch.long) in talker._predictor_decode_graph_disabled
    assert talker._predictor_decode_graph_fallback_counts == {
        "capture_failed": 1,
        "disabled": 1,
    }
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None


def test_replay_counts_and_info_are_json_serializable() -> None:
    talker = _talker()
    replays: list[int] = []
    talker._predictor_decode_graphs[(2, torch.long)] = SimpleNamespace(
        replay=lambda codes, hidden: replays.append(codes.shape[0]) or (codes, hidden)
    )
    talker._predictor_decode_graph_disabled.add((1, torch.long))
    talker._predictor_decode_graph_fallback_counts["no_bucket"] += 1

    assert _graph_call(talker, 2) is not None

    info = talker.predictor_decode_graph_info()
    assert info == {
        "batch_sizes": [1, 2],
        "captured": ["2:int64"],
        "disabled": ["1:int64"],
        "replay_count": 1,
        "fallback_counts": {"no_bucket": 1},
    }
    assert json.loads(json.dumps(info)) == info
    assert replays == [2]


def test_talker_admin_model_info_carries_predictor_graph_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    talker = _talker()
    talker._predictor_decode_graph_disabled.add((1, torch.long))
    talker._predictor_decode_graph_replay_count = 3
    talker._predictor_decode_graph_fallback_counts["no_bucket"] += 2
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler.model_worker = SimpleNamespace(model_runner=SimpleNamespace(model=talker))
    monkeypatch.setattr(
        OmniScheduler,
        "_admin_model_info",
        lambda self: {"success": True, "message": "ok", "data": {"tp_size": 1}},
    )

    response = scheduler._admin_model_info()

    assert response["data"]["tp_size"] == 1
    assert response["data"]["predictor_decode_graph"] == {
        "batch_sizes": [1, 2],
        "captured": [],
        "disabled": ["1:int64"],
        "replay_count": 3,
        "fallback_counts": {"no_bucket": 2},
    }
