# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from sglang_omni.models.qwen3_omni.components.talker_prefill import resolve_speaker_id
from sglang_omni.serve.openai_errors import is_bad_request_error

SPEAKERS = {"chelsie": 2301, "ethan": 2302, "aiden": 2303}


def test_named_speaker_resolves_case_insensitively() -> None:
    assert resolve_speaker_id({"speaker": "Chelsie"}, SPEAKERS) == 2301
    assert resolve_speaker_id({"speaker": "AIDEN"}, SPEAKERS) == 2303
    assert resolve_speaker_id({"speaker": " ethan "}, SPEAKERS) == 2302


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"speaker": None},
        {"speaker": ""},
        {"speaker": "   "},
        {"speaker": "default"},
        {"speaker": " Default "},
    ],
)
def test_absent_or_default_speaker_prefers_ethan_then_the_first_voice(params) -> None:
    assert resolve_speaker_id(params, SPEAKERS) == 2302
    assert resolve_speaker_id(params, {"vivian": 7, "ryan": 8}) == 7


def test_unknown_speaker_is_a_bad_request_listing_the_checkpoint_voices() -> None:
    with pytest.raises(ValueError) as raised:
        resolve_speaker_id({"speaker": "Nobody"}, SPEAKERS)
    message = str(raised.value)
    assert message == "Unknown voice 'Nobody'. Supported voices: chelsie, ethan, aiden"
    assert is_bad_request_error(raised.value)


def test_speaker_id_param_is_the_only_path_without_a_speaker_map() -> None:
    assert resolve_speaker_id({"speaker_id": 5}, {}) == 5
    assert resolve_speaker_id({"speaker": "anything", "speaker_id": 5}, {}) == 5
    assert resolve_speaker_id({}, {}) == 0
