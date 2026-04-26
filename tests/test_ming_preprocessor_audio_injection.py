# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang_omni.models.ming_omni.components.preprocessor import (
    _inject_top_level_audios,
    _inject_top_level_images,
)


def test_inject_top_level_audios_into_string_content():
    messages = [{"role": "user", "content": "describe this"}]
    out = _inject_top_level_audios(messages, ["a.wav"])
    assert out[0]["content"] == [
        {"type": "audio_url", "audio_url": {"url": "a.wav"}},
        {"type": "text", "text": "describe this"},
    ]


def test_inject_top_level_audios_into_list_content():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "q"}]},
    ]
    out = _inject_top_level_audios(messages, ["a.wav", "b.wav"])
    assert out[0]["content"] == [
        {"type": "audio_url", "audio_url": {"url": "a.wav"}},
        {"type": "audio_url", "audio_url": {"url": "b.wav"}},
        {"type": "text", "text": "q"},
    ]


def test_inject_top_level_audios_skips_system_targets_first_user():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "second"},
    ]
    out = _inject_top_level_audios(messages, ["a.wav"])
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1]["content"][0] == {"type": "audio_url", "audio_url": {"url": "a.wav"}}
    assert out[3] == {"role": "user", "content": "second"}


def test_inject_top_level_audios_does_not_mutate_input():
    messages = [{"role": "user", "content": "hi"}]
    _inject_top_level_audios(messages, ["a.wav"])
    assert messages == [{"role": "user", "content": "hi"}]


def test_inject_top_level_audios_empty_list_is_noop():
    messages = [{"role": "user", "content": "hi"}]
    out = _inject_top_level_audios(messages, [])
    assert out == messages


def test_inject_helpers_have_symmetric_shape():
    img_out = _inject_top_level_images([{"role": "user", "content": "x"}], ["i.jpg"])
    aud_out = _inject_top_level_audios([{"role": "user", "content": "x"}], ["a.wav"])
    assert img_out[0]["content"][0]["type"] == "image_url"
    assert aud_out[0]["content"][0]["type"] == "audio_url"
    assert len(img_out[0]["content"]) == len(aud_out[0]["content"])
