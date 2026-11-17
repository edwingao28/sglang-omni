# SPDX-License-Identifier: Apache-2.0
"""The floor pauses a reply under speech, resumes it when that speech was not addressed to the assistant, and cancels it otherwise."""

import math

import pytest

from sglang_omni.audio.floor import Answer, Ask, Cancel, Floor, Pause, Resume


def paused_floor() -> Floor:
    floor = Floor("pause", 0.0)
    assert floor.reply_started(1) == []
    assert floor.speech_started() == [Pause(1)]
    return floor


def test_a_second_overlap_while_deciding_waits_for_a_verdict_on_both():
    floor = paused_floor()
    floor.speech_stopped()
    assert floor.transcribed(1, has_words=True) == [Ask(0, 1, (1,))]
    assert floor.speech_started() == []
    assert floor.decided(0, -2.0) == []
    assert floor.speech_stopped() == []
    assert floor.transcribed(2, has_words=True) == [Ask(1, 1, (1, 2))]
    assert floor.decided(0, 5.0) == []
    assert floor.decided(1, 5.0) == [Answer((1, 2)), Cancel(1)]


@pytest.mark.parametrize(
    "margin, acts",
    [
        (None, [Answer((1,)), Cancel(1)]),
        (math.nan, [Answer((1,)), Cancel(1)]),
        (1.5, [Answer((1,)), Cancel(1)]),
        (1.49, [Resume(1, (1,))]),
    ],
)
def test_no_evidence_or_a_margin_at_the_threshold_cancels_and_answers_what_was_heard(
    margin, acts
):
    floor = Floor("pause", 1.5)
    floor.reply_started(1)
    floor.speech_started()
    floor.speech_stopped()
    floor.transcribed(1, has_words=True)
    assert floor.decided(0, margin) == acts
    assert floor.reply_ended(1) == []


def test_the_interrupt_policy_cancels_at_once_and_never_pauses():
    floor = Floor("interrupt", 0.0)
    assert floor.reply_started(1) == []
    assert floor.speech_started() == [Cancel(1)]
    assert floor.transcribed(1, has_words=True) == [Answer((1,))]
    assert floor.reply_started(2) == [Cancel(2)]
    floor.speech_stopped()
    assert floor.reply_started(3) == []
    floor.speech_started()
    assert floor.input_ended() == []
    assert floor.reply_started(4) == [Cancel(4)]


def test_a_client_cancel_while_paused_answers_what_was_heard_and_later_content_is_plain():
    floor = paused_floor()
    floor.speech_stopped()
    floor.transcribed(1, has_words=True)
    floor.speech_started()
    assert floor.reply_ended(1) == [Answer((1,))]
    assert floor.reply_ended(1) == []
    assert floor.decided(0, 5.0) == []
    floor.speech_stopped()
    assert floor.transcribed(2, has_words=True) == [Answer((2,))]


def test_a_reply_ending_while_paused_with_nothing_heard_answers_nothing():
    floor = paused_floor()
    assert floor.reply_ended(1) == []
    floor.speech_stopped()
    assert floor.transcribed(1, has_words=True) == [Answer((1,))]


def test_content_without_words_resumes_without_asking():
    floor = paused_floor()
    floor.speech_stopped()
    assert floor.transcribed(1, has_words=False) == [Resume(1, (1,))]
    assert floor.speech_started() == [Pause(1)]
    floor.speech_stopped()
    floor.speech_started()
    assert floor.transcribed(2, has_words=False) == []
    floor.speech_stopped()
    assert floor.transcribed(3, has_words=True) == [Ask(0, 1, (2, 3))]


def test_a_reply_opening_over_speech_pauses_with_that_speech_over_it():
    floor = Floor("pause", 0.0)
    assert floor.speech_started() == []
    floor.speech_stopped()
    floor.speech_started()
    assert floor.transcribed(1, has_words=True) == [Answer((1,))]
    assert floor.reply_started(1) == [Pause(1)]
    floor.speech_stopped()
    assert floor.transcribed(2, has_words=True) == [Ask(0, 1, (2,))]
    assert floor.decided(0, -1.0) == [Resume(1, (2,))]


def test_the_final_transcript_of_the_utterance_still_open_ends_it():
    """With no stop after the final transcript of the open utterance, a not-addressed verdict left the reply paused, and the reply answering addressed speech paused as it started."""
    floor = paused_floor()
    assert floor.transcribed(1, has_words=True) == [Ask(0, 1, (1,))]
    assert floor.decided(0, -1.0) == [Resume(1, (1,))]
    assert floor.speech_started() == [Pause(1)]
    floor.transcribed(2, has_words=True)
    assert floor.decided(1, 1.0) == [Answer((2,)), Cancel(1)]
    assert floor.reply_started(2) == []


def test_an_utterance_committed_outside_speech_is_answered_and_keeps_the_order():
    floor = paused_floor()
    assert floor.speech_started() == []
    floor.speech_stopped()
    floor.utterance_committed()
    assert floor.transcribed(1, has_words=True) == [Ask(0, 1, (1,))]
    assert floor.decided(0, -1.0) == [Resume(1, (1,))]
    assert floor.transcribed(2, has_words=False) == [Answer((2,))]
