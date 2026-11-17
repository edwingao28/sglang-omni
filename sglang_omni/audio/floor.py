# SPDX-License-Identifier: Apache-2.0
"""Who holds a session's floor: the reply speaking, or a reply paused by speech over it until a verdict says whether that speech was addressed to the assistant."""

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

OverlapPolicy = Literal["interrupt", "pause"]


@dataclass(frozen=True)
class Pause:
    """Stop the reply of turn where the listener is; its audio waits."""

    turn: int


@dataclass(frozen=True)
class Resume:
    """Continue the paused reply of turn from where it stopped; the utterances spoken over it, not addressed to the assistant or without words, get no answer."""

    turn: int
    utterances: tuple[int, ...]


@dataclass(frozen=True)
class Cancel:
    """End the reply of turn as an interrupt does."""

    turn: int


@dataclass(frozen=True)
class Ask:
    """Request evidence on whether these utterances, all spoken over the paused reply of turn, were said to the assistant; answered with Floor.decided(id, margin)."""

    id: int
    turn: int
    utterances: tuple[int, ...]


@dataclass(frozen=True)
class Answer:
    """Queue one user turn made of these utterances, in order."""

    utterances: tuple[int, ...]


FloorAct = Pause | Resume | Cancel | Ask | Answer


@dataclass(kw_only=True)
class Episode:
    """One pause of one reply: from the first speech over it to its resume or cancel."""

    utterances: list[int] = field(default_factory=list)
    has_words: bool = False
    open_ask: int | None = None


class Floor:
    """Who holds a session's floor; the caller reports speech, each final transcript under a number of its own, replies, verdicts and the end of its input, and carries out the acts each report returns."""

    def __init__(self, policy: OverlapPolicy, respond_margin: float) -> None:
        self.policy: OverlapPolicy = policy
        self.respond_margin = respond_margin
        self.reply: int | None = None
        # note (luojiaxuan): From a speech.started to the speech.stopped or commit that ends the utterance; the node that reports speech also stops it at the end of the input.
        # note (wenyao): under the pause policy the final transcript of that utterance, or the end of the input, ends it too: a client that reports speech may never stop it.
        self.is_user_speaking = False
        self.episode: Episode | None = None
        # note (wenyao): speech and its transcript share no id, so they pair by order: one flag per utterance still to be transcribed, set when it was spoken over the paused reply. Only the pause policy sets a flag, so the interrupt policy keeps none.
        self.untranscribed: deque[bool] = deque()
        self.ask_count = 0

    @property
    def is_paused(self) -> bool:
        return self.episode is not None

    def reply_started(self, turn: int) -> list[FloorAct]:
        self.reply = turn
        if not self.is_user_speaking:
            return []
        elif self.policy == "interrupt":
            self.reply = None
            return [Cancel(turn)]
        else:
            return self.overlap(turn)

    def reply_ended(self, turn: int) -> list[FloorAct]:
        """What was spoken over a reply that ends while paused is answered unjudged."""
        if self.reply != turn:
            return []
        else:
            self.reply = None
            episode = self.end_episode()
        if episode is not None and episode.utterances:
            return [Answer(tuple(episode.utterances))]
        else:
            return []

    def speech_started(self) -> list[FloorAct]:
        if self.is_user_speaking:
            return []
        else:
            self.is_user_speaking = True
        if self.policy == "pause":
            self.untranscribed.append(False)
        else:
            pass
        if self.reply is None:
            return []
        elif self.policy == "interrupt":
            turn, self.reply = self.reply, None
            return [Cancel(turn)]
        else:
            return self.overlap(self.reply)

    def speech_stopped(self) -> list[FloorAct]:
        self.is_user_speaking = False
        return self.resume_if_quiet()

    def utterance_committed(self) -> None:
        """A commit outside detected speech also brings one transcript, of an utterance no reply paused for."""
        if self.policy == "pause":
            self.untranscribed.append(False)
        else:
            pass

    def transcribed(self, utterance: int, *, has_words: bool) -> list[FloorAct]:
        """The next final transcript, under the caller's number for it; transcripts arrive in the order their utterances began, and under the pause policy the transcript of the utterance still open ends it."""
        is_over_reply = self.untranscribed.popleft() if self.untranscribed else False
        if self.policy == "pause" and not self.untranscribed:
            # note (wenyao): a paused reply cannot wait for the stop of this utterance: the stop may be the client's EOS chunk, which the graph hands this node only once no turn is open, and the paused reply keeps its turn open.
            self.is_user_speaking = False
        else:
            pass
        if not is_over_reply:
            return [Answer((utterance,))]
        else:
            episode = self.episode
            assert episode is not None and self.reply is not None
            episode.utterances.append(utterance)
            episode.has_words = episode.has_words or has_words
        if episode.has_words:
            episode.open_ask = self.ask_count
            self.ask_count += 1
            return [Ask(episode.open_ask, self.reply, tuple(episode.utterances))]
        else:
            return self.resume_if_quiet()

    def input_ended(self) -> list[FloorAct]:
        """No speech boundary or transcript is reported after this, so under the pause policy every utterance still open or untranscribed ends without words."""
        if self.policy == "interrupt":
            return []
        else:
            self.is_user_speaking = False
            self.untranscribed.clear()
            return self.resume_if_quiet()

    def decided(self, ask: int, margin: float | None) -> list[FloorAct]:
        """Evidence on an Ask: log P(addressed) - log P(other), or None for no evidence, which cancels like an addressed verdict; a verdict on any Ask but the latest is ignored."""
        episode = self.episode
        if episode is None or ask != episode.open_ask:
            return []
        elif margin is not None and margin < self.respond_margin:
            episode.open_ask = None
            return self.resume_if_quiet()
        else:
            turn, self.reply = self.reply, None
            assert turn is not None
            self.end_episode()
            # note (wenyao): the answer is first: the caller queues this speech before it ends the reply (Dialog.interrupt_reply).
            return [Answer(tuple(episode.utterances)), Cancel(turn)]

    def resume_if_quiet(self) -> list[FloorAct]:
        episode = self.episode
        if (
            episode is None
            or episode.open_ask is not None
            or any(self.untranscribed)
            or self.is_user_speaking
        ):
            return []
        else:
            assert self.reply is not None
            self.episode = None
            return [Resume(self.reply, tuple(episode.utterances))]

    def overlap(self, turn: int) -> list[FloorAct]:
        self.untranscribed[-1] = True
        if self.episode is None:
            self.episode = Episode()
            return [Pause(turn)]
        else:
            return []

    def end_episode(self) -> Episode | None:
        episode, self.episode = self.episode, None
        # note (wenyao): an utterance spoken over a reply that has since ended is answered like any other.
        self.untranscribed = deque(False for _ in self.untranscribed)
        return episode
