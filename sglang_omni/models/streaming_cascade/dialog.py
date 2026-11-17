# SPDX-License-Identifier: Apache-2.0
"""The dialog kind: the conversation history and the turns."""

import asyncio
import logging
import math
import re
from collections import deque
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

import httpx
from pydantic import Field

from sglang_omni.audio.floor import (
    Ask,
    Cancel,
    Floor,
    FloorAct,
    OverlapPolicy,
    Pause,
    Resume,
)
from sglang_omni.models.streaming_cascade.addressee import (
    AddresseeJudge,
    addressee_messages,
)
from sglang_omni.models.streaming_cascade.base import (
    ChatMessage,
    ComponentConfig,
    StreamComponent,
    control_chunk,
    messages_chunk,
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    COMMIT,
    INTERRUPT,
    MAX_HISTORY_CHARS,
    REPLY_DELIVERED,
    REPLY_HEARD,
    REPLY_PAUSED,
    REPLY_PROGRESS,
    REPLY_RESUMED,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TEXT_DELTA,
    TRANSCRIPT_FORMAT,
    TRUNCATE,
    TURN_DONE,
    TURN_EMPTY,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    chunk_language,
    control_type,
)
from sglang_omni.scheduling.session import ChunkEmitter

if TYPE_CHECKING:
    from sglang_omni.models.streaming_cascade.config import ComponentGraphConfig
else:
    pass

logger = logging.getLogger(__name__)


def heard_prefix(
    progress: list[tuple[int, float]], heard_ms: float, text: str
) -> tuple[int, bool]:
    """The characters of `text` up to the last sentence boundary whose audio ends at or before heard_ms, and whether the listener heard past it into text the cut drops."""
    text_end, audio_ms = max(
        (
            (boundary, boundary_ms)
            for boundary, boundary_ms in progress
            if boundary_ms <= heard_ms
        ),
        default=(0, 0.0),
    )
    return text_end, heard_ms > audio_ms and bool(text[text_end:].strip())


# note (Haoling Pu): a short utterance or a noise ("嗯。", "Okay.") is often recognized in the wrong language, so it neither changes the session language nor leads the reply into its language.
MIN_LANGUAGE_WORDS = 3
MIN_LANGUAGE_CJK_CHARS = 3
CJK_CHAR = re.compile("[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]")
SHORT_UTTERANCE_NOTE = "Reply in {language}, the language of this conversation; the user's last message is too short to tell their language."
# note (Haoling Pu): a cut reply shown as is reads to the model as how it answers, and after a few interruptions it replies with fragments (#3063).
CUT_REPLY_NOTE = "A reply of yours that ends with … was cut off when the user interrupted it; answer the latest message in full."
# note (wenyao): the longest an answer waits for the text heard of the reply that a pause-policy dialog just ended. With one session at a time the report came 21 to 63 ms after the delivered audio in 139 cuts of two live runs, and the first transcription after a server start took up to 211 ms. No report comes when the transcript differs in the first words of the sentence at the cut, or under method forced when a known word end of that sentence lies at or after the cut, because the aligner then sends no transcription request. The answer then waits the full time and joins the two user messages.
HEARD_CUT_WAIT_S = 0.25
# note (wenyao): how long after the end of the speech over a paused reply its final transcript may take: 20 times the longest of a live run with Qwen3-ASR, where it came 40 to 49 ms after the speech stopped in 8 utterances. A reply still paused this long plus decision_timeout_s after that speech ended has lost a transcript or a verdict.
FINAL_TRANSCRIPT_WAIT_S = 1.0


def prompt_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """The history as the model reads it: a reply the listener cut ends with … under a system note, and only such a reply ends with …."""
    prompt: list[ChatMessage] = []
    has_cut = False
    for message in messages:
        if message.is_cut:
            content = message.content.removesuffix("…")
            prompt.append(ChatMessage(role="assistant", content=f"{content}…"))
            has_cut = True
        elif message.role == "assistant":
            # note (Haoling Pu): the model sometimes ends a finished reply with …, which would read as cut and was followed by an empty reply in a live run.
            content = message.content.rstrip().removesuffix("…")
            prompt.append(ChatMessage(role="assistant", content=content))
        else:
            prompt.append(message)
    if has_cut:
        prompt[0] = ChatMessage(
            role="system", content=f"{prompt[0].content} {CUT_REPLY_NOTE}"
        )
    else:
        pass
    return prompt


class DialogConfig(ComponentConfig):
    """Conversation history; one messages chunk per answered utterance."""

    kind: Literal["dialog"] = "dialog"
    system_prompt: str = "You are a helpful voice assistant. Answer concisely."
    max_history_chars: int = Field(default=MAX_HISTORY_CHARS, gt=0)
    max_text_chars: int = Field(default=16000, gt=0)
    # What speech over a reply does: end it at once, or pause it until the chat service says whether the speech was addressed to the assistant.
    on_overlap: OverlapPolicy = "interrupt"
    # The least log P(addressed) - log P(other) of the chat service's answer that ends a paused reply.
    # note (wenyao): 0.34 was chosen with Qwen3.8-27B as the chat service on Full-Duplex-Bench v1.0 and development speech, and the Full-Duplex-Bench v1.5 comparison of the two policies ran with it. Another chat model needs its own value.
    respond_margin: float = 0.34
    # The deadline of the whole request for that answer; a request that misses it ends the paused reply as if the speech was addressed.
    decision_timeout_s: float = Field(default=2.0, gt=0)


class Dialog(StreamComponent):
    """Turns final transcripts into prompts and records the replies.

    Each final transcript opens the next turn with turn.queued. An utterance
    that arrives while a reply is running waits for the turn.done of that
    reply's turn, so history always alternates user and assistant. Speech
    over a reply, or an `interrupt` control, ends the reply: the dialog
    emits `interrupt` for the reply's turn to the nodes that take it as a
    control and publishes `turn.interrupted`; `reply.progress` controls from
    the synthesizer tell it how much of the reply was spoken. The interrupted
    reply stays in the history only up to the last sentence whose audio the
    graph delivered (`reply.delivered`), and a client `truncate` of that
    reply's turn moves the cut back to what the listener heard; from then on
    each cut is published as reply.heard. In a graph that sends audio, the
    next reply after an interrupted one begins once the delivered audio or a
    truncate bounds the cut, so its prompt holds what could have been heard of
    it, and these reports keep cutting the ended reply until a later reply
    ends, other than one interrupted before its first text, so one that
    arrives after the next reply began still cuts the history later prompts
    read. A reply the history keeps nothing of leaves no entry once an
    utterance follows it, and that utterance joins the user message the reply
    answered. The language the recognizer heard in the last utterance of at
    least 3 words or 3 Chinese, Japanese or Korean characters is the
    Session's language, and every messages chunk carries it; a shorter
    utterance leaves it unchanged, and when the recognizer heard another
    language in it, that turn's latest user message asks for a reply in the
    Session's language. A replacement adopts the history, the open reply and
    the waiting utterances, so new settings apply at once and reach the next
    prompt.

    Speech that started after an utterance ended and has not stopped when
    the utterance's reply begins is speech over that reply, so the reply
    ends as soon as it begins.

    Under on_overlap pause, speech over a reply pauses it (reply.paused) and
    its final transcripts are put to the chat service as one question. Speech
    not addressed to the assistant is dropped and the reply resumes
    (reply.resumed) once the user is silent; otherwise, or without an answer,
    the speech is queued as the next turn and the reply then ends as above.
    A reply still paused FINAL_TRANSCRIPT_WAIT_S plus decision_timeout_s
    after the speech over it ended ends as under on_overlap interrupt, and
    that speech is answered when its transcript arrives.
    Speech over a paused reply takes a turn number only when it is queued.
    That speech is queued before the reply ends, so its answer could begin
    on the delivered audio alone, before a node that recognizes the audio at
    the cut reports the text heard. When the delivered audio keeps nothing
    of a reply the listener heard part of, the answer waits for that report,
    HEARD_CUT_WAIT_S at most, so its prompt shows the heard part of the
    reply instead of joining the two utterances.
    """

    def __init__(self, config: DialogConfig) -> None:
        self.config = config
        self.emit: ChunkEmitter | None = None
        self.messages = [ChatMessage(role="system", content=config.system_prompt)]
        # Waiting utterances as (turn, transcript, language the recognizer heard).
        self.waiting: deque[tuple[int, str, str | None]] = deque()
        self.reply: list[str] | None = None
        self.reply_turn = 0
        self.turns = 0
        # Reply audio produced so far as (characters of the reply text with audio, audio ms), one entry per spoken sentence, for truncating an interrupted reply to what was heard.
        self.progress: list[tuple[int, float]] = []
        # note (luojiaxuan): The ended reply a truncate, its delivered audio or its progress may still cut, as (turn, history index), until a later reply ends, other than one interrupted before its first text, or its entry leaves the history.
        self.truncatable: tuple[int, int] | None = None
        # Note (Junnan Li): The ended reply's full text, the reply audio the graph delivered (unknown, or all of it once the reply is done) and the last truncate's heard position, so progress that arrives later re-cuts it.
        self.truncatable_text = ""
        # note (Haoling Pu): an interrupted reply was cut even when the listener heard all the text it had produced.
        self.is_truncatable_interrupted = False
        # note (luojiaxuan): The ended reply's progress, which keeps arriving while the next reply runs.
        self.truncatable_progress: list[tuple[int, float]] = []
        self.delivered_ms: float | None = math.inf
        self.heard_ms: float | None = None
        self.language: str | None = None
        # note (luojiaxuan): Whether the graph reports the audio it delivered of a reply this dialog interrupts; a graph without audio does not.
        self.awaits_delivery = True
        self.floor = Floor(config.on_overlap, config.respond_margin)
        # Final transcripts the floor has not yet answered or dropped, by arrival number, as (transcript, language the recognizer heard).
        self.transcripts: dict[int, tuple[str, str | None]] = {}
        self.transcript_count = 0
        self.judge: AddresseeJudge | None = None
        self.judging: asyncio.Task[None] | None = None
        # Whether an alignment node of the graph recognizes the audio at a cut and reports the text heard. Only a pause-policy dialog sets it, because only its answers wait for that report.
        self.hears_cuts = False
        # The loop time until which an answer waits for that report.
        self.heard_cut_deadline: float | None = None
        # The loop time at which a paused reply ends without a verdict, set while the speech over it has ended.
        self.pause_deadline: float | None = None

    def place(self, settings: "ComponentGraphConfig", name: str) -> None:
        self.awaits_delivery = bool(settings.delivery_reporters(name))
        if self.config.on_overlap == "pause":
            self.judge = AddresseeJudge(
                settings.chat_reader(name), self.config.decision_timeout_s
            )
            self.hears_cuts = bool(settings.heard_cut_reporters(name))
        else:
            pass

    def adopt(self, previous: StreamComponent) -> None:
        if isinstance(previous, Dialog):
            self.emit = previous.emit
            self.messages = [self.messages[0], *previous.messages[1:]]
            self.waiting = previous.waiting
            self.reply = previous.reply
            self.reply_turn = previous.reply_turn
            self.turns = previous.turns
            self.progress = previous.progress
            self.truncatable = previous.truncatable
            self.truncatable_text = previous.truncatable_text
            self.is_truncatable_interrupted = previous.is_truncatable_interrupted
            self.truncatable_progress = previous.truncatable_progress
            self.delivered_ms = previous.delivered_ms
            self.heard_ms = previous.heard_ms
            self.language = previous.language
            self.floor = previous.floor
            self.floor.policy = self.config.on_overlap
            self.floor.respond_margin = self.config.respond_margin
            self.transcripts = previous.transcripts
            self.transcript_count = previous.transcript_count
        else:
            pass

    def can_replace(self) -> bool:
        return not self.floor.is_paused and not self.awaits_heard_cut()

    async def close(self) -> None:
        if self.judging is not None:
            self.judging.cancel()
            await asyncio.gather(self.judging, return_exceptions=True)
        else:
            pass
        if self.judge is not None:
            await self.judge.close()
        else:
            pass

    def state_bytes(self) -> int:
        chars = sum(len(message.content) for message in self.messages)
        chars += sum(len(text) for _, text, _ in self.waiting)
        chars += sum(len(text) for text, _ in self.transcripts.values())
        chars += sum(len(part) for part in self.reply or ())
        return chars * 4

    def is_reply(self, chunk: TimedChunk) -> bool:
        return self.reply is not None and chunk.turn == self.reply_turn

    async def process(
        self, inputs: AsyncIterator[TimedChunk], emit: ChunkEmitter
    ) -> None:
        self.emit = emit
        if self.config.on_overlap == "pause":
            chunks = self.with_deadlines(inputs)
        else:
            chunks = inputs
        async for chunk in chunks:
            self.handle(chunk)
            self.respond_to_waiting()
        # note (wenyao): only the true end of the input reaches here; a replaced dialog is cancelled after its floor passes to the new one.
        self.act(self.floor.input_ended())

    async def with_deadlines(
        self, inputs: AsyncIterator[TimedChunk]
    ) -> AsyncIterator[TimedChunk]:
        """The input chunks. When the wait for the text heard or the limit of a pause runs out between two chunks, this loop answers the waiting speech or ends the paused reply, so a prompt the dialog refuses fails the session."""
        # note (wenyao): the wait cannot hold the chunk that began it. A session takes one append or control at a time, so the report would queue behind that chunk.
        loop = asyncio.get_running_loop()
        upcoming = asyncio.ensure_future(anext(inputs))
        try:
            while True:
                deadlines = [
                    deadline
                    for deadline in (self.heard_cut_deadline, self.pause_deadline)
                    if deadline is not None
                ]
                if deadlines:
                    wait_s = min(deadlines) - loop.time()
                else:
                    wait_s = None
                await asyncio.wait({upcoming}, timeout=wait_s)
                if upcoming.done():
                    try:
                        chunk = upcoming.result()
                    except StopAsyncIteration:
                        return
                    yield chunk
                    upcoming = asyncio.ensure_future(anext(inputs))
                else:
                    self.end_due_waits(loop.time())
        finally:
            upcoming.cancel()

    def end_due_waits(self, now: float) -> None:
        """Stop waiting for the text heard once its deadline has passed, and end a reply whose pause has reached its limit as an interrupt does."""
        if self.heard_cut_deadline is not None and self.heard_cut_deadline <= now:
            self.heard_cut_deadline = None
        else:
            pass
        if self.pause_deadline is not None and self.pause_deadline <= now:
            logger.warning(
                f"paused turn {self.reply_turn} got no verdict in time after the speech over it ended; it ends as an interrupt"
            )
            self.interrupt_reply()
        else:
            pass
        self.respond_to_waiting()

    async def control(self, event: TimedChunk) -> None:
        if control_type(event) == INTERRUPT and self.reply is not None:
            self.interrupt_reply()
            self.respond_to_waiting()
        elif control_type(event) == REPLY_PROGRESS:
            boundary = (
                int(event.payload["text_end"]),
                float(event.payload["audio_ms"]),
            )
            if self.truncatable is not None and event.turn == self.truncatable[0]:
                self.truncatable_progress.append(boundary)
                self.truncate()
                if self.heard_cut_deadline is not None:
                    self.respond_to_waiting()
                else:
                    pass
            elif self.reply is not None and event.turn == self.reply_turn:
                self.progress.append(boundary)
            else:
                pass
        else:
            pass

    def handle(self, chunk: TimedChunk) -> None:
        assert self.emit is not None
        if chunk.format == TRANSCRIPT_FORMAT and chunk.payload["is_final"]:
            text = str(chunk.payload["text"]).strip()
            self.transcript_count += 1
            self.transcripts[self.transcript_count] = (text, chunk_language(chunk))
            self.act(
                self.floor.transcribed(self.transcript_count, has_words=bool(text))
            )
        elif chunk.format == TEXT_DELTA and self.is_reply(chunk):
            self.reply.append(str(chunk.payload["text"]))
            if sum(len(part) for part in self.reply) > self.config.max_text_chars:
                raise ValueError("reply exceeds max_text_chars")
            else:
                pass
        elif control_type(chunk) == TURN_DONE and self.is_reply(chunk):
            reply = "".join(self.reply)
            self.messages.append(ChatMessage(role="assistant", content=reply))
            self.truncatable = (self.reply_turn, len(self.messages) - 1)
            self.truncatable_text = reply
            self.is_truncatable_interrupted = False
            self.truncatable_progress, self.progress = self.progress, []
            self.delivered_ms = math.inf
            self.heard_ms = None
            self.reply = None
            self.act(self.floor.reply_ended(chunk.turn))
        elif control_type(chunk) == SPEECH_STARTED:
            self.act(self.floor.speech_started())
        elif control_type(chunk) == SPEECH_STOPPED:
            self.act(self.floor.speech_stopped())
        elif control_type(chunk) == COMMIT and self.floor.is_user_speaking:
            self.act(self.floor.speech_stopped())
        elif control_type(chunk) == COMMIT:
            self.floor.utterance_committed()
        elif (
            control_type(chunk) == REPLY_DELIVERED
            and self.truncatable is not None
            and chunk.turn == self.truncatable[0]
        ):
            self.delivered_ms = float(chunk.payload["audio_ms"])
            self.truncate()
            if self.hears_cuts and self.waiting:
                self.heard_cut_deadline = (
                    asyncio.get_running_loop().time() + HEARD_CUT_WAIT_S
                )
            else:
                pass
        elif control_type(chunk) == TRUNCATE:
            # note (luojiaxuan): A truncate without a turn names the latest reply, which may have begun after the one a truncate can still cut.
            turn = self.reply_turn if chunk.turn is None else chunk.turn
            if self.truncatable is not None and turn == self.truncatable[0]:
                self.heard_ms = float(chunk.payload["t_ms"])
                self.truncate()
            else:
                logger.info(
                    "truncate of turn %s ignored: the reply that can still be cut is %s",
                    chunk.turn,
                    None if self.truncatable is None else self.truncatable[0],
                )
        else:
            pass

    def interrupt_reply(self) -> None:
        """End the open reply; the history keeps none of it until the graph reports the audio it delivered or a truncate says what was heard. A reply without text, such as one interrupted as it opens, keeps nothing whatever was delivered, so the reply before it stays the one a later report cuts."""
        assert self.emit is not None and self.reply is not None
        turn = self.reply_turn
        # note (wenyao): the graph hands this node the client's end of input once no turn is open, so speech over the paused reply is queued, which opens its turn, before the reply's turn ends.
        self.act(self.floor.reply_ended(turn))
        text = "".join(self.reply)
        self.messages.append(ChatMessage(role="assistant"))
        if text:
            self.truncatable_text = text
            self.is_truncatable_interrupted = True
            self.truncatable = (turn, len(self.messages) - 1)
            self.truncatable_progress, self.progress = self.progress, []
            self.delivered_ms = None
            self.heard_ms = None
            self.truncate()
        else:
            pass
        self.reply = None
        self.emit(control_chunk(INTERRUPT, turn=turn))
        self.emit(control_chunk(TURN_INTERRUPTED, turn=turn))

    def act(self, acts: list[FloorAct]) -> None:
        assert self.emit is not None
        for step in acts:
            if isinstance(step, Pause):
                self.emit(control_chunk(REPLY_PAUSED, turn=step.turn))
            elif isinstance(step, Resume):
                for utterance in step.utterances:
                    self.transcripts.pop(utterance)
                self.emit(control_chunk(REPLY_RESUMED, turn=step.turn))
            elif isinstance(step, Cancel):
                assert step.turn == self.reply_turn
                self.interrupt_reply()
            elif isinstance(step, Ask):
                assert self.reply is not None
                if self.judging is not None:
                    self.judging.cancel()
                else:
                    pass
                spoken = [self.transcripts[utterance] for utterance in step.utterances]
                question = addressee_messages(
                    self.messages,
                    "".join(self.reply),
                    [text for text, _ in spoken if text],
                )
                self.judging = asyncio.create_task(self.judge_ask(step, question))
            else:
                spoken = [
                    self.transcripts.pop(utterance) for utterance in step.utterances
                ]
                self.turns += 1
                self.waiting.append(
                    (
                        self.turns,
                        " ".join(text for text, _ in spoken if text),
                        next(
                            (
                                language
                                for _, language in reversed(spoken)
                                if language is not None
                            ),
                            None,
                        ),
                    )
                )
                self.emit(control_chunk(TURN_QUEUED, turn=self.turns))
        # note (wenyao): every report to the floor ends here, so the limit of a pause starts when the speech over the paused reply ends and is dropped when the reply resumes or ends, or speech begins again.
        if not self.floor.is_paused or self.floor.is_user_speaking:
            self.pause_deadline = None
        elif self.pause_deadline is None:
            self.pause_deadline = (
                asyncio.get_running_loop().time()
                + FINAL_TRANSCRIPT_WAIT_S
                + self.config.decision_timeout_s
            )
        else:
            pass

    async def judge_ask(self, ask: Ask, question: list[ChatMessage]) -> None:
        assert self.judge is not None
        started_at = asyncio.get_running_loop().time()
        try:
            margin = await self.judge.margin(question)
        except (httpx.HTTPError, ValueError, TimeoutError) as error:
            logger.warning(f"addressee judge of turn {ask.turn} failed: {error!r}")
            margin = None
        judge_ms = (asyncio.get_running_loop().time() - started_at) * 1000
        logger.info(
            f"speech over paused turn {ask.turn} judged {margin} in {judge_ms:.0f} ms"
        )
        # note (wenyao): the speech a verdict queues is answered from the input loop or a control, once the graph reports the audio it delivered of the reply the verdict ended, so a prompt the dialog refuses fails the session there.
        self.act(self.floor.decided(ask.id, margin))

    def awaits_heard_cut(self) -> bool:
        """Whether an answer waits for the text heard of the ended reply. It waits while the deadline is set and the history keeps nothing of a reply the listener heard part of."""
        if self.heard_cut_deadline is None or self.truncatable is None:
            return False
        else:
            cut = self.messages[self.truncatable[1]]
            return cut.uncertain and not cut.content

    def respond_to_waiting(self) -> None:
        assert self.emit is not None
        # note (luojiaxuan): The next prompt shows the interrupted reply's cut, which has no bound until the graph reports the audio it delivered or a truncate arrives.
        while (
            self.reply is None
            and self.waiting
            and (
                not self.awaits_delivery
                or self.delivered_ms is not None
                or self.heard_ms is not None
            )
            and not self.awaits_heard_cut()
        ):
            turn, text, language = self.waiting.popleft()
            for output in self.respond(turn, text, language):
                self.emit(output)
            if self.reply is not None:
                self.act(self.floor.reply_started(turn))
            else:
                pass

    def truncate(self) -> None:
        """Cut the ended reply in the history after the last sentence whose audio ended by the heard position: the client's truncate, never past the audio the graph delivered, and nothing when neither is known. Once the client's truncate arrived, every cut is reported as reply.heard so the client's caption matches the history."""
        assert self.emit is not None and self.truncatable is not None
        turn, index = self.truncatable
        heard_ms = min(
            (ms for ms in (self.heard_ms, self.delivered_ms) if ms is not None),
            default=0.0,
        )
        if math.isinf(heard_ms):
            pass
        else:
            content = self.truncatable_text
            text_end, uncertain = heard_prefix(
                self.truncatable_progress, heard_ms, content
            )
            logger.info(
                "truncate turn %d at %.0f ms (client %s, delivered %s) keeps %d of %d chars (uncertain=%s, %d progress points)",
                turn,
                heard_ms,
                self.heard_ms,
                self.delivered_ms,
                text_end,
                len(content),
                uncertain,
                len(self.truncatable_progress),
            )
            kept = content[:text_end].rstrip()
            self.messages[index] = ChatMessage(
                role="assistant",
                content=kept,
                uncertain=uncertain,
                is_cut=self.is_truncatable_interrupted or kept != content.rstrip(),
            )
            if self.heard_ms is not None:
                self.emit(control_chunk(REPLY_HEARD, turn=turn, text_end=text_end))
            else:
                pass
            if not self.messages[index].content and index < len(self.messages) - 1:
                self.join_around(index)
            else:
                pass

    def join_around(self, index: int) -> None:
        """Drop the reply at index, which the history keeps nothing of, and join the user messages around it: nothing of the reply was heard, so the user kept talking, and an empty assistant entry would only show the model an empty answer to copy."""
        self.messages[index - 1] = ChatMessage(
            role="user",
            content=f"{self.messages[index - 1].content} {self.messages[index + 1].content}",
        )
        del self.messages[index : index + 2]
        if self.truncatable is not None and self.truncatable[1] == index:
            self.truncatable = None
        else:
            pass

    def respond(self, turn: int, text: str, language: str | None) -> list[TimedChunk]:
        history_chars = sum(len(message.content) for message in self.messages)
        if not text:
            return [control_chunk(TURN_EMPTY, turn=turn)]
        elif len(text) > self.config.max_text_chars:
            raise ValueError("transcript exceeds max_text_chars")
        elif (
            history_chars + len(text) + self.config.max_text_chars
            > self.config.max_history_chars
        ):
            raise ValueError(
                "conversation exceeds max_history_chars; open a new session"
            )
        else:
            self.messages.append(ChatMessage(role="user", content=text))
            if self.messages[-2].role == "assistant" and not self.messages[-2].content:
                self.join_around(len(self.messages) - 2)
            else:
                pass
            is_clear = (
                len(CJK_CHAR.findall(text)) >= MIN_LANGUAGE_CJK_CHARS
                or len(text.split()) >= MIN_LANGUAGE_WORDS
            )
            is_short_switch = (
                not is_clear
                and language is not None
                and self.language is not None
                and language != self.language
            )
            if language is not None and is_clear:
                self.language = language
            else:
                pass
            prompt = prompt_messages(self.messages)
            if is_short_switch:
                note = SHORT_UTTERANCE_NOTE.format(language=self.language)
                prompt[-1] = ChatMessage(
                    role="user", content=f"{prompt[-1].content}\n\n[{note}]"
                )
            else:
                pass
            self.reply = []
            self.reply_turn = turn
            self.progress = []
            return [
                control_chunk(TURN_STARTED, turn=turn),
                messages_chunk(prompt, turn, self.language),
            ]
