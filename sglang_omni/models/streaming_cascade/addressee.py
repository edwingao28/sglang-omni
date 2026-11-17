# SPDX-License-Identifier: Apache-2.0
"""Whether speech over a paused reply was addressed to the assistant, asked of the chat service that writes the replies."""

import asyncio
import math
from collections.abc import Sequence

import httpx
from pydantic import BaseModel

from sglang_omni.models.streaming_cascade.base import (
    ChatMessage,
    bearer_headers,
    message_dict,
)
from sglang_omni.models.streaming_cascade.generators import OpenAIChatGeneratorConfig

QUESTION = (
    "You were speaking when this speech was heard. Answer yes if the speaker "
    "is talking to you: a question, a request, a correction, disagreement, "
    "asking you to stop, or a sudden change of topic, even one about their "
    "own plans or errands. Answer no only if it is clearly a short "
    "acknowledgement that lets you go on (such as mm-hmm, yeah, right, okay), "
    "clearly said to another person nearby, or background speech or noise "
    "such as a TV. If unsure, answer yes. Answer yes or no."
)
ADDRESSED_LABELS = frozenset({"yes", "Yes"})
OTHER_LABELS = frozenset({"no", "No"})
HISTORY_MESSAGES = 4
TOP_LOGPROBS = 20


class TopLogprob(BaseModel):
    token: str
    logprob: float


class TokenLogprobs(BaseModel):
    top_logprobs: list[TopLogprob] = []


class ChoiceLogprobs(BaseModel):
    content: list[TokenLogprobs] | None = None


class JudgeChoice(BaseModel):
    logprobs: ChoiceLogprobs | None = None


class JudgeResponse(BaseModel):
    choices: list[JudgeChoice]


def addressee_messages(
    history: Sequence[ChatMessage], reply: str, spoken: Sequence[str]
) -> list[ChatMessage]:
    recent = [message for message in history if message.role != "system"]
    lines = [
        f"{message.role.capitalize()}: {message.content}"
        for message in recent[-HISTORY_MESSAGES:]
    ]
    lines.append(f"Assistant (speaking): {reply}")
    lines.append(f'Heard while speaking: "{" ".join(spoken)}"')
    return [
        ChatMessage(role="system", content=QUESTION),
        ChatMessage(role="user", content="\n".join(lines)),
    ]


def logsumexp(logprobs: list[float]) -> float:
    peak = max(logprobs, default=-math.inf)
    if math.isinf(peak):
        return peak
    else:
        return peak + math.log(sum(math.exp(value - peak) for value in logprobs))


def addressed_margin(top_logprobs: Sequence[TopLogprob]) -> float | None:
    """log P(addressed) - log P(other) among an answer's first-token candidates, each side the candidates that read as one of its labels once stripped; None when neither side has one."""
    addressed = logsumexp(
        [
            entry.logprob
            for entry in top_logprobs
            if entry.token.strip() in ADDRESSED_LABELS
        ]
    )
    other = logsumexp(
        [entry.logprob for entry in top_logprobs if entry.token.strip() in OTHER_LABELS]
    )
    if math.isinf(addressed) and math.isinf(other):
        return None
    else:
        return addressed - other


class AddresseeJudge:
    def __init__(self, chat: OpenAIChatGeneratorConfig, timeout_s: float) -> None:
        self.chat = chat
        self.timeout_s = timeout_s
        # note (wenyao): the HTTP client would apply a timeout to each step of a request on its own, so margin sets one deadline for the whole request instead.
        self.client = httpx.AsyncClient(
            timeout=None, headers=bearer_headers(chat.api_key_env)
        )

    def request_body(self, messages: list[ChatMessage]) -> dict[str, object]:
        extra = dict(self.chat.extra_body)
        if "min_tokens" in extra:
            extra["min_tokens"] = 0
        else:
            pass
        template_kwargs = extra.get("chat_template_kwargs")
        extra["chat_template_kwargs"] = {
            **(template_kwargs if isinstance(template_kwargs, dict) else {}),
            "enable_thinking": False,
        }
        return {
            **extra,
            "model": self.chat.model,
            "messages": [message_dict(message) for message in messages],
            "stream": False,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
        }

    async def margin(self, messages: list[ChatMessage]) -> float | None:
        """The margin of the service's one-token answer to the question, or None when the answer carries no label; a service error raises, and so does an answer that is not complete timeout_s after the request began."""
        async with asyncio.timeout(self.timeout_s):
            response = await self.client.post(
                f"{self.chat.base_url.rstrip('/')}/chat/completions",
                json=self.request_body(messages),
            )
        response.raise_for_status()
        parsed = JudgeResponse.model_validate_json(response.content)
        logprobs = parsed.choices[0].logprobs if parsed.choices else None
        if logprobs is None or not logprobs.content:
            return None
        else:
            return addressed_margin(logprobs.content[0].top_logprobs)

    async def close(self) -> None:
        await self.client.aclose()
