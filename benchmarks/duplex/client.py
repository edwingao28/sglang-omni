# SPDX-License-Identifier: Apache-2.0
"""Record the native full-duplex session protocol over a real WebSocket."""

from __future__ import annotations

import asyncio
import base64
import json
import queue
import time
import uuid
from pathlib import Path
from typing import Literal, TypedDict

import websockets
from pydantic import JsonValue

from benchmarks.duplex.profiles import DEFAULT_PROFILE, PROFILES, ProfileName

SAMPLE_RATE = 16000
PACKET_MS = 80
PACKET_BYTES = SAMPLE_RATE * PACKET_MS // 1000 * 2
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
POST_CLOSE_SECONDS = 0.2
FRAME_EXCERPT_CHARS = 200
# note (wenyao): A closing session can still occupy the only admission slot.
ADMISSION_DENIED_STATUS = 503
ADMISSION_RETRIES = 3
ADMISSION_BACKOFF_S = 0.25
SEND_RECEIPTS_FILE = "input-send-receipts.json"


class AdmissionRejected(RuntimeError):
    """The bounded handshake retry exhausted server admission capacity."""


# note (wenyao): Keepalive and compression would perturb packet timing.
TRANSPORT: dict[str, JsonValue] = {
    "max_message_bytes": MAX_MESSAGE_BYTES,
    "compression": None,
    "keepalive_ping": False,
    "post_close_s": POST_CLOSE_SECONDS,
    "admission_retries": ADMISSION_RETRIES,
    "admission_backoff_s": ADMISSION_BACKOFF_S,
    "input_end": "first append send start + unpadded input duration",
    "input_send_receipts": SEND_RECEIPTS_FILE,
}


class InputAudioMetadata(TypedDict):
    seq: int
    t_start_ms: float


class SendReceipt(TypedDict):
    event_id: str
    seq: int
    scheduled_s: float
    start_s: float
    completed_s: float


def scheduled_send_s(start_s: float, sequence: int) -> float:
    return start_s + sequence * PACKET_MS / 1000


async def run_session(
    url: str,
    pcm: bytes,
    *,
    scenario: Literal["continuous"],
    trace_path: Path,
    timeout_s: float = 90.0,
    profile: ProfileName = DEFAULT_PROFILE,
    start_gate: asyncio.Future[float] | None = None,
    ready: asyncio.Future[bool] | None = None,
    start_offset_s: float = 0.0,
) -> None:
    """Save observations and failures; classification belongs to offline replay."""
    if scenario != "continuous":
        raise ValueError(f"unsupported scenario: {scenario}")
    else:
        pass
    if not pcm or len(pcm) % 2:
        raise ValueError("Input must be nonempty PCM16")
    else:
        pass

    receipts: list[SendReceipt] = []
    session_start_s: float | None = None
    admitted = False
    with trace_path.open("x", encoding="utf-8", buffering=1) as trace_file:
        trace_queue: queue.Queue[dict[str, JsonValue] | None] = queue.Queue()

        def write_trace() -> None:
            while True:
                observation = trace_queue.get()
                if observation is None:
                    return
                else:
                    trace_file.write(json.dumps(observation, allow_nan=False) + "\n")

        writer = asyncio.create_task(asyncio.to_thread(write_trace))

        def record(
            direction: Literal["send", "receive", "error", "admission"],
            event: dict[str, JsonValue],
            observed_time_s: float | None = None,
        ) -> float:
            if observed_time_s is None:
                observed_time_s = time.perf_counter()
            else:
                pass
            trace_queue.put(
                {
                    "direction": direction,
                    "time_s": observed_time_s,
                    "event": event,
                }
            )
            return observed_time_s

        async def admit() -> websockets.ClientConnection:
            nonlocal admitted
            attempt = 0
            while True:
                attempt += 1
                try:
                    connection = await websockets.connect(
                        url,
                        max_size=MAX_MESSAGE_BYTES,
                        compression=None,
                        ping_interval=None,
                    )
                    admitted = True
                    return connection
                except websockets.InvalidStatus as exc:
                    if exc.response.status_code != ADMISSION_DENIED_STATUS:
                        raise
                    else:
                        record(
                            "admission",
                            {
                                "type": "connection_denied",
                                "http_status": exc.response.status_code,
                                "attempt": attempt,
                                **(
                                    {"exhausted": True}
                                    if attempt > ADMISSION_RETRIES
                                    else {}
                                ),
                            },
                        )
                        if attempt > ADMISSION_RETRIES:
                            raise AdmissionRejected(
                                f"admission denied {attempt} times with HTTP "
                                f"{ADMISSION_DENIED_STATUS}"
                            ) from exc
                        else:
                            await asyncio.sleep(ADMISSION_BACKOFF_S)

        async def exchange() -> None:
            async with await admit() as websocket:
                seen: dict[str, asyncio.Event] = {}
                aborted = asyncio.Event()
                fatal = False

                async def send(
                    event_type: Literal[
                        "session.update",
                        "input_audio_buffer.append",
                        "sglang.input_audio.end",
                        "session.close",
                    ],
                    *,
                    session: dict[str, list[str]] | None = None,
                    audio: str | None = None,
                    sglang: InputAudioMetadata | None = None,
                    scheduled_s: float | None = None,
                ) -> None:
                    event_id = uuid.uuid4().hex
                    event: dict[str, JsonValue] = {
                        "type": event_type,
                        "event_id": event_id,
                    }
                    if session is not None:
                        event["session"] = session
                    else:
                        pass
                    if audio is not None:
                        event["audio"] = audio
                    else:
                        pass
                    if sglang is not None:
                        event["sglang"] = sglang
                    else:
                        pass
                    frame = json.dumps(event)
                    send_started_s = time.perf_counter()
                    record("send", event, send_started_s)
                    await websocket.send(frame)
                    send_completed_s = time.perf_counter()
                    if event_type == "input_audio_buffer.append":
                        assert sglang is not None
                        assert scheduled_s is not None
                        receipts.append(
                            {
                                "event_id": event_id,
                                "seq": sglang["seq"],
                                "scheduled_s": scheduled_s,
                                "start_s": send_started_s,
                                "completed_s": send_completed_s,
                            }
                        )
                    else:
                        pass

                async def settle(event_type: str) -> bool:
                    receipt = seen.setdefault(event_type, asyncio.Event())
                    waiters = (
                        asyncio.create_task(receipt.wait()),
                        asyncio.create_task(aborted.wait()),
                    )
                    try:
                        await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for waiter in waiters:
                            waiter.cancel()
                        await asyncio.gather(*waiters, return_exceptions=True)
                    return receipt.is_set()

                async def receive() -> None:
                    nonlocal fatal
                    async for frame in websocket:
                        received_time_s = time.perf_counter()
                        try:

                            def reject_constant(value: str) -> None:
                                raise ValueError(f"nonfinite JSON number: {value}")

                            event = json.loads(frame, parse_constant=reject_constant)
                            if not isinstance(event, dict):
                                raise ValueError("server event must be a JSON object")
                            else:
                                pass
                            event_type = event.get("type")
                            if not isinstance(event_type, str):
                                raise ValueError("server event type must be a string")
                            else:
                                pass
                            record("receive", event, received_time_s)
                        except ValueError as exc:
                            raise ValueError(
                                "malformed server frame "
                                f"{frame[:FRAME_EXCERPT_CHARS]!r}: "
                                f"{type(exc).__name__}: {exc}"
                            ) from exc
                        seen.setdefault(event_type, asyncio.Event()).set()
                        if event_type == "error":
                            # note (wenyao): Close waits for adapter teardown.
                            extension = event.get("sglang")
                            fatal = fatal or bool(
                                isinstance(extension, dict) and extension.get("fatal")
                            )
                            aborted.set()
                        else:
                            pass
                    if not seen.get("session.closed", asyncio.Event()).is_set():
                        raise RuntimeError("Connection closed without session.closed")
                    else:
                        pass

                async def drive() -> None:
                    nonlocal session_start_s
                    streamed = False
                    if await settle("session.created"):
                        await send(
                            "session.update",
                            session={
                                "output_modalities": list(
                                    PROFILES[profile].output_modalities
                                )
                            },
                        )
                    else:
                        pass
                    if await settle("session.updated"):
                        streamed = True
                        if ready is not None:
                            ready.set_result(True)
                        else:
                            pass
                        session_start_s = (
                            await start_gate
                            if start_gate is not None
                            else time.perf_counter()
                        ) + start_offset_s
                        for sequence, byte_offset in enumerate(
                            range(0, len(pcm), PACKET_BYTES)
                        ):
                            deadline_s = scheduled_send_s(session_start_s, sequence)
                            await asyncio.sleep(
                                max(0.0, deadline_s - time.perf_counter())
                            )
                            if aborted.is_set():
                                streamed = False
                                break
                            else:
                                pass
                            await send(
                                "input_audio_buffer.append",
                                audio=base64.b64encode(
                                    pcm[byte_offset : byte_offset + PACKET_BYTES]
                                ).decode("ascii"),
                                sglang={
                                    "seq": sequence,
                                    "t_start_ms": byte_offset / 2 / SAMPLE_RATE * 1000,
                                },
                                scheduled_s=deadline_s,
                            )
                    else:
                        pass
                    if streamed:
                        await asyncio.sleep(
                            max(
                                0.0,
                                max(
                                    session_start_s + len(pcm) / (2 * SAMPLE_RATE),
                                    receipts[-1]["start_s"]
                                    + min(
                                        PACKET_MS / 1000,
                                        (len(pcm) - receipts[-1]["seq"] * PACKET_BYTES)
                                        / (2 * SAMPLE_RATE),
                                    ),
                                )
                                - time.perf_counter(),
                            )
                        )
                        if not aborted.is_set():
                            await send("sglang.input_audio.end")
                            await settle("sglang.input_audio.drained")
                        else:
                            pass
                    else:
                        pass
                    closed = seen.setdefault("session.closed", asyncio.Event())
                    if not fatal and not closed.is_set():
                        await send("session.close")
                    else:
                        pass
                    await closed.wait()

                receiver = asyncio.create_task(receive())
                driver = asyncio.create_task(drive())
                try:
                    done, _ = await asyncio.wait(
                        {receiver, driver}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if receiver in done:
                        await receiver
                        # note (wenyao): No receipts can arrive after receiver exit.
                        try:
                            await asyncio.wait_for(driver, POST_CLOSE_SECONDS)
                        except asyncio.TimeoutError:
                            record(
                                "error",
                                {"message": "driver stalled after the receive loop"},
                            )
                    else:
                        await driver
                        try:
                            await asyncio.wait_for(receiver, POST_CLOSE_SECONDS)
                        except asyncio.TimeoutError:
                            pass
                finally:
                    receiver.cancel()
                    driver.cancel()
                    await asyncio.gather(receiver, driver, return_exceptions=True)

        try:
            await asyncio.wait_for(exchange(), timeout=timeout_s)
        except asyncio.TimeoutError:
            record("error", {"message": f"Session timeout after {timeout_s}s"})
        except AdmissionRejected:
            pass
        except (
            OSError,
            websockets.WebSocketException,
            ValueError,
            RuntimeError,
        ) as exc:
            record("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            if ready is not None and not ready.done():
                ready.set_result(False)
            else:
                pass
            trace_queue.put(None)
            await writer

            def write_receipts() -> None:
                trace_path.with_name(SEND_RECEIPTS_FILE).write_text(
                    json.dumps(
                        {
                            "session_start_s": session_start_s,
                            "admitted": admitted,
                            "appends": receipts,
                        },
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            await asyncio.to_thread(write_receipts)
