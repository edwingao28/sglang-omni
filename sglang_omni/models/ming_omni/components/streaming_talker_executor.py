"""Streaming Ming-Omni talker executor."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from sglang_omni.executors.interface import Executor
from sglang_omni.models.ming_omni.components.streaming_text import (
    uint8_tensor_to_text,
)
from sglang_omni.models.ming_omni.pipeline.next_stage import TALKER_STREAM_STAGE
from sglang_omni.pipeline.stage.stream_queue import StreamItem, StreamSignal
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "DB30"
DEFAULT_SAMPLE_RATE = 44100


@dataclass
class _RequestState:
    payload: StagePayload
    output_queue: asyncio.Queue[dict[str, Any] | None]
    abort_event: threading.Event
    request_t_start_s: float = 0.0
    segmenter_first_emit_ms: float | None = None
    talker_first_audio_ms: float | None = None
    task: asyncio.Task[None] | None = None
    segment_count: int = 0
    result_enqueued: bool = False


@dataclass
class _CompletedResult:
    request_id: str
    payload: StagePayload | None = None
    error: BaseException | None = None


class MingTalkerStreamExecutor(Executor):
    """Consume text segments and stream audio chunks from MingOmniTalker."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        talker_model_path: str | None = None,
        device: str = "cuda",
        voice: str = DEFAULT_VOICE,
        talker: Any | None = None,
        audio_detokenizer: Any | None = None,
        sample_rate: int | None = None,
        loader: Callable[[], Any] | None = None,
    ) -> None:
        self._model_path = model_path
        self._talker_model_path = talker_model_path
        self._device = device
        self._voice = voice
        self._loader = loader

        self._talker = talker
        self._audio_detokenizer = audio_detokenizer
        self._sample_rate = sample_rate
        self._started = talker is not None

        self._stream_queue: Any | None = None
        self._results: asyncio.Queue[_CompletedResult] = asyncio.Queue()
        self._states: dict[str, _RequestState] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._output_queues: dict[
            str, asyncio.Queue[dict[str, Any] | None]
        ] = {}

    async def start(self) -> None:
        """Load talker resources lazily unless a test double was injected."""
        t_start = time.perf_counter()
        logger.info(
            "[TALKER_STREAM] executor.start begin device=%s model_path=%s",
            self._device,
            self._model_path,
        )
        if self._talker is not None:
            self._sample_rate = self._resolve_sample_rate()
            self._started = True
            logger.info(
                "[TALKER_STREAM] executor.start using injected talker sample_rate=%s elapsed=%.2fs",
                self._sample_rate,
                time.perf_counter() - t_start,
            )
            return

        if self._loader is not None:
            logger.info("[TALKER_STREAM] executor.start invoking injected loader")
            loaded = await asyncio.to_thread(self._loader)
            self._apply_loaded_models(loaded)
            if self._talker is None:
                raise RuntimeError(
                    "MingTalkerStreamExecutor loader did not provide a talker"
                )
            self._sample_rate = self._resolve_sample_rate()
            self._started = True
            logger.info(
                "[TALKER_STREAM] executor.start loaded via injected loader sample_rate=%s elapsed=%.2fs",
                self._sample_rate,
                time.perf_counter() - t_start,
            )
            return

        if not self._model_path:
            raise RuntimeError("MingTalkerStreamExecutor requires model_path to start")

        logger.info("[TALKER_STREAM] executor.start loading production models")
        await asyncio.to_thread(self._load_production_models)
        if self._talker is None:
            raise RuntimeError("MingTalkerStreamExecutor did not load a talker")
        self._sample_rate = self._resolve_sample_rate()
        self._started = True
        logger.info(
            "[TALKER_STREAM] executor.start complete sample_rate=%s elapsed=%.2fs",
            self._sample_rate,
            time.perf_counter() - t_start,
        )

    async def stop(self) -> None:
        for request_id in list(self._states):
            await self.abort(request_id)

    async def add_request(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if self._talker is None:
            raise RuntimeError(
                "MingTalkerStreamExecutor.add_request() called before start(); "
                "call start() or inject a talker for tests"
            )
        if self._stream_queue is None:
            raise RuntimeError("Ming streaming talker requires a stream queue")
        if request_id in self._states or request_id in self._tasks:
            raise RuntimeError(f"Request {request_id} is already running")

        output_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        state = _RequestState(
            payload=payload,
            output_queue=output_queue,
            abort_event=threading.Event(),
            request_t_start_s=time.perf_counter(),
        )
        self._states[request_id] = state
        self._output_queues[request_id] = output_queue
        task = asyncio.create_task(self._run_request(request_id, state))
        state.task = task
        self._tasks[request_id] = task

    async def get_result(self) -> StagePayload:
        completed = await self._results.get()
        if completed.error is not None:
            completed.error.request_id = completed.request_id
            raise completed.error
        if completed.payload is None:
            raise RuntimeError(f"Missing result payload for {completed.request_id}")
        return completed.payload

    async def abort(self, request_id: str) -> None:
        state = self._states.get(request_id)
        task = self._tasks.get(request_id)
        queue = self._output_queues.get(request_id)

        if state is not None:
            state.abort_event.set()
            self._set_payload_result(state.payload, aborted=True)
            self._enqueue_result_once(
                state,
                _CompletedResult(request_id=request_id, payload=state.payload),
            )
        if queue is not None:
            self._close_output_queue(queue)
        if task is not None and not task.done():
            task.cancel()
        self._close_stream_queue(request_id)

    def stream(self, request_id: str):
        queue = self._output_queues.get(request_id)

        async def generator():
            if queue is None:
                return
            while True:
                item = await queue.get()
                if item is None:
                    return
                yield item

        return generator()

    async def _run_request(self, request_id: str, state: _RequestState) -> None:
        loop = asyncio.get_running_loop()
        try:
            while not state.abort_event.is_set():
                item = await self._get_inbound(request_id)
                if self._is_done_signal(item):
                    break
                if isinstance(item, StreamSignal):
                    if item.error is not None:
                        raise item.error
                    continue
                if not isinstance(item, StreamItem):
                    raise TypeError(f"Unexpected stream item type: {type(item)!r}")

                metadata = dict(item.metadata or {})
                segment_id = int(metadata.get("segment_id", item.chunk_id))
                is_final_segment = bool(metadata.get("is_final_segment", False))
                text = uint8_tensor_to_text(item.data)
                if text:
                    if state.segmenter_first_emit_ms is None:
                        state.segmenter_first_emit_ms = (
                            time.perf_counter() - state.request_t_start_s
                        ) * 1000.0
                    await self._run_generation_thread(
                        loop,
                        request_id,
                        text,
                        segment_id,
                        state.abort_event,
                        state,
                    )
                    state.segment_count += 1
                if is_final_segment:
                    break

            if state.abort_event.is_set():
                return
            self._set_payload_result(state.payload, aborted=False)
            self._enqueue_result_once(
                state,
                _CompletedResult(request_id=request_id, payload=state.payload),
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            exc.request_id = request_id
            self._enqueue_result_once(
                state,
                _CompletedResult(request_id=request_id, error=exc),
            )
        finally:
            if self._states.get(request_id) is state:
                self._put_sentinel(state.output_queue)
                self._close_stream_queue(request_id)
            self._cleanup_request(request_id, state)

    async def _get_inbound(self, request_id: str) -> StreamItem | StreamSignal | None:
        queue = self._stream_queue
        if queue is None:
            raise RuntimeError("Ming streaming talker requires a stream queue")

        get_with_source = getattr(queue, "get_with_source", None)
        if callable(get_with_source):
            return await get_with_source(request_id)

        get = getattr(queue, "get", None)
        if not callable(get):
            raise RuntimeError("Stream queue must provide get_with_source() or get()")
        try:
            return await get(request_id)
        except TypeError:
            return await get()

    def _drain_generation_sync(
        self,
        request_id: str,
        text: str,
        segment_id: int,
        abort_event: threading.Event,
        loop: asyncio.AbstractEventLoop,
        state: _RequestState,
    ) -> None:
        generator = self._build_generation_iterator(text, abort_event)
        for item in generator:
            if abort_event.is_set():
                break
            waveform = self._extract_waveform(item)
            if waveform is None or self._waveform_numel(waveform) == 0:
                continue
            payload = self._build_audio_payload(waveform, segment_id=segment_id)
            payload["talker_queue_depth"] = state.output_queue.qsize()
            if state.talker_first_audio_ms is None:
                state.talker_first_audio_ms = (
                    time.perf_counter() - state.request_t_start_s
                ) * 1000.0
            payload.setdefault("stage_times_ms", {})
            payload["stage_times_ms"]["talker_first_audio"] = (
                state.talker_first_audio_ms
            )
            if state.segmenter_first_emit_ms is not None:
                payload["stage_times_ms"]["segmenter_first_emit"] = (
                    state.segmenter_first_emit_ms
                )
                payload["segmenter_first_emit_ms"] = state.segmenter_first_emit_ms
            loop.call_soon_threadsafe(
                self._put_output_if_active,
                request_id,
                state,
                payload,
            )

    async def _run_generation_thread(
        self,
        loop: asyncio.AbstractEventLoop,
        request_id: str,
        text: str,
        segment_id: int,
        abort_event: threading.Event,
        state: _RequestState,
    ) -> None:
        """Run the talker generator off the event loop in a daemon thread.

        Cooperative abort contract:
        - The generator (``omni_audio_generation`` / ``instruct_audio_generation``)
          MUST honor ``abort_event`` within ~50 ms once the event is set, by
          breaking the inner loop and returning. ``_drain_generation_sync``
          rechecks ``abort_event`` between every yielded chunk to back this up.
        - ``abort()`` returns immediately after setting the event and
          cancelling the awaiting task. The thread future is NOT awaited;
          if the generator hangs in CUDA or third-party code that ignores
          ``abort_event``, the daemon thread leaks until process exit.
        - Daemon=True is intentional so a leaked thread does not block
          interpreter shutdown, but server operators should monitor for
          thread accumulation as a signal that a generator is non-cooperative.
        """
        future: asyncio.Future[None] = loop.create_future()

        def runner() -> None:
            try:
                self._drain_generation_sync(
                    request_id,
                    text,
                    segment_id,
                    abort_event,
                    loop,
                    state,
                )
            except BaseException as exc:
                self._complete_thread_future(loop, future, exc)
            else:
                self._complete_thread_future(loop, future, None)

        thread = threading.Thread(
            target=runner,
            name=f"ming-talker-stream-{request_id}",
            daemon=True,
        )
        thread.start()
        await future

    def _complete_thread_future(
        self,
        loop: asyncio.AbstractEventLoop,
        future: asyncio.Future[None],
        exc: BaseException | None,
    ) -> None:
        def complete() -> None:
            if future.cancelled() or future.done():
                return
            if exc is None:
                future.set_result(None)
            else:
                future.set_exception(exc)

        try:
            loop.call_soon_threadsafe(complete)
        except RuntimeError:
            pass

    def _build_generation_iterator(
        self, text: str, abort_event: threading.Event
    ) -> Any:
        if self._talker is None:
            raise RuntimeError("Talker model not loaded")
        if hasattr(self._talker, "omni_audio_generation"):
            return self._talker.omni_audio_generation(
                tts_text=text,
                voice_name=self._voice,
                audio_detokenizer=self._audio_detokenizer,
                stream=True,
                abort_event=abort_event,
            )
        if hasattr(self._talker, "instruct_audio_generation"):
            return self._talker.instruct_audio_generation(
                prompt="Please generate speech based on the following description.\n",
                text=text,
                audio_detokenizer=self._audio_detokenizer,
                stream=True,
                abort_event=abort_event,
            )
        raise RuntimeError("Talker has no supported streaming generation method")

    def _put_output_if_active(
        self,
        request_id: str,
        state: _RequestState,
        payload: dict[str, Any],
    ) -> None:
        if self._states.get(request_id) is not state:
            return
        if self._output_queues.get(request_id) is not state.output_queue:
            return
        if state.abort_event.is_set():
            return
        state.output_queue.put_nowait(payload)

    def _enqueue_result_once(
        self,
        state: _RequestState,
        completed: _CompletedResult,
    ) -> None:
        if state.result_enqueued:
            return
        state.result_enqueued = True
        self._results.put_nowait(completed)

    def _put_sentinel(
        self, queue: asyncio.Queue[dict[str, Any] | None]
    ) -> None:
        queue.put_nowait(None)

    def _close_output_queue(
        self, queue: asyncio.Queue[dict[str, Any] | None]
    ) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        queue.put_nowait(None)

    def _cleanup_request(self, request_id: str, state: _RequestState) -> None:
        # Ownership invariant: only the task that still owns the current
        # state may remove per-request dictionaries for this request_id.
        if self._states.get(request_id) is not state:
            return
        self._states.pop(request_id, None)
        task = state.task
        if task is not None and self._tasks.get(request_id) is task:
            self._tasks.pop(request_id, None)
        if self._output_queues.get(request_id) is state.output_queue:
            self._output_queues.pop(request_id, None)

    def _close_stream_queue(self, request_id: str) -> None:
        close = getattr(self._stream_queue, "close", None)
        if callable(close):
            try:
                close(request_id)
            except TypeError:
                close()

    def _is_done_signal(self, item: Any) -> bool:
        if item is None:
            return True
        return isinstance(item, StreamSignal) and item.is_done and item.error is None

    def _extract_waveform(self, item: Any) -> Any | None:
        if isinstance(item, dict):
            waveform = item.get("tts_speech")
            if waveform is not None:
                return waveform
            return item.get("audio_waveform")
        if isinstance(item, tuple):
            return item[0] if item else None
        return item

    def _waveform_numel(self, waveform: Any) -> int:
        if isinstance(waveform, torch.Tensor):
            return int(waveform.numel())
        if isinstance(waveform, np.ndarray):
            return int(waveform.size)
        if isinstance(waveform, (bytes, bytearray, memoryview)):
            return len(waveform)
        return int(np.asarray(waveform).size)

    def _build_audio_payload(self, waveform: Any, *, segment_id: int) -> dict[str, Any]:
        audio_bytes, shape, dtype = self._serialize_waveform(waveform)
        return {
            "modality": "audio",
            "audio_waveform": audio_bytes,
            "audio_waveform_shape": shape,
            "audio_waveform_dtype": dtype,
            "sample_rate": self._resolve_sample_rate(),
            "stage_name": TALKER_STREAM_STAGE,
            "segment_id": segment_id,
        }

    def _serialize_waveform(self, waveform: Any) -> tuple[bytes, list[int], str]:
        if isinstance(waveform, torch.Tensor):
            array = waveform.detach().cpu().float().numpy()
        elif isinstance(waveform, np.ndarray):
            array = waveform.astype(np.float32, copy=False)
        elif isinstance(waveform, (bytes, bytearray, memoryview)):
            raw = bytes(waveform)
            return raw, [len(raw)], "uint8"
        else:
            array = np.asarray(waveform, dtype=np.float32)
        array = np.asarray(array, dtype=np.float32)
        return array.tobytes(), list(array.shape), str(array.dtype)

    def _resolve_sample_rate(self) -> int:
        if self._sample_rate is not None:
            return int(self._sample_rate)
        for owner in (self._audio_detokenizer, self._talker):
            sample_rate = self._sample_rate_from(owner)
            if sample_rate is not None:
                self._sample_rate = sample_rate
                return sample_rate
        self._sample_rate = DEFAULT_SAMPLE_RATE
        return self._sample_rate

    def _sample_rate_from(self, owner: Any) -> int | None:
        if owner is None:
            return None
        config = getattr(owner, "config", None)
        sample_rate = getattr(config, "sample_rate", None)
        if sample_rate is None:
            sample_rate = getattr(owner, "sample_rate", None)
        return int(sample_rate) if sample_rate is not None else None

    def _set_payload_result(self, payload: StagePayload, *, aborted: bool) -> None:
        result = {"aborted": aborted}
        payload.data = result

    def _apply_loaded_models(self, loaded: Any) -> None:
        if inspect.isawaitable(loaded):
            raise RuntimeError("MingTalkerStreamExecutor loader must be synchronous")
        if isinstance(loaded, dict):
            self._talker = loaded.get("talker")
            self._audio_detokenizer = loaded.get(
                "audio_detokenizer", loaded.get("vae")
            )
            if loaded.get("sample_rate") is not None:
                self._sample_rate = int(loaded["sample_rate"])
            return
        if isinstance(loaded, tuple):
            if loaded:
                self._talker = loaded[0]
            if len(loaded) > 1:
                self._audio_detokenizer = loaded[1]
            if len(loaded) > 2 and loaded[2] is not None:
                self._sample_rate = int(loaded[2])
            return
        self._talker = loaded

    def _load_production_models(self) -> None:
        from transformers import AutoTokenizer

        from sglang_omni.models.ming_omni.talker import (
            MingOmniTalker,
            MingOmniTalkerConfig,
            SpkembExtractor,
        )
        from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import (
            AudioVAE,
        )
        from sglang_omni.models.weight_loader import load_weights_by_prefix

        import json
        import os

        if self._model_path is None:
            raise RuntimeError("MingTalkerStreamExecutor requires model_path to start")

        t_start = time.perf_counter()
        talker_model_path = self._talker_model_path or str(
            Path(self._model_path) / "talker"
        )
        logger.info(
            "[TALKER_STREAM] load begin talker_model_path=%s device=%s",
            talker_model_path,
            self._device,
        )
        t_step = time.perf_counter()
        config = MingOmniTalkerConfig.from_pretrained_dir(talker_model_path)
        logger.info(
            "[TALKER_STREAM] config loaded elapsed=%.2fs",
            time.perf_counter() - t_step,
        )
        t_step = time.perf_counter()
        talker = MingOmniTalker(config)
        talker.eval()
        logger.info(
            "[TALKER_STREAM] talker constructed elapsed=%.2fs",
            time.perf_counter() - t_step,
        )
        t_step = time.perf_counter()
        logger.info("[TALKER_STREAM] loading talker weights")
        weights = load_weights_by_prefix(talker_model_path, prefix="")
        logger.info(
            "[TALKER_STREAM] talker weights read elapsed=%.2fs",
            time.perf_counter() - t_step,
        )
        t_step = time.perf_counter()
        logger.info("[TALKER_STREAM] applying talker weights")
        talker.load_weights(weights.items())
        logger.info(
            "[TALKER_STREAM] talker weights applied elapsed=%.2fs",
            time.perf_counter() - t_step,
        )
        t_step = time.perf_counter()
        logger.info("[TALKER_STREAM] moving talker to %s", self._device)
        talker.to(device=self._device, dtype=torch.bfloat16)
        logger.info(
            "[TALKER_STREAM] talker moved to device elapsed=%.2fs",
            time.perf_counter() - t_step,
        )
        t_step = time.perf_counter()
        logger.info("[TALKER_STREAM] loading talker tokenizer")
        talker.set_tokenizer(
            AutoTokenizer.from_pretrained(str(Path(talker_model_path) / "llm"))
        )
        logger.info(
            "[TALKER_STREAM] tokenizer loaded elapsed=%.2fs",
            time.perf_counter() - t_step,
        )

        voice_json_path = os.path.join(talker_model_path, "data", "voice_name.json")
        t_step = time.perf_counter()
        if os.path.exists(voice_json_path):
            logger.info("[TALKER_STREAM] loading voice presets from %s", voice_json_path)
            with open(voice_json_path) as voice_file:
                voice_dict = json.load(voice_file)
            for value in voice_dict.values():
                value["prompt_wav_path"] = os.path.join(
                    talker_model_path,
                    value["prompt_wav_path"],
                )
            talker.set_voice_presets(voice_dict)
            logger.info(
                "[TALKER_STREAM] voice presets loaded count=%d elapsed=%.2fs",
                len(voice_dict),
                time.perf_counter() - t_step,
            )
        else:
            logger.warning(
                "[TALKER_STREAM] voice_name.json not found at %s",
                voice_json_path,
            )

        campplus_path = os.path.join(talker_model_path, "campplus.onnx")
        t_step = time.perf_counter()
        logger.info("[TALKER_STREAM] loading speaker embedding extractor")
        try:
            talker.set_spkemb_extractor(SpkembExtractor(campplus_path))
        except (ImportError, Exception) as exc:
            logger.warning("[TALKER_STREAM] SpkembExtractor not available: %s", exc)
        else:
            logger.info(
                "[TALKER_STREAM] speaker embedding extractor loaded elapsed=%.2fs",
                time.perf_counter() - t_step,
            )

        t_step = time.perf_counter()
        logger.info("[TALKER_STREAM] loading text normalizer")
        try:
            from talker_tn.talker_tn import TalkerTN

            talker.set_normalizer(TalkerTN())
        except ImportError:
            logger.warning(
                "[TALKER_STREAM] TalkerTN unavailable; using identity normalizer"
            )
        else:
            logger.info(
                "[TALKER_STREAM] text normalizer loaded elapsed=%.2fs",
                time.perf_counter() - t_step,
            )

        vae_path = str(Path(talker_model_path) / "vae")
        vae = None
        t_step = time.perf_counter()
        if Path(vae_path).exists():
            logger.info("[TALKER_STREAM] loading AudioVAE from %s", vae_path)
            vae = AudioVAE.from_pretrained(vae_path, dtype=torch.bfloat16)
            vae.to(self._device)
            vae.eval()
            logger.info(
                "[TALKER_STREAM] AudioVAE loaded sample_rate=%s elapsed=%.2fs",
                getattr(getattr(vae, "config", None), "sample_rate", None),
                time.perf_counter() - t_step,
            )
        else:
            logger.warning("[TALKER_STREAM] AudioVAE not found at %s", vae_path)

        t_step = time.perf_counter()
        logger.info("[TALKER_STREAM] initializing CUDA graphs")
        talker.initial_graph()
        logger.info(
            "[TALKER_STREAM] CUDA graphs initialized elapsed=%.2fs",
            time.perf_counter() - t_step,
        )
        self._talker = talker
        self._audio_detokenizer = vae
        logger.info(
            "[TALKER_STREAM] load complete total_elapsed=%.2fs",
            time.perf_counter() - t_start,
        )
