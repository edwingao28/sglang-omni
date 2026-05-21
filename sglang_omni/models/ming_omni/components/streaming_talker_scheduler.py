# SPDX-License-Identifier: Apache-2.0
"""Streaming talker scheduler for Ming streaming TTS."""

from __future__ import annotations

import inspect
import logging
import queue as _queue_mod
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sglang_omni.models.ming_omni.components.streaming_text import uint8_tensor_to_text
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "DB30"
DEFAULT_SAMPLE_RATE = 44100

_DONE_SEEN_MAX = 10000
_DONE_SEEN_EVICT_TO = 5000
_ABORTED_SEEN_MAX = 10000
_ABORTED_SEEN_EVICT_TO = 5000
_FINISHED_SEEN_MAX = 10000
_FINISHED_SEEN_EVICT_TO = 5000


@dataclass
class _PendingChunk:
    text: str
    metadata: dict[str, Any]
    chunk_id: int


@dataclass
class _RequestState:
    abort_event: threading.Event = field(default_factory=threading.Event)
    input_queue: _queue_mod.Queue[_PendingChunk | None] = field(
        default_factory=_queue_mod.Queue
    )
    payload: StagePayload | None = None
    done: bool = False
    pending_chunks: list[_PendingChunk] = field(default_factory=list)
    worker_thread: threading.Thread | None = None
    audio_pieces: list[Any] = field(default_factory=list)
    request_start_ms: int | None = None
    generation_started: bool = False
    audio_chunk_count: int = 0
    segment_count: int = 0
    segmenter_first_emit_ms: int | None = None
    talker_first_audio_ms: int | None = None


class MingTalkerStreamScheduler:
    """Scheduler-style Ming talker stage for streaming text-to-speech."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        device: str = "cuda",
        voice: str = DEFAULT_VOICE,
        talker: Any | None = None,
        audio_detokenizer: Any | None = None,
        sample_rate: int | None = None,
        loader: Callable[[], Any] | None = None,
        stage_name: str = "talker_stream",
        now_ms_fn: Callable[[], int] | None = None,
    ) -> None:
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self.stage_name = stage_name
        self._model_path = model_path
        self._device = device
        self._voice = voice
        self._talker = talker
        self._audio_detokenizer = audio_detokenizer
        self._sample_rate = int(sample_rate) if sample_rate is not None else None
        self._loader = loader
        self._now_ms_fn = now_ms_fn or self._default_now_ms
        self._running = False
        self._stopping = False
        self._state_lock = threading.RLock()
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._state: dict[str, _RequestState] = {}
        self._done_seen: OrderedDict[str, None] = OrderedDict()
        self._aborted_seen: OrderedDict[str, None] = OrderedDict()
        self._finished_seen: OrderedDict[str, None] = OrderedDict()

    def start(self) -> None:
        with self._state_lock:
            self._running = True
            self._stopping = False
        while self._running:
            try:
                msg = self.inbox.get(timeout=0.1)
            except _queue_mod.Empty:
                continue

            try:
                if self._is_inactive(msg.request_id):
                    continue
                if msg.type == "new_request":
                    self._on_new_request(msg.request_id, msg.data)
                elif msg.type == "stream_chunk":
                    self._on_stream_chunk(msg.request_id, msg.data)
                elif msg.type == "stream_done":
                    self._on_stream_done(msg.request_id)
                else:
                    raise ValueError(f"Unexpected message type: {msg.type!r}")
            except Exception as exc:
                logger.exception(
                    "MingStreamingTalkerScheduler failed request %s",
                    msg.request_id,
                )
                self.abort(msg.request_id)
                self.outbox.put(
                    OutgoingMessage(
                        request_id=msg.request_id,
                        type="error",
                        data=exc,
                    )
                )

    def stop(self) -> None:
        with self._state_lock:
            self._running = False
            self._stopping = True
            states = list(self._state.items())
            self._state.clear()
            self._done_seen.clear()
            for request_id, _state in states:
                self._remember_aborted(request_id)

        workers = []
        for _request_id, state in states:
            state.abort_event.set()
            state.input_queue.put(None)
            if state.worker_thread is not None:
                workers.append(state.worker_thread)
        current = threading.current_thread()
        for worker in workers:
            if worker is not current and worker.ident is not None:
                worker.join(timeout=0.25)

    def abort(self, request_id: str) -> _RequestState | None:
        with self._state_lock:
            state = self._state.pop(request_id, None)
            self._done_seen.pop(request_id, None)
            self._remember_aborted(request_id)
        if state is not None:
            state.abort_event.set()
            state.input_queue.put(None)
        return state

    def _ensure_state(self, request_id: str) -> _RequestState | None:
        with self._state_lock:
            if self._stopping or self._is_inactive_locked(request_id):
                return None
            state = self._state.get(request_id)
            if state is None:
                state = _RequestState()
                self._state[request_id] = state
            return state

    def _on_new_request(self, request_id: str, payload: StagePayload) -> None:
        if not isinstance(payload, StagePayload):
            raise TypeError("new_request data must be a StagePayload")

        state = self._ensure_state(request_id)
        if state is None:
            return
        state.payload = payload
        state.request_start_ms = self._now_ms()
        with self._state_lock:
            if request_id in self._done_seen:
                state.done = True
                self._done_seen.pop(request_id, None)
        if not self._is_active_state(request_id, state):
            return

        if not self._payload_includes_audio(payload):
            state.pending_chunks.clear()
            self._finalize(request_id, state, skipped=True)
            return

        self._start_worker(request_id, state)
        for chunk in list(state.pending_chunks):
            state.input_queue.put(chunk)
        state.pending_chunks.clear()

        if state.done:
            state.input_queue.put(None)

    def _on_stream_chunk(self, request_id: str, item: Any) -> None:
        chunk = self._decode_chunk(item)
        state = self._ensure_state(request_id)
        if state is None:
            return
        if not self._is_active_state(request_id, state):
            return
        if state.payload is None:
            state.pending_chunks.append(chunk)
            return
        if not self._payload_includes_audio(state.payload):
            return
        self._start_worker(request_id, state)
        state.input_queue.put(chunk)

    def _on_stream_done(self, request_id: str) -> None:
        with self._state_lock:
            state = self._state.get(request_id)
        if state is None:
            self._remember_done(request_id)
            return
        state.done = True
        if state.payload is not None:
            if not self._payload_includes_audio(state.payload):
                self._finalize(request_id, state, skipped=True)
            else:
                self._start_worker(request_id, state)
                state.input_queue.put(None)

    def _decode_chunk(self, item: Any) -> _PendingChunk:
        if not isinstance(item, StreamItem):
            raise TypeError(f"Unexpected stream item type: {type(item)!r}")
        return _PendingChunk(
            text=uint8_tensor_to_text(item.data),
            metadata=dict(item.metadata or {}),
            chunk_id=int(item.chunk_id),
        )

    def _start_worker(self, request_id: str, state: _RequestState) -> None:
        with self._state_lock:
            if self._stopping or not self._is_active_state_locked(request_id, state):
                return
            if state.worker_thread is not None and state.worker_thread.is_alive():
                return
            worker = threading.Thread(
                target=self._run_worker,
                args=(request_id, state),
                name=f"ming-talker-stream-{request_id}",
                daemon=True,
            )
            state.worker_thread = worker
            worker.start()

    def _run_worker(self, request_id: str, state: _RequestState) -> None:
        try:
            while not state.abort_event.is_set():
                chunk = state.input_queue.get()
                if chunk is None:
                    break
                self._generate_for_chunk(request_id, state, chunk)
            if state.abort_event.is_set():
                return
            self._finalize(request_id, state, skipped=False)
        except BaseException as exc:
            if state.abort_event.is_set() or not self._is_active_state(
                request_id,
                state,
            ):
                return
            logger.exception(
                "MingTalkerStreamScheduler worker failed request %s",
                request_id,
            )
            self.abort(request_id)
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="error",
                    data=exc,
                )
            )

    def _generate_for_chunk(
        self,
        request_id: str,
        state: _RequestState,
        chunk: _PendingChunk,
    ) -> None:
        if state.abort_event.is_set() or not chunk.text.strip():
            return

        state.segment_count += 1
        state.generation_started = True
        if state.segmenter_first_emit_ms is None:
            value = chunk.metadata.get("segmenter_first_emit_ms")
            if value is not None:
                state.segmenter_first_emit_ms = int(value)

        segment_id = int(chunk.metadata.get("segment_id", chunk.chunk_id))
        is_streaming = self._payload_is_streaming(state.payload)
        with self._generation_lock:
            if not self._is_active_state(request_id, state):
                return
            with torch.no_grad():
                for generated in self._build_generation_iterator(
                    chunk.text,
                    state.abort_event,
                ):
                    if state.abort_event.is_set():
                        return
                    waveform = self._extract_waveform(generated)
                    if waveform is None or self._waveform_numel(waveform) == 0:
                        continue
                    if state.abort_event.is_set():
                        return

                    if not is_streaming:
                        state.audio_pieces.append(waveform)
                    state.audio_chunk_count += 1
                    audio_payload = self._build_audio_payload(
                        waveform,
                        segment_id=segment_id,
                        state=state,
                    )
                    if not is_streaming:
                        continue
                    if not self._is_active_state(request_id, state):
                        return
                    self._put_outgoing_if_active(
                        request_id,
                        state,
                        OutgoingMessage(
                            request_id=request_id,
                            type="stream",
                            target=None,
                            data=audio_payload,
                            metadata={
                                "modality": "audio",
                                "stage_name": self.stage_name,
                                "sample_rate": self._resolve_sample_rate(),
                                "segment_id": segment_id,
                            },
                        ),
                    )

    def _finalize(
        self,
        request_id: str,
        state: _RequestState,
        *,
        skipped: bool,
    ) -> None:
        if state.payload is None:
            return

        payload = state.payload
        no_audio_generated = not skipped and state.audio_chunk_count == 0
        if skipped or no_audio_generated:
            if self._payload_is_streaming(payload):
                payload.data = self._build_streaming_skipped_result(state)
            else:
                payload.data = self._build_skipped_result(state)
        elif self._payload_is_streaming(payload):
            payload.data = self._build_streaming_result(state)
        else:
            payload.data = self._build_non_streaming_result(state)

        self._put_final_if_active(
            request_id,
            state,
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=payload,
            ),
        )

    def _build_generation_iterator(
        self,
        text: str,
        abort_event: threading.Event,
    ) -> Iterable[Any]:
        self._ensure_talker_loaded()
        if self._talker is None:
            raise RuntimeError("Talker model not loaded")
        if hasattr(self._talker, "omni_audio_generation"):
            method = self._talker.omni_audio_generation
            kwargs = {
                "tts_text": text,
                "voice_name": self._voice,
                "audio_detokenizer": self._audio_detokenizer,
                "stream": True,
                "abort_event": abort_event,
            }
            return self._call_generation_method(method, kwargs)
        if hasattr(self._talker, "instruct_audio_generation"):
            method = self._talker.instruct_audio_generation
            kwargs = {
                "prompt": "Please generate speech based on the following description.\n",
                "text": text,
                "audio_detokenizer": self._audio_detokenizer,
                "stream": True,
                "abort_event": abort_event,
            }
            return self._call_generation_method(method, kwargs)
        raise RuntimeError("Talker has no supported streaming generation method")

    def _call_generation_method(
        self,
        method: Callable[..., Any],
        kwargs: dict[str, Any],
    ) -> Iterable[Any]:
        filtered, allow_abort_retry = self._filter_kwargs(method, kwargs)
        try:
            result = method(**filtered)
        except TypeError:
            if not allow_abort_retry:
                raise
            without_abort = dict(filtered)
            without_abort.pop("abort_event", None)
            result = method(**without_abort)
        if self._is_direct_waveform(result):
            return (result,)
        if isinstance(result, Iterable):
            return result
        return (result,)

    def _filter_kwargs(
        self,
        method: Callable[..., Any],
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return kwargs, True
        params = signature.parameters
        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        ):
            return kwargs, True
        return {key: value for key, value in kwargs.items() if key in params}, False

    def _build_audio_payload(
        self,
        waveform: Any,
        *,
        segment_id: int,
        state: _RequestState,
    ) -> dict[str, Any]:
        audio_bytes, shape, dtype = self._serialize_waveform(waveform)
        now_ms = self._now_ms()
        if state.talker_first_audio_ms is None and state.request_start_ms is not None:
            state.talker_first_audio_ms = now_ms - state.request_start_ms

        stage_times_ms = {
            "talker_elapsed": (
                now_ms - state.request_start_ms
                if state.request_start_ms is not None
                else None
            )
        }
        if state.talker_first_audio_ms is not None:
            stage_times_ms["talker_first_audio"] = state.talker_first_audio_ms
        if state.segmenter_first_emit_ms is not None:
            stage_times_ms["segmenter_first_emit"] = state.segmenter_first_emit_ms

        payload: dict[str, Any] = {
            "modality": "audio",
            "audio_waveform": audio_bytes,
            "audio_waveform_shape": shape,
            "audio_waveform_dtype": dtype,
            "sample_rate": self._resolve_sample_rate(),
            "stage_name": self.stage_name,
            "segment_id": segment_id,
            "talker_queue_depth": self.outbox.qsize(),
            "stage_times_ms": stage_times_ms,
        }
        if state.segmenter_first_emit_ms is not None:
            payload["segmenter_first_emit_ms"] = state.segmenter_first_emit_ms
        if state.talker_first_audio_ms is not None:
            payload["talker_first_audio_ms"] = state.talker_first_audio_ms
        return payload

    def _build_streaming_result(self, state: _RequestState) -> dict[str, Any]:
        result = self._base_result(state)
        result.update(
            {
                "audio_chunk_count": state.audio_chunk_count,
                "streaming": True,
                "aborted": False,
            }
        )
        return result

    def _build_non_streaming_result(self, state: _RequestState) -> dict[str, Any]:
        if state.audio_pieces:
            waveform = self._concat_waveforms(state.audio_pieces)
            audio_bytes, shape, dtype = self._serialize_waveform(waveform)
            duration = (shape[-1] / self._resolve_sample_rate()) if shape else 0.0
        else:
            audio_bytes = None
            shape = []
            dtype = "float32"
            duration = 0.0

        result = self._base_result(state)
        result.update(
            {
                "audio_waveform": audio_bytes,
                "audio_waveform_shape": shape,
                "audio_waveform_dtype": dtype,
                "audio_chunk_count": state.audio_chunk_count,
                "duration": duration,
                "streaming": False,
                "aborted": False,
            }
        )
        return result

    def _build_skipped_result(self, state: _RequestState) -> dict[str, Any]:
        result = self._base_result(state)
        result.update(
            {
                "audio_waveform": None,
                "audio_waveform_shape": [],
                "audio_waveform_dtype": "float32",
                "audio_chunk_count": 0,
                "duration": 0.0,
                "skipped": True,
                "aborted": False,
            }
        )
        return result

    def _build_streaming_skipped_result(self, state: _RequestState) -> dict[str, Any]:
        result = self._base_result(state)
        result.update(
            {
                "audio_chunk_count": 0,
                "streaming": True,
                "skipped": True,
                "aborted": False,
            }
        )
        return result

    def _base_result(self, state: _RequestState) -> dict[str, Any]:
        now_ms = self._now_ms()
        stage_times_ms = {}
        if state.request_start_ms is not None:
            stage_times_ms["talker_total"] = now_ms - state.request_start_ms
        if state.talker_first_audio_ms is not None:
            stage_times_ms["talker_first_audio"] = state.talker_first_audio_ms
        if state.segmenter_first_emit_ms is not None:
            stage_times_ms["segmenter_first_emit"] = state.segmenter_first_emit_ms
        return {
            "modality": "audio",
            "sample_rate": self._resolve_sample_rate(),
            "stage_name": self.stage_name,
            "segment_count": state.segment_count,
            "stage_times_ms": stage_times_ms,
            "segmenter_first_emit_ms": state.segmenter_first_emit_ms,
            "talker_first_audio_ms": state.talker_first_audio_ms,
            "talker_queue_depth": self.outbox.qsize(),
        }

    def _extract_waveform(self, item: Any) -> Any | None:
        if isinstance(item, tuple):
            return item[0] if item else None
        return item

    def _is_direct_waveform(self, value: Any) -> bool:
        return isinstance(
            value,
            (torch.Tensor, np.ndarray, bytes, bytearray, memoryview),
        )

    def _waveform_numel(self, waveform: Any) -> int:
        if isinstance(waveform, torch.Tensor):
            return int(waveform.numel())
        if isinstance(waveform, np.ndarray):
            return int(waveform.size)
        if isinstance(waveform, (bytes, bytearray, memoryview)):
            return len(waveform)
        return int(np.asarray(waveform).size)

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

    def _concat_waveforms(self, waveforms: list[Any]) -> Any:
        arrays = []
        for waveform in waveforms:
            if isinstance(waveform, torch.Tensor):
                arrays.append(waveform.detach().cpu().float().numpy())
            elif isinstance(waveform, np.ndarray):
                arrays.append(waveform.astype(np.float32, copy=False))
            elif isinstance(waveform, (bytes, bytearray, memoryview)):
                arrays.append(np.frombuffer(bytes(waveform), dtype=np.uint8))
            else:
                arrays.append(np.asarray(waveform, dtype=np.float32))
        if not arrays:
            return np.asarray([], dtype=np.float32)
        if self._can_concat_last_dim(arrays):
            return np.concatenate(arrays, axis=-1)
        return np.concatenate([np.ravel(array) for array in arrays], axis=-1)

    def _can_concat_last_dim(self, arrays: list[np.ndarray]) -> bool:
        first = arrays[0]
        if first.ndim == 0:
            return False
        leading_shape = first.shape[:-1]
        return all(
            array.ndim == first.ndim and array.shape[:-1] == leading_shape
            for array in arrays
        )

    def _payload_includes_audio(self, payload: StagePayload | None) -> bool:
        if payload is None:
            return False
        metadata = getattr(payload.request, "metadata", None)
        if not isinstance(metadata, dict):
            return True
        modalities = metadata.get("output_modalities")
        if modalities is None:
            return True
        if isinstance(modalities, str):
            return modalities == "audio"
        if isinstance(modalities, (list, tuple, set)):
            return "audio" in {str(modality) for modality in modalities}
        return True

    def _payload_is_streaming(self, payload: StagePayload | None) -> bool:
        if payload is None:
            return False
        return bool((payload.request.params or {}).get("stream") is True)

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

    def _load_models(self) -> None:
        if self._loader is not None:
            loaded = self._loader()
            self._apply_loaded_models(loaded)
            if self._talker is None:
                raise RuntimeError(
                    "Ming streaming talker loader did not provide a talker"
                )
            return
        self._load_production_models()

    def _ensure_talker_loaded(self) -> None:
        if self._talker is not None:
            return
        with self._load_lock:
            if self._talker is None:
                self._load_models()

    def _apply_loaded_models(self, loaded: Any) -> None:
        if not isinstance(loaded, dict):
            raise RuntimeError(
                "Ming streaming talker loader must return a dict with 'talker'"
            )
        self._talker = loaded.get("talker")
        self._audio_detokenizer = loaded.get("audio_detokenizer", loaded.get("vae"))
        if loaded.get("sample_rate") is not None:
            self._sample_rate = int(loaded["sample_rate"])

    def _load_production_models(self) -> None:
        import json
        import os

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

        if self._model_path is None:
            raise RuntimeError(
                "MingStreamingTalkerScheduler requires model_path, injected talker, "
                "or loader"
            )

        talker_model_path = str(Path(self._model_path) / "talker")
        config = MingOmniTalkerConfig.from_pretrained_dir(talker_model_path)
        talker = MingOmniTalker(config)
        talker.eval()
        weights = load_weights_by_prefix(talker_model_path, prefix="")
        talker.load_weights(weights.items())
        talker.to(device=self._device, dtype=torch.bfloat16)
        talker.set_tokenizer(
            AutoTokenizer.from_pretrained(str(Path(talker_model_path) / "llm"))
        )

        voice_json_path = os.path.join(talker_model_path, "data", "voice_name.json")
        if os.path.exists(voice_json_path):
            with open(voice_json_path) as voice_file:
                voice_dict = json.load(voice_file)
            for value in voice_dict.values():
                value["prompt_wav_path"] = os.path.join(
                    talker_model_path,
                    value["prompt_wav_path"],
                )
            talker.set_voice_presets(voice_dict)
        else:
            logger.warning(
                "[TALKER_STREAM] voice_name.json not found at %s",
                voice_json_path,
            )

        campplus_path = os.path.join(talker_model_path, "campplus.onnx")
        try:
            talker.set_spkemb_extractor(SpkembExtractor(campplus_path))
        except (ImportError, Exception) as exc:
            logger.warning("[TALKER_STREAM] SpkembExtractor not available: %s", exc)

        try:
            from talker_tn.talker_tn import TalkerTN

            talker.set_normalizer(TalkerTN())
        except ImportError:
            logger.warning(
                "[TALKER_STREAM] TalkerTN unavailable; using identity normalizer"
            )

        vae_path = str(Path(talker_model_path) / "vae")
        vae = None
        if Path(vae_path).exists():
            vae = AudioVAE.from_pretrained(vae_path, dtype=torch.bfloat16)
            vae.to(self._device)
            vae.eval()
        else:
            logger.warning("[TALKER_STREAM] AudioVAE not found at %s", vae_path)

        talker.initial_graph()
        self._talker = talker
        self._audio_detokenizer = vae

    def _remember_done(self, request_id: str) -> None:
        with self._state_lock:
            if self._is_inactive_locked(request_id):
                return
            self._done_seen[request_id] = None
            self._done_seen.move_to_end(request_id)
            if len(self._done_seen) > _DONE_SEEN_MAX:
                for _ in range(len(self._done_seen) - _DONE_SEEN_EVICT_TO):
                    self._done_seen.popitem(last=False)

    def _remember_aborted(self, request_id: str) -> None:
        self._aborted_seen[request_id] = None
        self._aborted_seen.move_to_end(request_id)
        if len(self._aborted_seen) > _ABORTED_SEEN_MAX:
            for _ in range(len(self._aborted_seen) - _ABORTED_SEEN_EVICT_TO):
                self._aborted_seen.popitem(last=False)

    def _remember_finished(self, request_id: str) -> None:
        self._finished_seen[request_id] = None
        self._finished_seen.move_to_end(request_id)
        if len(self._finished_seen) > _FINISHED_SEEN_MAX:
            for _ in range(len(self._finished_seen) - _FINISHED_SEEN_EVICT_TO):
                self._finished_seen.popitem(last=False)

    def _is_inactive(self, request_id: str) -> bool:
        with self._state_lock:
            return self._is_inactive_locked(request_id)

    def _is_inactive_locked(self, request_id: str) -> bool:
        return request_id in self._aborted_seen or request_id in self._finished_seen

    def _is_active_state(self, request_id: str, state: _RequestState) -> bool:
        with self._state_lock:
            return self._is_active_state_locked(request_id, state)

    def _is_active_state_locked(
        self,
        request_id: str,
        state: _RequestState,
    ) -> bool:
        return (
            self._state.get(request_id) is state
            and not state.abort_event.is_set()
            and request_id not in self._aborted_seen
            and not self._stopping
        )

    def _put_outgoing_if_active(
        self,
        request_id: str,
        state: _RequestState,
        message: OutgoingMessage,
    ) -> bool:
        with self._state_lock:
            if not self._is_active_state(request_id, state):
                return False
            self.outbox.put(message)
            return True

    def _put_final_if_active(
        self,
        request_id: str,
        state: _RequestState,
        message: OutgoingMessage,
    ) -> bool:
        with self._state_lock:
            if not self._is_active_state(request_id, state):
                return False
            self.outbox.put(message)
            self._state.pop(request_id, None)
            self._done_seen.pop(request_id, None)
            self._remember_finished(request_id)
            return True

    def _now_ms(self) -> int:
        return int(self._now_ms_fn())

    def _default_now_ms(self) -> int:
        return int(time.monotonic() * 1000)


MingStreamingTalkerScheduler = MingTalkerStreamScheduler
