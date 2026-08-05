# SPDX-License-Identifier: Apache-2.0
"""Shared test doubles."""

from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace
from typing import Any


class FakeExecutionBridge:
    """SGLangExecutionBridge double for scheduler-owned ModelRunner tests."""

    def __init__(self, device: object | None = None) -> None:
        import torch

        self.published: list[tuple[object, object]] = []
        self.isolate_sampling_calls: list[bool] = []
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.device_module = torch.get_device_module(self.device)

    @contextlib.contextmanager
    def forward_context(self, batch: object, *, isolate_sampling: bool = False):
        del batch
        self.isolate_sampling_calls.append(isolate_sampling)
        yield

    def publish_next_tokens(self, batch: object, next_token_ids: object) -> None:
        self.published.append((batch, next_token_ids))

    def record_completion(self):
        return self.device_module.Event()


class FakeServerArgs(SimpleNamespace):
    """ServerArgs double exposing the 0.5.16 override() mutation entry point."""

    def override(self, source: str, **fields: object) -> None:
        del source
        for name, value in fields.items():
            setattr(self, name, value)


def init_terminal_output_state(scheduler: Any) -> None:
    scheduler._request_admission_lock = threading.RLock()
    scheduler.is_entry_rank = True
    scheduler._model_runner = None
    scheduler._stream_output_builder = None
    scheduler._request_finished_callback = None
    scheduler._completed_request_ids = {}
    scheduler._pending_stream_ingress = {}
