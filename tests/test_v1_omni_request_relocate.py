# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Payload:
    embeds: torch.Tensor | None = None
    deepstack: list[torch.Tensor] | None = None


def test_relocate_moves_top_level_tensor():
    from sglang_omni_v1.scheduling.omni_request_relocate import (
        relocate_request_tensors,
    )

    p = Payload(embeds=torch.randn(4))
    relocate_request_tensors(p, torch.device("cpu"))
    assert p.embeds.device.type == "cpu"


def test_relocate_walks_nested_dict():
    from sglang_omni_v1.scheduling.omni_request_relocate import (
        relocate_request_tensors,
    )

    p = {"a": torch.randn(2), "b": {"c": torch.randn(3)}}
    relocate_request_tensors(p, torch.device("cpu"))
    assert p["a"].device.type == "cpu"
    assert p["b"]["c"].device.type == "cpu"


def test_relocate_walks_list_of_tensors():
    from sglang_omni_v1.scheduling.omni_request_relocate import (
        relocate_request_tensors,
    )

    p = Payload(deepstack=[torch.randn(4), torch.randn(4)])
    relocate_request_tensors(p, torch.device("cpu"))
    assert all(t.device.type == "cpu" for t in p.deepstack)


def test_relocate_handles_cycle():
    from sglang_omni_v1.scheduling.omni_request_relocate import (
        relocate_request_tensors,
    )

    p: dict = {"t": torch.randn(2)}
    p["self"] = p
    relocate_request_tensors(p, torch.device("cpu"))
    assert p["t"].device.type == "cpu"
