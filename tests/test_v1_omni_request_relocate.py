# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Payload:
    embeds: torch.Tensor | None = None
    deepstack: list[object] | None = None
    extra: object | None = None


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


def test_relocate_replaces_tuple_attr_with_moved_tensor():
    from sglang_omni_v1.scheduling.omni_request_relocate import (
        relocate_request_tensors,
    )

    original_tuple = (torch.randn(4),)
    p = Payload(extra=original_tuple)

    relocate_request_tensors(p, torch.device("cpu"))

    assert p.extra is not original_tuple
    assert isinstance(p.extra, tuple)
    assert p.extra[0].device.type == "cpu"


def test_relocate_preserves_shared_tuple_identity():
    from sglang_omni_v1.scheduling.omni_request_relocate import (
        relocate_request_tensors,
    )

    shared_tuple = (torch.randn(4),)
    p = Payload(deepstack=[shared_tuple], extra=shared_tuple)

    relocate_request_tensors(p, torch.device("cpu"))

    assert p.extra is p.deepstack[0]
    assert p.extra is not shared_tuple
    assert p.extra[0].device.type == "cpu"


def test_relocate_preserves_tuple_list_cycle_backref():
    from sglang_omni_v1.scheduling.omni_request_relocate import (
        relocate_request_tensors,
    )

    holder = []
    p = Payload(deepstack=None, extra=(holder,))
    holder.append(p.extra)

    relocate_request_tensors(p, torch.device("cpu"))

    assert p.extra[0][0] is p.extra
