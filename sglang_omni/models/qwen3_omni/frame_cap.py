# SPDX-License-Identifier: Apache-2.0
"""Per-request cap on Talker codec frames, scaled by the text rows it was given."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TalkerFrameCap:
    floor: int
    per_text_token: int
    eos_id: int


def resolve_talker_frame_cap(
    *, floor: int, per_text_token: int, eos_id: int | None, vocab_size: int
) -> TalkerFrameCap | None:
    # Note (wenyao): HF configs spell "no codec EOS" as -1, which would unmask the
    # last codec id instead of EOS.
    if per_text_token <= 0 or eos_id is None or not 0 <= eos_id < vocab_size:
        return None
    return TalkerFrameCap(
        floor=max(int(floor), 0),
        per_text_token=int(per_text_token),
        eos_id=int(eos_id),
    )


def talker_frame_limit(data: Any) -> int | None:
    cap = data.talker_frame_cap
    # Note (wenyao): while text still streams the Talker eats one text row per frame
    # and stalls on an empty queue, so only the pad phase after it can run away.
    if cap is None or not data.thinker_chunks_done:
        return None
    appended_total = getattr(data.pending_text_queue, "appended_total", None)
    if appended_total is None:
        return None
    return cap.floor + cap.per_text_token * int(appended_total)
