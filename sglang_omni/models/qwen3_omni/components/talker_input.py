# SPDX-License-Identifier: Apache-2.0
"""TalkerInputBuilder — construct HF-parity talker prefill inputs.

Both the per-request path and the pooled path go through ONE copy of the
segment plan and ONE copy of the assembly helpers.  The projections are the
only thing that differs: serially they run per segment, pooled they run once
for the whole batch.  Keeping the plan and the assembly shared is what makes
the two paths equivalent by construction rather than by two copies of the
mask/ordering logic staying accidentally in sync.
"""
from __future__ import annotations

import torch


def segment_chat_template(
    input_ids: torch.Tensor,
    *,
    im_start_token_id: int,
    system_token_id: int,
    user_token_id: int,
    assistant_token_id: int,
) -> list[dict]:
    """Parse input_ids into chat template segments by <|im_start|> boundaries.

    Returns list of {"role": str, "start": int, "end": int}.
    """
    role_map = {
        system_token_id: "system",
        user_token_id: "user",
        assistant_token_id: "assistant",
    }

    ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
    segments = []
    im_start_positions = [i for i, tok in enumerate(ids) if tok == im_start_token_id]

    for idx, pos in enumerate(im_start_positions):
        # Role token is the one after im_start
        role_token = ids[pos + 1] if pos + 1 < len(ids) else None
        role = role_map.get(role_token, "unknown")
        start = pos
        end = (
            im_start_positions[idx + 1]
            if idx + 1 < len(im_start_positions)
            else len(ids)
        )
        segments.append({"role": role, "start": start, "end": end})

    return segments


def plan_prefill_segments(
    thinker_input_ids: torch.Tensor,
    *,
    im_start_token_id: int,
    system_token_id: int,
    user_token_id: int,
    assistant_token_id: int,
    im_end_token_id: int | None = None,
) -> list[dict]:
    """Resolve which segments contribute rows, and which rows, before any math.

    Everything order- and mask-sensitive is decided here: system segments and
    every assistant segment but the last are dropped, and the ``<|im_end|>``
    strip is resolved into an explicit ``embed_end``.  The result is pure
    bookkeeping on token ids, so both build paths can share it.
    """
    segments = segment_chat_template(
        thinker_input_ids,
        im_start_token_id=im_start_token_id,
        system_token_id=system_token_id,
        user_token_id=user_token_id,
        assistant_token_id=assistant_token_id,
    )
    assistant_segment_indices = [
        idx for idx, seg in enumerate(segments) if seg["role"] == "assistant"
    ]
    last_assistant_idx = (
        assistant_segment_indices[-1] if assistant_segment_indices else None
    )

    plans: list[dict] = []
    for seg_idx, seg in enumerate(segments):
        start, end = seg["start"], seg["end"]
        if seg["role"] == "user":
            plans.append({"role": "user", "start": start, "end": end})
        elif seg["role"] == "assistant":
            if last_assistant_idx is not None and seg_idx != last_assistant_idx:
                continue
            # Strip <|im_end|> from the assistant segment to match HF, whose
            # thinker_embed never contains a hidden state for the EOS token
            # produced by generate().
            embed_end = end
            if (
                im_end_token_id is not None
                and end > start
                and int(thinker_input_ids[end - 1].item()) == im_end_token_id
            ):
                embed_end = end - 1
            plans.append(
                {"role": "assistant", "start": start, "end": end, "embed_end": embed_end}
            )
    return plans


def _resolve_out_size(projection, probe_row: torch.Tensor | None) -> int | None:
    """Output width of a projection without paying for a probe forward.

    The width is a property of the module, not of the call.  ``ResizeMLP``
    exposes it via ``out_features``; anything else falls back to a one-row
    forward, which is the only reason the probe still exists.
    """
    out_size = getattr(projection, "out_features", None)
    if out_size is not None:
        return int(out_size)
    if probe_row is None or probe_row.shape[0] == 0:
        return None
    return int(projection(probe_row[:1]).shape[-1])


def _scatter_user_part(
    *,
    text_rows: torch.Tensor,
    hidden_rows: torch.Tensor,
    multimodal_mask: torch.Tensor,
    out_size: int,
    device,
    dtype,
) -> torch.Tensor:
    """Place already-projected rows back under their mask, in mask order."""
    result = torch.empty(
        (multimodal_mask.shape[0], out_size),
        device=device,
        dtype=dtype,
    )
    # No `mask.any()` guards. The masks live on the accelerator, so each guard
    # was a device-to-host SYNC -- together ~10% of build time, measured. An
    # empty mask makes the gather, the projection and the scatter all no-ops on
    # zero rows, so dropping the guards is exactly equivalent and trades a sync
    # for a launch on the empty branch.
    result[multimodal_mask] = hidden_rows
    text_mask = ~multimodal_mask
    result[text_mask] = text_rows
    return result


def build_user_part(
    *,
    thinker_embed: torch.Tensor,
    thinker_hidden: torch.Tensor,
    multimodal_mask: torch.Tensor,
    text_projection,
    hidden_projection,
) -> torch.Tensor:
    """Build user segment: text_projection for text, hidden_projection for multimodal."""
    out_size = _resolve_out_size(text_projection, thinker_embed)
    if out_size is None:
        out_size = text_projection(thinker_embed[:1]).shape[-1]
    hidden_rows = hidden_projection(thinker_hidden[multimodal_mask])
    text_rows = text_projection(thinker_embed[~multimodal_mask])
    return _scatter_user_part(
        text_rows=text_rows,
        hidden_rows=hidden_rows,
        multimodal_mask=multimodal_mask,
        out_size=out_size,
        device=thinker_embed.device,
        dtype=thinker_embed.dtype,
    )


def codec_special_ids(
    *,
    speaker_id: int,
    codec_nothink_id: int,
    codec_think_bos_id: int,
    codec_think_eos_id: int,
    codec_pad_id: int,
    codec_bos_id: int,
    device,
) -> torch.Tensor:
    return torch.tensor(
        [
            codec_nothink_id,
            codec_think_bos_id,
            codec_think_eos_id,
            speaker_id,
            codec_pad_id,
            codec_bos_id,
        ],
        device=device,
        dtype=torch.long,
    )


def _assemble_assistant_part(
    *,
    projected: torch.Tensor,
    codec_embeds: torch.Tensor,
    tts_bos_embed: torch.Tensor,
    tts_eos_embed: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    tts_pad_token_id: int,
    device,
    dtype,
) -> dict[str, torch.Tensor]:
    """Assemble the 9-row assistant prompt tail from already-projected rows."""
    # Text side: [first 3 rows] + [4x pad] + [bos] + [4th row] = 9 rows.
    # The segment starts at <|im_start|>, so rows 0-2 are the generation prompt
    # (``<|im_start|>``, ``assistant``, ``\n``) and row 3 is the FIRST thinker
    # text token. One thinker chunk is therefore enough to assemble the tail;
    # a shorter segment cannot be padded into shape, so say so instead of
    # letting the tensor add report a broadcast mismatch.
    if projected.shape[0] < 3:
        raise RuntimeError(
            "talker assistant segment needs at least 3 rows (the "
            "<|im_start|>assistant chat-template prefix) to assemble the 9-row "
            f"prompt tail; got {projected.shape[0]}"
        )
    fourth_token = (
        projected[3:4]
        if projected.shape[0] > 3
        else torch.zeros((1, projected.shape[-1]), device=device, dtype=dtype)
    )
    text_hidden = torch.cat(
        [
            projected[:3],
            tts_pad_embed.expand(4, -1),
            tts_bos_embed,
            fourth_token,
        ],
        dim=0,
    )  # [9, hidden]

    # Codec side: [3x zeros] + [embed(6 special tokens)]
    codec_hidden = torch.cat(
        [
            torch.zeros((3, text_hidden.shape[-1]), device=device, dtype=dtype),
            codec_embeds,
        ],
        dim=0,
    )  # [9, hidden]

    input_embeds = text_hidden + codec_hidden

    input_ids = torch.full(
        (text_hidden.shape[0],),
        tts_pad_token_id,
        dtype=torch.long,
        device=device,
    )

    # HF's trailing_text_hidden is a FIFO stream of future assistant text rows:
    # assistant tokens after the first spoken token, then TTS EOS.
    if projected.shape[0] > 4:
        future_text_rows = torch.cat([projected[4:], tts_eos_embed], dim=0)
    else:
        future_text_rows = tts_eos_embed.clone()

    return {
        "input_embeds": input_embeds,
        "input_ids": input_ids,
        "future_text_rows": future_text_rows,
    }


def build_assistant_part(
    *,
    assistant_embed: torch.Tensor,
    text_projection,
    codec_embed_fn,
    tts_bos_embed: torch.Tensor,
    tts_eos_embed: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    speaker_id: int,
    codec_nothink_id: int,
    codec_think_bos_id: int,
    codec_think_eos_id: int,
    codec_pad_id: int,
    codec_bos_id: int,
    tts_pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Build assistant segment matching HF's _get_talker_assistant_parts."""
    device = assistant_embed.device
    dtype = assistant_embed.dtype

    special_ids = codec_special_ids(
        speaker_id=speaker_id,
        codec_nothink_id=codec_nothink_id,
        codec_think_bos_id=codec_think_bos_id,
        codec_think_eos_id=codec_think_eos_id,
        codec_pad_id=codec_pad_id,
        codec_bos_id=codec_bos_id,
        device=device,
    )
    return _assemble_assistant_part(
        projected=text_projection(assistant_embed),  # [N, hidden]
        codec_embeds=codec_embed_fn(special_ids),  # [6, hidden]
        tts_bos_embed=tts_bos_embed,
        tts_eos_embed=tts_eos_embed,
        tts_pad_embed=tts_pad_embed,
        tts_pad_token_id=tts_pad_token_id,
        device=device,
        dtype=dtype,
    )


def _trim_assistant_eos(
    future_text_rows: torch.Tensor | None, include_assistant_eos: bool
) -> torch.Tensor | None:
    if (
        not include_assistant_eos
        and future_text_rows is not None
        and future_text_rows.shape[0] > 0
    ):
        return future_text_rows[:-1]
    return future_text_rows


def build_prefill_input(
    *,
    thinker_embed: torch.Tensor,
    thinker_hidden: torch.Tensor,
    thinker_input_ids: torch.Tensor,
    multimodal_mask: torch.Tensor,
    text_projection,
    hidden_projection,
    codec_embed_fn,
    tts_bos_embed: torch.Tensor,
    tts_eos_embed: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    im_start_token_id: int,
    system_token_id: int,
    user_token_id: int,
    assistant_token_id: int,
    speaker_id: int,
    codec_nothink_id: int,
    codec_think_bos_id: int,
    codec_think_eos_id: int,
    codec_pad_id: int,
    codec_bos_id: int,
    tts_pad_token_id: int,
    include_assistant_eos: bool = True,
    im_end_token_id: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build full talker prefill input from thinker outputs.

    When *im_end_token_id* is provided, the ``<|im_end|>`` token is stripped
    from the assistant segment before projection.  HF's ``thinker_embed``
    never contains a hidden state for the EOS token (``generate()`` stops
    before emitting one), so including the raw embedding introduces an
    off-by-one that shifts the future text-row queue.
    """
    plans = plan_prefill_segments(
        thinker_input_ids,
        im_start_token_id=im_start_token_id,
        system_token_id=system_token_id,
        user_token_id=user_token_id,
        assistant_token_id=assistant_token_id,
        im_end_token_id=im_end_token_id,
    )

    all_embeds = []
    all_ids = []
    future_text_rows = None

    for plan in plans:
        start, end = plan["start"], plan["end"]
        if plan["role"] == "user":
            all_embeds.append(
                build_user_part(
                    thinker_embed=thinker_embed[start:end],
                    thinker_hidden=thinker_hidden[start:end],
                    multimodal_mask=multimodal_mask[start:end],
                    text_projection=text_projection,
                    hidden_projection=hidden_projection,
                )
            )
            all_ids.append(thinker_input_ids[start:end].to(dtype=torch.long))
        else:
            assistant_result = build_assistant_part(
                assistant_embed=thinker_embed[start : plan["embed_end"]],
                text_projection=text_projection,
                codec_embed_fn=codec_embed_fn,
                tts_bos_embed=tts_bos_embed,
                tts_eos_embed=tts_eos_embed,
                tts_pad_embed=tts_pad_embed,
                speaker_id=speaker_id,
                codec_nothink_id=codec_nothink_id,
                codec_think_bos_id=codec_think_bos_id,
                codec_think_eos_id=codec_think_eos_id,
                codec_pad_id=codec_pad_id,
                codec_bos_id=codec_bos_id,
                tts_pad_token_id=tts_pad_token_id,
            )
            all_embeds.append(assistant_result["input_embeds"])
            all_ids.append(
                assistant_result["input_ids"].to(
                    device=thinker_input_ids.device,
                    dtype=torch.long,
                )
            )
            future_text_rows = _trim_assistant_eos(
                assistant_result["future_text_rows"], include_assistant_eos
            )

    return {
        "input_embeds": torch.cat(all_embeds, dim=0),
        "input_ids": torch.cat(all_ids, dim=0),
        "future_text_rows": future_text_rows,
    }


def _pooled_apply(projection, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """One projection call for every collected row group, split back in order."""
    if not tensors:
        return []
    sizes = [int(tensor.shape[0]) for tensor in tensors]
    pooled = projection(torch.cat(tensors, dim=0))
    return list(torch.split(pooled, sizes, dim=0))


def build_prefill_input_batch(
    items: list[dict],
    *,
    text_projection,
    hidden_projection,
    codec_embed_fn,
    im_start_token_id: int,
    system_token_id: int,
    user_token_id: int,
    assistant_token_id: int,
    codec_nothink_id: int,
    codec_think_bos_id: int,
    codec_think_eos_id: int,
    codec_pad_id: int,
    codec_bos_id: int,
    tts_pad_token_id: int,
    im_end_token_id: int | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Pooled sibling of :func:`build_prefill_input`.

    Rows from every request AND every segment are concatenated into a single
    ``text_projection`` call and a single ``hidden_projection`` call, then split
    back in exact order under their original masks.  ``items`` carries the
    per-request tensors and the two per-request scalars (speaker_id,
    include_assistant_eos); everything else is shared by the batch.

    The projections are row-independent (ResizeMLP is Linear-SiLU-Linear), so
    pooling cannot change a value in exact arithmetic; it is bit-exact in the
    talker's bf16 on device as well, which the equivalence suite asserts.
    """
    # Phase 1 -- plan every request and collect the rows each projection owes,
    # without running any of them.
    collected: list[list[dict]] = []
    text_inputs: list[torch.Tensor] = []
    hidden_inputs: list[torch.Tensor] = []
    codec_inputs: list[torch.Tensor] = []

    for item in items:
        thinker_embed = item["thinker_embed"]
        thinker_hidden = item["thinker_hidden"]
        thinker_input_ids = item["thinker_input_ids"]
        multimodal_mask = item["multimodal_mask"]

        plans = plan_prefill_segments(
            thinker_input_ids,
            im_start_token_id=im_start_token_id,
            system_token_id=system_token_id,
            user_token_id=user_token_id,
            assistant_token_id=assistant_token_id,
            im_end_token_id=im_end_token_id,
        )

        records: list[dict] = []
        for plan in plans:
            start, end = plan["start"], plan["end"]
            if plan["role"] == "user":
                seg_mask = multimodal_mask[start:end]
                text_slot = len(text_inputs)
                text_inputs.append(thinker_embed[start:end][~seg_mask])
                hidden_slot = len(hidden_inputs)
                hidden_inputs.append(thinker_hidden[start:end][seg_mask])
                records.append(
                    {
                        "role": "user",
                        "mask": seg_mask,
                        "text_slot": text_slot,
                        "hidden_slot": hidden_slot,
                        "start": start,
                        "end": end,
                    }
                )
            else:
                text_slot = len(text_inputs)
                text_inputs.append(thinker_embed[start : plan["embed_end"]])
                codec_slot = len(codec_inputs)
                codec_inputs.append(
                    codec_special_ids(
                        speaker_id=item["speaker_id"],
                        codec_nothink_id=codec_nothink_id,
                        codec_think_bos_id=codec_think_bos_id,
                        codec_think_eos_id=codec_think_eos_id,
                        codec_pad_id=codec_pad_id,
                        codec_bos_id=codec_bos_id,
                        device=thinker_embed.device,
                    )
                )
                records.append(
                    {"role": "assistant", "text_slot": text_slot, "codec_slot": codec_slot}
                )
        collected.append(records)

    # Phase 2 -- one call per projection for the whole batch.
    text_outputs = _pooled_apply(text_projection, text_inputs)
    hidden_outputs = _pooled_apply(hidden_projection, hidden_inputs)
    codec_outputs = (
        list(
            torch.split(
                codec_embed_fn(torch.cat(codec_inputs, dim=0)),
                [int(ids.shape[0]) for ids in codec_inputs],
                dim=0,
            )
        )
        if codec_inputs
        else []
    )

    out_size = _resolve_out_size(
        text_projection, text_inputs[0] if text_inputs else None
    )
    if out_size is None and text_outputs:
        out_size = int(text_outputs[0].shape[-1])

    # Phase 3 -- assemble each request from its own slices, in its own order.
    results: list[dict[str, torch.Tensor]] = []
    for item, records in zip(items, collected):
        thinker_embed = item["thinker_embed"]
        thinker_input_ids = item["thinker_input_ids"]
        all_embeds = []
        all_ids = []
        future_text_rows = None

        for record in records:
            if record["role"] == "user":
                all_embeds.append(
                    _scatter_user_part(
                        text_rows=text_outputs[record["text_slot"]],
                        hidden_rows=hidden_outputs[record["hidden_slot"]],
                        multimodal_mask=record["mask"],
                        out_size=out_size,
                        device=thinker_embed.device,
                        dtype=thinker_embed.dtype,
                    )
                )
                all_ids.append(
                    thinker_input_ids[record["start"] : record["end"]].to(
                        dtype=torch.long
                    )
                )
            else:
                assistant_result = _assemble_assistant_part(
                    projected=text_outputs[record["text_slot"]],
                    codec_embeds=codec_outputs[record["codec_slot"]],
                    tts_bos_embed=item["tts_bos_embed"],
                    tts_eos_embed=item["tts_eos_embed"],
                    tts_pad_embed=item["tts_pad_embed"],
                    tts_pad_token_id=tts_pad_token_id,
                    device=thinker_embed.device,
                    dtype=thinker_embed.dtype,
                )
                all_embeds.append(assistant_result["input_embeds"])
                all_ids.append(
                    assistant_result["input_ids"].to(
                        device=thinker_input_ids.device,
                        dtype=torch.long,
                    )
                )
                future_text_rows = _trim_assistant_eos(
                    assistant_result["future_text_rows"],
                    item["include_assistant_eos"],
                )

        results.append(
            {
                "input_embeds": torch.cat(all_embeds, dim=0),
                "input_ids": torch.cat(all_ids, dim=0),
                "future_text_rows": future_text_rows,
            }
        )

    return results
