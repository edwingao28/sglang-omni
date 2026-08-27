# SPDX-License-Identifier: Apache-2.0
"""Exact-equivalence tests for pooled talker request builds.

The bar is EXACT equality, not tolerance: pooling rows from several requests
into one projection call must not move a single bit relative to building each
request on its own.

TWO KINDS OF TEST LIVE HERE, and the split is deliberate.

*Structural* tests use a projection that is elementwise, so its output cannot
depend on how many rows were passed at once. Any inequality they report is a
real ordering, masking or segmentation bug in the pooled path -- which is the
entire risk surface of the restructure.

*Numeric* tests use a real Linear-SiLU-Linear at the talker's own dimensions
and dtype on device. A GEMM may select a different kernel as M changes, so
bit-exactness across a batching change is a property worth asserting rather
than assuming. It holds for the talker's bf16 on device (probed across split
patterns before this code was written); it does NOT hold for fp32 or on CPU,
where kernel selection differs. That is a property of GEMM kernels and exists
independently of this change, which is why the structural tests do not use a
real GEMM to prove ordering.
"""
from __future__ import annotations

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.talker_input import (
    build_prefill_input,
    build_prefill_input_batch,
    plan_prefill_segments,
)

IM_START, SYSTEM, USER, ASSISTANT, IM_END = 101, 102, 103, 104, 105
AUDIO_TOKEN = 106
IN_DIM, OUT_DIM = 12, 8
CODEC_VOCAB = 64

TOKEN_IDS = dict(
    im_start_token_id=IM_START,
    system_token_id=SYSTEM,
    user_token_id=USER,
    assistant_token_id=ASSISTANT,
)
CODEC_IDS = dict(
    codec_nothink_id=1,
    codec_think_bos_id=2,
    codec_think_eos_id=3,
    codec_pad_id=4,
    codec_bos_id=5,
    tts_pad_token_id=9,
)


class StableProjection(torch.nn.Module):
    """Row-independent projection whose result cannot depend on row count.

    Elementwise by construction, so pooling is bit-exact for reasons that have
    nothing to do with kernel selection.
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.scale = torch.randn(out_dim, generator=generator)
        self.shift = torch.randn(out_dim, generator=generator)
        self.out_features = out_dim
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x[:, : self.out_features] * self.scale + self.shift


class RealProjection(torch.nn.Module):
    """Shape-faithful stand-in for ResizeMLP: Linear-SiLU-Linear."""

    def __init__(self, in_dim, inter_dim, out_dim, dtype, device, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = torch.nn.Linear(in_dim, inter_dim, dtype=dtype, device=device)
        self.act = torch.nn.SiLU()
        self.fc2 = torch.nn.Linear(inter_dim, out_dim, dtype=dtype, device=device)
        self.out_features = out_dim
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.fc2(self.act(self.fc1(x)))


class CodecEmbed:
    """Embedding lookup: a pure gather, exact under any batching."""

    def __init__(self, out_dim, dtype=torch.float32, device="cpu", seed=7):
        generator = torch.Generator().manual_seed(seed)
        table = torch.randn(CODEC_VOCAB, out_dim, generator=generator)
        self.table = table.to(dtype=dtype, device=device)
        self.calls = 0

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.table[ids]


def make_case(
    *,
    user_segments: int = 1,
    user_len: int = 6,
    assistant_len: int = 6,
    mask_kind: str = "none",
    with_system: bool = True,
    trailing_im_end: bool = True,
    with_assistant: bool = True,
    extra_assistant_segments: int = 0,
    dtype=torch.float32,
    device="cpu",
    seed: int = 0,
) -> dict:
    """Build one synthetic request with an explicit chat-template layout."""
    ids: list[int] = []
    if with_system:
        ids += [IM_START, SYSTEM] + [200 + i for i in range(3)]
    for segment in range(user_segments):
        body = []
        for position in range(user_len):
            if mask_kind == "all":
                body.append(AUDIO_TOKEN)
            elif mask_kind == "mixed" and position % 3 == 0:
                body.append(AUDIO_TOKEN)
            else:
                body.append(300 + segment * 20 + position)
        ids += [IM_START, USER] + body
    # Assistant segments that are NOT last must be dropped by the planner.
    for extra in range(extra_assistant_segments):
        ids += [IM_START, ASSISTANT] + [400 + extra * 10 + i for i in range(4)] + [IM_END]
    if with_assistant:
        ids += [IM_START, ASSISTANT] + [500 + i for i in range(assistant_len)]
        if trailing_im_end:
            ids += [IM_END]

    input_ids = torch.tensor(ids, dtype=torch.long)
    n = input_ids.shape[0]
    generator = torch.Generator().manual_seed(seed)
    embed = torch.randn(n, IN_DIM, generator=generator).to(dtype=dtype, device=device)
    hidden = torch.randn(n, IN_DIM, generator=generator).to(dtype=dtype, device=device)
    mask = (input_ids == AUDIO_TOKEN).to(device=device)
    return {
        "thinker_embed": embed,
        "thinker_hidden": hidden,
        "thinker_input_ids": input_ids,
        "multimodal_mask": mask,
    }


def run_serial(case, *, text_projection, hidden_projection, codec_embed_fn, specials,
               speaker_id=11, include_assistant_eos=True):
    tts_bos, tts_eos, tts_pad = specials
    return build_prefill_input(
        thinker_embed=case["thinker_embed"],
        thinker_hidden=case["thinker_hidden"],
        thinker_input_ids=case["thinker_input_ids"],
        multimodal_mask=case["multimodal_mask"],
        text_projection=text_projection,
        hidden_projection=hidden_projection,
        codec_embed_fn=codec_embed_fn,
        tts_bos_embed=tts_bos,
        tts_eos_embed=tts_eos,
        tts_pad_embed=tts_pad,
        speaker_id=speaker_id,
        include_assistant_eos=include_assistant_eos,
        im_end_token_id=IM_END,
        **TOKEN_IDS,
        **CODEC_IDS,
    )


def run_batch(cases, *, text_projection, hidden_projection, codec_embed_fn, specials,
              speaker_ids=None, include_flags=None):
    tts_bos, tts_eos, tts_pad = specials
    speaker_ids = speaker_ids or [11] * len(cases)
    include_flags = include_flags if include_flags is not None else [True] * len(cases)
    items = [
        {
            **case,
            "tts_bos_embed": tts_bos,
            "tts_eos_embed": tts_eos,
            "tts_pad_embed": tts_pad,
            "speaker_id": speaker_ids[index],
            "include_assistant_eos": include_flags[index],
        }
        for index, case in enumerate(cases)
    ]
    return build_prefill_input_batch(
        items,
        text_projection=text_projection,
        hidden_projection=hidden_projection,
        codec_embed_fn=codec_embed_fn,
        im_end_token_id=IM_END,
        **TOKEN_IDS,
        **CODEC_IDS,
    )


def stable_kit(dtype=torch.float32, device="cpu"):
    text = StableProjection(IN_DIM, OUT_DIM, seed=1).to(device)
    hidden = StableProjection(IN_DIM, OUT_DIM, seed=2).to(device)
    codec = CodecEmbed(OUT_DIM, dtype=dtype, device=device)
    generator = torch.Generator().manual_seed(3)
    specials = tuple(
        torch.randn(1, OUT_DIM, generator=generator).to(dtype=dtype, device=device)
        for _ in range(3)
    )
    return text, hidden, codec, specials


def assert_same(got: dict, want: dict, label: str) -> None:
    assert got["input_embeds"].shape == want["input_embeds"].shape, label
    assert got["input_embeds"].dtype == want["input_embeds"].dtype, label
    assert torch.equal(got["input_embeds"], want["input_embeds"]), f"{label}: embeds"
    assert torch.equal(got["input_ids"], want["input_ids"]), f"{label}: ids"
    if want["future_text_rows"] is None:
        assert got["future_text_rows"] is None, f"{label}: future should be None"
    else:
        assert got["future_text_rows"] is not None, f"{label}: future missing"
        assert torch.equal(
            got["future_text_rows"], want["future_text_rows"]
        ), f"{label}: future"


CASE_MATRIX = {
    "text-only-single": dict(user_segments=1, mask_kind="none"),
    "text-only-long": dict(user_segments=1, user_len=64, assistant_len=40),
    "multimodal-all": dict(user_segments=1, mask_kind="all"),
    "multimodal-mixed": dict(user_segments=1, mask_kind="mixed"),
    "multi-segment": dict(user_segments=3, mask_kind="mixed"),
    "multi-segment-long": dict(user_segments=4, user_len=33, mask_kind="mixed"),
    "no-system": dict(with_system=False, mask_kind="mixed"),
    "no-trailing-im-end": dict(trailing_im_end=False),
    "short-assistant": dict(assistant_len=3),
    "extra-assistant-segments": dict(extra_assistant_segments=2, mask_kind="mixed"),
    "user-only": dict(with_assistant=False, mask_kind="mixed"),
}


@pytest.mark.parametrize("name", sorted(CASE_MATRIX))
def test_single_request_batch_matches_serial(name):
    """A batch of one must be bit-identical to the serial build."""
    case = make_case(seed=hash(name) % 1000, **CASE_MATRIX[name])
    text, hidden, codec, specials = stable_kit()
    want = run_serial(
        case, text_projection=text, hidden_projection=hidden,
        codec_embed_fn=codec, specials=specials,
    )
    text2, hidden2, codec2, _ = stable_kit()
    got = run_batch(
        [case], text_projection=text2, hidden_projection=hidden2,
        codec_embed_fn=codec2, specials=specials,
    )[0]
    assert_same(got, want, name)


@pytest.mark.parametrize("batch_size", [2, 8, 32])
def test_homogeneous_batch_matches_serial(batch_size):
    cases = [
        make_case(user_segments=2, mask_kind="mixed", seed=index)
        for index in range(batch_size)
    ]
    text, hidden, codec, specials = stable_kit()
    want = [
        run_serial(case, text_projection=text, hidden_projection=hidden,
                   codec_embed_fn=codec, specials=specials)
        for case in cases
    ]
    text2, hidden2, codec2, _ = stable_kit()
    got = run_batch(cases, text_projection=text2, hidden_projection=hidden2,
                    codec_embed_fn=codec2, specials=specials)
    for index, (g, w) in enumerate(zip(got, want)):
        assert_same(g, w, f"batch{batch_size}[{index}]")


def test_heterogeneous_batch_matches_serial():
    """The case that actually breaks ordering: every shape in one batch."""
    names = sorted(CASE_MATRIX)
    cases = [
        make_case(seed=index, **CASE_MATRIX[name])
        for index, name in enumerate(names)
    ]
    speaker_ids = [10 + index for index in range(len(cases))]
    include_flags = [index % 2 == 0 for index in range(len(cases))]

    text, hidden, codec, specials = stable_kit()
    want = [
        run_serial(case, text_projection=text, hidden_projection=hidden,
                   codec_embed_fn=codec, specials=specials,
                   speaker_id=speaker_ids[index],
                   include_assistant_eos=include_flags[index])
        for index, case in enumerate(cases)
    ]
    text2, hidden2, codec2, _ = stable_kit()
    got = run_batch(cases, text_projection=text2, hidden_projection=hidden2,
                    codec_embed_fn=codec2, specials=specials,
                    speaker_ids=speaker_ids, include_flags=include_flags)
    for index, (g, w) in enumerate(zip(got, want)):
        assert_same(g, w, f"hetero[{names[index]}]")


def test_reversed_batch_order_gives_same_per_request_results():
    """Position within the batch must not change a request's own output."""
    names = sorted(CASE_MATRIX)
    cases = [make_case(seed=i, **CASE_MATRIX[n]) for i, n in enumerate(names)]

    text, hidden, codec, specials = stable_kit()
    forward = run_batch(cases, text_projection=text, hidden_projection=hidden,
                        codec_embed_fn=codec, specials=specials)
    text2, hidden2, codec2, _ = stable_kit()
    backward = run_batch(list(reversed(cases)), text_projection=text2,
                         hidden_projection=hidden2, codec_embed_fn=codec2,
                         specials=specials)
    for index in range(len(cases)):
        assert_same(backward[len(cases) - 1 - index], forward[index], f"rev[{index}]")


def test_batch_makes_exactly_one_call_per_projection():
    """Proof the batch is pooled, not a loop wearing a batch's clothes."""
    cases = [make_case(user_segments=3, mask_kind="mixed", seed=i) for i in range(16)]
    text, hidden, codec, specials = stable_kit()
    run_batch(cases, text_projection=text, hidden_projection=hidden,
              codec_embed_fn=codec, specials=specials)
    assert text.calls == 1, f"text projection called {text.calls} times, expected 1"
    assert hidden.calls == 1, f"hidden projection called {hidden.calls} times"
    assert codec.calls == 1, f"codec embed called {codec.calls} times"


def test_serial_path_still_calls_per_segment():
    """Guard the control arm: the serial path is unchanged, not secretly pooled."""
    case = make_case(user_segments=3, mask_kind="mixed", seed=1)
    text, hidden, codec, specials = stable_kit()
    run_serial(case, text_projection=text, hidden_projection=hidden,
               codec_embed_fn=codec, specials=specials)
    # three user segments + one assistant segment
    assert text.calls == 4
    assert hidden.calls == 3


def test_empty_batch_returns_empty():
    text, hidden, codec, specials = stable_kit()
    assert run_batch([], text_projection=text, hidden_projection=hidden,
                     codec_embed_fn=codec, specials=specials) == []


def test_too_short_assistant_raises_in_both_paths():
    """A 2-row assistant segment cannot make the 9-row tail; both paths say so."""
    ids = [IM_START, USER, 300, 301] + [IM_START, ASSISTANT]
    input_ids = torch.tensor(ids, dtype=torch.long)
    generator = torch.Generator().manual_seed(5)
    case = {
        "thinker_embed": torch.randn(len(ids), IN_DIM, generator=generator),
        "thinker_hidden": torch.randn(len(ids), IN_DIM, generator=generator),
        "thinker_input_ids": input_ids,
        "multimodal_mask": torch.zeros(len(ids), dtype=torch.bool),
    }
    text, hidden, codec, specials = stable_kit()
    with pytest.raises(RuntimeError, match="at least 3 rows"):
        run_serial(case, text_projection=text, hidden_projection=hidden,
                   codec_embed_fn=codec, specials=specials)
    with pytest.raises(RuntimeError, match="at least 3 rows"):
        run_batch([case], text_projection=text, hidden_projection=hidden,
                  codec_embed_fn=codec, specials=specials)


def test_planner_keeps_only_the_last_assistant_segment():
    case = make_case(extra_assistant_segments=2, user_segments=2)
    plans = plan_prefill_segments(
        case["thinker_input_ids"], im_end_token_id=IM_END, **TOKEN_IDS
    )
    roles = [plan["role"] for plan in plans]
    assert roles.count("assistant") == 1
    assert roles.count("user") == 2
    assert "system" not in roles
    assert roles[-1] == "assistant"


def test_planner_resolves_the_im_end_strip():
    with_end = make_case(trailing_im_end=True)
    without_end = make_case(trailing_im_end=False)
    plan_with = [
        p for p in plan_prefill_segments(
            with_end["thinker_input_ids"], im_end_token_id=IM_END, **TOKEN_IDS)
        if p["role"] == "assistant"
    ][0]
    plan_without = [
        p for p in plan_prefill_segments(
            without_end["thinker_input_ids"], im_end_token_id=IM_END, **TOKEN_IDS)
        if p["role"] == "assistant"
    ][0]
    assert plan_with["embed_end"] == plan_with["end"] - 1
    assert plan_without["embed_end"] == plan_without["end"]


# --- numeric gate: the real projection, the real dtype, on device -----------

REAL_IN, REAL_INTER, REAL_OUT = 2048, 2048, 1024


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("batch_size", [2, 8, 32])
def test_pooled_projection_is_bit_exact_on_device_in_bf16(batch_size):
    """Rider-1 gate: pooling must not move a bit at the talker's real dims.

    A GEMM can switch kernel on M, so this is asserted rather than assumed.
    """
    device, dtype = "cuda", torch.bfloat16
    global IN_DIM, OUT_DIM
    saved_in, saved_out = IN_DIM, OUT_DIM
    IN_DIM, OUT_DIM = REAL_IN, REAL_OUT
    try:
        cases = [
            make_case(user_segments=1 + index % 3, user_len=8 + index,
                      mask_kind=["none", "mixed", "all"][index % 3],
                      dtype=dtype, device=device, seed=index)
            for index in range(batch_size)
        ]
        text = RealProjection(REAL_IN, REAL_INTER, REAL_OUT, dtype, device, seed=1)
        hidden = RealProjection(REAL_IN, REAL_INTER, REAL_OUT, dtype, device, seed=2)
        codec = CodecEmbed(REAL_OUT, dtype=dtype, device=device)
        generator = torch.Generator().manual_seed(3)
        specials = tuple(
            torch.randn(1, REAL_OUT, generator=generator).to(dtype=dtype, device=device)
            for _ in range(3)
        )
        want = [
            run_serial(case, text_projection=text, hidden_projection=hidden,
                       codec_embed_fn=codec, specials=specials)
            for case in cases
        ]
        got = run_batch(cases, text_projection=text, hidden_projection=hidden,
                        codec_embed_fn=codec, specials=specials)
        for index, (g, w) in enumerate(zip(got, want)):
            assert_same(g, w, f"cuda-bf16[{index}]")
    finally:
        IN_DIM, OUT_DIM = saved_in, saved_out
