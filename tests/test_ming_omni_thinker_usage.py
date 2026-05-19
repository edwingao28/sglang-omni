from types import SimpleNamespace

import pytest

from sglang_omni.models.ming_omni.bootstrap import make_thinker_scheduler_adapters


def test_result_adapter_emits_token_counts():
    class _Tok:
        eos_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return ""

    _, result_adapter = make_thinker_scheduler_adapters(
        tokenizer=_Tok(), vocab_size=10, stage_name="thinker"
    )
    payload_stub = SimpleNamespace(
        request_id="r1",
        request=SimpleNamespace(),
        data={"prompt": {"input_ids": None, "attention_mask": None, "prompt_text": ""}},
    )
    data = SimpleNamespace(
        output_ids=[1, 2, 3],
        finish_reason="stop",
        extra_model_outputs={},
        stage_payload=payload_stub,
        input_ids=[5, 6, 7, 8],
    )
    out_payload = result_adapter(data)
    thinker_out = out_payload.data["engine_outputs"]["thinker"]
    assert thinker_out["prompt_tokens"] == 4
    assert thinker_out["completion_tokens"] == 3
    assert thinker_out["finish_reason"] == "stop"


def test_request_builder_rejects_overlong_prompt():
    import torch

    class _Tok:
        eos_token_id = 0

        def __call__(self, *a, **kw):
            return None

    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=_Tok(), vocab_size=10, stage_name="thinker"
    )
    payload = SimpleNamespace(
        request_id="r2",
        request=SimpleNamespace(params={"max_new_tokens": 8, "max_seq_len": 4}),
        data={
            "prompt": {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "attention_mask": None,
                "prompt_text": "",
            },
            "thinker_inputs": {},
        },
    )
    with pytest.raises(ValueError, match="longer than the model's context length"):
        request_builder(payload)


def test_request_builder_rejects_prompt_plus_max_new_tokens_at_context_limit():
    import torch

    class _Tok:
        eos_token_id = 0

        def __call__(self, *a, **kw):
            return None

    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=_Tok(), vocab_size=10, stage_name="thinker"
    )
    payload = SimpleNamespace(
        request_id="r3",
        request=SimpleNamespace(params={"max_new_tokens": 2, "max_seq_len": 5}),
        data={
            "prompt": {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": None,
                "prompt_text": "",
            },
            "thinker_inputs": {},
        },
    )
    with pytest.raises(ValueError, match="Requested token count exceeds"):
        request_builder(payload)


def test_request_builder_uses_configured_max_seq_len_when_request_omits_it():
    import torch

    class _Tok:
        eos_token_id = 0

        def __call__(self, *a, **kw):
            return None

    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=_Tok(), vocab_size=10, max_seq_len=4, stage_name="thinker"
    )
    payload = SimpleNamespace(
        request_id="r4",
        request=SimpleNamespace(params={"max_new_tokens": 1}),
        data={
            "prompt": {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "attention_mask": None,
                "prompt_text": "",
            },
            "thinker_inputs": {},
        },
    )
    with pytest.raises(ValueError, match="longer than the model's context length"):
        request_builder(payload)


def test_request_builder_prefers_explicit_request_max_seq_len():
    import torch

    class _Tok:
        eos_token_id = 0

        def __call__(self, *a, **kw):
            return None

    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=_Tok(), vocab_size=10, max_seq_len=4, stage_name="thinker"
    )
    payload = SimpleNamespace(
        request_id="r5",
        request=SimpleNamespace(params={"max_new_tokens": 1, "max_seq_len": 8}),
        data={
            "prompt": {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "attention_mask": None,
                "prompt_text": "",
            },
            "thinker_inputs": {},
        },
    )

    try:
        request_builder(payload)
    except ValueError as exc:
        pytest.fail(f"request max_seq_len should override configured limit: {exc}")
    except ImportError:
        pass


def test_request_builder_does_not_mutate_params_when_injecting_configured_max_seq_len():
    import torch

    class _Tok:
        eos_token_id = 0

        def __call__(self, *a, **kw):
            return None

    request_builder, _ = make_thinker_scheduler_adapters(
        tokenizer=_Tok(), vocab_size=10, max_seq_len=4, stage_name="thinker"
    )
    params = {"max_new_tokens": 1}
    payload = SimpleNamespace(
        request_id="r6",
        request=SimpleNamespace(params=params),
        data={
            "prompt": {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "attention_mask": None,
                "prompt_text": "",
            },
            "thinker_inputs": {},
        },
    )

    with pytest.raises(ValueError, match="longer than the model's context length"):
        request_builder(payload)
    assert params == {"max_new_tokens": 1}
