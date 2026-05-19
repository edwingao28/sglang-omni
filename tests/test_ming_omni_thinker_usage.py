from types import SimpleNamespace

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
