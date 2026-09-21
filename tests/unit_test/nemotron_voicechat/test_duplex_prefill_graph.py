# SPDX-License-Identifier: Apache-2.0

from bisect import bisect_left

from sglang_omni.models.nemotron_voicechat import CAPABILITIES
from sglang_omni.models.nemotron_voicechat.duplex_stages import (
    TALKER_PREFILL_CUDA_GRAPH_BS,
    TalkerBuilder,
    ThinkerBuilder,
)
from sglang_omni.models.nemotron_voicechat.engine_builder import (
    NemotronVoiceChatEngineBuilder,
    NemotronVoiceChatTalkerEngineBuilder,
)
from sglang_omni.models.nemotron_voicechat.talker import NemotronVoiceChatTalker
from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    build_generation_batch_overrides,
)

TALKER_EXTEND_SHAPES = (1, 38)


def _replayed_bucket(tokens):
    ladder = TALKER_PREFILL_CUDA_GRAPH_BS
    index = bisect_left(ladder, tokens)
    if index >= len(ladder):
        return None
    bucket = ladder[index]
    return bucket if bucket <= 2 * tokens else None


def test_one_token_extend_replays_instead_of_falling_back():

    assert TALKER_PREFILL_CUDA_GRAPH_BS[0] == 1
    assert _replayed_bucket(1) == 1


def test_every_talker_extend_shape_replays():
    assert all(_replayed_bucket(n) is not None for n in TALKER_EXTEND_SHAPES)


def test_ladder_is_sorted_and_unique():

    assert sorted(set(TALKER_PREFILL_CUDA_GRAPH_BS)) == list(
        TALKER_PREFILL_CUDA_GRAPH_BS
    )


def test_talker_builder_opts_in():
    builder = TalkerBuilder()
    defaults = builder.generation_defaults(dtype="bfloat16")
    assert builder.supports_breakable_prefill_cuda_graph is True
    assert defaults["disable_cuda_graph"] is False
    assert defaults["cuda_graph_backend_prefill"] == CudaGraphBackend.BREAKABLE
    assert defaults["cuda_graph_bs_prefill"][0] == 1


def test_thinker_builder_stays_eager():

    builder = ThinkerBuilder()
    defaults = builder.generation_defaults(dtype="bfloat16")
    assert getattr(builder, "supports_breakable_prefill_cuda_graph", False) is False
    assert defaults["disable_cuda_graph"] is True
    assert "cuda_graph_backend_prefill" not in defaults
    assert "cuda_graph_bs_prefill" not in defaults


def test_talker_decode_capture_stays_disabled():

    defaults = TalkerBuilder().generation_defaults(dtype="bfloat16")
    assert defaults["disable_decode_cuda_graph"] is True


def test_operator_disable_still_wins_over_talker_defaults():

    overrides = build_generation_batch_overrides(
        server_args_overrides={"disable_cuda_graph": True},
        **TalkerBuilder().generation_defaults(dtype="bfloat16"),
    )
    assert overrides["disable_cuda_graph"] is True
    assert overrides["cuda_graph_backend_prefill"] == CudaGraphBackend.DISABLED
    assert "cuda_graph_bs_prefill" not in overrides


def test_talker_defaults_keep_breakable_prefill_without_operator_override():
    overrides = build_generation_batch_overrides(
        **TalkerBuilder().generation_defaults(dtype="bfloat16"),
    )
    assert overrides["cuda_graph_backend_prefill"] == CudaGraphBackend.BREAKABLE
    assert overrides["cuda_graph_bs_prefill"][0] == 1


def test_talker_exposes_language_model_for_the_installed_resolver():
    from sglang.srt.model_loader.utils import resolve_language_model

    class _Talker:
        language_model = NemotronVoiceChatTalker.language_model

        def __init__(self):
            self.llm = object()

    talker = _Talker()
    assert not hasattr(talker, "model")
    assert resolve_language_model(talker) is talker.llm


def test_language_model_is_read_only_and_not_a_parameter_alias():

    assert isinstance(NemotronVoiceChatTalker.language_model, property)
    assert NemotronVoiceChatTalker.language_model.fset is None


def test_model_level_capabilities_are_not_flipped():

    assert CAPABILITIES.supports_cuda_graph is False
    assert CAPABILITIES.supports_breakable_prefill_cuda_graph is False


def test_offline_voicechat_builders_keep_eager_defaults():
    for builder in (
        NemotronVoiceChatEngineBuilder(),
        NemotronVoiceChatTalkerEngineBuilder(),
    ):
        defaults = builder.generation_defaults(dtype="bfloat16")
        assert defaults["disable_cuda_graph"] is True
        assert "cuda_graph_backend_prefill" not in defaults
        assert getattr(builder, "supports_breakable_prefill_cuda_graph", False) is False


def test_duplex_builders_keep_single_forward_session_settings():
    for builder in (ThinkerBuilder(), TalkerBuilder()):
        defaults = builder.generation_defaults(dtype="bfloat16")
        assert defaults["enable_streaming_session"] is True
        assert defaults["max_running_requests"] == 2
        assert defaults["attention_backend"] == "triton"
        assert defaults["page_size"] == 1
