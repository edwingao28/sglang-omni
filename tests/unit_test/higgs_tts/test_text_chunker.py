# SPDX-License-Identifier: Apache-2.0

from sglang_omni.models.higgs_tts.text_chunker import chunk_text


def test_short_input_returns_single_chunk_unchanged() -> None:
    text = "Hello world."
    assert chunk_text(text, max_chars=200) == [text]


def test_disabled_when_max_chars_non_positive() -> None:
    text = "a" * 1000
    assert chunk_text(text, max_chars=0) == [text]
    assert chunk_text(text, max_chars=-1) == [text]


def test_every_chunk_within_limit_and_reconstructs() -> None:
    text = ". ".join(f"Sentence number {i} here" for i in range(60)) + "."
    max_chars = 80
    chunks = chunk_text(text, max_chars=max_chars)

    assert len(chunks) > 1
    assert all(len(c) <= max_chars for c in chunks)
    assert "".join(chunks) == text


def test_cjk_sentence_packing_and_reconstruction() -> None:
    text = "你好世界。" * 100
    max_chars = 50
    chunks = chunk_text(text, max_chars=max_chars)

    assert all(len(c) <= max_chars for c in chunks)
    assert "".join(chunks) == text


def test_long_unit_falls_back_to_clause_then_hard_slice() -> None:
    # One "sentence" with no terminal punctuation, only commas.
    text = "word, " * 100  # 600 chars, no period
    max_chars = 40
    chunks = chunk_text(text, max_chars=max_chars)

    assert all(len(c) <= max_chars for c in chunks)
    assert "".join(chunks) == text


def test_single_token_longer_than_max_is_hard_sliced() -> None:
    text = "x" * 500  # no delimiters at all
    max_chars = 100
    chunks = chunk_text(text, max_chars=max_chars)

    assert all(len(c) <= max_chars for c in chunks)
    assert "".join(chunks) == text
    assert len(chunks) == 5
