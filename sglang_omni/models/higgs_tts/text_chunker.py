# SPDX-License-Identifier: Apache-2.0
"""Sentence/clause text chunking for Higgs TTS long-prompt synthesis.

Higgs Audio v3 loses text-audio alignment on long input and emits
end-of-chunk early (or loops the middle of repetitive text). Splitting the
text into reliable-window chunks at the request-orchestration boundary keeps
each synthesized chunk inside the alignment window; the resulting PCM is
concatenated by the caller. Short inputs (``len(text) <= max_chars``) take a
byte-identical single-chunk passthrough.
"""

from __future__ import annotations

import re

DEFAULT_MAX_CHARS = 200

# Sentence-terminating punctuation, Latin + CJK, plus newline. Kept on the
# preceding sentence so reconstruction only differs at split points.
_SENTENCE_RE = re.compile(r"[^.!?;\n。！？；\n]*[.!?;\n。！？；]+|[^.!?;\n。！？；]+")
# Clause-level fallback delimiters (commas, CJK comma/enumeration).
_CLAUSE_RE = re.compile(r"[^,，、]*[,，、]+|[^,，、]+")


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split ``text`` into chunks no longer than ``max_chars`` characters.

    Greedily packs adjacent sentences; if a single sentence still exceeds
    ``max_chars`` it is split on clause boundaries, then hard-sliced as a last
    resort. Returns ``[text]`` unchanged when the input already fits, so the
    common short-prompt path is byte-identical to no chunking.

    Args:
        text: Input text to synthesize.
        max_chars: Maximum characters per chunk. ``<= 0`` disables chunking.

    Returns:
        Ordered list of chunks whose concatenation reconstructs ``text`` at
        the split points. Always non-empty.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    units: list[str] = []
    for sentence in _SENTENCE_RE.findall(text):
        if len(sentence) <= max_chars:
            units.append(sentence)
        else:
            units.extend(_split_long_unit(sentence, max_chars))

    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current += unit
    if current:
        chunks.append(current)

    return chunks or [text]


def _split_long_unit(unit: str, max_chars: int) -> list[str]:
    """Split a single over-long unit on clause boundaries, then hard-slice."""
    pieces: list[str] = []
    current = ""
    for clause in _CLAUSE_RE.findall(unit):
        if len(clause) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_slice(clause, max_chars))
        elif current and len(current) + len(clause) > max_chars:
            pieces.append(current)
            current = clause
        else:
            current += clause
    if current:
        pieces.append(current)
    return pieces


def _hard_slice(text: str, max_chars: int) -> list[str]:
    """Last-resort fixed-width slice on character boundaries."""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
