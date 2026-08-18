#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audio CONTENT check for prefill-graph on/off: prefix WER via whisper-large-v3.

G3 validated the contract (no failures, exact duration signature) but never what
the audio *says*. Hidden-state capture runs as forward-pre hooks on thinker
layers 0/24; if those copies were not recorded into the CUDA graph, replay would
feed the talker stale hidden states and produce fluent-but-wrong speech at
exactly the right duration — invisible to every G3 gate.

Prefix WER follows /data/omni-paper-20260816/wer/README.md: the fixed-256
contract forces ~20.4 s of audio for ~3.5 s of text, so trailing material scores
as insertions. Truncate the hypothesis to the reference word count, then score.
"""

from __future__ import annotations

import json
import os
import sys

import torch
from jiwer import process_words
from transformers import pipeline
from whisper.normalizers import EnglishTextNormalizer


def refs(run_dir: str) -> dict[str, str]:
    d = json.load(open(os.path.join(run_dir, "speed_results.json")))
    out = {}
    for r in d["per_request"]:
        if r.get("is_success") and r.get("wav_path"):
            out[os.path.basename(r["wav_path"])] = r["text"]
    return out


def score(asr, run_dir: str, norm) -> dict:
    reference = refs(run_dir)
    audio_dir = os.path.join(run_dir, "audio")
    files = sorted(reference)
    paths = [os.path.join(audio_dir, f) for f in files]
    outs = asr(paths, batch_size=8, generate_kwargs={"language": "en", "task": "transcribe"})
    tot_err = tot_words = 0
    per_utt, empties = [], 0
    for f, o in zip(files, outs):
        ref = norm(reference[f]).strip()
        hyp_full = norm(o["text"]).strip()
        if not hyp_full:
            empties += 1
        nref = len(ref.split())
        hyp = " ".join(hyp_full.split()[:nref])
        if not ref:
            continue
        m = process_words(ref, hyp if hyp else "*")
        err = m.substitutions + m.deletions + m.insertions
        tot_err += err
        tot_words += nref
        per_utt.append(err / max(nref, 1))
    per_utt.sort()
    return {
        "clips": len(per_utt),
        "corpus_prefix_wer": round(tot_err / max(tot_words, 1), 4),
        "utt_p50": round(per_utt[len(per_utt) // 2], 4) if per_utt else None,
        "utt_p95": round(per_utt[max(0, int(0.95 * len(per_utt)) - 1)], 4) if per_utt else None,
        "empty_transcripts": empties,
    }


if __name__ == "__main__":
    norm = EnglishTextNormalizer()
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16,
        device="cuda:0",
        chunk_length_s=30,
        stride_length_s=5,
    )
    for run_dir in sys.argv[1:]:
        res = score(asr, run_dir, norm)
        print(os.path.basename(run_dir.rstrip("/")), json.dumps(res), flush=True)
