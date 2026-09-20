"""Audit occurrence WAVs and reuse one already-running ASR for explicit closed cells.

No ASR/service/Slurm process is started here. Run separately from timed traffic.
Every directory is explicit. Existing quality outputs are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from sustained_protocol import ASR_MODEL, INPUT_SHA, sha, write_new


def verify_population(manifest, capture, generated, base):
    expected = manifest['occurrences']
    ids = [r['occurrence_id'] for r in expected]
    if len(ids) != len(set(ids)) or len(base) != 128:
        raise ValueError('Unique occurrence IDs and exactly128 base inputs required')
    if manifest['input_sha256'] != INPUT_SHA or manifest['base_sample_ids'] != [s.sample_id for s in base]:
        raise ValueError('Pinned base cohort differs')
    if [r['occurrence_id'] for r in capture['per_request']] != ids or [r['sample_id'] for r in generated] != ids:
        raise ValueError('Capture/generated occurrence population differs')
    for i, (entry, row, produced) in enumerate(zip(expected, capture['per_request'], generated)):
        index = entry['base_index']
        if type(index) is not int or not 0 <= index < 128:
            raise ValueError('Invalid base index')
        sample = base[index]
        if any(entry[k] != getattr(sample, attr) for k, attr in (
                ('base_id','sample_id'), ('ref_audio','ref_audio'), ('ref_text','ref_text'), ('target_text','target_text'))):
            raise ValueError('Occurrence full input does not match sealed base sample')
        if entry['phase'] == 'service' and (index != i % 128 or entry['cycle'] != i // 128):
            raise ValueError('Service cyclic mapping differs')
        if produced['target_text'] != entry['target_text']:
            raise ValueError('Generated target text differs from occurrence')
        raw = row.get('result') or {}
        if produced['is_success'] != bool(raw.get('is_success')) or produced['wav_path'] != raw.get('wav_path', ''):
            raise ValueError('Generated success/WAV differs from captured upstream result')
    return expected


def summarize_wer(rows, expected_ids):
    if [r['sample_id'] for r in rows] != expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError('ASR occurrence population differs')
    scored = [r for r in rows if r['is_success']]
    keys = ('substitutions', 'deletions', 'insertions', 'hits')
    for row in scored:
        if any(type(row[k]) is not int or row[k] < 0 for k in keys):
            raise ValueError('ASR counts must be nonnegative integers')
        if row['substitutions'] + row['deletions'] + row['hits'] != len(row['ref_norm'].split()):
            raise ValueError('ASR reference denominator does not match normalized text')
    counts = {k: sum(r[k] for r in scored) for k in keys}
    words = counts['substitutions']+counts['deletions']+counts['hits']
    errors = counts['substitutions']+counts['deletions']+counts['insertions']
    return dict(expected_occurrences=len(rows), scored_occurrences=len(scored),
        unscored_occurrences=len(rows)-len(scored), counts=counts, reference_words=words,
        errors=errors, corpus_wer=errors/words if words else None,
        scope='Full S/D/I/H, no filtered WER or tolerance. Failed/no-WAV occurrences remain unscored, not invented word errors; repeated inputs are not independent samples.')


def prepare_quality(cell, meta):
    import numpy as np
    import soundfile as sf
    from benchmarks.dataset.seedtts import load_seedtts_samples
    if sha(meta) != INPUT_SHA:
        raise ValueError('Pinned meta SHA differs')
    manifest_path, capture_path, generated_path = (cell/name for name in (
        'occurrence-manifest.json', 'sustained-capture.json', 'generated.json'))
    manifest, capture, generated = (json.loads(p.read_text()) for p in (manifest_path,capture_path,generated_path))
    if capture['manifest_sha256'] != sha(manifest_path):
        raise ValueError('Capture manifest byte SHA differs')
    expected = verify_population(manifest, capture, generated, load_seedtts_samples(str(meta),128))
    wavs, asr_input, files_expected = [], [], set()
    for entry, row, produced in zip(expected, capture['per_request'], generated):
        path = cell/'audio'/f'{entry["occurrence_id"]}.wav'
        mapped = dict(produced)
        if path.exists():
            if produced['wav_path'] and Path(produced['wav_path']).resolve() != path.resolve():
                raise ValueError('WAV path is not this occurrence in this cell')
            files_expected.add(path.resolve())
            audio, rate = sf.read(path, dtype='float32')
            if audio.ndim != 1 or not audio.size or rate <= 0 or not np.isfinite(audio).all():
                raise ValueError(f'Undecodable/nonfinite WAV retained; stop before ASR: {path}')
            peak = float(np.max(np.abs(audio)))
            wavs.append(dict(occurrence_id=entry['occurrence_id'], base_id=entry['base_id'],
                wav_path=str(path.resolve()), sha256=sha(path), frames=len(audio), sample_rate=rate,
                duration_s=len(audio)/rate, peak=peak, rms=float(np.sqrt(np.mean(audio*audio))),
                fraction_at_full_scale=float(np.mean(np.abs(audio)>=.9999)), positive_finite_audio=peak>0,
                generation_outcome=row['outcome'], generation_success=produced['is_success']))
            # Feed every actual decodable WAV, including silent/adverse or partial
            # outputs; generation outcome is retained separately and never promoted.
            mapped.update(wav_path=str(path.resolve()), is_success=True, audio_duration_s=len(audio)/rate)
        else:
            if produced['is_success'] or produced['wav_path']:
                raise ValueError('Claimed generated WAV is missing')
            mapped.update(is_success=False, error='No produced WAV: '+row['outcome'])
        asr_input.append(mapped)
    observed_files = {p.resolve() for p in (cell/'audio').glob('*.wav')}
    if observed_files != files_expected:
        raise ValueError('Unmatched/orphan WAV outside occurrence manifest')
    return dict(cell=str(cell), phase=manifest['phase'], expected_occurrences=len(expected),
        generation_outcomes=dict(Counter(r['outcome'] for r in capture['per_request'])),
        wav_count=len(wavs), positive_finite_wavs=sum(r['positive_finite_audio'] for r in wavs), rows=wavs,
        source_sha256={str(p):sha(p) for p in (meta,manifest_path,capture_path,generated_path)},
        input_sha256=INPUT_SHA, occurrence_quality_scope='ASR transcript fidelity and WAV integrity; no speaker/perceptual equivalence'), asr_input, capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=Path, nargs='+', required=True)
    parser.add_argument('--meta', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--asr-port', type=int, required=True)
    args = parser.parse_args()
    if len({p.resolve() for p in args.cells}) != len(args.cells):
        parser.error('Explicit cells must be unique')
    if args.receipt.exists() or any((p/'sustained-quality').exists() for p in args.cells):
        raise FileExistsError('Quality receipt/output exists; preserve scored evidence')
    state = dict(status='running', cells=[str(p) for p in args.cells], results=[],
        asr_model=ASR_MODEL, asr_port=args.asr_port, asr_concurrency=8, script_sha256=sha(__file__),
        ownership='Caller owns one already-running ASR and timing isolation; no service started here')
    write_new(args.receipt, state)
    try:
        from benchmarks.eval.benchmark_tts_seedtts import TtsSeedttsBenchmarkConfig, run_tts_seedtts_transcribe
        for cell in args.cells:
            out = cell/'sustained-quality'
            out.mkdir()
            audit, generated, capture = prepare_quality(cell, args.meta)
            write_new(out/'audio-audit.json', audit)
            write_new(out/'generated.json', generated)
            config = TtsSeedttsBenchmarkConfig(model='qwen3-tts', meta=str(args.meta),
                output_dir=str(out), max_samples=len(generated), concurrency=0, seed=42,
                max_new_tokens=2048, asr_model_path=ASR_MODEL, asr_concurrency=8)
            response = run_tts_seedtts_transcribe(config, asr_router_port=args.asr_port)
            rows = [asdict(r) for r in response['per_sample']]
            full = summarize_wer(rows, [r['sample_id'] for r in generated])
            for row in audit['rows']:
                if sha(row['wav_path']) != row['sha256']:
                    raise ValueError('Actual WAV bytes changed during ASR')
            full.update(audio_audit_sha256=sha(out/'audio-audit.json'),
                generated_asr_input_sha256=sha(out/'generated.json'),
                upstream_wer_sha256=sha(out/'wer_results.json'),
                generation_outcomes=audit['generation_outcomes'], actual_wavs=audit['wav_count'])
            if 'origin' in capture:
                origin = capture['origin']['mono_ns']
                ids = {r['occurrence_id'] for r in capture['per_request']
                       if origin+30*10**9 <= r.get('send_mono_ns', -1) < origin+150*10**9}
                subset = [r for r in rows if r['sample_id'] in ids]
                full['sent_window'] = summarize_wer(subset, [r['sample_id'] for r in subset])
            write_new(out/'quality-audit.json', full)
            state['results'].append(dict(cell=str(cell), audit=full))
            args.receipt.write_text(json.dumps(state, indent=2)+'\n')
            if full['scored_occurrences'] != audit['wav_count']:
                raise RuntimeError('Not every actual WAV received ASR; retain failure and stop')
        state['status'] = 'scored_all_actual_wavs_generation_outcomes_retained'
    except BaseException as exc:
        state.update(status='failed_preserved', error=repr(exc))
        raise
    finally:
        args.receipt.write_text(json.dumps(state, indent=2)+'\n')


if __name__ == '__main__':
    main()
