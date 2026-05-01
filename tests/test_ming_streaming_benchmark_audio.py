import base64
import io
import wave

from benchmarks.performance.tts.benchmark_ming_streaming_tts import (
    collect_pcm_from_audio_payloads,
    write_wav_from_audio_payloads,
)


def _wav_b64(pcm: bytes, sample_rate: int = 44100) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_collect_pcm_from_audio_payloads_decodes_wav_chunks_before_concat():
    first_pcm = b"\x01\x00\x02\x00"
    second_pcm = b"\x03\x00\x04\x00"

    pcm, sample_rate = collect_pcm_from_audio_payloads(
        [
            {"data": _wav_b64(first_pcm, sample_rate=24000)},
            {"data": _wav_b64(second_pcm, sample_rate=24000)},
        ]
    )

    assert pcm == first_pcm + second_pcm
    assert b"RIFF" not in pcm
    assert sample_rate == 24000


def test_write_wav_from_audio_payloads_rejects_empty_pcm(tmp_path):
    output_path = tmp_path / "empty.wav"

    try:
        write_wav_from_audio_payloads(
            [
                {"data": "not valid base64"},
                {"audio": {"data": 123, "sample_rate": 44100}},
            ],
            output_path,
        )
    except RuntimeError as exc:
        assert "No valid PCM audio" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")

    assert not output_path.exists()


def test_write_wav_from_audio_payloads_writes_valid_pcm(tmp_path):
    output_path = tmp_path / "audio.wav"

    write_wav_from_audio_payloads(
        [{"data": _wav_b64(b"\x07\x00\x08\x00", sample_rate=22050)}],
        output_path,
    )

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getframerate() == 22050
        assert wav_file.readframes(wav_file.getnframes()) == b"\x07\x00\x08\x00"
