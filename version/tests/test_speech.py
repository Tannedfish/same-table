"""Pipeline 3 (voice notes) with fake HF pipelines: no model download."""

import io
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from same_table.speech import (
    MAX_SECONDS, SAMPLE_RATE, SpeechError, Transcriber, char_error_rate, load_audio, normalize_transcript,
)


def tone(seconds=1.0, sr=16000, amp=0.3, fmt="WAV", channels=1):
    t = np.arange(int(seconds * sr)) / sr
    y = (amp * np.sin(2 * np.pi * 440 * t)).astype("float32")
    if channels == 2:
        y = np.stack([y, y], axis=1)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format=fmt)
    return buf.getvalue()


# ------------------------------------------------------------------ decoding
def test_wav_is_resampled_to_16k_mono():
    y = load_audio(tone(1.0, sr=44100, channels=2))
    assert y.dtype == np.float32 and y.ndim == 1 and abs(len(y) - SAMPLE_RATE) <= 2


def test_flac_and_ogg():
    assert len(load_audio(tone(fmt="FLAC"))) == SAMPLE_RATE
    assert abs(len(load_audio(tone(fmt="OGG"))) - SAMPLE_RATE) < 400


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_m4a_goes_through_ffmpeg(tmp_path):
    path = tmp_path / "note.m4a"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5",
                    "-c:a", "aac", str(path)], check=True)
    assert abs(len(load_audio(path.read_bytes())) - 1.5 * SAMPLE_RATE) < 2000


@pytest.mark.parametrize("data,code", [
    (b"", "empty"),
    (tone(0.1), "too_short"),
    (tone(MAX_SECONDS + 1, sr=8000), "too_long"),
    (tone(1.0, amp=0.0), "silent"),
    (b"x" * (11 * 1024 * 1024), "too_big"),
])
def test_bad_audio_gives_a_clear_code(data, code):
    with pytest.raises(SpeechError, match=code):
        load_audio(data)


def test_garbage_bytes_are_rejected():
    with pytest.raises(SpeechError):
        load_audio(b"not audio at all" * 100)


# -------------------------------------------------------------------- text
def test_normalize_transcript():
    assert normalize_transcript(" 阿爸 想食 软熟 嘅蒸餸 ") == "阿爸想食軟熟嘅蒸餸"
    assert normalize_transcript("quick  stir fry") == "quick stir fry"


def test_char_error_rate():
    assert char_error_rate("想食蒸魚。", "想食蒸魚") == 0.0
    assert char_error_rate("想食蒸魚", "想食真魚") == 0.25
    assert char_error_rate("想食蒸魚", "") == 1.0
    assert char_error_rate("想食蒸鱼", "想食蒸魚") == 0.0          # simplified vs traditional is not an error


# ---------------------------------------------------------------- pipeline
class FakeWhisper:
    calls = []

    def __init__(self, task, model):
        assert task == "automatic-speech-recognition"
        self.model = type("M", (), {"config": type("C", (), {"model_type": "whisper"})()})()

    def __call__(self, inputs, generate_kwargs=None):
        FakeWhisper.calls.append(generate_kwargs)
        assert inputs["sampling_rate"] == SAMPLE_RATE and inputs["raw"].dtype == np.float32
        return {"text": "阿爸想食 蒸鱼"}


class FakeCTC:
    def __init__(self, task, model):
        self.model = type("M", (), {"config": type("C", (), {"model_type": "wav2vec2"})()})()

    def __call__(self, inputs, **kwargs):
        assert not kwargs                     # CTC models take no generation settings
        return {"text": "炒 菜"}


def test_whisper_path(settings):
    tr = Transcriber(replace(settings, p3_language="zh"), pipeline_factory=FakeWhisper)
    assert tr.transcribe(tone()) == "阿爸想食蒸魚"
    assert FakeWhisper.calls[-1]["language"] == "zh" and FakeWhisper.calls[-1]["num_beams"] == 1


def test_language_none_lets_whisper_detect(settings):
    Transcriber(replace(settings, p3_language="none"), pipeline_factory=FakeWhisper).transcribe(tone())
    assert "language" not in FakeWhisper.calls[-1]


def test_ctc_path(settings):
    assert Transcriber(settings, pipeline_factory=FakeCTC).transcribe(tone()) == "炒菜"


def test_offline_and_broken_model(settings):
    off = Transcriber(settings)              # tests run with SAME_TABLE_OFFLINE=1
    assert not off.enabled
    with pytest.raises(SpeechError, match="model_unavailable"):
        off.transcribe(tone())

    def broken(*a, **k):
        raise OSError("no network")
    tr = Transcriber(settings, pipeline_factory=broken)
    with pytest.raises(SpeechError, match="model_unavailable"):
        tr.transcribe(tone())
    assert "no network" in tr.error


def test_bad_audio_never_reaches_the_model(settings):
    tr = Transcriber(settings, pipeline_factory=FakeWhisper)
    n = len(FakeWhisper.calls)
    with pytest.raises(SpeechError):
        tr.transcribe(tone(0.05))
    assert len(FakeWhisper.calls) == n
