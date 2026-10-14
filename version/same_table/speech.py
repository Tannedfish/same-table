"""Pipeline 3 — voice note -> text (Cantonese speech recognition).

The cook records or uploads a short voice note; a Hugging Face
`automatic-speech-recognition` pipeline turns it into text, which goes into
the request box (the cook can still edit it) and then through Pipelines 1 and 2.

Audio is decoded here, not by the pipeline: soundfile reads WAV/FLAC/OGG/MP3;
anything else (iPhone .m4a, browser .webm) goes through ffmpeg if it is installed
(`packages.txt` installs it on Streamlit Cloud). Everything is converted to
16 kHz mono float32, the input all candidate models expect.
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import subprocess
from functools import lru_cache
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
MIN_SECONDS = 0.3
MAX_SECONDS = 60.0
MAX_BYTES = 10 * 1024 * 1024
AUDIO_TYPES = ("wav", "mp3", "m4a", "ogg", "flac", "webm", "aac")


class SpeechError(ValueError):
    """Audio that cannot be used; the message is safe to show to the cook."""


# ------------------------------------------------------------------ decoding
def _resample(y: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return y
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr), SAMPLE_RATE)
    return resample_poly(y, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)


def _decode_ffmpeg(data: bytes) -> np.ndarray:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise SpeechError("unsupported_format")
    proc = subprocess.run(
        [exe, "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        input=data, capture_output=True, timeout=60,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise SpeechError("unreadable")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def load_audio(data: bytes) -> np.ndarray:
    """Any common audio file -> 16 kHz mono float32 in [-1, 1]."""
    if not data:
        raise SpeechError("empty")
    if len(data) > MAX_BYTES:
        raise SpeechError("too_big")
    try:
        import soundfile as sf

        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        y = _resample(y.mean(axis=1), sr)
    except SpeechError:
        raise
    except Exception:  # m4a / webm / anything libsndfile cannot open
        y = _decode_ffmpeg(data)
    seconds = len(y) / SAMPLE_RATE
    if seconds < MIN_SECONDS:
        raise SpeechError("too_short")
    if seconds > MAX_SECONDS:
        raise SpeechError("too_long")
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak < 1e-3:
        raise SpeechError("silent")
    return y.astype(np.float32)


# --------------------------------------------------------------- text output
@lru_cache(maxsize=1)
def _to_hk():
    try:
        from opencc import OpenCC

        return OpenCC("s2hk").convert
    except Exception:  # optional dependency: keep the text as it is
        return lambda text: text


def normalize_transcript(text: str) -> str:
    """Hong Kong traditional characters (Whisper sometimes writes simplified), tidy spacing."""
    text = _to_hk()(text or "")
    text = re.sub(r"(?<=[一-鿿])\s+(?=[一-鿿])", "", text)  # no spaces between Chinese characters
    return re.sub(r"\s+", " ", text).strip()


def _clean_for_scoring(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", normalize_transcript(text)).lower()


def char_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """(character edits, reference characters), ignoring punctuation, spaces and simplified/traditional."""
    ref, hyp = _clean_for_scoring(reference), _clean_for_scoring(hypothesis)
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1], len(ref)


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Character edit distance / reference length."""
    edits, n = char_errors(reference, hypothesis)
    if n == 0:
        return 0.0 if edits == 0 else 1.0
    return edits / n


# --------------------------------------------------------------- the pipeline
class Transcriber:
    """Wraps the Hugging Face ASR pipeline. Loaded on first use (the model is the largest of the three)."""

    def __init__(self, settings, pipeline_factory: Callable | None = None):
        self.settings = settings
        self._factory = pipeline_factory
        self._pipe = None
        self.error: str | None = None

    @property
    def enabled(self) -> bool:
        return not (self.settings.offline and self._factory is None)

    @property
    def available(self) -> bool:
        return self._pipe is not None

    def load(self) -> "Transcriber":
        if self._pipe is not None or not self.enabled:
            if not self.enabled:
                self.error = "offline mode"
            return self
        try:
            factory = self._factory
            if factory is None:
                from transformers import pipeline as factory
            self._pipe = factory("automatic-speech-recognition", model=self.settings.p3_model_id)
        except Exception as exc:
            log.warning("Pipeline 3 unavailable: %s", exc)
            self._pipe, self.error = None, str(exc)
        return self

    def _generate_kwargs(self) -> dict:
        language = (self.settings.p3_language or "").strip().lower()
        model_type = getattr(getattr(getattr(self._pipe, "model", None), "config", None), "model_type", "")
        if model_type != "whisper":
            return {}
        kwargs = {"task": "transcribe", "num_beams": 1, "max_new_tokens": 128}
        if language and language != "none":
            kwargs["language"] = language
        return kwargs

    def transcribe_array(self, audio: np.ndarray) -> str:
        if self._pipe is None:
            self.load()
        if self._pipe is None:
            raise SpeechError("model_unavailable")
        kwargs = self._generate_kwargs()
        inputs = {"raw": audio, "sampling_rate": SAMPLE_RATE}
        out = self._pipe(inputs, generate_kwargs=kwargs) if kwargs else self._pipe(inputs)
        return normalize_transcript(out["text"] if isinstance(out, dict) else str(out))

    def transcribe(self, data: bytes) -> str:
        return self.transcribe_array(load_audio(data))
