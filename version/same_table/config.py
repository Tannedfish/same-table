"""Runtime settings.

Every value can be set with an environment variable or a Streamlit secret of
the same name (Streamlit Cloud: App settings -> Secrets). After the notebook
fine-tune is pushed to the Hub, set P1_MODEL_ID / P2_MODEL_ID to those ids so
the app model equals the notebook model (course rule).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Base models used until the fine-tuned ids exist.
DEFAULT_P1_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  # multilingual zero-shot
DEFAULT_P1_TASK = "zero-shot-classification"
DEFAULT_P2_MODEL = "BAAI/bge-small-zh-v1.5"  # small Chinese sentence encoder
DEFAULT_P2_POOLING = "cls"  # bge uses CLS pooling; use "mean" for most sentence-transformers
# The untuned bge model expects this instruction in front of short queries. Our fine-tune is trained
# on plain queries, so the prefix is used only with the untuned default model.
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
DEFAULT_P3_MODEL = "alvanlii/whisper-small-cantonese"  # Cantonese speech recognition (voice notes)
DEFAULT_P3_LANGUAGE = "zh"  # Whisper language token; "" for CTC models such as wav2vec2
DEFAULT_P2_HYBRID_WEIGHT = 3.0  # tuned on 600 development queries in notebooks/Model2_experiments.ipynb


def _read(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    try:  # Streamlit secrets are optional; any failure means "not set".
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def _flag(name: str, default: bool) -> bool:
    return _read(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    p1_model_id: str
    p1_task: str  # "zero-shot-classification" (base) or "text-classification" (fine-tuned)
    p1_threshold: float
    p2_model_id: str
    p2_pooling: str
    p2_query_prefix: str
    p2_hybrid_weight: float  # weight of the character-match score added to the cosine similarity
    p3_model_id: str
    p3_language: str
    library_path: Path
    offline: bool  # skip model downloads, use keyword/char-gram fallbacks
    demo_mode: bool  # show one unlabeled demo dish on the menu

    @classmethod
    def load(cls) -> "Settings":
        library = Path(_read("LIBRARY_PATH", str(DATA_DIR / "library.json")))
        if not library.exists():
            library = DATA_DIR / "seed_library.json"
        task = _read("P1_TASK", DEFAULT_P1_TASK)
        if task not in {"zero-shot-classification", "text-classification"}:
            raise ValueError(f"P1_TASK must be zero-shot-classification or text-classification, got {task!r}")
        pooling = _read("P2_POOLING", DEFAULT_P2_POOLING)
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"P2_POOLING must be cls or mean, got {pooling!r}")
        p2_model = _read("P2_MODEL_ID", DEFAULT_P2_MODEL)
        return cls(
            p1_model_id=_read("P1_MODEL_ID", DEFAULT_P1_MODEL),
            p1_task=task,
            p1_threshold=float(_read("P1_THRESHOLD", "0.5" if task == "text-classification" else "0.7")),
            p2_model_id=p2_model,
            p2_pooling=pooling,
            p2_query_prefix=_read("P2_QUERY_PREFIX", BGE_QUERY_PREFIX if p2_model == DEFAULT_P2_MODEL else ""),
            p2_hybrid_weight=float(_read("P2_HYBRID_WEIGHT", str(DEFAULT_P2_HYBRID_WEIGHT))),
            p3_model_id=_read("P3_MODEL_ID", DEFAULT_P3_MODEL),
            p3_language=_read("P3_LANGUAGE", DEFAULT_P3_LANGUAGE).strip(),
            library_path=library,
            offline=_flag("SAME_TABLE_OFFLINE", False),
            demo_mode=_flag("SAME_TABLE_DEMO", True),
        )
