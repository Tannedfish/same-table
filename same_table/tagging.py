"""Pipeline 1 — understand tonight's request.

Input: the cook's sentence (Cantonese, Mandarin or English) plus chips.
Output: Tags (cooking methods, balance, less-sweet, within-20-min).

Model modes (set in config):
  * zero-shot-classification : base multilingual NLI model, no training.
  * text-classification      : our fine-tuned multi-label model on the Hub.
    Its id2label must use the tag names below (steam, blanch, soup, stir,
    more_veg, more_meat, less_sweet, quick).
A keyword lexicon always runs too: it is high-precision and it keeps the app
working if the model cannot load.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable

log = logging.getLogger(__name__)

METHODS = ("steam", "blanch", "soup", "stir")
BALANCES = ("more_veg", "balanced", "more_meat")
FLAGS = ("less_sweet", "quick")
ALL_TAGS = METHODS + ("more_veg", "more_meat") + FLAGS
CHIPS = ("steam", "blanch", "soup", "stir", "more_veg", "more_meat", "less_sweet", "quick")

TAG_WORDS_ZH = {
    "steam": "蒸", "blanch": "白灼", "soup": "湯", "stir": "炒",
    "more_veg": "多菜 清淡", "more_meat": "多肉", "less_sweet": "少甜", "quick": "快手",
}

KEYWORDS: dict[str, tuple[str, ...]] = {
    "steam": ("蒸", "steam"),
    "blanch": ("灼", "焯", "燙", "烚", "浸", "blanch", "poach", "boiled"),
    "soup": ("湯", "soup", "broth"),
    "stir": ("炒", "stir", "wok"),
    "more_veg": ("多菜", "多啲菜", "多點菜", "多些菜", "蔬菜", "清淡", "素", "more veg", "vegetable", "light"),
    "more_meat": ("多肉", "多啲肉", "多點肉", "肉多", "more meat", "meaty", "protein"),
    "less_sweet": ("少甜", "唔好甜", "唔甜", "不甜", "少糖", "走糖", "無糖", "less sweet", "no sugar", "less sugar"),
    "quick": ("快", "趕", "20分鐘", "二十分鐘", "唔夠時間", "冇時間", "沒時間", "quick", "fast", "20 min", "hurry"),
}

ZERO_SHOT_LABELS = {
    "steam": "steamed dishes",
    "blanch": "blanched or poached dishes",
    "soup": "soup",
    "stir": "stir-fried dishes",
    "more_veg": "more vegetables and a lighter meal",
    "more_meat": "more meat",
    "less_sweet": "less sweet food with no sugar",
    "quick": "quick cooking, very little time",
}
HYPOTHESIS = "This dinner request asks for {}."


@dataclass
class Tags:
    methods: set[str] = field(default_factory=set)
    balance: str = "balanced"
    less_sweet: bool = False
    quick: bool = False
    source: str = "none"  # chips | keyword | model (joined with +)

    def query_words(self) -> str:
        words = [TAG_WORDS_ZH[m] for m in sorted(self.methods)]
        if self.balance != "balanced":
            words.append(TAG_WORDS_ZH[self.balance])
        if self.less_sweet:
            words.append(TAG_WORDS_ZH["less_sweet"])
        if self.quick:
            words.append(TAG_WORDS_ZH["quick"])
        return " ".join(words)


def keyword_tags(text: str) -> set[str]:
    low = text.lower()
    return {tag for tag, words in KEYWORDS.items() if any(w in low for w in words)}


def _balance(found: set[str], scores: dict[str, float] | None = None) -> str | None:
    veg, meat = "more_veg" in found, "more_meat" in found
    if veg and meat:
        if scores and scores.get("more_veg", 0) != scores.get("more_meat", 0):
            return "more_veg" if scores["more_veg"] > scores["more_meat"] else "more_meat"
        return "balanced"
    if veg:
        return "more_veg"
    if meat:
        return "more_meat"
    return None


class Tagger:
    """Wraps the Hugging Face pipeline; falls back to keywords if it fails."""

    def __init__(self, settings, pipeline_factory: Callable | None = None):
        self.settings = settings
        self._factory = pipeline_factory
        self._pipe = None
        self.error: str | None = None

    @property
    def available(self) -> bool:
        return self._pipe is not None

    def load(self) -> "Tagger":
        if self.settings.offline and self._factory is None:
            self.error = "offline mode"
            return self
        try:
            factory = self._factory
            if factory is None:
                from transformers import pipeline as factory  # heavy import, done lazily
            self._pipe = factory(self.settings.p1_task, model=self.settings.p1_model_id)
        except Exception as exc:  # network, missing weights, bad id, OOM ...
            log.warning("Pipeline 1 unavailable, using keywords: %s", exc)
            self._pipe, self.error = None, str(exc)
        return self

    # -- model calls -------------------------------------------------------
    def _model_scores(self, text: str) -> dict[str, float]:
        if self.settings.p1_task == "zero-shot-classification":
            out = self._pipe(
                text,
                candidate_labels=list(ZERO_SHOT_LABELS.values()),
                hypothesis_template=HYPOTHESIS,
                multi_label=True,
            )
            back = {v: k for k, v in ZERO_SHOT_LABELS.items()}
            return {back[l]: float(s) for l, s in zip(out["labels"], out["scores"]) if l in back}
        out = self._pipe(text, top_k=None)
        if out and isinstance(out[0], list):  # batched shape
            out = out[0]
        return {d["label"]: float(d["score"]) for d in out if d["label"] in ALL_TAGS}

    def tag(self, text: str, chips: Iterable[str] = ()) -> Tags:
        chips = {c for c in chips if c in CHIPS}
        text = (text or "").strip()[:500]
        found = set(keyword_tags(text)) if text else set()
        sources = ["chips"] if chips else []
        if found:
            sources.append("keyword")
        scores: dict[str, float] = {}
        if text and self._pipe is not None:
            try:
                scores = self._model_scores(text)
                found |= {t for t, s in scores.items() if s >= self.settings.p1_threshold}
                sources.append("model")
            except Exception as exc:
                log.warning("Pipeline 1 call failed, keywords only: %s", exc)

        balance = _balance(chips) or _balance(found, scores) or "balanced"
        return Tags(
            methods={t for t in (chips | found) if t in METHODS},
            balance=balance,
            less_sweet="less_sweet" in chips or "less_sweet" in found,
            quick="quick" in chips or "quick" in found,
            source="+".join(sources) or "none",
        )
