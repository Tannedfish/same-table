"""Pipeline 2 — retrieve dishes from the labeled library.

Embeds every library dish once with a Hugging Face `feature-extraction`
pipeline, then ranks dishes by cosine similarity to the cook's request.
The fine-tune (notebook) teaches the encoder which requests match which
dishes; it never writes new recipes.

Fallback when the model cannot load: character n-gram TF-IDF, which works
well enough on Chinese dish names to keep the app usable.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Callable

import numpy as np

from .library import Library

log = logging.getLogger(__name__)

# bge-zh recommends this instruction for short queries against passages.
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def _pool(output, pooling: str) -> np.ndarray:
    arr = np.asarray(output, dtype=np.float32)
    while arr.ndim > 2:  # [batch=1, tokens, hidden] -> [tokens, hidden]
        arr = arr[0]
    if arr.ndim == 1:
        return arr
    return arr[0] if pooling == "cls" else arr.mean(axis=0)


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


def _grams(text: str) -> Counter:
    s = re.sub(r"\s+", "", text.lower())
    return Counter(list(s) + [s[i : i + 2] for i in range(len(s) - 1)])


class CharTfidf:
    """Tiny char uni+bi-gram TF-IDF index (fallback ranker, no dependencies)."""

    def __init__(self, docs: dict[str, str]):
        self.ids = list(docs)
        grams = [_grams(docs[i]) for i in self.ids]
        df = Counter(g for doc in grams for g in doc)
        n = len(self.ids)
        self.idf = {g: math.log((1 + n) / (1 + c)) + 1 for g, c in df.items()}
        self.vecs = [self._vec(doc) for doc in grams]

    def _vec(self, grams: Counter) -> dict[str, float]:
        v = {g: c * self.idf.get(g, 0.0) for g, c in grams.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / norm for g, x in v.items()}

    def scores(self, query: str) -> dict[str, float]:
        q = self._vec(_grams(query))
        return {i: sum(w * v.get(g, 0.0) for g, w in q.items()) for i, v in zip(self.ids, self.vecs)}


class Retriever:
    def __init__(self, settings, library: Library, pipeline_factory: Callable | None = None):
        self.settings = settings
        self.library = library
        self._factory = pipeline_factory
        self._pipe = None
        self._matrix: np.ndarray | None = None
        self._ids = [d.id for d in library if d.labeled]
        self._fallback = CharTfidf({d.id: d.search_text() for d in library if d.labeled})
        self.error: str | None = None

    @property
    def available(self) -> bool:
        return self._matrix is not None

    def _embed(self, texts: list[str]) -> np.ndarray:
        outputs = self._pipe(texts, truncation=True)
        return _normalize(np.stack([_pool(o, self.settings.p2_pooling) for o in outputs]))

    def load(self) -> "Retriever":
        if self.settings.offline and self._factory is None:
            self.error = "offline mode"
            return self
        try:
            factory = self._factory
            if factory is None:
                from transformers import pipeline as factory
            self._pipe = factory("feature-extraction", model=self.settings.p2_model_id)
            docs = [self.library.get(i).search_text() for i in self._ids]
            self._matrix = self._embed(docs)
        except Exception as exc:
            log.warning("Pipeline 2 unavailable, using char TF-IDF: %s", exc)
            self._pipe, self._matrix, self.error = None, None, str(exc)
        return self

    def rank(self, query: str) -> dict[str, float]:
        """Similarity in [0, 1]-ish for every labeled library dish."""
        query = (query or "").strip() or "家常晚餐"
        if self._matrix is not None:
            try:
                prefix = BGE_QUERY_PREFIX if "bge" in self.settings.p2_model_id.lower() else ""
                q = self._embed([prefix + query])[0]
                sims = self._matrix @ q
                return {i: float(s) for i, s in zip(self._ids, sims)}
            except Exception as exc:
                log.warning("Pipeline 2 call failed, using char TF-IDF: %s", exc)
        return self._fallback.scores(query)
