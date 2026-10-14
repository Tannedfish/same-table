"""Pipeline wrappers tested with fake HF pipelines (no downloads)."""

import numpy as np

from same_table.planner import build_query, plan_tonight
from same_table.retrieval import Retriever
from same_table.rules import seed_log
from same_table.tagging import ZERO_SHOT_LABELS, Tagger, Tags, keyword_tags
from tests.conftest import TODAY


# ------------------------------------------------------------------ Pipeline 1
def test_keywords_cantonese_sentence():
    tags = keyword_tags("阿爸牙唔好，想清淡啲，蒸魚或者煲湯，唔好太甜，今晚趕時間")
    assert {"steam", "soup", "more_veg", "quick"} <= tags


def test_keywords_english():
    assert {"stir", "more_meat"} <= keyword_tags("A quick stir fry with more meat please")


def test_offline_tagger_uses_keywords_and_chips(settings):
    tagger = Tagger(settings).load()
    assert not tagger.available
    tags = tagger.tag("想食蒸嘅嘢", chips=["soup", "more_veg", "bogus"])
    assert tags.methods == {"steam", "soup"} and tags.balance == "more_veg"
    assert tags.source == "chips+keyword"


def test_chips_override_text_balance(settings):
    tags = Tagger(settings).load().tag("多啲肉", chips=["more_veg"])
    assert tags.balance == "more_veg"


def test_empty_input(settings):
    tags = Tagger(settings).load().tag("", [])
    assert tags == Tags(source="none")


class FakeZeroShot:
    def __init__(self, task, model):
        assert task == "zero-shot-classification"

    def __call__(self, text, candidate_labels, hypothesis_template, multi_label):
        scores = {ZERO_SHOT_LABELS["steam"]: 0.93, ZERO_SHOT_LABELS["less_sweet"]: 0.81}
        return {"labels": candidate_labels, "scores": [scores.get(l, 0.05) for l in candidate_labels]}


def test_zero_shot_mode(settings):
    tagger = Tagger(settings, pipeline_factory=FakeZeroShot).load()
    tags = tagger.tag("老人家想食啲軟熟嘢")  # no keywords at all
    assert tagger.available and tags.methods == {"steam"} and tags.less_sweet and tags.source == "model"


class FakeClassifier:
    def __init__(self, task, model):
        pass

    def __call__(self, text, top_k=None):
        return [[{"label": "blanch", "score": 0.9}, {"label": "more_meat", "score": 0.7},
                 {"label": "quick", "score": 0.2}, {"label": "LABEL_99", "score": 0.99}]]


def test_fine_tuned_classifier_mode(settings):
    from dataclasses import replace
    s = replace(settings, p1_task="text-classification", p1_threshold=0.5)
    tags = Tagger(s, pipeline_factory=FakeClassifier).load().tag("隨便")
    assert tags.methods == {"blanch"} and tags.balance == "more_meat" and not tags.quick


def test_model_failure_falls_back(settings):
    def broken(*a, **k):
        raise OSError("no network")
    tagger = Tagger(settings, pipeline_factory=broken).load()
    assert not tagger.available and "no network" in tagger.error
    assert "steam" in tagger.tag("蒸魚").methods


# ------------------------------------------------------------------ Pipeline 2
def test_fallback_retrieval_ranks_relevant_dish(settings, library):
    r = Retriever(settings, library).load()
    scores = r.rank("清蒸魚")
    assert max(scores, key=scores.get) == "s01"
    assert set(scores) == set(library.ids())


class FakeEncoder:
    """Returns [1, tokens, hidden] like HF feature-extraction; 'hidden' = char flags."""
    VOCAB = "蒸魚湯炒菜肉"

    def __init__(self, task, model):
        assert task == "feature-extraction"

    def __call__(self, texts, truncation=True):
        out = []
        for t in texts:
            cls = [float(c in t) for c in self.VOCAB]
            out.append([[cls, [0.0] * len(self.VOCAB)]])
        return out


def test_model_retrieval_uses_embeddings(settings, library):
    r = Retriever(settings, library, pipeline_factory=FakeEncoder).load()
    assert r.available
    scores = r.rank("蒸魚")
    top = sorted(scores, key=scores.get, reverse=True)[:3]
    assert "s01" in top
    bound = 1.001 + settings.p2_hybrid_weight  # cosine + weight x lexical (both at most 1)
    assert all(-1.001 <= v <= bound for v in scores.values())


# ------------------------------------------------------------------ end to end
def test_plan_tonight_end_to_end(settings, library):
    tagger, retriever = Tagger(settings).load(), Retriever(settings, library).load()
    log = seed_log(library, TODAY)
    res = plan_tonight(people=4, text="想食蒸魚，清淡啲，20分鐘內", chips=["less_sweet"], library=library,
                       tagger=tagger, retriever=retriever, log=log, today=TODAY)
    assert res.n_dishes == 6 and len(res.plan.dish_ids) == 6
    assert res.tags.quick and res.tags.less_sweet and "steam" in res.tags.methods
    cats = [library.get(i).category for i in res.plan.dish_ids]
    assert cats.count("soup") == 1 and cats.count("veg") >= 2


def test_build_query():
    assert build_query("  想食魚 ", Tags(methods={"steam"}, quick=True)) == "想食魚 蒸 快手"


def test_hybrid_weight_zero_is_pure_cosine(settings, library):
    from dataclasses import replace
    dense_only = Retriever(replace(settings, p2_hybrid_weight=0.0), library, pipeline_factory=FakeEncoder).load()
    mixed = Retriever(replace(settings, p2_hybrid_weight=2.0), library, pipeline_factory=FakeEncoder).load()
    a, b = dense_only.rank("蒸魚"), mixed.rank("蒸魚")
    lexical = mixed._fallback.scores("蒸魚")
    assert all(abs(b[i] - (a[i] + 2.0 * lexical[i])) < 1e-6 for i in a)
