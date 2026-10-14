"""Glue: headcount + sentence + chips -> tonight's menu plan."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from .library import Library
from .retrieval import Retriever
from .rules import MealLog, MenuPlan, adjust_scores, category_targets, dish_count, plan_menu
from .tagging import Tagger, Tags


@dataclass
class TonightResult:
    people: int
    n_dishes: int
    tags: Tags
    scores: dict[str, float]
    plan: MenuPlan


def build_query(text: str, tags: Tags) -> str:
    return " ".join(p for p in ((text or "").strip(), tags.query_words()) if p)


def plan_tonight(
    *,
    people: int,
    text: str,
    chips: Iterable[str],
    library: Library,
    tagger: Tagger,
    retriever: Retriever,
    log: MealLog,
    today: date,
) -> TonightResult:
    n = dish_count(people)
    tags = tagger.tag(text, chips)                        # Pipeline 1
    similarity = retriever.rank(build_query(text, tags))  # Pipeline 2
    scores = adjust_scores(similarity, library, tags)     # rules
    targets = category_targets(n, tags.methods, tags.balance)
    plan = plan_menu(scores, library, targets, log.recent_ids(today), quick=tags.quick)
    return TonightResult(people=people, n_dishes=n, tags=tags, scores=scores, plan=plan)
