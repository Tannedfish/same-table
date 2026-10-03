from datetime import timedelta

import pytest

from same_table.library import Dish, Ingredient, Library, make_unlabeled
from same_table.rules import (
    MealLog, build_shopping_list, category_targets, classify_ingredient, dish_count,
    novelty_ratio, plan_menu, seed_log, swap_dish,
)
from same_table.tagging import Tags
from tests.conftest import TODAY


@pytest.mark.parametrize("people,dishes", [(1, 2), (2, 3), (3, 4), (4, 6), (5, 7), (6, 8), (7, 8), (12, 8)])
def test_dish_count_table(people, dishes):
    assert dish_count(people) == dishes


@pytest.mark.parametrize("bad", [0, -1, 2.5, "4", True, None])
def test_dish_count_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        dish_count(bad)


@pytest.mark.parametrize("n", range(1, 9))
@pytest.mark.parametrize("balance", ["balanced", "more_veg", "more_meat"])
def test_category_targets_add_up(n, balance):
    targets = category_targets(n, (), balance)
    assert sum(targets.values()) == n
    assert all(v >= 0 for v in targets.values())
    assert targets["veg"] >= 1 or n - targets["soup"] == 0
    assert targets["soup"] == (1 if n >= 4 else 0)


def test_soup_requested_on_small_table():
    assert category_targets(2, {"soup"})["soup"] == 1


def test_balance_moves_slots():
    veg = category_targets(8, (), "more_veg")
    meat = category_targets(8, (), "more_meat")
    assert veg["veg"] > meat["veg"] and meat["meat"] > veg["meat"]


def uniform_scores(library):
    return {d.id: 1.0 for d in library}


def test_plan_fills_every_slot_with_unique_dishes(library):
    targets = category_targets(8)
    plan = plan_menu(uniform_scores(library), library, targets, recent_ids=set())
    assert len(plan.dish_ids) == 8 == len(set(plan.dish_ids))
    for cat, k in targets.items():
        assert sum(library.get(i).category == cat for i in plan.dish_ids) == k
    assert plan.notes == [] and plan.shortfall == 0


def test_plan_prefers_new_dishes(library):
    recent = {d.id for d in library if d.category == "veg"}
    recent.pop()  # leave one new veg
    targets = category_targets(6)
    plan = plan_menu(uniform_scores(library), library, targets, recent)
    veg = [i for i in plan.dish_ids if library.get(i).category == "veg"]
    assert any(i not in recent for i in veg)
    assert "repeat_allowed" in plan.notes  # needed repeats to fill veg


def test_quick_filter_respected_when_possible(library):
    plan = plan_menu(uniform_scores(library), library, category_targets(4), set(), quick=True)
    assert all(library.get(i).is_quick for i in plan.dish_ids)


def test_quick_filter_relaxes_when_library_short():
    lib = Library([
        Dish("a", "慢湯", "soup", "soup", cooking_time=90),
        Dish("b", "快菜", "veg", "stir", cooking_time=5),
        Dish("c", "快肉", "meat", "stir", cooking_time=10),
        Dish("d", "快魚", "seafood", "steam", cooking_time=10),
    ])
    plan = plan_menu({i: 1.0 for i in lib.ids()}, lib, category_targets(4), set(), quick=True)
    assert "a" in plan.dish_ids and "time_relaxed" in plan.notes


def test_library_short_reports_shortfall():
    lib = Library([Dish("a", "菜", "veg", "stir"), Dish("b", "肉", "meat", "stir")])
    plan = plan_menu({"a": 1, "b": 1}, lib, category_targets(6), set())
    assert plan.shortfall == 4 and "library_short" in plan.notes
    assert sorted(plan.dish_ids) == ["a", "b"]


def test_scores_decide_order(library):
    scores = uniform_scores(library)
    scores["s01"] = 5.0
    plan = plan_menu(scores, library, {"meat": 0, "seafood": 1, "veg": 0, "soup": 0}, set())
    assert plan.dish_ids == ["s01"]


def test_swap_keeps_category_and_avoids_menu(library):
    scores = uniform_scores(library)
    menu = plan_menu(scores, library, category_targets(6), set()).dish_ids
    old = next(i for i in menu if library.get(i).category == "veg")
    new = swap_dish(menu, old, scores, library, set())
    assert new not in menu and library.get(new).category == "veg"


def test_swap_returns_none_when_nothing_left():
    lib = Library([Dish("a", "菜", "veg", "stir")])
    assert swap_dish(["a"], "a", {"a": 1.0}, lib, set()) is None


def test_novelty_ratio():
    assert novelty_ratio(["a", "b", "c", "d", "e"], {"a", "b"}) == 0.6
    assert novelty_ratio([], set()) == 0.0


def test_seed_log_shape(library):
    log = seed_log(library, TODAY)
    assert len(log.recent_entries(TODAY)) == 24
    assert sum(log.category_counts(library, TODAY).values()) == 24
    assert all(TODAY - timedelta(days=14) <= d < TODAY for d, _ in log.entries)
    assert len(log.recent_ids(TODAY)) < len(library)  # new dishes remain


def test_log_window_excludes_old_and_today():
    log = MealLog([(TODAY - timedelta(days=20), "old"), (TODAY, "tonight"), (TODAY - timedelta(days=3), "x")])
    assert log.recent_ids(TODAY) == {"x"}


@pytest.mark.parametrize("name,group", [
    ("豬肉絲", "meat"), ("雞蛋", "meat"), ("鱸魚", "seafood"), ("蠔油", "pantry"), ("雞粉", "pantry"),
    ("洋蔥", "veg"), ("菜心", "veg"), ("蒜蓉", "pantry"), ("水", None), ("魚露", "pantry"),
    ("「烹大師」干貝風味", "pantry"),
])
def test_classify_ingredient(name, group):
    assert classify_ingredient(name) == group


def test_shopping_list_merges_and_never_invents():
    a = Dish("a", "炒菜", "veg", "stir", ingredients=(Ingredient("菜心", "300克"), Ingredient("油", "1湯匙")))
    b = Dish("b", "灼菜", "veg", "blanch", ingredients=(Ingredient("菜心", "200克"), Ingredient("水", "1升")))
    u = make_unlabeled("糖醋排骨")
    shop = build_shopping_list([a, b, u])
    veg = {i.name: i for i in shop.groups["veg"]}
    assert veg["菜心"].amounts == ["300克", "200克"] and veg["菜心"].dishes == ["炒菜", "灼菜"]
    oil = shop.groups["pantry"][0]
    assert oil.name == "油" and oil.maybe_at_home
    assert shop.no_list == ["糖醋排骨"]
    assert not any(i.name == "水" for g in shop.groups.values() for i in g)
    assert shop.groups["meat"] == []  # nothing guessed for the unlabeled ribs


def test_less_sweet_penalty(library):
    from same_table.rules import adjust_scores
    sims = {d.id: 0.5 for d in library}
    scores = adjust_scores(sims, library, Tags(less_sweet=True))
    assert scores["t06"] < 0.5  # 節瓜瘦肉湯 has 蜜棗
    assert scores["v01"] == 0.5
