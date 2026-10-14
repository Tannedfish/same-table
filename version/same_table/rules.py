"""Business rules. Plain Python, deterministic, no model involved.

The models only tag the request and rank dishes. Everything that must be
exact — how many dishes, which categories, the 14-day log, the 20-minute
filter, the patient actions and the shopping list — lives here.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable

from .library import CATEGORIES, Dish, Library

# ---------------------------------------------------------------- dish count
DISH_COUNT = {1: 2, 2: 3, 3: 4, 4: 6, 5: 7, 6: 8}
MAX_DISHES = 8


def dish_count(people: int) -> int:
    """People at the table -> number of shared dishes (7+ people caps at 8)."""
    if isinstance(people, bool) or not isinstance(people, int) or people < 1:
        raise ValueError(f"people must be a positive integer, got {people!r}")
    return DISH_COUNT.get(people, MAX_DISHES)


# ---------------------------------------------------------- category balance
BALANCE_WEIGHTS = {
    "balanced": {"veg": 0.40, "meat": 0.35, "seafood": 0.25},
    "more_veg": {"veg": 0.55, "meat": 0.25, "seafood": 0.20},
    "more_meat": {"veg": 0.30, "meat": 0.45, "seafood": 0.25},
}


def category_targets(n: int, methods: Iterable[str] = (), balance: str = "balanced") -> dict[str, int]:
    """Split n dishes into meat / seafood / veg / soup slots.

    One soup from 4 dishes up (or whenever soup is asked for). The rest is
    split by the balance weights using largest remainders, with at least one
    vegetable dish and, from 2 non-soup slots up, at least one meat or seafood.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if balance not in BALANCE_WEIGHTS:
        raise ValueError(f"unknown balance {balance!r}")
    methods = set(methods)
    soup = 1 if (n >= 4 or "soup" in methods) else 0
    rest = n - soup
    weights = BALANCE_WEIGHTS[balance]
    raw = {c: rest * w for c, w in weights.items()}
    alloc = {c: int(v) for c, v in raw.items()}
    leftover = rest - sum(alloc.values())
    for c in sorted(raw, key=lambda c: (raw[c] - alloc[c], weights[c]), reverse=True)[:leftover]:
        alloc[c] += 1
    if rest >= 1 and alloc["veg"] == 0:
        donor = max(("meat", "seafood"), key=lambda c: alloc[c])
        alloc[donor] -= 1
        alloc["veg"] += 1
    if rest >= 2 and alloc["meat"] + alloc["seafood"] == 0:
        alloc["veg"] -= 1
        alloc["meat"] += 1
    return {"meat": alloc["meat"], "seafood": alloc["seafood"], "veg": alloc["veg"], "soup": soup}


# ------------------------------------------------------------------ scoring
METHOD_BONUS = 0.25
QUICK_BONUS = 0.10
SWEET_PENALTY = 0.30
SWEET_MARKERS = ("糖", "蜜", "茄汁", "甜酸", "煉奶", "冰糖")


def has_sweet_ingredient(dish: Dish) -> bool:
    return any(m in ing.name for ing in dish.ingredients for m in SWEET_MARKERS)


def adjust_scores(similarity: dict[str, float], library: Library, tags) -> dict[str, float]:
    """Add transparent rule bonuses on top of the Pipeline 2 similarity."""
    scores = {}
    for dish_id, sim in similarity.items():
        dish = library.get(dish_id)
        score = float(sim)
        if tags.methods and dish.method in tags.methods:
            score += METHOD_BONUS
        if tags.quick and dish.is_quick:
            score += QUICK_BONUS
        if tags.less_sweet and has_sweet_ingredient(dish):
            score -= SWEET_PENALTY
        scores[dish_id] = score
    return scores


# --------------------------------------------------------------- menu plan
@dataclass
class MenuPlan:
    dish_ids: list[str]
    targets: dict[str, int]
    notes: list[str] = field(default_factory=list)  # relaxation codes, see i18n "note_*"
    shortfall: int = 0


# Each pass loosens one rule, only if the stricter pass left empty slots.
_PASSES = (
    ("strict", dict(allow_recent=False, enforce_time=True, any_category=False)),
    ("repeat_allowed", dict(allow_recent=True, enforce_time=True, any_category=False)),
    ("time_relaxed", dict(allow_recent=True, enforce_time=False, any_category=False)),
    ("category_swapped", dict(allow_recent=True, enforce_time=False, any_category=True)),
)


def plan_menu(
    scores: dict[str, float],
    library: Library,
    targets: dict[str, int],
    recent_ids: set[str],
    quick: bool = False,
) -> MenuPlan:
    ranked = sorted(scores, key=lambda i: (-scores[i], i))
    need = dict(targets)
    chosen: list[str] = []
    notes: list[str] = []

    def eligible(dish: Dish, allow_recent: bool, enforce_time: bool) -> bool:
        if dish.id in chosen or not dish.labeled:
            return False
        if not allow_recent and dish.id in recent_ids:
            return False
        if quick and enforce_time and not dish.is_quick:
            return False
        return True

    for code, rule in _PASSES:
        if sum(need.values()) == 0:
            break
        before = len(chosen)
        for dish_id in ranked:
            if sum(need.values()) == 0:
                break
            dish = library.get(dish_id)
            if not eligible(dish, rule["allow_recent"], rule["enforce_time"]):
                continue
            if need.get(dish.category, 0) > 0:
                need[dish.category] -= 1
                chosen.append(dish_id)
            elif rule["any_category"]:
                # take from the largest open slot so the total still adds up
                open_cat = max(need, key=lambda c: need[c])
                need[open_cat] -= 1
                chosen.append(dish_id)
        if code != "strict" and len(chosen) > before:
            notes.append(code)

    shortfall = sum(need.values())
    if shortfall:
        notes.append("library_short")
    order = {c: i for i, c in enumerate(CATEGORIES)}
    chosen.sort(key=lambda i: (order[library.get(i).category], -scores[i]))
    return MenuPlan(dish_ids=chosen, targets=dict(targets), notes=notes, shortfall=shortfall)


def swap_dish(
    menu_ids: list[str],
    dish_id: str,
    scores: dict[str, float],
    library: Library,
    recent_ids: set[str],
    quick: bool = False,
    exclude: Iterable[str] = (),
) -> str | None:
    """Next-best library dish of the same category that is not on the menu."""
    current = library.get(dish_id)
    blocked = set(menu_ids) | set(exclude)
    pool = [
        library.get(i)
        for i in sorted(scores, key=lambda i: (-scores[i], i))
        if i not in blocked and library.get(i).category == current.category
    ]
    tiers = (
        [d for d in pool if d.id not in recent_ids and (not quick or d.is_quick)],
        [d for d in pool if d.id not in recent_ids],
        pool,
    )
    for tier in tiers:
        if tier:
            return tier[0].id
    return None


def novelty_ratio(dish_ids: Iterable[str], recent_ids: set[str]) -> float:
    ids = list(dish_ids)
    if not ids:
        return 0.0
    return sum(1 for i in ids if i not in recent_ids) / len(ids)


# ---------------------------------------------------------- patient plating
PATIENT_ACTIONS = ("sauce_aside", "less_rice", "no_added_sugar")  # fixed, max three


# ------------------------------------------------------------ 14-day log
@dataclass
class MealLog:
    entries: list[tuple[date, str]] = field(default_factory=list)

    def recent_entries(self, today: date, days: int = 14) -> list[tuple[date, str]]:
        start = today - timedelta(days=days)
        return [(d, i) for d, i in self.entries if start <= d < today]

    def recent_ids(self, today: date, days: int = 14) -> set[str]:
        return {i for _, i in self.recent_entries(today, days)}

    def category_counts(self, library: Library, today: date, days: int = 14) -> dict[str, int]:
        counts = {c: 0 for c in CATEGORIES}
        for _, dish_id in self.recent_entries(today, days):
            if dish_id in library:
                counts[library.get(dish_id).category] += 1
        return counts

    def add(self, dish_ids: Iterable[str], day: date) -> None:
        self.entries.extend((day, i) for i in dish_ids)


def seed_log(library: Library, today: date, n_entries: int = 24, days: int = 14, seed: int = 7) -> MealLog:
    """Mock history for the demo (Streamlit Cloud has no database)."""
    rng = random.Random(seed)
    ids = sorted(library.ids())
    pool = rng.sample(ids, k=max(1, min(len(ids) // 2, 12)))  # ~12 dishes eaten twice
    entries = []
    for k in range(n_entries):
        offset = days - (k * (days - 1)) // max(n_entries - 1, 1)
        entries.append((today - timedelta(days=offset), pool[k % len(pool)]))
    return MealLog(entries)


# --------------------------------------------------------- shopping list
GROUPS = ("meat", "seafood", "veg", "pantry")
_SKIP = {"水", "清水", "熱水", "冷水", "溫水", "冰水"}
_VEG_OVERRIDE = ("洋蔥", "蒜苗", "蒜心", "青蒜", "薑芽")
_PANTRY = (
    "油", "鹽", "糖", "豉油", "生抽", "老抽", "醬", "酒", "胡椒", "醋", "蒜", "薑", "蔥",
    "豆豉", "蠔油", "雞粉", "味精", "生粉", "粟粉", "澱粉", "淀粉", "麵粉", "八角", "花椒",
    "陳皮", "魚露", "上湯", "高湯", "芡", "風味", "調味", "烹大師", "雞精", "鮮味",
)
_SEAFOOD = ("魚", "蝦", "蟹", "蜆", "蠔", "扇貝", "帶子", "魷", "墨魚", "蟶", "貝", "蚌", "海")
_MEAT = ("豬", "牛", "雞", "鴨", "羊", "肉", "排骨", "骨", "蛋", "火腿", "臘")
MAYBE_AT_HOME = {
    "油", "食油", "生油", "花生油", "粟米油", "橄欖油", "植物油",
    "鹽", "鹽巴", "幼鹽", "糖", "砂糖", "白糖", "冰糖",
    "蒜", "蒜頭", "蒜蓉", "蒜片", "薑", "薑片", "薑絲", "薑蓉", "老薑",
}


def normalize_ingredient(name: str) -> str:
    name = re.sub(r"[（(].*?[）)]", "", name)
    return re.sub(r"\s+", "", name).strip()


def classify_ingredient(name: str) -> str | None:
    """meat / seafood / veg / pantry, or None for water."""
    if name in _SKIP:
        return None
    if any(k in name for k in _VEG_OVERRIDE):
        return "veg"
    if any(k in name for k in _PANTRY):
        return "pantry"
    if any(k in name for k in _SEAFOOD):
        return "seafood"
    if any(k in name for k in _MEAT):
        return "meat"
    return "veg"


@dataclass
class ShoppingItem:
    name: str  # normalized Chinese name: the merge key
    group: str
    label: str = ""  # what the cook sees, in the chosen language
    amounts: list[str] = field(default_factory=list)
    dishes: list[str] = field(default_factory=list)
    maybe_at_home: bool = False


@dataclass
class ShoppingList:
    groups: dict[str, list[ShoppingItem]]
    no_list: list[str]  # dish names with no ingredients: never guessed

    def is_empty(self) -> bool:
        return not any(self.groups.values())


def build_shopping_list(dishes: Iterable[Dish], lang: str = "zh") -> ShoppingList:
    """Merge the ingredients of the chosen dishes. Never invent missing ones.

    Ingredients are merged and grouped on their Chinese names (the source of truth);
    `label`, amounts and dish names are shown in `lang` when a translation exists."""
    items: dict[str, ShoppingItem] = {}
    no_list: list[str] = []
    for dish in dishes:
        if not dish.ingredients:
            no_list.append(dish.display_name(lang))
            continue
        shown = dish.display_ingredients(lang)
        dish_label = dish.display_name(lang)
        for ing, show in zip(dish.ingredients, shown):
            key = normalize_ingredient(ing.name)
            if not key:
                continue
            group = classify_ingredient(key)
            if group is None:
                continue
            item = items.get(key)
            if item is None:
                label = key if show is ing else (show.name[:1].upper() + show.name[1:])
                item = items[key] = ShoppingItem(
                    name=key, group=group, label=label, maybe_at_home=group == "pantry" and key in MAYBE_AT_HOME
                )
            if show.amount:
                item.amounts.append(show.amount)
            if dish_label not in item.dishes:
                item.dishes.append(dish_label)
    groups = {g: [] for g in GROUPS}
    seen: dict[tuple[str, str], ShoppingItem] = {}
    for item in items.values():
        twin = seen.get((item.group, item.label.casefold()))
        if twin is None:                       # different source spellings (鹽 / 鹽巴) that read the same: one line
            seen[(item.group, item.label.casefold())] = item
            groups[item.group].append(item)
            continue
        twin.amounts += item.amounts
        twin.dishes += [d for d in item.dishes if d not in twin.dishes]
        twin.maybe_at_home = twin.maybe_at_home or item.maybe_at_home
    for g in groups:
        groups[g].sort(key=lambda it: (it.maybe_at_home, -len(it.dishes), it.label))
    return ShoppingList(groups=groups, no_list=no_list)
