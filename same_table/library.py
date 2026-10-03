"""Curated dish library. The app only ever picks dishes from here.

A library row is *labeled*: a person (or a reviewed rule) gave it exactly one
category and one cooking method. Dishes the cook types in that are not in the
library become *unlabeled* add-ons: no category, no ingredients, risk warning.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

CATEGORIES = ("meat", "seafood", "veg", "soup")
METHODS = ("steam", "blanch", "soup", "stir", "other")
METHOD_ZH = {"steam": "蒸", "blanch": "白灼", "soup": "湯", "stir": "炒", "other": ""}
CATEGORY_ZH = {"meat": "肉", "seafood": "海鮮", "veg": "菜", "soup": "湯"}


class LibraryError(ValueError):
    """Raised when the library file or a row is invalid."""


@dataclass(frozen=True)
class Ingredient:
    name: str
    amount: str = ""


@dataclass(frozen=True)
class Dish:
    id: str
    name: str
    category: str | None
    method: str | None
    cooking_time: int | None = None  # minutes
    servings: int | None = None
    cover_url: str | None = None
    cover_path: str | None = None  # local copy under data/covers/, relative to the project root
    source_url: str | None = None
    description: str = ""
    ingredients: tuple[Ingredient, ...] = field(default_factory=tuple)
    steps: tuple[str, ...] = field(default_factory=tuple)
    labeled: bool = True

    @property
    def is_quick(self) -> bool:
        return self.cooking_time is not None and self.cooking_time <= 20

    def search_text(self) -> str:
        """Text that Pipeline 2 embeds for this dish."""
        parts = [
            self.name,
            METHOD_ZH.get(self.method or "", ""),
            CATEGORY_ZH.get(self.category or "", ""),
            self.description,
            " ".join(i.name for i in self.ingredients),
        ]
        return " ".join(p for p in parts if p)


def _opt_int(value, field_name: str, row: int) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise LibraryError(f"row {row}: {field_name} must be an integer, got {value!r}") from exc
    if number <= 0:
        raise LibraryError(f"row {row}: {field_name} must be positive, got {number}")
    return number


def dish_from_dict(raw: dict, row: int = 0) -> Dish:
    if not isinstance(raw, dict):
        raise LibraryError(f"row {row}: expected an object, got {type(raw).__name__}")
    dish_id = str(raw.get("id", "")).strip()
    name = str(raw.get("name", "")).strip()
    if not dish_id or not name:
        raise LibraryError(f"row {row}: id and name are required")
    category = raw.get("category")
    if category not in CATEGORIES:
        raise LibraryError(f"row {row} ({name}): category must be one of {CATEGORIES}, got {category!r}")
    method = raw.get("method") or "other"
    if method not in METHODS:
        raise LibraryError(f"row {row} ({name}): method must be one of {METHODS}, got {method!r}")

    ingredients = []
    for item in raw.get("ingredients") or []:
        if isinstance(item, str):
            item = {"name": item}
        ing_name = str(item.get("name", "")).strip()
        if ing_name:
            ingredients.append(Ingredient(ing_name, str(item.get("amount", "") or "").strip()))

    return Dish(
        id=dish_id,
        name=name,
        category=category,
        method=method,
        cooking_time=_opt_int(raw.get("cooking_time"), "cooking_time", row),
        servings=_opt_int(raw.get("servings"), "servings", row),
        cover_url=raw.get("cover_url") or None,
        cover_path=raw.get("cover_path") or None,
        source_url=raw.get("source_url") or None,
        description=str(raw.get("description", "") or ""),
        ingredients=tuple(ingredients),
        steps=tuple(str(s).strip() for s in raw.get("steps") or [] if str(s).strip()),
        labeled=True,
    )


class Library:
    def __init__(self, dishes: list[Dish]):
        self._by_id: dict[str, Dish] = {}
        for dish in dishes:
            if dish.id in self._by_id:
                raise LibraryError(f"duplicate dish id {dish.id!r}")
            self._by_id[dish.id] = dish
        if not self._by_id:
            raise LibraryError("library is empty")
        self._by_name = {d.name: d for d in dishes}

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[Dish]:
        return iter(self._by_id.values())

    def __contains__(self, dish_id: str) -> bool:
        return dish_id in self._by_id

    def get(self, dish_id: str) -> Dish:
        try:
            return self._by_id[dish_id]
        except KeyError as exc:
            raise LibraryError(f"unknown dish id {dish_id!r}") from exc

    def by_name(self, name: str) -> Dish | None:
        return self._by_name.get(name.strip())

    def ids(self) -> list[str]:
        return list(self._by_id)


def load_library(path: str | Path) -> Library:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LibraryError(f"library file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LibraryError(f"library file is not valid JSON: {path} ({exc})") from exc
    rows = payload.get("dishes") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise LibraryError("library JSON must be a list or an object with a 'dishes' list")
    return Library([dish_from_dict(r, i) for i, r in enumerate(rows, start=1)])


def make_unlabeled(name: str) -> Dish:
    """A cook-typed dish outside the library. Never guessed or enriched."""
    clean = " ".join(name.split())
    if not clean:
        raise LibraryError("dish name is empty")
    if len(clean) > 40:
        raise LibraryError("dish name is too long (max 40 characters)")
    digest = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:8]
    return Dish(id=f"u-{digest}", name=clean, category=None, method=None, labeled=False)
