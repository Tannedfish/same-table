import json

import pytest

from same_table.library import LibraryError, dish_from_dict, load_library, make_unlabeled


def test_seed_library_is_valid(library):
    assert len(library) >= 28
    cats = {d.category for d in library}
    assert cats == {"meat", "seafood", "veg", "soup"}
    assert all(d.labeled and d.ingredients and d.steps for d in library)


@pytest.mark.parametrize("row,msg", [
    ({"id": "x", "name": "", "category": "veg"}, "required"),
    ({"id": "x", "name": "菜", "category": "dessert"}, "category"),
    ({"id": "x", "name": "菜", "category": "veg", "method": "bake"}, "method"),
    ({"id": "x", "name": "菜", "category": "veg", "cooking_time": "abc"}, "integer"),
    ({"id": "x", "name": "菜", "category": "veg", "servings": -2}, "positive"),
])
def test_bad_rows_rejected(row, msg):
    with pytest.raises(LibraryError, match=msg):
        dish_from_dict(row, 1)


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "lib.json"
    row = {"id": "a", "name": "菜", "category": "veg", "method": "stir"}
    p.write_text(json.dumps([row, row]), encoding="utf-8")
    with pytest.raises(LibraryError, match="duplicate"):
        load_library(p)


def test_missing_and_broken_files(tmp_path):
    with pytest.raises(LibraryError, match="not found"):
        load_library(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(LibraryError, match="valid JSON"):
        load_library(bad)


def test_unlabeled_dish():
    d = make_unlabeled("  糖醋  排骨 ")
    assert d.name == "糖醋 排骨" and not d.labeled and d.ingredients == () and d.category is None
    assert make_unlabeled("糖醋 排骨").id == d.id
    with pytest.raises(LibraryError):
        make_unlabeled("   ")
    with pytest.raises(LibraryError):
        make_unlabeled("x" * 41)


# ---------------------------------------------------------------- English text
def test_translations_attach(library):
    assert all(d.has_english() for d in library)
    d = library.get("s01")
    assert d.display_name("en") != d.name and d.display_name("zh") == d.name
    assert len(d.display_ingredients("en")) == len(d.ingredients)
    assert len(d.display_steps("en")) == len(d.steps)
    assert library.by_name(d.display_name("en").upper()) is d     # English lookup ignores case


def _write(tmp_path, dishes, english):
    (tmp_path / "lib.json").write_text(json.dumps(dishes, ensure_ascii=False), encoding="utf-8")
    if english is not None:
        (tmp_path / "lib_en.json").write_text(english, encoding="utf-8")
    return load_library(tmp_path / "lib.json")


ROW = {"id": "a", "name": "炒菜心", "category": "veg", "method": "stir",
       "ingredients": [{"name": "菜心", "amount": "1斤"}], "steps": ["洗菜", "炒"]}


def test_mismatched_translation_is_ignored(tmp_path):
    bad = json.dumps([{"id": "a", "name": "Choy sum", "ingredients": [], "steps": ["Wash", "Fry"]}])
    assert not _write(tmp_path, [ROW], bad).get("a").has_english()


def test_broken_or_missing_translation_file(tmp_path):
    assert not _write(tmp_path, [ROW], "{not json").get("a").has_english()
    (tmp_path / "lib_en.json").unlink()
    assert not _write(tmp_path, [ROW], None).get("a").has_english()


def test_invisible_characters_are_stripped(tmp_path):
    row = dict(ROW, steps=["洗菜", "\U000e0020\U000e0020", "炒​"])
    good = json.dumps([{"id": "a", "name": "Choy sum", "ingredients": [{"name": "choy sum", "amount": "600 g"}],
                        "steps": ["Wash", "", "Fry"]}])
    dish = _write(tmp_path, [row], good).get("a")
    assert dish.steps == ("洗菜", "炒") and dish.display_steps("en") == ("Wash", "Fry")
