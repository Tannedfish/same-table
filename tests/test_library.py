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
