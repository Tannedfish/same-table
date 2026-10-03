"""Build the ~100-dish library from the Hugging Face recipe dataset.

Two steps, so a person reviews every label (the library is *labeled*):

  1) python scripts/build_library.py export --out data/library_review.csv
     Downloads AWeirdDev/zh-tw-recipes-sm, keeps dinner dishes, auto-suggests
     category + method with keyword rules, samples a balanced ~100 rows and
     writes a CSV. Open it, fix `category` / `method`, set keep=0 to drop rows.

  2) python scripts/build_library.py import --csv data/library_review.csv
     Validates the reviewed CSV, downloads each cover photo to data/covers/
     and writes data/library.json for the app.

  python scripts/build_library.py covers   (re-try photo downloads only)

Photos: the dataset's `image` links use iCook's retired resize proxy
(imageproxy.icook.network/resize?...url=<original>). It now only redirects, and
the original S3 address returns 403. We extract <original> and rebuild the URL
on iCook's current proxy (imgproxy.icook.network/safe/...), falling back to
the current direct host (tokyo-kitchen.icook.network). Copyright stays with the
iCook authors: caption every photo as "image from the library, not generated".

Needs:  pip install datasets pandas requests
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from same_table.library import CATEGORIES, METHODS, LibraryError, dish_from_dict  # noqa: E402
from same_table.rules import classify_ingredient, normalize_ingredient  # noqa: E402

DATASET = "AWeirdDev/zh-tw-recipes-sm"

# Column names differ between dataset versions; the first match wins.
FIELD_CANDIDATES = {
    "name": ("name", "title", "recipe_name"),
    "description": ("description", "descriotion", "desc", "intro"),  # dataset has the typo
    "cover_url": ("cover_url", "image", "image_url", "img", "cover"),
    "source_url": ("url", "source_url", "link"),
    "cooking_time": ("cooking_time", "time", "duration", "cook_time"),
    "servings": ("servings", "serving", "portion", "yield"),
    "ingredients": ("ingredients", "ingredient", "materials"),
    "steps": ("steps", "instructions", "directions", "method"),
}

# Not dinner dishes (buns, bakes, desserts, drinks, sauces).
EXCLUDE = (
    "饅頭", "包子", "蛋糕", "餅乾", "麵包", "吐司", "司康", "布丁", "果凍", "甜點", "點心", "甜湯",
    "果醬", "飲", "茶", "咖啡", "奶昔", "冰", "塔", "派", "餅", "糕", "醬汁", "沾醬", "粽", "月餅",
    "便當", "早餐", "三明治", "披薩", "義大利麵", "寵物",
    # not a shared Cantonese-style dinner dish
    "零嘴", "沙拉", "飯", "麵", "粥", "烏龍", "米粉", "冬粉", "河粉", "米線", "洋芋片", "烤", "焗", "漢堡", "鬆餅", "餃", "壽司", "丼", "捲", "炸",
)
CATEGORY_RULES = (
    ("soup", ("湯", "羹")),
    ("seafood", ("魚", "蝦", "蟹", "蛤", "蜆", "蚵", "蠔", "干貝", "扇貝", "花枝", "魷", "透抽", "中卷", "海鮮")),
    ("meat", ("豬", "牛", "雞", "鴨", "羊", "排骨", "肉", "蛋")),
)
METHOD_RULES = (
    ("soup", ("湯", "羹")),
    ("steam", ("蒸",)),
    ("blanch", ("燙", "汆", "白灼", "涼拌", "水煮", "白切")),
    ("stir", ("炒", "爆", "煸")),
)


COVERS_DIR = ROOT / "data" / "covers"
COVER_WIDTH = 600
UA = {"User-Agent": "Mozilla/5.0 (SameTable classroom prototype; HKUST ISOM5240)"}


def original_image_url(url: str | None) -> str | None:
    """Pull the original photo address out of any iCook proxy URL."""
    if not url:
        return None
    from urllib.parse import parse_qs, unquote, urlparse

    url = str(url).strip()
    parsed = urlparse(url)
    if "url" in parse_qs(parsed.query):                       # old: /resize?...&url=<original>
        return parse_qs(parsed.query)["url"][0]
    if "/plain/" in parsed.path:                              # current: /safe/.../plain/<original>
        return unquote(url.split("/plain/", 1)[1].split("@", 1)[0])
    return url


def cover_candidates(url: str | None, width: int = COVER_WIDTH) -> list[str]:
    """Working URLs to try, best first."""
    original = original_image_url(url)
    if not original:
        return []
    urls = [f"https://imgproxy.icook.network/safe/rt:fit/w:{width}/el:0/q:80/plain/{original}"]
    m = re.search(r"(/uploads/.+)$", original)
    if m:
        urls.append("https://tokyo-kitchen.icook.network" + m.group(1))
    if url not in urls:
        urls.append(str(url))
    return urls


def download_cover(dish_id: str, url: str | None, session=None, retries: int = 2) -> tuple[str | None, str | None]:
    """Save the photo to data/covers/<id>.jpg. Returns (relative path, working url) or (None, None)."""
    import time

    import requests

    session = session or requests.Session()
    target = COVERS_DIR / f"{dish_id}.jpg"
    rel = target.relative_to(ROOT).as_posix()
    for candidate in cover_candidates(url):
        if target.exists() and target.stat().st_size > 1000:
            return rel, candidate
        for attempt in range(retries + 1):
            try:
                r = session.get(candidate, headers=UA, timeout=20, allow_redirects=True)
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/") and len(r.content) > 1000:
                    COVERS_DIR.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(r.content)
                    return rel, candidate
                break  # 403/404/not an image: try the next candidate
            except requests.RequestException:
                time.sleep(1.5 * (attempt + 1))
    return None, None


def attach_covers(dishes: list[dict]) -> None:
    import requests

    session = requests.Session()
    failed = []
    for d in dishes:
        path, url = download_cover(d["id"], d.get("cover_url"), session)
        d["cover_path"] = path
        if url:
            d["cover_url"] = url
        if path is None:
            failed.append(d["name"])
    print(f"Photos: {len(dishes) - len(failed)}/{len(dishes)} saved to {COVERS_DIR.relative_to(ROOT)}/")
    if failed:
        print("  No photo (the app shows an icon instead): " + "、".join(failed))


def pick_column(columns, key):
    for cand in FIELD_CANDIDATES[key]:
        if cand in columns:
            return cand
    return None


def parse_minutes(value) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return int(value) or None
    text = str(value)
    hours = re.search(r"(\d+)\s*(小時|hr|hour)", text)
    mins = re.search(r"(\d+)\s*(分|min)", text)
    total = (int(hours.group(1)) * 60 if hours else 0) + (int(mins.group(1)) if mins else 0)
    if total == 0 and re.fullmatch(r"\s*\d+\s*", text):
        total = int(text)
    return total or None


def parse_int(value) -> int | None:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group()) if match else None


def split_steps(value) -> list[str]:
    """Steps arrive as one string like '1. 備料：\n1.肉片解凍…'. One step per line, numbers stripped."""
    if isinstance(value, (list, tuple)):
        lines = [str(v) for v in value]
    else:
        lines = re.split(r"\n+", str(value or ""))
    out = []
    for line in lines:
        line = re.sub(r"^\s*(?:步驟)?\s*\d+\s*[.、:：)）]\s*", "", line).strip()
        if line:
            out.append(line)
    return out


def as_list(value) -> list:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [s for s in re.split(r"[\n；;]", value) if s.strip()]
    return list(value)


def parse_ingredient(item) -> dict:
    if isinstance(item, dict):
        name = item.get("name") or item.get("ingredient") or ""
        amount = item.get("amount") or item.get("quantity") or item.get("unit") or ""
        return {"name": str(name).strip(), "amount": str(amount).strip()}
    parts = str(item).strip().split(maxsplit=1)
    return {"name": parts[0] if parts else "", "amount": parts[1] if len(parts) > 1 else ""}


def suggest(rules, text, default):
    for label, words in rules:
        if any(w in text for w in words):
            return label
    return default


def cmd_export(args) -> None:
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train")
    df = ds.to_pandas()
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
    cols = {k: pick_column(df.columns, k) for k in FIELD_CANDIDATES}
    print("Column mapping:", cols)
    if cols["name"] is None:
        raise SystemExit("Could not find a title column; edit FIELD_CANDIDATES.")

    rows = []
    for i, r in df.iterrows():
        name = str(r[cols["name"]]).strip()
        if not name or any(x in name for x in EXCLUDE):
            continue
        ingredients = [parse_ingredient(x) for x in as_list(r[cols["ingredients"]])] if cols["ingredients"] else []
        steps = split_steps(r[cols["steps"]]) if cols["steps"] else []
        if not ingredients or not steps:
            continue
        # main ingredients only: 蠔油 / 魚露 / 雞粉 must not make a dish seafood or meat
        mains = [x["name"] for x in ingredients if classify_ingredient(normalize_ingredient(x["name"])) not in (None, "pantry")]
        text = name + " " + " ".join(mains)
        rows.append({
            "keep": 1,
            "id": f"r{i:05d}",
            "name": name,
            "category": suggest(CATEGORY_RULES, name, None) or suggest(CATEGORY_RULES, text, "veg"),
            "method": suggest(METHOD_RULES, name, "other"),
            "cooking_time": parse_minutes(r[cols["cooking_time"]]) if cols["cooking_time"] else None,
            "servings": parse_int(r[cols["servings"]]) if cols["servings"] else None,
            "cover_url": (cover_candidates(r[cols["cover_url"]]) or [None])[0] if cols["cover_url"] else None,
            "source_url": r[cols["source_url"]] if cols["source_url"] else None,
            "description": str(r[cols["description"]] or "")[:200] if cols["description"] else "",
            "ingredients": json.dumps(ingredients, ensure_ascii=False),
            "steps": json.dumps(steps, ensure_ascii=False),
        })
    print(f"{len(rows)} dinner candidates after filtering")

    # Balanced sample: roughly 30 meat, 20 seafood, 35 veg, 15 soup (x target/100).
    rng = random.Random(args.seed)
    shares = {"meat": 0.30, "seafood": 0.20, "veg": 0.35, "soup": 0.15}
    sample = []
    for cat, share in shares.items():
        pool = [r for r in rows if r["category"] == cat]
        rng.shuffle(pool)
        # prefer rows whose title names one of our four methods, then rows with a cooking time
        pool.sort(key=lambda r: (r["method"] == "other", r["cooking_time"] is None))
        sample += pool[: round(args.target * share)]
    out = pd.DataFrame(sample)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} rows to {args.out}. Review category/method, then run `import`.")


def cmd_import(args) -> None:
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    dishes, errors = [], []
    for n, r in enumerate(df.to_dict("records"), start=2):
        if int(r.get("keep", 1) or 0) != 1:
            continue
        raw = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in r.items()}
        raw["ingredients"] = json.loads(raw.get("ingredients") or "[]")
        raw["steps"] = json.loads(raw.get("steps") or "[]")
        try:
            dish = dish_from_dict(raw, n)
        except LibraryError as exc:
            errors.append(str(exc))
            continue
        dishes.append({
            "id": dish.id, "name": dish.name, "category": dish.category, "method": dish.method,
            "cooking_time": dish.cooking_time, "servings": dish.servings, "cover_url": dish.cover_url,
            "cover_path": None, "source_url": dish.source_url, "description": dish.description,
            "ingredients": [{"name": i.name, "amount": i.amount} for i in dish.ingredients],
            "steps": list(dish.steps),
        })
    if errors:
        print("Fix these rows and re-run:\n  " + "\n  ".join(errors))
        raise SystemExit(1)
    if not args.no_photos:
        attach_covers(dishes)
    write_library(dishes, args.out)


def write_library(dishes: list[dict], out) -> None:
    payload = {"source": f"{DATASET} (reviewed)", "categories": CATEGORIES, "methods": METHODS, "dishes": dishes}
    Path(out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = pd.Series([d["category"] for d in dishes]).value_counts().to_dict()
    print(f"Wrote {len(dishes)} dishes to {out}: {counts}")


def cmd_covers(args) -> None:
    payload = json.loads(Path(args.library).read_text(encoding="utf-8"))
    dishes = payload["dishes"]
    attach_covers(dishes)
    write_library(dishes, args.library)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", default=str(ROOT / "data" / "library_review.csv"))
    e.add_argument("--target", type=int, default=100)
    e.add_argument("--seed", type=int, default=42)
    i = sub.add_parser("import")
    i.add_argument("--csv", default=str(ROOT / "data" / "library_review.csv"))
    i.add_argument("--out", default=str(ROOT / "data" / "library.json"))
    i.add_argument("--no-photos", action="store_true", help="skip downloading cover photos")
    c = sub.add_parser("covers", help="download / re-try cover photos for an existing library.json")
    c.add_argument("--library", default=str(ROOT / "data" / "library.json"))
    args = parser.parse_args()
    {"export": cmd_export, "import": cmd_import, "covers": cmd_covers}[args.cmd](args)


if __name__ == "__main__":
    main()
