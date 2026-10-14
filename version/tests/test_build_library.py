import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_library as bl  # noqa: E402


def test_parse_minutes():
    assert bl.parse_minutes("10分鐘") == 10
    assert bl.parse_minutes("1小時30分鐘") == 90
    assert bl.parse_minutes("45") == 45
    assert bl.parse_minutes(None) is None and bl.parse_minutes("少許") is None


def test_parse_ingredient_and_labels():
    assert bl.parse_ingredient("青椒 2個") == {"name": "青椒", "amount": "2個"}
    assert bl.parse_ingredient({"name": "豬肉", "amount": "200g"}) == {"name": "豬肉", "amount": "200g"}
    assert bl.suggest(bl.CATEGORY_RULES, "青椒番茄炒肉絲", None) == "meat"
    assert bl.suggest(bl.METHOD_RULES, "清蒸鱸魚", "other") == "steam"
    assert bl.suggest(bl.CATEGORY_RULES, "蒜炒高麗菜", "veg") == "veg"


OLD = ("https://imageproxy.icook.network/resize?background=255%2C255%2C255&height=150&nocrop=false"
       "&stripmeta=true&type=auto&url=http%3A%2F%2Ftokyo-kitchen.icook.tw.s3.amazonaws.com%2Fuploads"
       "%2Frecipe%2Fcover%2F455340%2F7e9435a22d6413f3.jpg&width=200")
ORIGINAL = "http://tokyo-kitchen.icook.tw.s3.amazonaws.com/uploads/recipe/cover/455340/7e9435a22d6413f3.jpg"


def test_old_proxy_url_rewritten():
    assert bl.original_image_url(OLD) == ORIGINAL
    urls = bl.cover_candidates(OLD)
    assert urls[0] == f"https://imgproxy.icook.network/safe/rt:fit/w:600/el:0/q:80/plain/{ORIGINAL}"
    assert urls[1] == "https://tokyo-kitchen.icook.network/uploads/recipe/cover/455340/7e9435a22d6413f3.jpg"
    assert bl.original_image_url(urls[0]) == ORIGINAL  # idempotent on re-runs
    assert bl.cover_candidates(None) == []


def test_split_steps():
    raw = "1. 備料：\n1.肉片解凍備用\n2.洋蔥切片\n\n料理步驟：熱鍋加油"
    assert bl.split_steps(raw) == ["備料：", "肉片解凍備用", "洋蔥切片", "料理步驟：熱鍋加油"]


def test_seasoning_does_not_set_category():
    from same_table.rules import classify_ingredient
    mains = [n for n in ["芹菜", "豆乾", "蠔油", "魚露"] if classify_ingredient(n) not in (None, "pantry")]
    assert bl.suggest(bl.CATEGORY_RULES, "芹菜鮮蔬炒豆乾 " + " ".join(mains), "veg") == "veg"
