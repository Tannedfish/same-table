"""Create the training and test data for both pipelines.

    python scripts/make_training_data.py

Writes into data/train/:
    p1_train.jsonl   template-generated cook sentences + 8 tag labels (fine-tuning)
    p1_eval.jsonl    same generator, but built ONLY from phrases held out of training (evaluation)
    p1_test.jsonl    HAND-WRITTEN sentences (never seen in training) -> the final test set
    p2_dev.jsonl     600 literal (query, dish_id) pairs built from each library dish
    p2_test.jsonl    200 more literal pairs with different sentence templates
    p2_paraphrase_test.jsonl  HAND-WRITTEN requests that name a food family or use English
                     ("poultry", "leafy greens", "tofu"); relevance comes from a rule on the dish name

Why templates: there is no public dataset of Cantonese dinner requests. The
training sentences are assembled from phrase banks written for this project;
the test sentences were written separately, in free form, so the test score
measures generalisation rather than template memorisation. Add real sentences
from your own group to p1_test.jsonl to make the test stronger.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from same_table.config import Settings  # noqa: E402
from same_table.library import load_library  # noqa: E402
from same_table.rules import classify_ingredient, normalize_ingredient  # noqa: E402
from same_table.tagging import ALL_TAGS  # noqa: E402

OUT = ROOT / "data" / "train"

# ----------------------------------------------------------------- Pipeline 1
# Each tag has explicit phrases (contain an obvious keyword) and implicit ones
# (no keyword: the model has to learn the meaning). Cantonese, Mandarin, English.
PHRASES: dict[str, list[str]] = {
    "steam": [
        "想食蒸嘅嘢", "整啲蒸餸", "清蒸就最好", "蒸條魚", "蒸水蛋都好", "用蒸嘅方法", "想吃蒸的", "來一道清蒸的",
        "steamed please", "steamed dishes please", "隔水蒸", "放入電飯煲上面一齊整", "唔想開油鑊，用蒸籠",
        "蒸排骨", "蒸雞", "蒸豆腐", "蒸嘅餸", "蒸多幾味", "steam some fish", "I'd like it steamed", "用電鍋蒸", "蒸熟就食得", "清蒸最健康", "蒸的料理",
    ],
    "blanch": [
        "白灼就得", "灼啲菜", "烚菜", "燙個青菜", "水煮的", "白灼蝦", "灼吓就食得", "blanched vegetables", "just poach it",
        "滾水拖一拖", "淥熟就算", "川燙一下", "唔落油，過水煮熟",
        "白灼菜心", "灼熟佢", "烚熟就得", "燙一燙", "汆水就食", "白灼海鮮", "boiled greens", "poached, no oil", "blanch the veg", "水煮青菜", "灼個生菜", "清水煮",
    ],
    "soup": [
        "想飲湯", "煲個湯", "滾個湯", "要有湯水", "來個湯", "想喝湯", "a soup would be nice", "need a soup", "有碗湯就好",
        "想飲啲熱辣辣嘅", "天氣凍想暖吓胃", "老人家要飲啲湯水",
        "煲老火湯", "滾湯都好", "飲啖湯", "要一個湯", "煮個湯", "今晚要有湯", "湯水滋潤", "soup please", "a bowl of soup", "some broth", "想喝熱湯", "來碗湯",
    ],
    "stir": [
        "炒兩味", "炒個菜", "小炒", "快炒", "想吃熱炒", "stir-fry", "a quick stir fry", "炒餸", "要有鑊氣", "爆炒",
        "落鑊兜兩兜", "想食啲香口惹味嘅", "大火兜亂",
        "炒個肉", "炒蛋", "炒兩個餸", "快炒一碟", "用鑊炒", "想食炒嘅", "wok-fried", "熱炒一盤", "炒得香啲", "清炒", "蒜蓉炒",
    ],
    "more_veg": [
        "多啲菜", "清淡啲", "少肉多菜", "想食齋啲", "蔬菜多些", "多點青菜", "more vegetables", "something light",
        "唔好咁油膩", "食得健康啲", "呢排食得太肥膩", "想清吓腸胃", "膽固醇高要戒口", "eat a bit greener",
        "多食菜", "菜要多", "青菜多啲", "少啲肉", "食素", "齋啲", "想吃清淡的", "蔬菜為主", "mainly veg", "more greens", "cut down on meat", "a lighter meal", "veggie heavy", "唔想食咁多肉",
    ],
    "more_meat": [
        "多啲肉", "要有肉", "肉多啲", "多點肉", "想食肉", "extra meat", "extra protein", "無肉不歡", "大啖肉",
        "細路發育要食多啲蛋白質", "做完運動要補充", "阿仔話唔夠飽", "啲後生仔食量大",
        "肉要夠", "多啲肉類", "想大啖食肉", "要多蛋白質", "肉多一點", "要有牛有雞", "加多個肉", "more protein", "lots of meat", "a meaty dinner", "肉食為主", "唔夠肉唔飽",
    ],
    "less_sweet": [
        "唔好太甜", "少甜", "唔落糖", "走糖", "不要太甜", "少糖", "less sweet", "no sugar please", "戒甜",
        "唔好用咁多茄汁", "啲醬汁唔好蜜糖味", "阿爸血糖高", "醫生話要控糖", "唔要甜酸嗰類",
        "唔要甜", "不要加糖", "唔好加糖", "少啲糖", "甜嘅唔要", "怕甜", "要低糖", "控制糖分", "low sugar", "without sugar", "skip the sweet sauce", "糖尿要戒甜", "無糖最好",
    ],
    "quick": [
        "趕時間", "快手啲", "20分鐘內搞掂", "冇乜時間", "簡單快趣", "要快", "quick", "in a hurry", "二十分鐘內",
        "好夜先返到屋企", "遲放工", "得半個鐘煮", "細路等緊食", "今日好攰唔想搞咁耐", "趕住出門口",
        "快啲煮好", "半個鐘內", "十五分鐘", "要快手", "冇時間煮", "即刻有得食", "越快越好", "煮得快嘅", "short on time", "need it fast", "ready in 20 minutes", "no time to cook", "下班很晚", "沒時間",
    ],
}
NEUTRAL = [
    "今晚食乜好", "隨便啦", "有咩建議", "幫我諗下今晚煮咩", "四個人食飯", "阿媽今晚過嚟食飯", "what should I cook tonight",
    "any ideas for dinner", "今晚吃什麼", "照舊啦", "冇所謂", "屋企六個人", "同平時差唔多", "老公話求其", "諗唔到食乜",
    "今晚得我同阿爸兩個", "dinner for three", "你話事", "是但", "想轉下口味",
    "食飯啦", "今晚開飯", "有咩好介紹", "想食好啲", "家常便飯", "同屋企人食飯", "what's for dinner", "the usual dinner", "three of us tonight", "今晚吃飯", "普通晚餐", "求其煮",
]
OPENERS = ["", "", "", "今晚", "今晚", "阿媽話", "老公想", "屋企四個人，", "阿爸有糖尿，", "今日", "tonight ", "我哋", "婆婆嚟食飯，"]
CLOSERS = ["", "", "", "，唔該", "啦", "，得唔得", "，幫我諗下", " please", "，謝謝", "呀"]
JOINERS = ["，", "，", "，同埋", "，另外", " and ", "，仲有", "。"]


def split_phrases(rng: random.Random, eval_share: float = 0.25):
    """Hold out a quarter of the phrases of every tag for the evaluation split, so that
    trainer.evaluate() measures wording the model has never seen (not template recall)."""
    train, held = {}, {}
    for tag, phrases in PHRASES.items():
        shuffled = phrases[:]
        rng.shuffle(shuffled)
        k = max(3, round(len(shuffled) * eval_share))
        held[tag], train[tag] = shuffled[:k], shuffled[k:]
    neutral = NEUTRAL[:]
    rng.shuffle(neutral)
    k = round(len(neutral) * eval_share)
    return (train, neutral[k:]), (held, neutral[:k])


def make_p1(n: int, rng: random.Random, PHRASES: dict, NEUTRAL: list, limit: int) -> list[dict]:
    rows = []
    for _ in range(n):
        k = rng.choices([0, 1, 2, 3], weights=[1.2, 4, 3.5, 1.5])[0]
        if k == 0:
            rows.append({"text": rng.choice(OPENERS) + rng.choice(NEUTRAL) + rng.choice(CLOSERS), "tags": []})
            continue
        tags = rng.sample(list(PHRASES), k)
        if "more_veg" in tags and "more_meat" in tags:  # contradictory request: keep only one
            tags.remove(rng.choice(["more_veg", "more_meat"]))
        parts = [rng.choice(PHRASES[t]) for t in tags]
        text = parts[0]
        for p in parts[1:]:
            text += rng.choice(JOINERS) + p
        rows.append({"text": rng.choice(OPENERS) + text + rng.choice(CLOSERS), "tags": sorted(tags)})
    seen, unique = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            unique.append(r)
    return unique[:limit]


# Hand-written, free-form. None of these come from the phrase banks above.
P1_TEST: list[tuple[str, list[str]]] = [
    ("阿爸牙唔好，想食啲軟熟嘅蒸餸，唔好太甜", ["steam", "less_sweet"]),
    ("今晚好趕，求其炒兩個餸就算", ["stir", "quick"]),
    ("想煲湯，另外加個白灼菜", ["soup", "blanch"]),
    ("屋企有三個男仔，肉要多", ["more_meat"]),
    ("最近食得好油，想清清地", ["more_veg"]),
    ("可唔可以蒸魚加個湯", ["steam", "soup"]),
    ("七點先放工，要十五分鐘搞得掂", ["quick"]),
    ("婆婆糖尿，醬汁唔可以甜", ["less_sweet"]),
    ("今晚想吃清蒸的，配一個燙青菜", ["steam", "blanch"]),
    ("我想喝湯，還要一個快炒", ["soup", "stir"]),
    ("something steamed and not sweet", ["steam", "less_sweet"]),
    ("quick stir fry with lots of veggies", ["stir", "quick", "more_veg"]),
    ("we want more meat tonight and a soup", ["more_meat", "soup"]),
    ("冇特別要求", []),
    ("今晚五個人食", []),
    ("跟平時一樣就可以", []),
    ("唔知食乜", []),
    ("天氣熱冇胃口，想食得清啲，灼菜就夠", ["more_veg", "blanch"]),
    ("個仔打完波好肚餓，要多啲肉，快啲有得食", ["more_meat", "quick"]),
    ("唔想成身油煙，用蒸嘅", ["steam"]),
    ("落雨天想飲啖熱湯", ["soup"]),
    ("醫生叫佢控制血糖，啲餸唔好落糖", ["less_sweet"]),
    ("想食得有鑊氣，炒牛肉都好", ["stir"]),
    ("青菜白灼，蝦都白灼", ["blanch"]),
    ("要有菜有湯，菜要多", ["soup", "more_veg"]),
    ("半個鐘內要開飯，蒸啲嘢最方便", ["quick", "steam"]),
    ("唔要甜酸排骨嗰類，改做蒸嘅", ["less_sweet", "steam"]),
    ("老人家同小朋友都食，煲個湯，炒個菜", ["soup", "stir"]),
    ("多菜少肉，唔好甜", ["more_veg", "less_sweet"]),
    ("食肉獸，冇肉唔得", ["more_meat"]),
    ("不想太油膩，蔬菜多一點", ["more_veg"]),
    ("時間不多，簡單炒一炒", ["quick", "stir"]),
    ("想吃水煮的青菜和蒸蛋", ["blanch", "steam"]),
    ("今晚要控糖，不要加糖", ["less_sweet"]),
    ("來一鍋湯，肉多一點", ["soup", "more_meat"]),
    ("light dinner, mostly vegetables, no sugar", ["more_veg", "less_sweet"]),
    ("I only have 20 minutes", ["quick"]),
    ("poached chicken and a clear soup", ["blanch", "soup"]),
    ("six people coming over", []),
    ("surprise me", []),
    ("唔好煎炸，蒸同灼就好", ["steam", "blanch"]),
    ("返到屋企都八點，要快手，有湯更好", ["quick", "soup"]),
    ("呢排肥咗，想食多啲菜", ["more_veg"]),
    ("阿女減肥，唔食甜嘢，多菜", ["less_sweet", "more_veg"]),
    ("想整個滾湯，十分鐘嗰種", ["soup", "quick"]),
    ("蒸肉餅啦，再炒個菜", ["steam", "stir"]),
    ("大家都想食肉，炒嘅", ["more_meat", "stir"]),
    ("淨係想食灼菜", ["blanch"]),
    ("有冇唔使落糖嘅蒸餸", ["less_sweet", "steam"]),
    ("公公咬唔到硬嘢，煲湯同蒸水蛋", ["soup", "steam"]),
    ("好攰，唔想諗，快就得", ["quick"]),
    ("想食啲惹味小炒，但唔好甜", ["stir", "less_sweet"]),
    ("今晚吃素一點", ["more_veg"]),
    ("三菜一湯", ["soup"]),
    ("要有魚有菜", []),
    ("阿媽鍾意食魚", []),
    ("唔食辣", []),
    ("兩個大人一個小朋友", []),
    ("想要白灼海鮮，唔好落醬", ["blanch"]),
    ("蛋白質要夠，啲仔要長高", ["more_meat"]),
    ("急，蒸", ["quick", "steam"]),
    ("炒，多肉，快", ["stir", "more_meat", "quick"]),
    ("湯，菜多，少甜", ["soup", "more_veg", "less_sweet"]),
    ("唔想食得太膩，清蒸或者白灼都得", ["more_veg", "steam", "blanch"]),
    ("收工遲，淥個菜炒個蛋算", ["quick", "blanch", "stir"]),
    ("no sweet sauces, steamed fish would be great", ["less_sweet", "steam"]),
    ("lots of protein, stir fried", ["more_meat", "stir"]),
    ("a warm broth and blanched greens", ["soup", "blanch"]),
    ("想吃有肉的，不要甜", ["more_meat", "less_sweet"]),
    ("下班晚了，快點煮好，清淡一點", ["quick", "more_veg"]),
    ("蒸的炒的都可以，要有湯", ["steam", "stir", "soup"]),
    ("唔該整返啲家常小菜", []),
    ("阿嫲想飲老火湯", ["soup"]),
    ("兩餸一湯，炒嘅", ["soup", "stir"]),
    ("白烚雞同烚菜", ["blanch"]),
    ("想要蒸水蛋同埋灼生菜", ["steam", "blanch"]),
    ("八點半先開飯，快炒算", ["quick", "stir"]),
    ("唔好甜，唔好炒，蒸就得", ["less_sweet", "steam"]),
    ("肉少啲，菜多啲", ["more_veg"]),
    ("啲男仔話要多肉少菜", ["more_meat"]),
    ("阿爸要戒糖，煲個清湯", ["less_sweet", "soup"]),
    ("十五分鐘煮好，灼個菜蒸條魚", ["quick", "blanch", "steam"]),
    ("想食住家菜", []),
    ("今晚有客人", []),
    ("老婆大肚，想食得清淡", ["more_veg"]),
    ("我想要有汁撈飯嘅餸", []),
    ("唔想洗咁多碗", []),
    ("蒸魚，炒菜，再加個湯", ["steam", "stir", "soup"]),
    ("湯要足料，肉要多", ["soup", "more_meat"]),
    ("用最快嘅方法整餐飯", ["quick"]),
    ("唔好落糖同蜜糖", ["less_sweet"]),
    ("淨係想食菜，灼就得", ["more_veg", "blanch"]),
    ("今晚簡單吃，燙個青菜煮個湯", ["blanch", "soup"]),
    ("想吃肉多一點的熱炒", ["more_meat", "stir"]),
    ("沒什麼時間，蒸的比較方便", ["quick", "steam"]),
    ("長輩血糖高，不要甜的", ["less_sweet"]),
    ("多一點蔬菜，再來個湯", ["more_veg", "soup"]),
    ("隨便煮都可以", []),
    ("steamed egg and stir-fried greens", ["steam", "stir"]),
    ("need dinner ready fast, soup is fine", ["quick", "soup"]),
    ("more veggies, less meat", ["more_veg"]),
    ("no sugar in anything please", ["less_sweet"]),
    ("meat lovers tonight", ["more_meat"]),
    ("just the usual", []),
    ("blanched prawns and steamed fish", ["blanch", "steam"]),
    ("好肚餓，快啲有得食就得", ["quick"]),
    ("想食啲鑊氣小炒，多啲肉", ["stir", "more_meat"]),
    ("湯水唔好甜", ["soup", "less_sweet"]),
]


# ----------------------------------------------------------------- Pipeline 2
METHOD_WORDS = {
    "steam": ["蒸", "清蒸", "蒸嘅"], "blanch": ["白灼", "灼", "水煮"], "soup": ["湯", "煲湯", "滾湯"],
    "stir": ["炒", "小炒", "快炒"], "other": ["家常"],
}
CAT_WORDS = {"meat": ["肉", "肉類"], "seafood": ["海鮮", "海產"], "veg": ["菜", "蔬菜"], "soup": ["湯", "湯水"]}
P2_TRAIN_TEMPLATES = [
    "想食{a}", "屋企有{a}，可以煮乜", "{m}{a}", "今晚整{a}同{b}", "有冇{m}嘅{c}，用{a}", "{a}點煮好",
    "想吃{a}，{m}的", "買咗{a}同{b}", "{c}，{m}，有{a}",
]
P2_TEST_TEMPLATES = [
    "雪櫃得返{a}，點算", "想要一道{m}嘅{c}，材料有{a}", "{a}{b}可以點整", "街市買到靚{a}，想{m}",
]


# Paraphrase test: the request names a food *family* or uses English, so the words do not appear
# in the dish text. A dish is relevant when its name matches the rule. Hand-written, test only.
FISH = ["魚", "斑", "鯛"]
P2_PARAPHRASE: list[dict] = [
    {"query": "想食禽肉", "name_any": ["雞", "鴨"]},
    {"query": "紅肉嘅餸", "name_any": ["牛", "豬", "排骨", "肉絲", "肉末", "肉片", "臘肉", "回鍋肉"]},
    {"query": "想食綠葉菜", "name_any": ["高麗菜", "小白菜", "芥菜", "菜心", "青江菜", "菠菜", "芥藍", "萵筍", "蘿蔓", "韭菜"]},
    {"query": "豆製品", "name_any": ["豆腐", "豆干", "豆乾", "干絲", "豆皮"]},
    {"query": "有殼嘅海產", "name_any": ["蝦", "蛤", "蚵", "螺", "鮑魚"]},
    {"query": "今晚想食條魚", "name_any": FISH, "name_not": ["魷魚", "鮑魚"]},
    {"query": "用雞蛋整嘅餸", "name_any": ["蛋"]},
    {"query": "菇菌類", "name_any": ["菇"]},
    {"query": "瓜類嘅菜", "name_any": ["瓜"]},
    {"query": "凍食嘅前菜", "name_any": ["涼拌"]},
    {"query": "想飲啲熱辣辣嘅湯水", "category": "soup"},
    {"query": "隔水整熟嘅海產", "category": "seafood", "method": "steam"},
    {"query": "惹味帶辣嘅", "name_any": ["辣", "剁椒", "泡菜"]},
    {"query": "根莖類蔬菜", "name_any": ["蘿蔔", "山藥", "馬鈴薯", "蓮藕", "筍"]},
    {"query": "醃肉同臘味", "name_any": ["鹹豬肉", "臘肉", "鴨賞", "香腸", "火腿"]},
    {"query": "用豬骨煲嘅湯", "name_any": ["排骨"], "category": "soup"},
    {"query": "chicken dishes", "name_any": ["雞"]},
    {"query": "I want fish tonight", "name_any": FISH, "name_not": ["魷魚", "鮑魚"]},
    {"query": "something with tofu", "name_any": ["豆腐"]},
    {"query": "mushrooms", "name_any": ["菇"]},
    {"query": "prawns or shrimp", "name_any": ["蝦"]},
    {"query": "a hot soup", "category": "soup"},
    {"query": "pork", "name_any": ["豬", "排骨", "肉絲", "肉末", "臘肉", "回鍋肉"]},
    {"query": "eggs", "name_any": ["蛋"]},
]


def relevant_ids(rule: dict, library) -> list[str]:
    out = []
    for d in library:
        if "category" in rule and d.category != rule["category"]:
            continue
        if "method" in rule and d.method != rule["method"]:
            continue
        if "name_any" in rule and not any(w in d.name for w in rule["name_any"]):
            continue
        if any(w in d.name for w in rule.get("name_not", [])):
            continue
        out.append(d.id)
    return out


def main_ingredients(dish) -> list[str]:
    names = []
    for ing in dish.ingredients:
        name = normalize_ingredient(ing.name)
        if 1 < len(name) <= 6 and classify_ingredient(name) not in (None, "pantry") and name not in names:
            names.append(name)
    return names[:4]


def make_p2(library, rng: random.Random) -> tuple[list[dict], list[dict]]:
    train, test = [], []
    for dish in library:
        mains = main_ingredients(dish) or [dish.name]

        def fill(template: str) -> str:
            a = rng.choice(mains)
            others = [x for x in mains if x != a] or [a]
            return template.format(
                a=a, b=rng.choice(others),
                m=rng.choice(METHOD_WORDS.get(dish.method or "other", ["家常"])),
                c=rng.choice(CAT_WORDS[dish.category]),
            )

        for template in rng.sample(P2_TRAIN_TEMPLATES, 6):
            train.append({"query": fill(template), "dish_id": dish.id})
        for template in rng.sample(P2_TEST_TEMPLATES, 2):
            test.append({"query": fill(template), "dish_id": dish.id})
    return train, test


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"{path.relative_to(ROOT)}: {len(rows)} rows")


def main() -> None:
    rng = random.Random(5240)
    OUT.mkdir(parents=True, exist_ok=True)

    test_texts = {t for t, _ in P1_TEST}
    (train_phrases, train_neutral), (eval_phrases, eval_neutral) = split_phrases(rng)
    p1_train = [r for r in make_p1(9000, rng, train_phrases, train_neutral, limit=3000) if r["text"] not in test_texts]
    p1_eval = [r for r in make_p1(3000, rng, eval_phrases, eval_neutral, limit=600) if r["text"] not in test_texts]
    unknown = {tag for _, tags in P1_TEST for tag in tags} - set(ALL_TAGS)
    assert not unknown, f"unknown tags in P1_TEST: {unknown}"
    write(OUT / "p1_train.jsonl", p1_train)
    write(OUT / "p1_eval.jsonl", p1_eval)
    write(OUT / "p1_test.jsonl", [{"text": t, "tags": sorted(tags)} for t, tags in P1_TEST])

    library = load_library(Settings.load().library_path)
    p2_train, p2_test = make_p2(library, rng)
    write(OUT / "p2_dev.jsonl", p2_train)   # literal queries, set A (Pipeline 2 is not fine-tuned: A + B are both test queries)
    write(OUT / "p2_test.jsonl", p2_test)
    paraphrase = [{"query": r["query"], "relevant_ids": relevant_ids(r, library)} for r in P2_PARAPHRASE]
    paraphrase = [r for r in paraphrase if r["relevant_ids"]]  # skip families the library does not contain
    write(OUT / "p2_paraphrase_test.jsonl", paraphrase)
    write(OUT / "p2_corpus.jsonl", [{"dish_id": d.id, "text": d.search_text()} for d in library])


if __name__ == "__main__":
    main()
