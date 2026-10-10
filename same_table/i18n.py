"""zh / EN strings. Dish names always stay in library Chinese."""

from __future__ import annotations

STRINGS: dict[str, dict[str, str]] = {
    "app_title": {"zh": "糖友同枱", "en": "Same Table"},
    "tagline": {"zh": "一枱餸，全家食；只改糖友嗰碟。", "en": "One shared dinner. Only the patient's plate changes."},
    "disclaimer": {
        "zh": "醫院管理局課堂原型。並非醫療建議或處方，未連接 HA Go。",
        "en": "Classroom prototype for the Hospital Authority. Not medical advice or a prescription. Not connected to HA Go.",
    },
    "basic_mode": {"zh": "基本模式：以關鍵字配對菜式。", "en": "Basic mode: dishes matched by keywords."},
    # steps
    "step_tonight": {"zh": "今晚設定", "en": "Tonight"},
    "step_menu": {"zh": "今晚菜單", "en": "Menu"},
    "step_cook": {"zh": "呢道點煮", "en": "How to cook"},
    "step_plate": {"zh": "用餐建議", "en": "How to plate"},
    "step_list": {"zh": "清單", "en": "List"},
    # page 1
    "people": {"zh": "幾多人食飯？", "en": "How many people?"},
    "people_summary": {"zh": "{p} 人 · {n} 道餸", "en": "{p} people · {n} dishes"},
    "request": {"zh": "今晚想食乜？（可留空）", "en": "What would you like tonight? (optional)"},
    "request_ph": {"zh": "例如：阿爸牙唔好，想清淡啲，唔好太甜", "en": "e.g. Dad wants something light and not too sweet"},
    "chips": {"zh": "快選", "en": "Quick picks"},
    "chip_steam": {"zh": "蒸", "en": "Steam"},
    "chip_blanch": {"zh": "白灼", "en": "Blanch"},
    "chip_soup": {"zh": "湯", "en": "Soup"},
    "chip_stir": {"zh": "炒", "en": "Stir-fry"},
    "chip_more_veg": {"zh": "多啲菜", "en": "More veg"},
    "chip_more_meat": {"zh": "多啲肉", "en": "More meat"},
    "chip_less_sweet": {"zh": "少甜", "en": "Less sweet"},
    "chip_quick": {"zh": "20 分鐘內", "en": "Within 20 min"},
    "history": {"zh": "過去 14 日食過", "en": "Last 14 days"},
    "donut_center": {"zh": "{n} 道餸 · 14 日", "en": "{n} dishes · 14 days"},
    "donut_caption": {"zh": "只係數菜式數目，唔係飲食評分。", "en": "Dish counts only, not a diet score."},
    "plan_btn": {"zh": "排今晚菜單", "en": "Plan tonight's menu"},
    "planning": {"zh": "排緊菜單…", "en": "Planning…"},
    "loading": {"zh": "準備緊…", "en": "Getting ready…"},
    # categories / methods
    "cat_meat": {"zh": "肉", "en": "Meat"},
    "cat_seafood": {"zh": "海鮮", "en": "Seafood"},
    "cat_veg": {"zh": "菜", "en": "Veg"},
    "cat_soup": {"zh": "湯", "en": "Soup"},
    "cat_pantry": {"zh": "調味 / 乾貨", "en": "Pantry & seasoning"},
    "m_steam": {"zh": "蒸", "en": "Steamed"},
    "m_blanch": {"zh": "白灼", "en": "Blanched"},
    "m_soup": {"zh": "湯", "en": "Soup"},
    "m_stir": {"zh": "炒", "en": "Stir-fried"},
    "m_other": {"zh": "其他", "en": "Other"},
    # page 2
    "menu_header": {
        "zh": "{p} 人 · {n} 道餸 · 約 {pct}% 係 14 日內冇食過",
        "en": "{p} people · {n} dishes · about {pct}% new vs last 14 days",
    },
    "menu_header_note": {"zh": "以示範紀錄計算，並非模型評分。", "en": "Based on the demo log, not a model score."},
    "labeled": {"zh": "已標註", "en": "Labeled"},
    "unlabeled": {"zh": "未標註", "en": "Unlabeled"},
    "unlabeled_risk": {
        "zh": "未標註：唔喺菜式庫，冇材料資料，請自行判斷份量。",
        "en": "Unlabeled: not in the library, no ingredient data. Decide portions yourself.",
    },
    "minutes": {"zh": "{m} 分鐘", "en": "{m} min"},
    "servings": {"zh": "{s} 人份", "en": "Serves {s}"},
    "btn_cook": {"zh": "點煮", "en": "How to cook"},
    "btn_swap": {"zh": "換一道", "en": "Swap"},
    "btn_remove": {"zh": "移除", "en": "Remove"},
    "swap_none": {"zh": "菜式庫冇同類可換。", "en": "No other dish of this type in the library."},
    "add_unlabeled": {"zh": "加一道自己嘅餸（只填名）", "en": "Add your own dish (name only)"},
    "add_btn": {"zh": "加入", "en": "Add"},
    "added_library": {"zh": "「{name}」已喺菜式庫，已加入。", "en": "“{name}” is in the library and was added."},
    "already_on_menu": {"zh": "「{name}」已經喺菜單。", "en": "“{name}” is already on the menu."},
    "use_table": {"zh": "就食呢枱", "en": "Use this table"},
    "back_tonight": {"zh": "返回設定", "en": "Back to Tonight"},
    "note_repeat_allowed": {
        "zh": "新菜式唔夠，部分係 14 日內食過。",
        "en": "Not enough new dishes, so some repeat the last 14 days.",
    },
    "note_time_relaxed": {
        "zh": "20 分鐘內嘅餸唔夠，部分要耐啲。",
        "en": "Not enough dishes within 20 min, so some take longer.",
    },
    "note_category_swapped": {
        "zh": "某類菜式不足，已用其他類補上。",
        "en": "One category ran short, so other categories filled in.",
    },
    "note_library_short": {"zh": "菜式庫唔夠 {k} 道餸。", "en": "The library is {k} dish(es) short."},
    # page 3
    "ingredients": {"zh": "材料", "en": "Ingredients"},
    "steps": {"zh": "做法", "en": "Steps"},
    "no_recipe": {"zh": "菜式庫冇呢道餸嘅做法。", "en": "The library has no recipe for this dish."},
    "patient_box": {"zh": "糖友嗰碟", "en": "Patient's plate"},
    "image_caption": {"zh": "圖片來自菜式庫，非生成。", "en": "Image from the library, not generated."},
    "source": {"zh": "來源", "en": "Source"},
    "no_translation": {"zh": "", "en": "No English version of this recipe yet; shown in Chinese."},
    "translated_note": {"zh": "", "en": "English text translated from the original Chinese recipe."},
    "back_menu": {"zh": "返回菜單", "en": "Back to Menu"},
    # patient actions
    "act_sauce_aside": {"zh": "醬汁另上，唔好淋落飯", "en": "Sauce on the side, not on the rice"},
    "act_less_rice": {"zh": "飯比平時少啲", "en": "Less rice than usual"},
    "act_no_added_sugar": {"zh": "唔好另外加糖", "en": "No added sugar"},
    # page 4
    "plate_intro": {"zh": "同一枱餸。糖友只改碟上份量。", "en": "Same dishes for everyone. Only the patient's plate changes."},
    "col_patient": {"zh": "糖友", "en": "Patient"},
    "col_family": {"zh": "家人", "en": "Family"},
    "family_normal": {"zh": "同一道餸，照平常上碟", "en": "Same dish, normal plating"},
    "patient_unlabeled": {"zh": "自行判斷 · 有風險", "en": "Decide yourself · risk"},
    "go_list": {"zh": "睇清單", "en": "See the list"},
    # page 5
    "list_intro": {"zh": "合併今晚所有材料。唔係購物車。", "en": "All of tonight's ingredients, merged. Not a checkout."},
    "maybe_home": {"zh": "屋企可能已有", "en": "May already be at home"},
    "no_list_for": {"zh": "「{name}」：呢道餸冇清單", "en": "“{name}”: no list for this dish"},
    "for_dishes": {"zh": "用於：{d}", "en": "For: {d}"},
    "download": {"zh": "下載清單 (.txt)", "en": "Download list (.txt)"},
    "back_plate": {"zh": "返回用餐建議", "en": "Back to How to plate"},
    "empty_list": {"zh": "冇材料可以列出。", "en": "No ingredients to list."},
    "error_generic": {"zh": "出咗少少問題：{e}", "en": "Something went wrong: {e}"},
}


def t(key: str, lang: str = "zh", **kwargs) -> str:
    entry = STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(lang) or entry["zh"]
    return text.format(**kwargs) if kwargs else text
