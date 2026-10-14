"""Same Table / 糖友同枱 — Streamlit app.

Run:  streamlit run app.py
Pages: Tonight -> Menu -> (How to cook) -> How to plate -> List
"""

from __future__ import annotations

import html
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

from same_table.config import ROOT, Settings
from same_table.i18n import t
from same_table.library import CATEGORIES, Dish, Library, LibraryError, load_library, make_unlabeled
from same_table.planner import plan_tonight
from same_table.retrieval import Retriever
from same_table.rules import (
    GROUPS,
    PATIENT_ACTIONS,
    build_shopping_list,
    dish_count,
    novelty_ratio,
    seed_log,
    swap_dish,
)
from same_table.speech import AUDIO_TYPES, SpeechError, Transcriber
from same_table.tagging import CHIPS, Tagger

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Same Table 糖友同枱", page_icon="🍲", layout="centered")

HK = ZoneInfo("Asia/Hong_Kong")
PAGES = ("tonight", "menu", "cook", "plate", "list")
STEPPER = ("tonight", "menu", "plate", "list")
CAT_COLORS = {"meat": "#C8553D", "seafood": "#2E86AB", "veg": "#6A9A1F", "soup": "#D9A21B"}
CAT_EMOJI = {"meat": "🥩", "seafood": "🐟", "veg": "🥬", "soup": "🍲", None: "❔"}
DEMO_UNLABELED = "糖醋排骨"

CSS = """
<style>
:root { --forest:#1F4D3A; --lime:#C6E377; --amber:#FFF4D6; --amber-edge:#E0A526; --warn:#FFF7C2; }
.st-hero { background:var(--forest); color:#fff; border-radius:14px; padding:14px 18px; margin-bottom:6px; }
.st-hero h1 { color:#fff; font-size:1.5rem; margin:0; padding:0; }
.st-hero p { color:var(--lime); margin:2px 0 0; font-size:.92rem; }
.st-steps { display:flex; gap:6px; margin:6px 0 12px; flex-wrap:wrap; }
.st-steps span { font-size:.8rem; padding:3px 10px; border-radius:999px; background:#E8EFEA; color:#1F4D3A; }
.st-steps span.on { background:var(--forest); color:var(--lime); font-weight:600; }
.badge { display:inline-block; font-size:.75rem; padding:1px 8px; border-radius:999px; margin-right:4px; }
.b-lab { background:#E3F2D0; color:#2F5310; }
.b-unl { background:#FFE58A; color:#6B4E00; }
.b-cat { background:#EEF1EF; color:#1F4D3A; }
.dish-name { font-size:1.15rem; font-weight:700; margin:0; }
.dish-sub { font-size:.85rem; color:#6B7C72; margin:0 0 4px; }
.photo-ph { font-size:2.4rem; text-align:center; background:#F1F5F2; border-radius:10px; padding:10px 0; }
.patient { background:var(--amber); border-left:5px solid var(--amber-edge); border-radius:8px; padding:10px 14px; margin-top:8px; }
.patient b { color:#7A4B00; }
.unl-box { background:var(--warn); border-left:5px solid #E0C200; border-radius:8px; padding:8px 12px; font-size:.9rem; }
[data-testid="stImage"] img { width:100%; height:170px; object-fit:cover; border-radius:10px; }
.st-key-cover_big [data-testid="stImage"] img { height:300px; }
.disclaimer { font-size:.78rem; color:#6B7C72; border-top:1px solid #E1E7E3; margin-top:24px; padding-top:8px; }
</style>
"""


# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return Settings.load()


@st.cache_resource(show_spinner=False)
def get_library(path: str) -> Library:
    return load_library(path)


@st.cache_resource(show_spinner=False)
def get_tagger(_settings: Settings) -> Tagger:
    return Tagger(_settings).load()


@st.cache_resource(show_spinner=False)
def get_retriever(_settings: Settings, _library: Library) -> Retriever:
    return Retriever(_settings, _library).load()


@st.cache_resource(show_spinner=False)
def get_transcriber(_settings: Settings) -> Transcriber:
    return Transcriber(_settings)  # the speech model loads on the first voice note, not at start-up


# --------------------------------------------------------------------- state
def today():
    return datetime.now(HK).date()


def init_state(library: Library) -> None:
    ss = st.session_state
    defaults = {
        "page": "tonight",
        "people": 4,
        "text": "",
        "chips": [],
        "result": None,
        "menu_ids": [],
        "addons": [],
        "swapped_out": set(),
        "selected": None,
        "logged": False,
        "flash": [],
        "demo_done": False,
    }
    for key, value in defaults.items():
        ss.setdefault(key, value)
    if "log" not in ss:
        ss.log = seed_log(library, today())


def persist(widget_key: str, store_key: str) -> None:
    """Widgets lose state when their page is not drawn; keep a copy."""
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state[store_key]


def go(page: str, selected: str | None = None) -> None:
    st.session_state.page = page
    if selected is not None:
        st.session_state.selected = selected


def lang() -> str:
    return st.session_state.get("lang", "zh")


def menu_dishes(library: Library) -> list[Dish]:
    ss = st.session_state
    return [library.get(i) for i in ss.menu_ids] + list(ss.addons)


def find_dish(library: Library, dish_id: str) -> Dish | None:
    if dish_id in library:
        return library.get(dish_id)
    return next((d for d in st.session_state.addons if d.id == dish_id), None)


# --------------------------------------------------------------- components
def header(active: str) -> None:
    L = lang()
    left, right = st.columns([4, 1.3])
    with left:
        st.markdown(
            f"<div class='st-hero'><h1>{t('app_title', 'zh')} · Same Table</h1><p>{t('tagline', L)}</p></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.radio("lang", ["zh", "en"], key="lang", horizontal=True, label_visibility="collapsed",
                 format_func=lambda x: "中" if x == "zh" else "EN")
    shown = "menu" if active == "cook" else active
    pills = "".join(
        f"<span class='{'on' if p == shown else ''}'>{i}. {t('step_' + p, L)}</span>"
        for i, p in enumerate(STEPPER, start=1)
    )
    st.markdown(f"<div class='st-steps'>{pills}</div>", unsafe_allow_html=True)
    for msg, kind in st.session_state.flash:
        getattr(st, kind)(msg)
    st.session_state.flash = []


def footer(basic_mode: bool) -> None:
    L = lang()
    extra = f"<br>{t('basic_mode', L)}" if basic_mode else ""
    st.markdown(f"<div class='disclaimer'>{t('disclaimer', L)}{extra}</div>", unsafe_allow_html=True)


def dish_title(dish: Dish, size: str = "card") -> str:
    """Dish name in the chosen language; in English mode the Chinese name stays underneath."""
    L = lang()
    main = html.escape(dish.display_name(L))
    sub = html.escape(dish.name) if L == "en" and dish.has_english() else ""
    if size == "page":
        out = f"<h2 style='margin-bottom:0'>{main}</h2>"
    else:
        out = f"<p class='dish-name'>{main}</p>"
    if sub:
        out += f"<p class='dish-sub'>{sub}</p>"
    return out


def badges(dish: Dish) -> str:
    L = lang()
    if not dish.labeled:
        return f"<span class='badge b-unl'>{t('unlabeled', L)}</span>"
    parts = [f"<span class='badge b-lab'>{t('labeled', L)}</span>",
             f"<span class='badge b-cat'>{t('cat_' + dish.category, L)}</span>"]
    if dish.method not in (None, "other", dish.category):  # no "Soup · Soup"
        parts.append(f"<span class='badge b-cat'>{t('m_' + dish.method, L)}</span>")
    if dish.cooking_time:
        parts.append(f"<span class='badge b-cat'>⏱ {t('minutes', L, m=dish.cooking_time)}</span>")
    return "".join(parts)


def cover_source(dish: Dish) -> str | None:
    """Local copy first (works offline and if iCook moves again), then the URL."""
    if dish.cover_path:
        local = ROOT / dish.cover_path
        if local.is_file():
            return str(local)
    return dish.cover_url


def photo(dish: Dish, big: bool = False) -> None:
    source = cover_source(dish)
    if source:
        st.image(source, width="stretch")
        if big:
            st.caption(t("image_caption", lang()))
    else:
        st.markdown(f"<div class='photo-ph'>{CAT_EMOJI.get(dish.category, '❔')}</div>", unsafe_allow_html=True)


def patient_box() -> None:
    L = lang()
    items = "".join(f"<li>{t('act_' + a, L)}</li>" for a in PATIENT_ACTIONS)
    st.markdown(f"<div class='patient'><b>{t('patient_box', L)}</b><ol>{items}</ol></div>", unsafe_allow_html=True)


def donut(counts: dict[str, int]) -> None:
    L = lang()
    df = pd.DataFrame({
        "category": [t("cat_" + c, L) for c in CATEGORIES],
        "count": [counts[c] for c in CATEGORIES],
    })
    chart = (
        alt.Chart(df)
        .mark_arc(innerRadius=55, outerRadius=90)
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color("category:N", sort=list(df["category"]),
                            scale=alt.Scale(domain=list(df["category"]), range=[CAT_COLORS[c] for c in CATEGORIES]),
                            legend=alt.Legend(orient="right", title=None)),
            tooltip=["category", "count"],
        )
        .properties(height=200)
    )
    st.altair_chart(chart)


# --------------------------------------------------------------------- pages
def _transcribe(transcriber: Transcriber) -> None:
    """Callback: voice note -> text appended to the request box (runs before the box is drawn)."""
    ss, L = st.session_state, lang()
    clip = ss.get("w_voice_rec") or ss.get("w_voice_file")
    if clip is None:
        ss.flash.append((t("voice_none", L), "info"))
        return
    try:
        with st.spinner(t("voice_working", L)):
            heard = transcriber.transcribe(clip.getvalue())
    except SpeechError as exc:
        ss.flash.append((t("voice_err_" + str(exc), L), "warning"))
        return
    except Exception as exc:  # model or runtime failure: keep the app usable
        ss.flash.append((t("error_generic", L, e=exc), "warning"))
        return
    if not heard:
        ss.flash.append((t("voice_err_nothing", L), "warning"))
        return
    combined = f"{ss.get('text', '')} {heard}".strip()[:300]
    ss.text = ss.w_text = combined
    ss.flash.append((t("voice_heard", L, text=heard), "success"))


def voice_note(transcriber: Transcriber) -> None:
    L = lang()
    with st.expander(t("voice_title", L), expanded=False):
        if not transcriber.enabled:
            st.caption(t("voice_off", L))
            return
        st.caption(t("voice_help", L))
        st.audio_input(t("voice_record", L), key="w_voice_rec", sample_rate=16000)
        st.file_uploader(t("voice_upload", L), type=list(AUDIO_TYPES), key="w_voice_file", max_upload_size=10)
        st.button(t("voice_btn", L), key="transcribe", on_click=_transcribe, args=(transcriber,), icon=":material/mic:")


def page_tonight(library, tagger, retriever, transcriber) -> None:
    L, ss = lang(), st.session_state
    persist("w_people", "people")
    persist("w_text", "text")
    persist("w_chips", "chips")

    st.number_input(t("people", L), min_value=1, max_value=6, step=1, key="w_people")
    ss.people = int(ss.w_people)
    st.markdown(f"**{t('people_summary', L, p=ss.people, n=dish_count(ss.people))}**")

    st.text_area(t("request", L), key="w_text", placeholder=t("request_ph", L), max_chars=300, height=90)
    ss.text = ss.w_text
    voice_note(transcriber)
    st.pills(t("chips", L), options=list(CHIPS), selection_mode="multi", key="w_chips",
             format_func=lambda c: t("chip_" + c, L))
    ss.chips = list(ss.w_chips or [])

    st.markdown(f"#### {t('history', L)}")
    counts = ss.log.category_counts(library, today())
    c1, c2 = st.columns([2.2, 1])
    with c1:
        donut(counts)
    with c2:
        st.markdown(f"<div style='font-size:1.25rem;font-weight:700;margin-top:48px'>"
                    f"{t('donut_center', L, n=sum(counts.values()))}</div>", unsafe_allow_html=True)
        st.caption(t("donut_caption", L))

    if st.button(t("plan_btn", L), type="primary", key="plan", width="stretch"):
        with st.spinner(t("planning", L)):
            try:
                result = plan_tonight(people=ss.people, text=ss.text, chips=ss.chips, library=library,
                                      tagger=tagger, retriever=retriever, log=ss.log, today=today())
            except (ValueError, LibraryError) as exc:
                st.error(t("error_generic", L, e=exc))
                return
        ss.result = result
        ss.menu_ids = list(result.plan.dish_ids)
        ss.swapped_out = set()
        ss.logged = False
        ss.addons = []
        if get_settings().demo_mode and not ss.demo_done:
            ss.addons = [make_unlabeled(DEMO_UNLABELED)]
            ss.demo_done = True
        go("menu")
        st.rerun()


def _add_unlabeled(library: Library) -> None:
    ss, L = st.session_state, lang()
    name = (ss.get("w_addname") or "").strip()
    if not name:
        return
    names_on_menu = {n.casefold() for d in menu_dishes(library) for n in (d.name, d.display_name("en"))}
    if name.casefold() in names_on_menu:
        ss.flash.append((t("already_on_menu", L, name=name), "info"))
    elif (hit := library.by_name(name)) is not None:
        ss.menu_ids.append(hit.id)
        ss.flash.append((t("added_library", L, name=name), "success"))
    else:
        try:
            ss.addons.append(make_unlabeled(name))
        except LibraryError as exc:
            ss.flash.append((t("error_generic", L, e=exc), "warning"))
    ss.w_addname = ""


def _swap(library: Library, dish_id: str) -> None:
    ss, L = st.session_state, lang()
    new = swap_dish(ss.menu_ids, dish_id, ss.result.scores, library, ss.log.recent_ids(today()),
                    quick=ss.result.tags.quick, exclude=ss.swapped_out)
    if new is None:
        ss.flash.append((t("swap_none", L), "info"))
        return
    ss.swapped_out.add(dish_id)
    ss.menu_ids[ss.menu_ids.index(dish_id)] = new


def _remove(dish_id: str) -> None:
    ss = st.session_state
    ss.addons = [d for d in ss.addons if d.id != dish_id]
    ss.menu_ids = [i for i in ss.menu_ids if i != dish_id]


def _use_table(library: Library) -> None:
    ss = st.session_state
    if not ss.logged:
        ss.log.add([d.id for d in menu_dishes(library) if d.labeled], today())
        ss.logged = True
    go("plate")


def page_menu(library) -> None:
    L, ss = lang(), st.session_state
    if ss.result is None:
        go("tonight")
        st.rerun()
    res = ss.result
    dishes = menu_dishes(library)
    recent = ss.log.recent_ids(today())  # tonight's own entries are dated today, so excluded
    pct = round(100 * novelty_ratio([d.id for d in dishes if d.labeled], recent))
    n_lib = sum(d.labeled for d in dishes)
    extra = f" (+{len(dishes) - n_lib})" if len(dishes) > n_lib else ""
    st.markdown(f"### {t('menu_header', L, p=res.people, n=f'{n_lib}{extra}', pct=pct)}")
    st.caption(t("menu_header_note", L))
    understood = sorted(res.tags.methods) + ([res.tags.balance] if res.tags.balance != "balanced" else []) \
        + [flag for flag in ("less_sweet", "quick") if getattr(res.tags, flag)]
    shown = " · ".join(t("chip_" + tag, L) for tag in understood) or t("understood_none", L)
    st.markdown(f"<p class='dish-sub'>{t('understood', L)}: <b>{html.escape(shown)}</b></p>", unsafe_allow_html=True)
    for note in res.plan.notes:
        st.info(t("note_" + note, L, k=res.plan.shortfall))

    for dish in dishes:
        with st.container(border=True):
            c_img, c_body = st.columns([1, 3.2])
            with c_img:
                photo(dish)
            with c_body:
                st.markdown(dish_title(dish) + badges(dish), unsafe_allow_html=True)
                if not dish.labeled:
                    st.markdown(f"<div class='unl-box'>{t('unlabeled_risk', L)}</div>", unsafe_allow_html=True)
                b1, b2, _ = st.columns([1, 1, 1])
                b1.button(t("btn_cook", L), key=f"cook_{dish.id}", on_click=go, args=("cook", dish.id))
                if dish.labeled:
                    b2.button(t("btn_swap", L), key=f"swap_{dish.id}", on_click=_swap, args=(library, dish.id))
                else:
                    b2.button(t("btn_remove", L), key=f"remove_{dish.id}", on_click=_remove, args=(dish.id,))

    with st.container(border=True):
        a1, a2 = st.columns([3, 1])
        a1.text_input(t("add_unlabeled", L), key="w_addname", max_chars=40, placeholder=DEMO_UNLABELED)
        a2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        a2.button(t("add_btn", L), key="add_unlabeled_btn", on_click=_add_unlabeled, args=(library,))

    c1, c2 = st.columns(2)
    c1.button(t("back_tonight", L), key="back_tonight", on_click=go, args=("tonight",), width="stretch")
    c2.button(t("use_table", L), key="use_table", type="primary", on_click=_use_table, args=(library,),
              width="stretch", disabled=not dishes)


def page_cook(library) -> None:
    L, ss = lang(), st.session_state
    dish = find_dish(library, ss.selected) if ss.selected else None
    if dish is None:
        go("menu")
        st.rerun()
    st.markdown(dish_title(dish, "page") + badges(dish), unsafe_allow_html=True)
    with st.container(key="cover_big"):
        photo(dish, big=True)
    meta = []
    if dish.servings:
        meta.append(t("servings", L, s=dish.servings))
    if dish.cooking_time:
        meta.append("⏱ " + t("minutes", L, m=dish.cooking_time))
    if meta:
        st.caption(" · ".join(meta))

    if not dish.labeled:
        st.markdown(f"<div class='unl-box'>{t('unlabeled_risk', L)}</div>", unsafe_allow_html=True)
    if L == "en" and dish.labeled and not dish.has_english():
        st.caption(t("no_translation", L))
    ingredients, steps = dish.display_ingredients(L), dish.display_steps(L)
    if ingredients:
        st.markdown(f"#### {t('ingredients', L)}")
        st.markdown("\n".join(f"- {i.name}" + (f" — {i.amount}" if i.amount else "") for i in ingredients))
    if steps:
        st.markdown(f"#### {t('steps', L)}")
        st.markdown("\n".join(f"{n}. {s}" for n, s in enumerate(steps, start=1)))
    if L == "en" and dish.has_english():
        st.caption(t("translated_note", L))
    if not dish.ingredients and not dish.steps:
        st.caption(t("no_recipe", L))
    if dish.labeled:
        patient_box()
    if dish.source_url:
        st.caption(f"{t('source', L)}: {dish.source_url}")
    st.button(t("back_menu", L), key="back_menu", on_click=go, args=("menu",))


def page_plate(library) -> None:
    L = lang()
    dishes = menu_dishes(library)
    st.markdown(f"### {t('step_plate', L)}")
    st.caption(t("plate_intro", L))
    actions = "<br>".join(f"{n}. {t('act_' + a, L)}" for n, a in enumerate(PATIENT_ACTIONS, start=1))
    for dish in dishes:
        with st.container(border=True):
            st.markdown(dish_title(dish) + badges(dish), unsafe_allow_html=True)
            p, f = st.columns(2)
            with p:
                st.markdown(f"**{t('col_patient', L)}**")
                if dish.labeled:
                    st.markdown(f"<div class='patient' style='margin-top:0'>{actions}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='unl-box'>{t('patient_unlabeled', L)}</div>", unsafe_allow_html=True)
            with f:
                st.markdown(f"**{t('col_family', L)}**")
                st.write(t("family_normal", L))
    c1, c2 = st.columns(2)
    c1.button(t("back_menu", L), key="back_menu_plate", on_click=go, args=("menu",), width="stretch")
    c2.button(t("go_list", L), key="go_list", type="primary", on_click=go, args=("list",), width="stretch")


def page_list(library) -> None:
    L = lang()
    shopping = build_shopping_list(menu_dishes(library), lang=L)
    st.markdown(f"### {t('step_list', L)}")
    st.caption(t("list_intro", L))
    lines = []
    if shopping.is_empty():
        st.info(t("empty_list", L))
    for group in GROUPS:
        items = shopping.groups[group]
        if not items:
            continue
        st.markdown(f"#### {t('cat_' + group, L)}")
        lines.append(f"[{t('cat_' + group, L)}]")
        for item in items:
            amount = "" if item.maybe_at_home else " + ".join(item.amounts)
            label = item.label + (f"  ({amount})" if amount else "")
            if item.maybe_at_home:
                label += f"  · {t('maybe_home', L)}"
            st.checkbox(label, key=f"chk_{group}_{item.name}", help=t("for_dishes", L, d=(", " if L == "en" else "、").join(item.dishes)))
            lines.append(f"- {label}")
        lines.append("")
    for name in shopping.no_list:
        st.warning(t("no_list_for", L, name=name))
        lines.append(t("no_list_for", L, name=name))
    c1, c2 = st.columns(2)
    c1.button(t("back_plate", L), key="back_plate", on_click=go, args=("plate",), width="stretch")
    c2.download_button(t("download", L), "\n".join(lines).encode("utf-8"), file_name="same_table_list.txt",
                       mime="text/plain", width="stretch", key="download")


# ---------------------------------------------------------------------- main
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    settings = get_settings()
    try:
        library = get_library(str(settings.library_path))
    except LibraryError as exc:
        st.error(f"Library error: {exc}")
        st.stop()
    with st.spinner(t("loading", st.session_state.get("lang", "zh"))):
        tagger = get_tagger(settings)
        retriever = get_retriever(settings, library)
    transcriber = get_transcriber(settings)
    init_state(library)

    page = st.session_state.page if st.session_state.page in PAGES else "tonight"
    header(page)
    if page == "tonight":
        page_tonight(library, tagger, retriever, transcriber)
    elif page == "menu":
        page_menu(library)
    elif page == "cook":
        page_cook(library)
    elif page == "plate":
        page_plate(library)
    else:
        page_list(library)
    footer(basic_mode=not (tagger.available and retriever.available))


main()
