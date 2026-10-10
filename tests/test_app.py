"""Click through all five pages with Streamlit's AppTest (offline mode)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def run_app():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def page(at):
    return at.session_state["page"]


def test_full_flow():
    at = run_app()
    assert page(at) == "tonight"
    at.text_area(key="w_text").set_value("想食蒸魚，唔好太甜").run()
    at.button(key="plan").click().run()
    assert not at.exception, at.exception
    assert page(at) == "menu"
    ids = list(at.session_state["menu_ids"])
    assert len(ids) == 6
    assert [d.name for d in at.session_state["addons"]] == ["糖醋排骨"]  # demo unlabeled

    at.button(key=f"swap_{ids[0]}").click().run()
    assert at.session_state["menu_ids"][0] != ids[0]

    at.text_input(key="w_addname").set_value("咖喱雞").run()
    at.button(key="add_unlabeled_btn").click().run()
    assert {d.name for d in at.session_state["addons"]} == {"糖醋排骨", "咖喱雞"}

    first = at.session_state["menu_ids"][0]
    at.button(key=f"cook_{first}").click().run()
    assert page(at) == "cook" and not at.exception
    at.button(key="back_menu").click().run()

    at.button(key="use_table").click().run()
    assert page(at) == "plate" and at.session_state["logged"]
    at.button(key="go_list").click().run()
    assert page(at) == "list" and not at.exception
    warnings = [w.value for w in at.warning]
    assert any("糖醋排骨" in w for w in warnings) and any("咖喱雞" in w for w in warnings)


def test_english_toggle_and_disclaimer():
    at = run_app()
    at.radio(key="lang").set_value("en").run()
    html = " ".join(m.value for m in at.markdown)
    assert "Not medical advice" in html and "Tonight" in html


def test_remove_unlabeled_and_add_library_dish(library):
    at = run_app()
    at.button(key="plan").click().run()
    demo_id = at.session_state["addons"][0].id
    at.button(key=f"remove_{demo_id}").click().run()
    assert at.session_state["addons"] == []
    menu = list(at.session_state["menu_ids"])
    extra = next(d for d in library if d.id not in menu)
    at.text_input(key="w_addname").set_value(extra.name).run()
    at.button(key="add_unlabeled_btn").click().run()
    assert at.session_state["menu_ids"] == menu + [extra.id]  # library hit -> labeled, not unlabeled
    assert at.session_state["addons"] == []


def test_english_mode_shows_translated_dishes(library):
    at = run_app()
    at.radio(key="lang").set_value("en").run()
    at.button(key="plan").click().run()
    first = library.get(at.session_state["menu_ids"][0])
    page_html = " ".join(m.value for m in at.markdown)
    assert first.display_name("en") in page_html and first.name in page_html   # English name + Chinese under it
    at.button(key=f"cook_{first.id}").click().run()
    page_html = " ".join(m.value for m in at.markdown)
    assert first.display_steps("en")[0] in page_html
    at.button(key="back_menu").click().run()
    at.button(key="use_table").click().run()
    at.button(key="go_list").click().run()
    labels = [c.label for c in at.checkbox]
    assert labels and not any(any("\u4e00" <= ch <= "\u9fff" for ch in label) for label in labels)
