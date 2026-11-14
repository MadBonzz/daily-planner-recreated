import sqlite3

import pytest

from planner_store import DEFAULT_CATEGORIES, PlannerStore


def make_task(store, category):
    return store.create_task(title="Review", description="Review a paper", category=category,
                             day="2026-11-14", estimate_minutes=50, prediction={})


def test_category_list_persists_and_is_separate_for_each_workspace(tmp_path):
    live = PlannerStore(tmp_path / "live")
    sample = PlannerStore(tmp_path / "dev")
    assert live.list_categories() == list(DEFAULT_CATEGORIES)
    assert live.add_category(" Reading  ") == "Reading"
    with pytest.raises(ValueError, match="already exists"):
        live.add_category("READING")
    with pytest.raises(ValueError, match="Enter a category name"):
        live.add_category("   ")
    assert make_task(live, "reading")["category"] == "Reading"
    assert "Reading" in PlannerStore(tmp_path / "live").list_categories()
    assert "Reading" not in sample.list_categories()
    live.clear_data()
    assert live.list_categories() == list(DEFAULT_CATEGORIES)


def test_existing_categories_are_available_after_upgrade(tmp_path):
    folder = tmp_path / "legacy"
    old = PlannerStore(folder)
    make_task(old, "Linear algebra")
    with sqlite3.connect(old.path) as db:
        db.execute("DROP TABLE categories")
        db.execute("DELETE FROM settings WHERE key='category_list_initialized'")
    reopened = PlannerStore(folder)
    assert reopened.list_categories() == [*DEFAULT_CATEGORIES, "Linear algebra"]
    assert reopened.list_tasks()[0]["category"] == "Linear algebra"
    assert make_task(reopened, "Hardware measurements")["category"] in reopened.list_categories()


def test_deleting_a_note_source_clears_saved_cautions(tmp_path):
    store = PlannerStore(tmp_path)
    prior = make_task(store, "Study")
    snapshot = {"point_min": 20, "evidence": [], "cautions": [
        {"advice": "A prior note was sensitive.", "sources": [{"task_id": prior["id"]}]}
    ]}
    new = store.create_task(title="Home reading", description="Appendix", day="2026-11-14",
                            category="Study", estimate_minutes=20, prediction=snapshot)
    store.delete_task(prior["id"])
    saved = store.get_task(new["id"])["prediction"]
    assert saved["point_min"] == 20 and not saved.get("cautions")
