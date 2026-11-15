"""Exercise the five Dev histories through the same picker and calendar a Mac demo uses."""

from streamlit.testing.v1 import AppTest

import app
import dev_workspace
import task_workspace


PROFILES = (
    "d1-systems-student", "d2-lab-researcher", "d3-clinic-coordinator",
    "d4-signal-student", "d5-field-engineer",
)


def test_demo_picker_opens_five_histories_and_their_analytics(tmp_path, monkeypatch):
    choices = dev_workspace.history_choices()
    assert set(choices) == set(PROFILES)
    monkeypatch.setattr(app, "DATA_ROOT", tmp_path)
    # This smoke covers navigation and the imported records. The packaged replay test
    # runs the real forecast workers, replacing only the external forecast HTTP response.
    monkeypatch.setattr(app, "schedule_pending", lambda *_: None)
    monkeypatch.setattr(app, "schedule_reports", lambda *_: None)
    monkeypatch.setattr(task_workspace, "availability", lambda *_: {"eligible": False})
    page = AppTest.from_string("from app import main\nmain()", default_timeout=30).run()
    assert not page.exception
    next(item for item in page.get("segmented_control") if item.key == "planner_mode").set_value("Dev").run()
    assert not page.exception
    assert set(page.selectbox(key="dev_history").options) == set(choices.values())

    for profile in PROFILES:
        page.selectbox(key="dev_history").set_value(profile).run()
        assert not page.exception
        assert page.selectbox(key=f"sample-task-{profile}").options
        store = dev_workspace.open_dev_workspace(tmp_path, profile)
        assert store.path.resolve().is_relative_to((tmp_path / "dev").resolve())
        assert any(task.get("report", {}).get("intervals") for task in store.completed_tasks() if task.get("report"))
        assert any("Date" in table.value.columns and "Recorded (min)" in table.value.columns for table in page.dataframe)
    # Follow the macOS quick start through the visible controls and verify that
    # the attached handout and pending forecast were saved in the selected workspace.
    page.selectbox(key="dev_history").set_value("d1-systems-student").run()
    sample = page.selectbox(key="sample-task-d1-systems-student")
    assert "Solve part B of the bridge handout" in sample.options
    sample.set_value("bridge-proof").run()
    next(button for button in page.button if button.label == "Add sample task").click().run()
    assert not page.exception
    systems = dev_workspace.open_dev_workspace(tmp_path, "d1-systems-student")
    added = next(task for task in systems.list_tasks() if task["title"] == "Solve part B of the bridge handout")
    assert added["prediction"]["status"] == "pending"
    attachment = systems.get_task(added["id"])["attachments"][0]
    assert systems.attachment_bytes(attachment["id"]).startswith(b"%PDF-")
    next(item for item in page.get("segmented_control") if item.key == "planner_view").set_value("Analytics & logs").run()
    assert not page.exception
    assert any("Device" in table.value.columns and "Classification" in table.value.columns for table in page.dataframe)
