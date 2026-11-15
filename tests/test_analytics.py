from datetime import datetime
import json
import sqlite3
from unittest.mock import patch

import pytest

from planner_analytics import analytics_rows, accuracy, ask_analytics, run_query


def _records():
    dataset = []
    for i, (budget, actual) in enumerate(((30, 40), (10, 5), (20, 20), (10, 0))):
        original = {"point_min": 120, "provenance": {"minutes_by": "history"}}
        if i == 0:
            original = {"point_min": 42, "provenance": {"minutes_by": "model"}}
        dataset.append({
            "id": str(i), "title": f"Flyer {i}", "category": "Office", "day": "2025-01-01",
            "completed_at": datetime(2025, 2, 10, 10).timestamp(), "actual_seconds": actual * 60,
            "estimate_minutes": budget, "creation_prediction": original, "prediction": {"point_min": 999},
            "archived": int(i == 1),
        })
    dataset[0]["report"] = {"intervals": [
        {"start": 1739181600, "seconds": 25, "app": "Browser", "domain": "docs.example",
         "title": "Public flyer draft", "label": "on_task"},
        {"start": 1739181630, "seconds": 30, "app": "Secret Bank", "domain": "secret.example", "excluded": True},
        {"start": 1739181660, "seconds": 60, "phone_app": "Private Mobile", "reason": "sensitive"},
    ]}
    return analytics_rows(dataset, ["PRIVATE"])


def test_metrics_keep_archived_history_and_exclude_untimed_and_non_model_forecasts():
    tasks, logs = _records()
    assert len(logs) == 1 and logs[0]["domain"] == "docs.example"
    assert tasks[1]["archived"] == 1 and all(t["completed_day"] == "2025-02-10" for t in tasks)
    assert tasks[-1]["actual_minutes"] is None
    assert run_query("SELECT COUNT(actual_minutes) AS timed FROM tasks", tasks, logs)[0] == [{"timed": 3}]
    stats, = accuracy(tasks, "user_minutes")
    assert (stats["Tasks"], stats["Mean absolute error (min)"], stats["Exact"]) == (3, 5.0, 1)
    assert (stats["Under (%)"], stats["Mean shortfall (min)"]) == (33.3, 10)
    assert (stats["Over (%)"], stats["Mean surplus (min)"]) == (33.3, 5)
    model, = accuracy(tasks, "model_minutes")
    assert model["Tasks"] == 1 and model["Mean surplus (min)"] == 2


@pytest.mark.parametrize("sql", [
    "DROP TABLE tasks", "ATTACH DATABASE 'should-never-exist' AS disk", "SELECT * FROM sqlite_master",
    "PRAGMA database_list", "SELECT load_extension('library')", "SELECT randomblob(9999999)",
    "SELECT CAST(title AS blob) FROM tasks", "SELECT 1e309 FROM tasks",
    "SELECT * FROM tasks; DELETE FROM tasks", "WITH RECURSIVE seq(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM seq) SELECT * FROM seq",
])
def test_sql_is_read_only_and_bounded(sql):
    with pytest.raises((sqlite3.Error, ValueError)):
        run_query(sql, *_records())


def test_query_joins_scoped_rows_and_answer_shows_evidence():
    tasks, logs = _records()
    query = ("SELECT title, domain, sum(seconds) AS total_seconds FROM tasks JOIN logs USING(task_id) "
             "GROUP BY title, domain ORDER BY total_seconds DESC")
    answers = [{"scope": "all", "tool": "custom_sql", "task_ids": [], "sql": query}, {"answer": "The flyer draft used docs.example for 25 seconds."}]
    with patch("planner_analytics.chat_json", side_effect=answers) as model:
        result = ask_analytics("Which task used docs.example?", tasks, logs)
    assert result["rows"] == [{"title": "Flyer 0", "domain": "docs.example", "total_seconds": 25}]
    assert not result["more"]
    assert json.loads(model.call_args.args[0][-1]["content"])["result_rows"] == result["rows"]
    assert "secret.example" not in str(model.call_args_list)


def test_query_aggregates_before_limiting_evidence_rows():
    tasks, _ = _records()
    many = [dict(tasks[0], task_id=str(i)) for i in range(45)]
    assert run_query("SELECT COUNT(*) AS n FROM tasks", many, []) == ([{"n": 45}], False)
    rows, truncated = run_query("SELECT task_id FROM tasks ORDER BY CAST(task_id AS INT)", many, [])
    assert len(rows) == 40 and truncated


def test_activity_crossing_midnight_is_credited_by_day_and_keeps_review_source():
    start = datetime(2025, 4, 15, 23, 59, 50).timestamp()
    _, logs = analytics_rows([{
        "id": "read", "title": "Read", "day": "2025-04-15", "estimate_minutes": 5, "actual_seconds": 30,
        "report": {"intervals": [{"start": start, "seconds": 30, "label": "on_task", "label_source": "user"}]},
    }], [])
    assert [(r["started"][:10], r["seconds"], r["source"]) for r in logs] == [
        ("2025-04-15", 10, "user"), ("2025-04-16", 20, "user")]


def test_purge_during_planning_does_not_send_old_results():
    with patch("planner_analytics.chat_json", return_value={"sql": "SELECT title FROM tasks"}) as model:
        result = ask_analytics("Which tasks?", *_records(), still_current=lambda: False)
    assert model.call_count == 1 and "rows" not in result and "Flyer" not in str(result)


def test_chat_cannot_answer_from_sql_constants_without_reading_records():
    with pytest.raises(ValueError):
        run_query("SELECT 'invented' AS answer", *_records())


@pytest.mark.parametrize("tool", ["custom_sql", "distraction_sources"])
def test_topic_query_scopes_distractions_to_gpu_and_preserves_phone_only_rows(tool):
    tasks, _ = _records()
    tasks[0].update(category="GPU lab", title="Bound the CUDA warp tail")
    base = {"started": "2025-02-10T10:20:00", "app": "", "window": "", "domain": "",
            "device": "Phone", "phone_app": "Word game", "seconds": 180, "source": "user", "reason": "",
            "classification": "off_task"}
    logs = [dict(base, task_id="0"), dict(base, task_id="1", seconds=7200, phone_app="Unrelated office app")]
    sql = "SELECT phone_app, SUM(seconds)/60 AS minutes FROM logs WHERE classification='off_task' GROUP BY phone_app"
    planned = {"scope": "category:GPU lab", "tool": tool, "task_ids": [], "sql": sql if tool == "custom_sql" else ""}
    with patch("planner_analytics.forecast_endpoint") as endpoint, patch("planner_analytics.chat_json") as model:
        endpoint.return_value.__enter__.return_value = {"provider": "Cohere", "model": "command-r7b-12-2024"}
        reply = {"finding_ids": [1]} if tool == "distraction_sources" else {
            "answer": "A phone word game appeared for three minutes."}
        model.side_effect = [planned, reply]
        result = ask_analytics("What distracts me on GPU programming tasks?", tasks, logs)
    evidence, = result["rows"]
    assert evidence["phone_app"] == "Word game" and evidence["minutes"] == 3.0
    if tool == "distraction_sources":
        assert (evidence["device"], evidence["tasks"], evidence["days"]) == ("Phone", 1, 1)
    assert result["scope"]["task_count"] == 1 and result["scope"]["interval_count"] == 1
    assert result["provider"] == "Cohere" and result["answer_source"] == result["query_source"] == "model"
    assert "Unrelated office app" not in str(model.call_args_list)
    assert "GPU programming" in model.call_args_list[0].args[0][-1]["content"]


def test_no_matches_still_goes_to_model_for_a_grounded_explanation():
    tasks, logs = _records()
    sql = "SELECT phone_app FROM logs WHERE classification='off_task'"
    plan = {"scope": "all", "tool": "custom_sql", "task_ids": [], "sql": sql}
    with patch("planner_analytics.chat_json", side_effect=[plan, {"sql": ""}, {"finding": "The selected data did not yield a matching result for this query.", "next_step": "Inspect the query and current filters below."}]) as model:
        result = ask_analytics("Was my phone a distraction?", tasks, logs)
    assert result["answer_source"] == "model" and result["rows"] == []
    payload = json.loads(model.call_args_list[-1].args[0][-1]["content"])
    assert payload["visible_interval_counts"] == {"on_task": 1}
    assert payload["result_rows"] == [] and payload["scope"]["interval_count"] == 1


def test_unavailable_narrator_does_not_claim_model_authorship():
    from model_client import ModelUnavailable
    plan = {"scope": "all", "tool": "custom_sql", "task_ids": [], "sql": "SELECT COUNT(*) AS tasks FROM tasks"}
    with patch("planner_analytics.chat_json", side_effect=[plan, ModelUnavailable("offline")]):
        result = ask_analytics("How many?", *_records())
    assert result["query_source"] == "model" and result["answer_source"] == "local status"
    assert result["rows"] == [{"tasks": 4}]


def test_analytics_page_charts_and_device_filter():
    from streamlit.testing.v1 import AppTest
    script = """
from pathlib import Path
import streamlit as st
from planner_analytics import render_analytics
class Sample:
    path = Path("test-analytics")
    def get_setting(self, key, default): return default
    def completed_tasks(self):
        return [{"id":"gpu", "title":"Debug a CUDA block", "category":"GPU lab", "day":"2025-01-02",
                 "estimate_minutes":5, "actual_seconds":360, "report":{"intervals":[
                 {"start":1735819200, "seconds":180, "phone_app":"Number game", "label":"off_task"},
                 {"start":1735819380, "seconds":180, "app":"Editor", "title":"kernel.cu", "label":"on_task"}]}}]
render_analytics(Sample())
"""
    app = AppTest.from_string(script).run()
    assert not app.exception
    charts = app.get("vega_lite_chart")
    assert len(charts) == 4
    scatter = json.loads(charts[1].proto.spec)
    assert "params" not in scatter and scatter["layer"][1]["params"][0]["bind"] == "scales"
    app.selectbox[-1].select("Phone").run()
    assert not app.exception
    logs = app.dataframe[-1].value
    assert list(logs["Device"]) == ["Phone"] and list(logs["Classification"]) == ["Distraction"]


def test_accuracy_query_stays_with_user_estimates_and_only_averages_actual_shortfalls():
    plan = {"scope": "all", "tool": "user_estimate_accuracy", "task_ids": [], "sql": ""}
    with patch("planner_analytics.chat_json", side_effect=[plan, {"answer": "Underestimated tasks ran 10 minutes over."}]):
        result = ask_analytics("When I underestimate, by how many minutes?", *_records())
    overall = next(row for row in result["rows"] if row["level"] == "overall")
    assert overall["mean_underestimated_by_minutes"] == 10 and overall["underestimated_tasks"] == 1
    assert overall["mean_overestimated_by_minutes"] == 5 and overall["overestimated_tasks"] == 1


def test_empty_sql_aggregate_returns_no_measurement():
    plan = {"scope": "all", "tool": "custom_sql", "task_ids": [],
            "sql": "SELECT AVG(model_minutes) AS average FROM tasks WHERE 0"}
    with patch("planner_analytics.chat_json", side_effect=[plan, {
        "finding": "The calculation returned no measured value for the selected records.",
        "next_step": "Inspect the query and current filters below."}]) as model:
        result = ask_analytics("What is the average?", *_records())
    assert result["rows"] == [{"average": None}]
    assert "no measured value" in result["answer"]
    assert "enum" in model.call_args.args[1]["properties"]["finding"]


def test_model_repairs_spurious_date_predicate_before_answering():
    old = "SELECT title, actual_minutes FROM tasks WHERE completed_day='2999-01-01'"
    new = "SELECT title, actual_minutes FROM tasks ORDER BY actual_minutes DESC LIMIT 1"
    plan = {"scope": "all", "tool": "custom_sql", "task_ids": [], "sql": old}
    with patch("planner_analytics.chat_json", side_effect=[plan, {"sql": new}, {"answer": "Flyer 0 took 40 minutes."}]):
        result = ask_analytics("Which task took the longest?", *_records())
    assert result["first_sql"] == old and result["sql"] == new
    assert result["rows"] == [{"title": "Flyer 0", "actual_minutes": 40}]



def test_last_week_query_uses_local_calendar_and_preserves_date_constraint():
    from planner_analytics import _local_calendar
    from datetime import date
    calendar = _local_calendar(date(2026, 1, 1))
    assert (calendar["last_week_starts"], calendar["last_week_ends"]) == ("2025-12-22", "2025-12-28")
    assert (calendar["last_month_starts"], calendar["last_month_ends"]) == ("2025-12-01", "2025-12-31")
    plan = {"scope": "all", "tool": "custom_sql", "task_ids": [],
            "sql": "SELECT COUNT(*) AS tasks FROM tasks WHERE completed_day BETWEEN '2025-12-22' AND '2025-12-28'"}
    with patch("planner_analytics.chat_json", side_effect=[plan, {"answer": "There were no completed tasks."}]) as model:
        ask_analytics("How many tasks did I finish last week?", *_records())
    messages, schema = model.call_args_list[0].args
    assert json.loads(messages[-1]["content"])["local_calendar"] == _local_calendar(date.today())
    assert schema["properties"]["tool"]["enum"] == ["custom_sql"]
