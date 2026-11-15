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
    answers = [{"sql": query}, {"answer": "The flyer draft used docs.example for 25 seconds."}]
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
