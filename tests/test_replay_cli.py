from datetime import date
import json
import socket

import pytest

from examples.estimation.replay import main, replay


def test_replay_offline_scores_only_observed_completions_and_refuses_overwrite(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Offline replay attempted a network connection")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    output = tmp_path / "profile-01"
    main(["01", "--offline", "--actions", "5", "--anchor-day", "2026-11-15", "--output", str(output)])
    entries = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert entries[0]["as_of"] == "2026-11-08T08:30:00"
    assert entries[0]["forecast_jobs"][0]["history_available"] == []
    first = entries[0]["changes"]["clock-two"]["forecast"]
    assert first["source"] == "fallback" and first["provenance"]["model"] is None
    assert first["low_min"] <= first["point_min"] <= first["high_min"]
    assert first["used_tasks"] == [] and first["explanation"]
    assert entries[-1]["status"] == "completed" and entries[-1]["recorded_min"] == 26
    assert entries[-1]["changes"]["clock-two"]["report"]["elapsed_seconds"] == 26 * 60
    summary = json.loads((output / "summary.json").read_text())
    assert summary["schedule_complete"] is False
    assert summary["overall"] == {"count": 1, "user_mae_min": 9, "initial_planner_mae_min": 9}
    assert {name: group["count"] for name, group in summary["by_source"].items()} == {
        "model": 0, "fallback": 1, "scheduled": 0}
    with pytest.raises(FileExistsError):
        replay("01", output, actions=1, offline=True)


def test_fixed_slots_have_their_own_score_and_no_forecaster_job(tmp_path):
    output = tmp_path / "theatre"
    summary = replay("08", output, offline=True, anchor=date(2026, 11, 15))
    assert summary["schedule_complete"]
    assert summary["by_source"]["scheduled"] == {"count": 3, "user_mae_min": 0, "initial_planner_mae_min": 0}
    assert summary["overall"]["count"] == 8
    first = json.loads((output / "events.jsonl").read_text().splitlines()[0])
    assert first["changes"]["radio-cabaret"]["forecast"]["source"] == "scheduled" and first["forecast_jobs"] == []
