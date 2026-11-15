"""Real loopback requests against a planner whose committed timer state changes."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import sqlite3
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

import focus_bridge as bridge
from focus_domains import save_domains
from planner_store import PlannerStore


def add(store, name):
    return store.create_task(title=name, description="Prepare the experiment", day="2026-11-14",
                             estimate_minutes=10, prediction={})["id"]


@pytest.fixture
def linked(tmp_path):
    store = PlannerStore(tmp_path / "live")
    key = bridge.bind(tmp_path, store.path)["token"]
    bridge.ensure_listener(tmp_path, port=0)
    server = bridge.LISTENERS[(tmp_path.resolve(), 0)]
    server.poll_seconds = 3
    yield store, key, server.server_port
    server.shutdown()
    server.server_close()
    bridge.LISTENERS.pop((tmp_path.resolve(), 0))


def fetch(port, token, *, after="", headers=None):
    req = Request(f"http://127.0.0.1:{port}/v1/sites?after={after}", headers={
        "Authorization": "Bearer " + token, "X-Planner-Browser": "profile-001", **(headers or {})})
    with build_opener(ProxyHandler({})).open(req, timeout=5) as response:
        return json.load(response)


def test_running_rule_follows_commits_with_dashboard_closed(linked, tmp_path):
    store, token, port = linked
    a, b = add(store, "Code"), add(store, "Review")
    save_domains(store, a, ["x.com"], [])
    save_domains(store, b, ["nba.com"], [])
    none = fetch(port, token)
    assert none["domains"] == []
    store.transition(a, "start")
    first = fetch(port, token, after=none["revision"])
    assert first["domains"] == ["x.com"]
    with ThreadPoolExecutor() as executor:
        pending = executor.submit(fetch, port, token, after=first["revision"])
        # The server may already have a long-poll outstanding when the timer switches.
        store.transition(b, "start")
        assert pending.result()["domains"] == ["nba.com"]
    store.transition(b, "pause")
    assert fetch(port, token)["domains"] == []
    store.transition(b, "resume")
    save_domains(store, b, ["nba.com", "x.com"], ["nba.com"])
    assert fetch(port, token)["domains"] == ["nba.com", "x.com"]
    store.transition(b, "stop")
    assert fetch(port, token)["domains"] == []
    store.transition(b, "finish")
    assert fetch(port, token)["domains"] == []
    bridge.disable(tmp_path)
    with pytest.raises(HTTPError) as error:
        fetch(port, token)
    assert error.value.code == 403
    assert bridge.bind(tmp_path, store.path)["token"] != token


def test_loopback_authentication_and_foreign_origins(linked):
    _, token, port = linked
    for key, headers in [("bad", {}), (token, {"Origin": "https://example.org"}),
                         (token, {"Host": "other.example"}), (token, {"X-Planner-Browser": ""})]:
        with pytest.raises(HTTPError) as error:
            fetch(port, key, headers=headers)
        assert error.value.code == 403
    assert set(fetch(port, token, headers={"Origin": "chrome-extension://" + "a" * 32})) == {"revision", "domains"}


def test_only_bound_editable_workspace_can_apply_rules_even_after_reset(linked, tmp_path):
    live, token, port = linked
    dev = PlannerStore(tmp_path / "dev" / "gpu" / "generation-a")
    with closing(sqlite3.connect(tmp_path / "dev/profiles.sqlite3")) as db, db:
        db.execute("CREATE TABLE profiles(id TEXT, generation TEXT)")
        db.execute("INSERT INTO profiles VALUES('gpu','generation-a')")
    sample = add(dev, "GPU")
    save_domains(dev, sample, ["youtube.com"], [])
    dev.transition(sample, "start")
    assert fetch(port, token)["domains"] == []
    assert bridge.bind(tmp_path, dev.path)["token"] == token
    assert fetch(port, token)["domains"] == ["youtube.com"]
    with closing(sqlite3.connect(tmp_path / "dev/profiles.sqlite3")) as db, db:
        db.execute("UPDATE profiles SET generation='generation-b'")
    assert fetch(port, token)["domains"] == []
    with pytest.raises(ValueError):
        bridge.bind(tmp_path, PlannerStore(tmp_path / "dev/replays/gpu/generation-r").path)
    assert bridge.bind(tmp_path, live.path)["token"] == token


def test_task_editor_and_settings_pair_then_revoke_across_streamlit_reruns(tmp_path, monkeypatch):
    from datetime import date
    from pathlib import Path
    import estimate_worker
    import report_worker
    from focus_domains import policy
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("PLANNER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(estimate_worker, "schedule_pending", lambda *_: None)
    monkeypatch.setattr(report_worker, "schedule_reports", lambda *_: None)
    monkeypatch.setattr(bridge, "ensure_listener", lambda *_: None)
    store = PlannerStore(tmp_path / "live")
    task = store.create_task(title="Capture a GPU trace", description="Diagnose the intermittent failure.",
                             estimate_minutes=15, day=date.today(), prediction={})
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()

    def click(label):
        next(b for b in app.button if b.label == label).click().run()
        assert not app.exception

    next(w for w in app.multiselect if w.label == "Blocked websites").set_value(["x.com"])
    click("Save websites")
    assert policy(store, task["id"])["domains"] == ["x.com"]
    click("Planner settings")
    click("Enable browser blocking for your personal planner")
    code = next(w for w in app.text_input if w.label == "Browser pairing code")
    assert code.value == bridge.binding(tmp_path)["token"]
    click("Disconnect all browsers and release sites")
    assert not bridge.binding(tmp_path)
