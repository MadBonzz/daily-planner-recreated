"""Read-only SQLite source, privacy boundaries and Windows store discovery."""

import json
from pathlib import Path
import sqlite3
from urllib.parse import quote

import pytest

import activity
import activity_db


BASE = 1_790_000_000
NS = 1_000_000_000
BUCKET = "aw-watcher-window_laptop"


def make_db(path, desktop="laptop", *, android_only=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.executescript("""
            PRAGMA user_version=6;
            CREATE TABLE buckets (id INTEGER PRIMARY KEY, name TEXT UNIQUE, type TEXT,
                                  client TEXT, hostname TEXT, created TEXT, data TEXT DEFAULT '{}');
            CREATE TABLE events (id INTEGER PRIMARY KEY, bucketrow INTEGER REFERENCES buckets(id),
                                 starttime INTEGER, endtime INTEGER, data TEXT);
            CREATE INDEX events_bucketrow_endtime_starttime_index
            ON events(bucketrow, endtime, starttime DESC);
        """)
        window = ("aw-watcher-android_phone", "currentwindow", "aw-watcher-android", "phone") if android_only else (
            BUCKET, "currentwindow", "aw-watcher-window", desktop)
        con.executemany("INSERT INTO buckets(name,type,client,hostname,created) VALUES(?,?,?,?, '2026-01-01')",
                        [window, ("aw-watcher-web-chrome_laptop", "web.tab.current", "aw-client-web", "unknown"),
                         ("aw-watcher-afk_laptop", "afkstatus", "aw-watcher-afk", desktop),
                         ("aw-watcher-web-firefox/slash", "web.tab.current", "aw-client-web", desktop)])
        for bucket, start, end, data in [
            (1, -30, 0, {"app": "Code", "title": "outside the timer"}),
            (1, -1, 2.125, {"app": "chrome.exe", "title": "Course notes"}),
            (1, 2.125, 10, {"app": "chrome.exe", "title": "Private window title"}),
            (1, 9, 9, {"app": "Code", "title": "zero duration"}),
            (1, 10, 15, {"app": "Code", "title": "outside the timer"}),
            (2, 0, 2.125, {"url": "https://notes.example.org/docs?nonce=hidden", "title": "Course notes"}),
            (2, 2.125, 10, {"url": "https://do-not-share.example.org/private", "title": "unrelated secret"}),
            (3, 0, 10, {"status": "not-afk"}),
            (4, 0, 10, {"url": "https://test.example.org/", "title": "encoded bucket"}),
        ]:
            con.execute("INSERT INTO events(bucketrow,starttime,endtime,data) VALUES(?,?,?,?)",
                        (bucket, BASE * NS + round(start * NS), BASE * NS + round(end * NS), json.dumps(data)))
    return path


def test_read_bounded_events_across_timer_edges(tmp_path):
    source = activity_db.SQLiteActivitySource(make_db(tmp_path / "test #1.db"), hostname="laptop")
    assert source.get("buckets/")[BUCKET]["type"] == "currentwindow"
    path = f"buckets/{BUCKET}/events"
    result = source.get(path, start=activity._iso(BASE), end=activity._iso(BASE + 10))
    assert [(r["timestamp"] - BASE, r["duration"]) for r in result] == [(2.125, 7.875), (-1, 3.125)]
    encoded = "buckets/" + quote("aw-watcher-web-firefox/slash", safe="") + "/events"
    assert source.get(encoded, start=BASE, end=BASE + 10)[0]["data"]["title"] == "encoded bucket"
    with pytest.raises(ValueError):
        source.get(path)
    assert source.get(path, start=BASE + 8, end=BASE + 7) == []


def test_demo_reads_database_through_same_filtered_model_path(tmp_path, monkeypatch):
    path = make_db(tmp_path / "test.db", desktop="foreign-computer")
    monkeypatch.setenv("PLANNER_ACTIVITYWATCH_DB", str(path))
    monkeypatch.setattr(activity, "_get", lambda *a, **kw: pytest.fail("attempted HTTP"))
    asked = []

    def model(messages, schema, **kwargs):
        asked.extend(json.loads(messages[1]["content"])["intervals"])
        return {key: {"label": "on_task", "reason": "Course notes open."} for key in schema["required"]}

    monkeypatch.setattr(activity, "chat_json", model)
    report = activity.build_report({"title": "Study"}, [(BASE, BASE + 10)], ["do-not-share"])
    preview = activity.live_preview(BASE, BASE + 10, ["do-not-share"])
    assert preview["warnings"] == []
    assert report["totals"] == {"on_task": 2.125, "off_task": 0, "uncertain": 7.875}
    assert len(asked) == 1 and asked[0]["domain"] == "notes.example.org"
    sanitized = json.dumps([asked, report, preview])
    for secret in ("do-not-share", "unrelated secret", "Private window title", "nonce", "encoded bucket"):
        assert secret not in sanitized


def test_missing_explicit_database_cannot_create_a_file_or_switch_to_other_devices(tmp_path, monkeypatch):
    path = tmp_path / "absent.db"
    monkeypatch.setenv("PLANNER_ACTIVITYWATCH_DB", str(path))
    monkeypatch.setattr(activity, "_get", lambda *a, **kw: pytest.fail("attempted HTTP"))
    result = activity.live_preview(BASE, BASE + 10)
    assert not path.exists()
    assert "SQLite" in result["warnings"][0] and result["intervals"][0]["app"] == ""


def test_read_sees_committed_wal_updates(tmp_path):
    store = make_db(tmp_path / "live.db")
    source = activity_db.SQLiteActivitySource(store, hostname="laptop")
    with sqlite3.connect(store) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("UPDATE events SET endtime=? WHERE id=2", ((BASE + 3) * NS,))
        writer.commit()
        assert Path(str(store) + "-wal").stat().st_size > 0
        result = source.get(f"buckets/{BUCKET}/events", start=BASE, end=BASE + 1)
        assert result[0]["duration"] == 4


def test_windows_discovery_prefers_authoritative_db_then_synced_desktop(tmp_path, monkeypatch):
    monkeypatch.setattr(activity_db.sys, "platform", "win32")
    profile = tmp_path / "User"
    local = tmp_path / "AppData" / "Local"
    sync = profile / "Downloads" / "syncthing-windows-amd64-v2.1.7" / "syncthing-windows-amd64-v2.1.7" / "activity-watch-one-plus"
    phone = make_db(sync / "0-phone" / "test.db", android_only=True)
    mirror = make_db(sync / "desktop-host" / "uuid" / "test.db", desktop="fixture-host")
    primary = make_db(local / "activitywatch" / "aw-server-rust" / "sqlite.db")
    config = {"USERPROFILE": str(profile), "LOCALAPPDATA": str(local)}
    assert activity_db.windows_database_candidates(config) == (primary, phone, mirror)
    assert activity_db.configured_desktop_source(config).path == primary
    primary.unlink()
    source = activity_db.configured_desktop_source(config)
    assert source.path == mirror and source.hostname == "fixture-host"
    # The explicitly configured live path is used even when path validation fails.
    assert activity_db.configured_desktop_source({**config, "PLANNER_ACTIVITYWATCH_DB": str(primary)}).path == primary


def test_macos_discovers_rust_database_and_honors_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setattr(activity_db.sys, "platform", "darwin")
    monkeypatch.setattr(activity_db.Path, "home", classmethod(lambda cls: tmp_path))
    native = tmp_path / "Library/Application Support/activitywatch/aw-server-rust/sqlite.db"
    assert activity_db.configured_desktop_source({}) is None
    assert not native.parent.exists()
    make_db(native, desktop="test-mac.local")
    source = activity_db.configured_desktop_source({})
    assert source.path == native and source.hostname == "test-mac.local"
    assert source.get(f"buckets/{BUCKET}/events", start=BASE, end=BASE + 1)
    other = make_db(tmp_path / "elsewhere.sqlite", desktop="other")
    assert activity_db.configured_desktop_source({"PLANNER_ACTIVITYWATCH_DB": str(other)}).path == other


def test_capture_off_skips_db_discovery_and_filesystem_reads(monkeypatch):
    monkeypatch.setattr(activity, "configured_desktop_source", lambda *a: pytest.fail("looked up desktop store"))
    monkeypatch.setattr(activity.activity_phone, "configured_phone_source", lambda *a: pytest.fail("looked up phone store"))
    result = activity.live_preview(BASE, BASE + 10, capture=False)
    assert all(row["excluded"] for row in result["intervals"])


def test_phone_source_in_parallel_with_desktop_is_sanitized_and_counts_once(tmp_path, monkeypatch):
    desktop = make_db(tmp_path / "desktop.db")
    phone = make_db(tmp_path / "phone.db", android_only=True)
    with sqlite3.connect(phone) as db:
        for event, app, classname in [(2, "Calendar", "internal.calendar"), (3, "Messages", "private.thread")]:
            db.execute("UPDATE events SET data=? WHERE id=?",
                       (json.dumps({"app": app, "package": "test.app", "classname": classname}), event))
    monkeypatch.setenv("PLANNER_ACTIVITYWATCH_DB", str(desktop))
    monkeypatch.setenv("PLANNER_ACTIVITYWATCH_PHONE_DB", str(phone))
    sent = []

    def classify(messages, schema, **kwargs):
        sent.extend(json.loads(messages[1]["content"])["intervals"])
        return {key: {"label": "uncertain", "reason": "Two apps are open."} for key in schema["required"]}

    monkeypatch.setattr(activity, "chat_json", classify)
    report = activity.build_report({"title": "Review notes"}, [(BASE, BASE + 10)], exclusions=["private.thread"])
    assert report["elapsed_seconds"] == sum(report["totals"].values()) == 10
    assert len(sent) == 1 and sent[0]["phone_app"] == "Calendar" and sent[0]["app"] == "chrome.exe"
    hidden = report["intervals"][-1]
    assert hidden["excluded"] and not hidden["app"] and "phone_app" not in hidden
    visible = json.dumps([report, sent])
    assert all(term not in visible for term in ("internal.calendar", "test.app", "private.thread", "Messages"))


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_synced_android_store_takes_precedence_over_stale_import(tmp_path, monkeypatch, platform):
    monkeypatch.setattr(activity_db.sys, "platform", platform)
    monkeypatch.setattr(activity_db.Path, "home", classmethod(lambda cls: tmp_path))
    if platform == "darwin":
        local = tmp_path / "Library" / "Application Support"
    else:
        local = tmp_path / "AppData" / "Local"
        monkeypatch.setenv("LOCALAPPDATA", str(local))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
    primary = make_db(local / "activitywatch" / "aw-server-rust" / "sqlite.db")
    with sqlite3.connect(primary) as db:
        db.execute("INSERT INTO buckets(name,type,client,hostname,created) VALUES(?,?,?,?,?)",
                   ("aw-watcher-android_phone", "currentwindow", "aw-watcher-android", "phone", "2026-01-01"))
        db.execute("INSERT INTO events(bucketrow,starttime,endtime,data) VALUES(5,?,?,?)",
                   (BASE * NS, (BASE + 2) * NS, json.dumps({"app": "Old copy"})))
    phone = make_db(tmp_path / "Downloads" / "activity-watch-one-plus" / "uuid" / "test.db", android_only=True)
    with sqlite3.connect(phone) as db:
        db.execute("UPDATE events SET data=? WHERE bucketrow=1", (json.dumps({"app": "Current phone app"}),))
    if platform == "darwin":
        monkeypatch.setenv("PLANNER_ACTIVITYWATCH_PHONE_DB", str(phone))
    result = activity.live_preview(BASE, BASE + 2)
    assert [(r["app"], r.get("phone_app"), r["seconds"]) for r in result["intervals"]] == [
        ("chrome.exe", "Current phone app", 2)]


def test_one_corrupted_synced_phone_event_preserves_its_valid_neighbor(tmp_path, monkeypatch):
    desktop = make_db(tmp_path / "desktop.db")
    phone = make_db(tmp_path / "phone.db", android_only=True)
    with sqlite3.connect(phone) as db:
        db.execute("UPDATE events SET data='not-json' WHERE id=2")
        db.execute("UPDATE events SET data=? WHERE id=3", (json.dumps({"app": "Garden notes"}),))
    monkeypatch.setenv("PLANNER_ACTIVITYWATCH_DB", str(desktop))
    monkeypatch.setenv("PLANNER_ACTIVITYWATCH_PHONE_DB", str(phone))
    report = activity.live_preview(BASE, BASE + 10)
    assert any("invalid phone event" in warning for warning in report["warnings"])
    assert sum(row["seconds"] for row in report["intervals"] if row.get("phone_app") == "Garden notes") == 7.875
