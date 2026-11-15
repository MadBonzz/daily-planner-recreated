import json
import socket
import unittest
from unittest.mock import patch

import activity


def event(start, seconds, **data):
    return {"timestamp": activity._iso(start), "duration": seconds, "data": data}


class ActivityTests(unittest.TestCase):
    def setUp(self):
        host = socket.gethostname()
        self.buckets = {
            "aw-watcher-window_test": {"type": "currentwindow", "hostname": host},
            "aw-watcher-web-chrome": {"type": "web.tab.current", "hostname": "unknown"},
            "aw-watcher-web-firefox": {"type": "web.tab.current", "hostname": host},
            "aw-watcher-afk_test": {"type": "afkstatus", "hostname": host},
            "aw-watcher-window_remote": {"type": "currentwindow", "hostname": "another-machine.invalid"},
        }
        self.streams = {
            "aw-watcher-window_test": [event(10, 20, app="chrome.exe", title="Docs"),
                                       event(-10, 20, app="code.exe", title="Kernel.cu")],
            "aw-watcher-web-chrome": [event(-5, 50, url="https://docs.example.org/code?a=secret", title="CUDA API")],
            "aw-watcher-web-firefox": [event(-5, 50, url="https://private.example.org", title="Unrelated Firefox tab")],
            "aw-watcher-afk_test": [event(18, 2, status="afk"), event(0, 18, status="not-afk")],
        }
        self.reads = []
        self.reader = patch("activity._get", side_effect=self.read)
        self.reader.start()
        self.addCleanup(self.reader.stop)

    def read(self, path, **query):
        if path == "buckets/":
            return self.buckets
        key = path.split("/")[1]
        self.reads.append((key, activity._stamp(query["start"]), activity._stamp(query["end"])))
        self.assertNotIn("limit", query)
        return self.streams[key]

    def test_overlay_clip_remove_pauses_and_unknown(self):
        with patch("activity.chat_json", side_effect=activity.ModelUnavailable()):
            report = activity.build_report({"description": "Write a CUDA kernel"}, [(0, 20), (25, 35)])
        intervals = report["intervals"]
        self.assertEqual([(r["start"], r["end"], r["app"], r["domain"]) for r in intervals],
                         [(0, 10, "code.exe", ""), (10, 18, "chrome.exe", "docs.example.org"),
                          (18, 20, "chrome.exe", "docs.example.org"),
                          (25, 30, "chrome.exe", "docs.example.org"), (30, 35, "", "")])
        self.assertEqual(report["elapsed_seconds"], 30)
        self.assertEqual(report["totals"]["uncertain"], 30)
        self.assertEqual(report["switches"], 1)
        self.assertNotIn("a=secret", json.dumps(report))

    def test_schema_requests_all_active_ids_and_skips_idle(self):
        self.streams["aw-watcher-window_test"].append(event(30, 10, app="editor", title="Review"))
        def classify(messages, schema, **kwargs):
            self.assertEqual(set(schema["required"]), {"0", "1", "3", "4"})
            self.assertFalse(schema["additionalProperties"])
            return {key: {"label": "on_task", "reason": "Reviewed source."} for key in schema["required"]}
        with patch("activity.chat_json", side_effect=classify) as model:
            report = activity.build_report({"title": "CUDA", "pdf_text": "Read section three"}, [(0, 40)])
        sent = json.loads(model.call_args.args[0][1]["content"])
        self.assertIn("Read section three", sent["task"]["pdf_text"])
        self.assertNotIn(2, [row["id"] for row in sent["intervals"]])
        self.assertEqual(report["totals"], {"on_task": 38, "off_task": 0, "uncertain": 2})

    def test_exclusions_apply_to_window_url_and_browser_title(self):
        self.streams["aw-watcher-window_test"][1]["data"]["title"] = "Secret project in IDE"
        with patch("activity.chat_json") as model:
            report = activity.build_report({}, [(0, 20)], ["secret PROJECT", "a=secret"])
        self.assertTrue(all(r["excluded"] for r in report["intervals"]))
        self.assertNotIn("secret", json.dumps(report).casefold())
        model.assert_not_called()

    def test_privacy_gaps_are_never_queried_and_are_counted_once(self):
        settings = {"disabled_periods": [(5, 8), (6, 9), (27, None)]}
        with patch("activity.chat_json", side_effect=activity.ModelUnavailable()):
            report = activity.build_report({}, [(0, 15), (10, 20), (25, 30)], **settings)
        self.assertEqual({(start, end) for _, start, end in self.reads}, {(0, 5), (9, 20), (25, 27)})
        self.assertEqual(sum(r["seconds"] for r in report["intervals"] if r["excluded"]), 7)
        self.assertEqual(sum(report["totals"].values()), 25)

    def test_disabled_does_not_connect(self):
        with patch("activity._get") as endpoint, patch("activity.chat_json") as model:
            report = activity.build_report({}, [(0, 50)], capture=False)
        endpoint.assert_not_called()
        model.assert_not_called()
        self.assertEqual(report["totals"]["uncertain"], 50)

    def test_missing_api_is_visible(self):
        with patch("activity._get", side_effect=ConnectionRefusedError()), patch("activity.chat_json") as model:
            report = activity.build_report({}, [(25, 60)])
        model.assert_not_called()
        self.assertIn("unavailable", report["warnings"][0])
        self.assertEqual(report["elapsed_seconds"], 35)
        self.assertEqual(report["totals"]["uncertain"], 35)

    def test_poll_rebuilds_for_extended_heartbeat(self):
        first = activity.live_preview(25, 35)["intervals"]
        self.streams["aw-watcher-window_test"][0]["duration"] += 6
        second = activity.live_preview(25, 35)["intervals"]
        self.assertEqual(len(first), 2)
        self.assertEqual([(r["seconds"], r["app"]) for r in second], [(10, "chrome.exe")])


def test_truncated_interval_does_not_erase_other_batches(monkeypatch):
    import model_client

    calls = []

    def classify(messages, schema, **kwargs):
        ids = schema["required"]
        calls.append(ids)
        if "3" in ids:
            raise model_client.ModelResponseTruncated("response length limit")
        return {key: {"label": "on_task", "reason": "Focused test output."} for key in ids}

    monkeypatch.setattr(activity, "chat_json", classify)
    rows = [{"period": 0, "app": "Terminal", "title": f"Test result {i}", "domain": "", "idle": False,
             "excluded": False, "seconds": 5} for i in range(10)]
    warnings = []
    activity._classify({"title": "Debug tests"}, rows, warnings)
    assert [r["label"] for r in rows].count("on_task") == 9
    assert rows[3]["label_source"] == "unavailable"
    assert len(warnings) == 1 and calls[-1] == ["8", "9"]


def test_concurrent_devices_are_judged_independently_and_counted_once(monkeypatch):
    def classify(messages, schema, **kwargs):
        assert schema["properties"]["0"]["required"] == ["laptop", "phone", "reason"]
        assert schema["properties"]["1"]["required"] == ["label", "reason"]
        return {"0": {"laptop": "on_task", "phone": "off_task", "reason": "Code and a phone game."},
                "1": {"label": "off_task", "reason": "Phone game while the laptop is idle."}}

    monkeypatch.setattr(activity, "chat_json", classify)
    base = {"period": 0, "app": "Code", "title": "cleanup_test.py", "domain": "", "phone_app": "Chess",
            "excluded": False, "seconds": 60}
    rows = [{**base, "idle": False}, {**base, "idle": True}]
    activity._classify({"title": "Debug the cleanup tests"}, rows, [])
    assert rows[0]["label"] == "uncertain" and "Phone: off-task" in rows[0]["reason"]
    report = activity._summarize({"intervals": rows})
    assert report["totals"] == {"on_task": 0, "off_task": 60, "uncertain": 60}
