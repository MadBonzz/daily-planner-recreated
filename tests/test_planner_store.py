from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from planner_store import PlannerStore


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = 1000
        self.base = Path(self.tmp.name)
        self.store = PlannerStore(self.base / "live", clock=lambda: self.now)
        self.budget = dict(point_min=20, low_min=10, high_min=30, evidence=[])

    def task(self, title="Read paper", **overrides):
        return self.store.create_task(**{
            "day": "2026-11-14", "title": title, "description": "Read the proof.",
            "estimate_minutes": 15, "prediction": self.budget, **overrides,
        })

    def test_stop_and_completion_across_sessions(self):
        task_id = self.task()["id"]
        self.store.transition(task_id, "start")
        self.now += 60
        self.store.transition(task_id, "pause")
        self.now += 10
        self.store.transition(task_id, "resume")
        self.now += 15
        self.store.transition(task_id, "stop")
        reopened = PlannerStore(self.base / "live", clock=lambda: self.now)
        self.assertEqual(reopened.pending_task()["id"], task_id)
        self.assertEqual(reopened.completed_tasks(), [])
        self.now += 86400
        self.assertEqual(reopened.transition(task_id, "continue_later")["actual_seconds"], 75)
        reopened.transition(task_id, "start")
        self.now += 45
        reopened.transition(task_id, "stop")
        self.now += 300
        result = reopened.transition(task_id, "finish")
        self.assertEqual(result["actual_seconds"], 120)
        self.assertEqual(result["estimate_minutes"], 15)
        self.assertEqual(result["prediction"], self.budget)
        self.assertEqual(reopened.transition(task_id, "finish"), result)
        self.assertEqual(len(reopened.periods(task_id)), 3)
        self.assertEqual(len(reopened.completed_tasks()), 1)
        reopened.save_report(task_id, {"elapsed_seconds": 120})
        self.assertEqual(self.store.get_task(task_id)["report"], {"elapsed_seconds": 120})

    def test_task_switch_pauses_previous_timer(self):
        a, b = self.task()["id"], self.task()["id"]
        self.store.transition(a, "start")
        self.now += 30
        self.store.transition(b, "start")
        self.now += 2
        self.assertEqual(self.store.get_task(a)["status"], "paused")
        self.assertEqual(self.store.get_task(a)["actual_seconds"], 30)
        self.assertEqual(self.store.get_task(b)["actual_seconds"], 2)
        self.assertEqual(self.store.running_task()["id"], b)
        self.store.transition(a, "stop")
        self.store.reschedule(a, "2026-11-15")
        self.assertEqual(self.store.list_tasks("2026-11-15")[0]["id"], a)

    def test_forgotten_stop_correction_trims_capture_and_preserves_earlier_work(self):
        task_id = self.task()["id"]
        self.store.transition(task_id, "start")
        self.now += 10
        self.store.transition(task_id, "pause")
        self.now += 30
        self.store.transition(task_id, "resume")
        self.now += 40
        self.store.set_capture(False)
        self.now += 20
        self.store.set_capture(True)
        self.now += 36000
        with self.assertRaises(ValueError):
            self.store.transition(task_id, "stop", ended_at=1039)
        self.assertEqual(self.store.get_task(task_id)["status"], "running")
        self.store.transition(task_id, "stop", ended_at=1060)
        task = self.store.transition(task_id, "finish")
        self.assertEqual(task["actual_seconds"], 30)
        self.assertEqual(self.store.capture_periods(task_id), [(1000, 1010), (1040, 1060)])
        self.assertEqual(self.store.disabled_periods(task_id), [])

    def test_capture_disabled_boundaries_and_switches(self):
        task_id = self.task()["id"]
        self.store.set_capture(False)
        self.store.transition(task_id, "start")
        self.assertEqual(self.store.disabled_periods(task_id), [(1000, None)])
        self.now += 12
        self.store.set_capture(True)
        self.now += 7
        self.store.set_capture(False)
        self.now += 5
        self.store.transition(task_id, "pause")
        self.now += 20
        self.store.set_capture(True)
        self.store.transition(task_id, "resume")
        self.now += 4
        self.store.transition(task_id, "stop")
        self.assertEqual(self.store.capture_periods(task_id), [(1012, 1019), (1044, 1048)])
        self.assertEqual(self.store.disabled_periods(task_id), [(1000, 1012), (1019, 1024)])
        self.assertEqual(self.store.periods(task_id), [(1000, 1024), (1044, 1048)])

    def test_atomic_creation_and_privacy_deletion(self):
        with self.assertRaises(ValueError):
            self.task(attachments=[("first.pdf", b"%PDF-example"), ("invalid.pdf", b"oops")])
        self.assertEqual(self.store.list_tasks(None), [])
        source = self.task(attachments=[("file.pdf", b"%PDF-example")])
        doc_id = source["attachments"][0]["id"]
        self.assertEqual(self.store.attachment_bytes(doc_id), b"%PDF-example")
        pred = {**self.budget, "evidence": [{"task_id": source["id"], "title": source["title"]}],
                "explanation": f'Like {source["title"]}.'}
        new = self.task(prediction=pred)
        self.store.delete_task(source["id"])
        with self.assertRaises(ValueError):
            self.store.attachment_bytes(doc_id)
        saved = self.store.get_task(new["id"])["prediction"]
        self.assertEqual(saved["point_min"], 20)
        self.assertNotIn(source["title"], repr(saved))

    def test_concurrent_start_and_stop(self):
        a, b = self.task()["id"], self.task()["id"]
        with ThreadPoolExecutor(2) as executor:
            list(executor.map(lambda t: self.store.transition(t, "start"), (a, b)))
        states = {t["status"] for t in self.store.list_tasks(None)}
        self.assertEqual(states, {"paused", "running"})
        self.now += 30
        running = self.store.running_task()["id"]
        with ThreadPoolExecutor(2) as executor:
            list(executor.map(lambda _: self.store.transition(running, "stop"), (1, 2)))
        self.assertEqual(self.store.get_task(running)["actual_seconds"], 30)

    def test_demo_isolation_and_reset(self):
        with self.assertRaises(ValueError):
            self.store.seed_demo()
        demo = PlannerStore(self.base / "demo")
        demo.seed_demo()
        demo.seed_demo()
        self.assertEqual(len(demo.completed_tasks()), 5)
        self.assertEqual(self.store.completed_tasks(), [])
        demo.clear_data()
        self.assertEqual(demo.list_tasks(None), [])
        self.assertFalse(demo.get_setting("example_history_seeded", False))


if __name__ == "__main__":
    unittest.main()
