from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
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

    def finish_one(self, title, prediction=None):
        task = self.task(title, prediction=self.budget if prediction is None else prediction)
        self.store.transition(task["id"], "start")
        self.now += 60
        self.store.transition(task["id"], "stop")
        return self.store.transition(task["id"], "finish")

    def decide_pairs(self, relevant):
        while job := self.store.claim_relation():
            related = relevant(*job["pair"])
            self.assertTrue(self.store.resolve_relation(job["pair"], job["token"], related))

    def test_background_estimate_freezes_history_and_survives_task_completion(self):
        earlier = self.finish_one("Earlier proof")
        target = self.task("Another proof", prediction=None)
        self.assertEqual(target["prediction"]["status"], "pending")
        later = self.finish_one("Later proof")
        self.store.transition(target["id"], "start")
        self.now += 15
        self.store.transition(target["id"], "stop")
        self.store.transition(target["id"], "finish")
        reopened = PlannerStore(self.base / "live", clock=lambda: self.now)
        self.assertTrue(reopened.has_pending_predictions())
        job = reopened.claim_prediction()
        self.assertEqual(job["task"]["id"], target["id"])
        self.assertEqual([row["id"] for row in job["history"]], [earlier["id"]])
        self.assertNotEqual(later["id"], earlier["id"])
        self.assertIsNone(reopened.claim_prediction())
        self.assertTrue(reopened.resolve_prediction(target["id"], job["token"], self.budget, pdf_text="Page 1"))
        self.assertFalse(self.store.resolve_prediction(target["id"], job["token"], {"point_min": 999}))
        final = reopened.get_task(target["id"])
        self.assertEqual((final["status"], final["actual_seconds"], final["pdf_text"]), ("completed", 15, "Page 1"))
        self.assertEqual(final["prediction"], self.budget)
        self.assertFalse(reopened.has_pending_predictions())

    def test_upgrade_retries_only_open_tokenizer_failures_once(self):
        failure = dict(self.budget, retrieval_error="The bare task embedding was unavailable. "
                       "Could not verify the complete input with the local model tokenizer.")
        self.finish_one("Completed task", prediction=failure)
        target = self.task("Unfinished task", prediction=failure)["id"]
        self.task("Healthy forecast")
        self.task("Other failure", prediction=dict(self.budget, retrieval_error="Index unavailable."))
        with sqlite3.connect(self.store.path) as db:
            db.execute("DELETE FROM settings WHERE key='retried_qwen_tokenizer_failures'")
        upgraded = PlannerStore(self.store.data_dir, clock=lambda: self.now)
        job = upgraded.claim_prediction()
        self.assertEqual(job["task"]["id"], target)
        reopened = PlannerStore(self.store.data_dir, clock=lambda: self.now)
        self.assertTrue(reopened.resolve_prediction(target, job["token"], self.budget))
        self.assertEqual(reopened.get_task(target)["creation_prediction"], failure)
        self.assertIsNone(reopened.claim_prediction())

    def test_upgrade_retries_only_saved_local_bonsai_json_failures_and_keeps_other_jobs(self):
        provenance = {"provider": "Local endpoint", "model": "ternary-bonsai-2-4b-mlx",
                      "decision": "fallback", "minutes_by": "user"}
        failed = dict(self.budget, provenance=provenance, model_error="The model could not supply a valid forecast. "
                      "ternary-bonsai-2-4b-mlx returned an incomplete or invalid JSON object.")
        self.finish_one("Completed", prediction=failed)
        self.store.archive_task(self.task("Archived", prediction=failed)["id"])
        fixed = self.task("Scheduled", prediction=failed)["id"]
        self.task("Hosted", prediction=dict(failed, provenance={**provenance, "provider": "Hosted endpoint"}))
        self.task("Other local model", prediction=dict(failed, provenance={**provenance, "model": "other-model"}))
        self.task("Unavailable endpoint", prediction=dict(failed, model_error="Bonsai is unavailable."))
        self.task("Healthy forecast")
        active = self.task("Retry already in progress", prediction=failed)["id"]
        self.store.retry_estimate(active)
        active_job = self.store.claim_prediction()
        target = self.task("Running task", prediction=failed)["id"]
        self.store.transition(target, "start")
        self.now += 45
        with sqlite3.connect(self.store.path) as db:
            db.execute("DELETE FROM settings WHERE key='retried_bonsai_mlx_invalid_json'")
            db.execute("UPDATE tasks SET fixed_duration=1 WHERE id=?", (fixed,))

        upgraded = PlannerStore(self.store.data_dir, clock=lambda: self.now)
        self.assertTrue(upgraded.resolve_prediction(active, active_job["token"], self.budget))
        job = upgraded.claim_prediction()
        self.assertEqual(job["task"]["id"], target)
        self.assertTrue(upgraded.resolve_prediction(target, job["token"], failed))
        reopened = PlannerStore(self.store.data_dir, clock=lambda: self.now)
        self.assertIsNone(reopened.claim_prediction())
        saved = reopened.get_task(target)
        self.assertEqual((saved["status"], saved["actual_seconds"], saved["creation_prediction"]),
                         ("running", 45, failed))

    def test_retry_saved_failure_is_idempotent_preserves_first_forecast_and_timer(self):
        failed = dict(self.budget, explanation="Model comparison unavailable. api.cohere.ai returned HTTP 422.")
        task_id = self.task(prediction=failed)["id"]
        self.store.transition(task_id, "start")
        self.now += 45
        self.store.transition(task_id, "pause")
        reopened = PlannerStore(self.store.data_dir, clock=lambda: self.now)
        self.assertTrue(reopened.retry_estimate(task_id))
        job = reopened.claim_prediction()
        self.assertFalse(self.store.retry_estimate(task_id))
        self.assertTrue(reopened.resolve_prediction(task_id, job["token"], self.budget))
        saved = reopened.get_task(task_id)
        self.assertEqual((saved["creation_prediction"], saved["prediction"]), (failed, self.budget))
        self.assertEqual((saved["status"], saved["actual_seconds"]), ("paused", 45))
        self.assertTrue(reopened.retry_estimate(task_id))
        self.assertFalse(self.store.retry_estimate(task_id))

    def test_completed_failed_forecast_stays_frozen(self):
        task = self.finish_one("Earlier work", prediction=dict(self.budget, retryable=True))
        self.assertFalse(self.store.retry_estimate(task["id"]))
        self.assertFalse(self.store.has_pending_predictions())

    def test_prediction_claims_are_atomic_and_restart_recovers_expired_lease(self):
        task = self.task(prediction=None)
        with ThreadPoolExecutor(2) as pool:
            claims = list(pool.map(lambda _: self.store.claim_prediction(lease_seconds=5), range(2)))
        self.assertEqual(sum(job is not None for job in claims), 1)
        job = next(job for job in claims if job)
        self.now += 4
        self.assertTrue(self.store.renew_prediction(task["id"], job["token"], lease_seconds=10))
        self.now += 7
        reopened = PlannerStore(self.base / "live", clock=lambda: self.now)
        self.assertIsNone(reopened.claim_prediction())
        self.now += 3
        self.assertFalse(self.store.renew_prediction(task["id"], job["token"]))
        self.assertFalse(self.store.resolve_prediction(task["id"], job["token"], self.budget))
        fresh = reopened.claim_prediction()
        self.assertNotEqual(fresh["token"], job["token"])
        self.assertFalse(self.store.resolve_prediction(task["id"], job["token"], self.budget))
        self.assertTrue(reopened.resolve_prediction(task["id"], fresh["token"], self.budget))

    def test_inflight_prediction_cannot_restore_deleted_tasks_or_sources(self):
        source = self.finish_one("Private source")
        task = self.task(prediction=None)
        first = self.store.claim_prediction()
        self.store.delete_task(source["id"])
        self.assertFalse(self.store.renew_prediction(task["id"], first["token"]))
        self.assertFalse(self.store.resolve_prediction(task["id"], first["token"], self.budget))
        claim = self.store.claim_prediction()
        self.assertEqual(claim["history"], [])
        self.store.delete_task(task["id"])
        self.assertFalse(self.store.resolve_prediction(task["id"], claim["token"], self.budget))
        self.assertFalse(self.store.has_pending_predictions())

    def test_creation_snapshot_then_latest_refresh_with_frozen_evidence(self):
        earlier = self.finish_one("Earlier reading")
        self.store.save_report(earlier["id"], {"summary": "first report"})
        target = self.task("Next reading", prediction=None)
        latest = self.task("Similar completion")
        self.store.transition(latest["id"], "start")
        self.now += 100
        self.store.transition(latest["id"], "stop")
        self.store.transition(latest["id"], "finish", completion_note="Had to revisit the proof.",
                              report={"summary": "reviewed derivation"})
        self.store.save_report(earlier["id"], {"summary": "later revision"})
        reopened = PlannerStore(self.base / "live", clock=lambda: self.now)
        initial = reopened.claim_prediction()
        self.assertEqual([r["id"] for r in initial["history"]], [earlier["id"]])
        self.assertEqual(initial["history"][0]["report"]["summary"], "first report")
        self.assertNotIn("completion_note", initial["task"])
        self.assertNotIn("actual_seconds", initial["task"])
        self.assertTrue(reopened.resolve_prediction(target["id"], initial["token"], self.budget))
        self.assertTrue(reopened.get_task(target["id"])["awaiting_related_update"])
        pair = reopened.claim_relation()
        self.assertEqual(set(pair["pair"]), {latest["id"], target["id"]})
        self.assertTrue(reopened.resolve_relation(pair["pair"], pair["token"], True))
        pending = reopened.get_task(target["id"])
        self.assertTrue(pending["estimate_updating"])
        self.assertEqual(pending["creation_prediction"], self.budget)
        self.assertEqual(pending["prediction"], self.budget)
        refresh = reopened.claim_prediction()
        by_id = {r["id"]: r for r in refresh["history"]}
        self.assertEqual(by_id[earlier["id"]]["report"]["summary"], "later revision")
        self.assertEqual(by_id[latest["id"]]["completion_note"], "Had to revisit the proof.")
        self.assertEqual(by_id[latest["id"]]["report"]["summary"], "reviewed derivation")
        # Repeated Stop confirmation leaves an existing refresh's lease intact.
        reopened.transition(latest["id"], "finish", completion_note="second submit")
        self.assertTrue(reopened.resolve_prediction(target["id"], refresh["token"], {"point_min": 44}))
        final = reopened.get_task(target["id"])
        self.assertEqual(final["prediction"]["point_min"], 44)
        self.assertEqual(final["creation_prediction"], self.budget)
        self.assertFalse(final["estimate_updating"])
        self.assertFalse(reopened.has_pending_predictions())

    def test_fixed_length_and_newer_completion_revokes_an_obsolete_refresh(self):
        fixed = self.task("Office hours", estimate_minutes=45.5, fixed_duration=True, prediction=None)
        self.assertEqual([fixed["prediction"][k] for k in ("point_min","low_min","high_min")], [45.5] * 3)
        self.assertFalse(fixed["estimate_updating"])
        target = self.task("Writing")
        source = self.finish_one("Revision")
        self.decide_pairs(lambda a,b: target["id"] in (a,b))
        stale = self.store.claim_prediction()
        self.assertEqual(stale["task"]["id"], target["id"])
        newer = self.task("A second related revision", prediction=self.budget)
        self.decide_pairs(lambda a,b: target["id"] in (a,b))
        self.store.transition(newer["id"], "start")
        self.now += 60
        self.store.transition(newer["id"], "stop")
        self.store.transition(newer["id"], "finish", completion_note="Verified on the second pass.",
                              pdf_text="Page 1: Agenda")
        self.assertFalse(self.store.resolve_prediction(target["id"], stale["token"], {"point_min": 900}))
        latest = self.store.claim_prediction()
        self.assertEqual({r["id"] for r in latest["history"]}, {source["id"], newer["id"]})
        self.assertEqual(next(r for r in latest["history"] if r["id"] == newer["id"])["pdf_text"], "Page 1: Agenda")
        self.store.transition(target["id"], "start")
        self.now += 30
        self.store.transition(target["id"], "stop")
        self.store.transition(target["id"], "finish")
        self.assertFalse(self.store.resolve_prediction(target["id"], latest["token"], {"point_min": 999}))
        self.assertEqual(self.store.get_task(target["id"])["prediction"], self.budget)
        self.assertFalse(self.store.has_pending_predictions())

    def test_delayed_matrix_job_keeps_newer_task_as_target_after_it_finishes(self):
        # A burst can insert both tasks within one clock tick.
        earlier = self.task("Earlier")["id"]
        newer = self.task("Newer")["id"]
        self.store.transition(newer, "start")
        self.now += 120
        self.store.transition(newer, "stop")
        self.store.transition(newer, "finish")
        job = self.store.claim_relation()
        self.assertEqual((job["target"]["id"], job["candidate"]["id"]), (newer, earlier))
        self.assertTrue(self.store.resolve_relation(job["pair"], job["token"], True))
        self.assertEqual(self.store.claim_prediction()["task"]["id"], earlier)

    def test_refreshes_only_related_tasks_in_other_days_and_timer_states(self):
        ready, paused, running = (self.task(title) for title in ("Tuesday", "Paused", "Active"))
        self.store.reschedule(ready["id"], "2026-11-15")
        self.store.transition(paused["id"], "start")
        self.store.transition(paused["id"], "pause")
        prior = self.task("Source")
        self.store.transition(prior["id"], "start")
        self.now += 20
        self.store.transition(prior["id"], "stop")
        self.store.transition(running["id"], "start")
        self.decide_pairs(lambda a,b: ready["id"] not in (a,b))
        self.store.transition(prior["id"], "finish")
        for task, state in ((ready, "ready"), (paused, "paused"), (running, "running")):
            refreshed = self.store.get_task(task["id"])
            self.assertEqual(refreshed["estimate_updating"], task["id"] != ready["id"])
            self.assertEqual(refreshed["status"], state)
        self.assertFalse(self.store.get_task(prior["id"])["estimate_updating"])

    def test_initial_job_has_priority_over_queued_refresh_across_workers(self):
        existing = self.task("Existing")
        self.finish_one("Completed")
        self.decide_pairs(lambda a,b: True)
        self.assertTrue(self.store.get_task(existing["id"])["estimate_updating"])
        first = self.task("Just added", prediction=None)
        claim = self.store.claim_prediction()
        self.assertEqual(claim["task"]["id"], first["id"])
        self.assertIsNone(self.store.claim_prediction())
        self.assertIsNone(self.store.claim_relation())
        self.store.resolve_prediction(first["id"], claim["token"], self.budget)
        self.assertEqual(self.store.claim_prediction()["task"]["id"], existing["id"])

    def test_pending_pair_recovers_after_outage_and_never_refreshes_deleted_source(self):
        self.task("Target")
        source = self.finish_one("Source")
        job = self.store.claim_relation(lease_seconds=3)
        self.assertIsNone(self.store.claim_relation())
        self.assertTrue(self.store.resolve_relation(job["pair"], job["token"], None))
        self.assertIsNone(self.store.claim_relation())
        self.assertIsNone(self.store.claim_prediction())
        self.now += 16
        reopened = PlannerStore(self.store.data_dir, clock=lambda: self.now)
        retry = reopened.claim_relation(lease_seconds=2)
        self.assertEqual(retry["pair"], job["pair"])
        self.now += 3
        self.assertFalse(reopened.resolve_relation(retry["pair"], retry["token"], True))
        last = reopened.claim_relation()
        reopened.delete_task(source["id"])
        self.assertFalse(reopened.resolve_relation(last["pair"], last["token"], True))
        self.assertFalse(reopened.has_estimation_work())

    def test_forgetting_an_example_rejudges_only_dependent_edges(self):
        example = self.finish_one("Reviewed old handout")
        a,b,c = [self.task(name)["id"] for name in ("A","B","C")]
        expected = tuple(sorted((a,b)))
        while job := self.store.claim_relation():
            used = [{"candidate_id": example["id"]}] if job["pair"] == expected else []
            self.assertTrue(self.store.resolve_relation(job["pair"], job["token"], True, examples=used))
        self.store.delete_task(example["id"])
        new = self.store.claim_relation()
        self.assertEqual(new["pair"], expected)
        self.assertTrue(self.store.resolve_relation(new["pair"], new["token"], False))
        self.assertIsNone(self.store.claim_relation())

    def test_matrix_backfill_only_unfinished_pairs_and_reopen_is_idempotent(self):
        self.finish_one("Finished")
        a,b,c = [self.task(name)["id"] for name in ("A","B","C")]
        with sqlite3.connect(self.store.path) as db:
            db.execute("DELETE FROM settings WHERE key='relation_matrix_initialized'")
            db.execute("DROP TABLE task_relations")
        restored = PlannerStore(self.store.data_dir)
        pairs = []
        while job := restored.claim_relation():
            pairs.append(set(job["pair"]))
            restored.resolve_relation(job["pair"], job["token"], False)
        self.assertCountEqual(pairs, [{a,b},{b,c},{a,c}])
        self.assertFalse(PlannerStore(self.store.data_dir).has_estimation_work())

    def test_replay_timeline_records_successful_commits_and_forgets_deleted_sources(self):
        source = self.finish_one("Source")
        self.assertEqual(self.store.forecast_timeline(source["id"]), [])
        self.store.set_setting("dev_replay_enabled", True)
        target = self.task("Target", prediction=None)
        first = self.store.claim_prediction()
        original = dict(self.budget, feedback_source_ids=[source["id"]], rationale="private feedback")
        first_at = self.now
        self.store.resolve_prediction(target["id"], first["token"], original)
        self.now += 120
        self.assertTrue(self.store.retry_estimate(target["id"]))
        retry = self.store.claim_prediction()
        self.assertFalse(self.store.resolve_prediction(target["id"], first["token"], original))
        self.store.resolve_prediction(target["id"], retry["token"], self.budget)
        self.assertEqual(self.store.forecast_timeline(target["id"]), [
            {"at": first_at, "initial": True, "prediction": original},
            {"at": self.now, "initial": False, "prediction": self.budget},
        ])
        self.store.delete_task(source["id"])
        self.assertEqual([event["initial"] for event in self.store.forecast_timeline(target["id"])], [False])
        self.store.delete_task(target["id"])
        self.assertEqual(self.store.forecast_timeline(target["id"]), [])
        fixed = self.task("Meeting", fixed_duration=True)
        self.assertEqual(self.store.forecast_timeline(fixed["id"])[0]["prediction"], fixed["prediction"])

    def test_previous_databases_preserve_creation_forecasts_and_pending_work(self):
        known = self.task()
        pending = self.task(prediction=None)
        with sqlite3.connect(self.store.path) as db:
            for column in ("creation_prediction", "completion_note", "fixed_duration", "refresh_waiting"):
                db.execute(f"ALTER TABLE tasks DROP COLUMN {column}")
            db.execute("ALTER TABLE prediction_history DROP COLUMN snapshot")
        reopened = PlannerStore(self.store.data_dir)
        self.assertEqual(reopened.get_task(known["id"])["creation_prediction"], self.budget)
        self.assertIsNone(reopened.get_task(pending["id"])["creation_prediction"])
        self.assertFalse(reopened.get_task(known["id"])["fixed_duration"])
        self.assertEqual(reopened.get_task(known["id"])["completion_note"], "")
        self.assertEqual(reopened.claim_prediction()["task"]["id"], pending["id"])

    def test_pending_job_rolls_back_with_invalid_pdf(self):
        with self.assertRaises(ValueError):
            self.task(prediction=None, attachments=[("broken.pdf", b"invalid")])
        self.assertFalse(self.store.has_pending_predictions())
        self.assertIsNone(self.store.claim_prediction())

    def test_clear_data_only_resets_its_workspace(self):
        live_task = self.task()
        demo = PlannerStore(self.base / "demo")
        demo.create_task(day="2026-11-14", title="Demo task", description="Check the example.", estimate_minutes=5)
        demo.set_setting("capture_enabled", False)
        demo.clear_data()
        self.assertEqual(demo.list_tasks(None), [])
        self.assertIsNone(demo.get_setting("capture_enabled"))
        self.assertEqual(self.store.get_task(live_task["id"]), live_task)


if __name__ == "__main__":
    unittest.main()
