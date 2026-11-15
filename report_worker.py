"""Generate and revise durable activity reports outside the Streamlit request."""

from concurrent.futures import ThreadPoolExecutor
import logging
import threading

import portalocker

import activity
from dev_workspace import dev_activity_report
from estimate_worker import schedule_pending
from task_files import document_text


_LOG = logging.getLogger(__name__)
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="activity-report")
_LOCK = threading.Lock()
_ACTIVE = {}


def schedule_reports(store, dev):
    """Wake the workspace worker, including any jobs left by a prior app process."""
    path = store.path.resolve()
    with _LOCK:
        if path in _ACTIVE:
            _ACTIVE[path].set()
        elif store.next_report_job():
            wake = threading.Event()
            _ACTIVE[path] = wake
            try:
                _POOL.submit(_drain, store, dev, path, wake)
            except RuntimeError:
                del _ACTIVE[path]
                raise


def _drain(store, dev, path, wake):
    # The OS releases this lock if a process exits during a model call. Jobs
    # remain in SQLite until a matching generation is atomically published.
    try:
        with (store.data_dir / "activity-worker.lock").open("ab") as lockfile:
            portalocker.lock(lockfile, portalocker.LOCK_EX)
            while True:
                wake.clear()
                if run_one_report(store, dev, wake_forecasts=True):
                    continue
                with _LOCK:
                    if not wake.is_set():
                        del _ACTIVE[path]
                        return
    finally:
        with _LOCK:
            if _ACTIVE.get(path) is wake:
                del _ACTIVE[path]


def run_one_report(store, dev, *, wake_forecasts=False):
    """Run one persisted report, optionally waking forecasts in a live workspace."""
    job = store.next_report_job()
    if not job:
        return False
    try:
        _process(store, dev, job, wake_forecasts)
    except Exception as exc:
        _LOG.warning("Activity report generation failed (%s).", type(exc).__name__)
        store.fail_report(job["task_id"], job["generation"],
                          "Activity report generation failed. You can retry it here.")
    return True


def _process(store, dev, job, wake_forecasts):
    task_id = job["task_id"]
    task = store.get_task(task_id)
    overrides, examples = store.report_feedback(task)
    pdf_text = None
    if job["action"] == "review" and task["report"]:
        report = activity.revise_report(task["report"], overrides)
    else:
        if not task["pdf_text"] and task["attachments"]:
            pdf_text = "\n".join(document_text(doc.get("kind", "pdf"), store.attachment_bytes(doc["id"]))
                                 for doc in task["attachments"])
            task["pdf_text"] = pdf_text
        if dev:
            report = dev_activity_report(store, task, examples=examples, overrides=overrides)
        else:
            report = activity.build_report(task, store.periods(task_id),
                                           exclusions=store.get_setting("exclusions", []),
                                           disabled_periods=store.disabled_periods(task_id),
                                           examples=examples, overrides=overrides)
    if store.save_report(task_id, report, generation=job["generation"], pdf_text=pdf_text) and wake_forecasts:
        schedule_pending(store, store.data_dir / "vectors")
