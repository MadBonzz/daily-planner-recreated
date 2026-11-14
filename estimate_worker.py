"""Run pending estimates independently of Streamlit sessions and reruns."""

from concurrent.futures import ThreadPoolExecutor
import logging
import sqlite3
import threading

import estimator
from pdf_reader import extract_pdf_text


_LOG = logging.getLogger(__name__)
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="planner-estimate")
_LOCK = threading.Lock()
# A set event requests another scan while a workspace is already being drained.
_ACTIVE = {}
_LEASE_SECONDS = 30
_RECHECK_SECONDS = 5


def schedule_pending(store, index_dir):
    """Wake one worker per workspace. Safe to call on startup or after a task save."""
    key = store.path.resolve()
    with _LOCK:
        if key in _ACTIVE:
            _ACTIVE[key].set()
            return
        wake = threading.Event()
        _ACTIVE[key] = wake
        try:
            _POOL.submit(_drain, store, index_dir, key, wake)
        except RuntimeError:
            del _ACTIVE[key]
            raise


def _drain(store, index_dir, key, wake):
    try:
        while True:
            wake.clear()
            job = store.claim_prediction(lease_seconds=_LEASE_SECONDS)
            if job:
                _process(store, index_dir, job)
                continue
            if store.has_pending_predictions():
                # A previous process may have stopped with a lease outstanding.
                wake.wait(_RECHECK_SECONDS)
                continue
            with _LOCK:
                if not wake.is_set():
                    del _ACTIVE[key]
                    return
    except Exception as exc:
        # A later UI render will reschedule durable jobs on a temporary DB error.
        _LOG.error("Estimate worker paused (%s).", type(exc).__name__)
    finally:
        with _LOCK:
            if _ACTIVE.get(key) is wake:
                del _ACTIVE[key]


def _process(store, index_dir, job):
    task, token = job["task"], job["token"]
    task_id = task["id"]
    finished = threading.Event()
    revoked = threading.Event()

    def renew():
        while not finished.wait(_LEASE_SECONDS / 3):
            try:
                if store.renew_prediction(task_id, token, lease_seconds=_LEASE_SECONDS):
                    continue
            except sqlite3.Error:
                # Recheck ownership before publishing a result.
                continue
            revoked.set()
            return

    heartbeat = threading.Thread(target=renew, daemon=True, name="planner-estimate-lease")
    heartbeat.start()
    pdf_text = None
    try:
        try:
            pages = []
            unreadable = []
            for document in task["attachments"]:
                text = extract_pdf_text(store.attachment_bytes(document["id"]))
                if text:
                    pages.append(text)
                else:
                    unreadable.append(document["name"])
            if pages or task["attachments"]:
                pdf_text = "\n".join(pages)
                task["pdf_text"] = pdf_text

            if revoked.is_set() or not store.renew_prediction(task_id, token, lease_seconds=_LEASE_SECONDS):
                return
            prediction = estimator.predict(task, job["history"], index_dir)
            if unreadable:
                prediction["explanation"] += " No readable text was available from: " + ", ".join(unreadable) + "."
        except Exception as exc:
            # The timer and original estimate are already saved. An unexpected
            # parsing or inference error must leave a usable, truthful estimate.
            _LOG.error("Could not prepare task estimate (%s).", type(exc).__name__)
            prediction = estimator._fallback(task["estimate_minutes"], "The background estimate could not be completed.",
                                             retryable=True)
        saved = store.resolve_prediction(task_id, token, prediction, pdf_text=pdf_text)
        if saved:
            try:
                estimator.index_current_task(index_dir, store, task_id)
            except (RuntimeError, ValueError, OSError, sqlite3.Error) as exc:
                # The committed SQLite forecast is usable. Retrieval repairs a
                # missing or stale vector when this task becomes a comparison.
                _LOG.warning("Could not refresh task vector (%s).", type(exc).__name__)
        elif job["history"]:
            # Search may have repaired a missing vector just as its source task
            # was deleted. Drop any such vectors without indexing stale targets.
            existing = {row["id"] for row in store.completed_tasks()}
            for row in job["history"]:
                if row["id"] not in existing:
                    estimator.forget_task(index_dir, row["id"])
    finally:
        finished.set()
        heartbeat.join()
