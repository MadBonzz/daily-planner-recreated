"""Run pending estimates independently of Streamlit sessions and reruns."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import logging
import sqlite3
import threading

import portalocker

import estimator
import focus_domains
import retrieval
import workspace_forecast
from model_client import ModelUnavailable
from pdf_reader import extract_pdf_text


_LOG = logging.getLogger(__name__)
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="planner-estimate")
_LOCK = threading.Lock()
# A set event requests another scan while a workspace is already being drained.
_ACTIVE = {}
_RETRIES = {}
_LEASE_SECONDS = 30
_RECHECK_SECONDS = 5


def schedule_pending(store, index_dir):
    """Wake one worker per workspace. Safe to call on startup or after a task save."""
    key = store.path.resolve()
    with _LOCK:
        retry = _RETRIES.pop(key, None)
        if retry:
            retry.cancel()
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


def run_one_job(store, index_dir):
    """Run one forecast or pair at a time across processes sharing the workspace."""
    # Two Streamlit processes have separate worker registries. Lock before
    # choosing the next job so a waiting worker rechecks first-forecast priority.
    # Timer changes and new submissions never acquire this inference lock.
    with (store.data_dir / "estimate-worker.lock").open("ab") as lockfile:
        portalocker.lock(lockfile, portalocker.LOCK_EX)
        job = store.claim_prediction(lease_seconds=_LEASE_SECONDS)
        if job:
            _process(store, index_dir, job)
            return True
        pair = store.claim_relation(lease_seconds=_LEASE_SECONDS)
        if pair:
            _process_relation(store, pair)
            return True
        return False


def _drain(store, index_dir, key, wake):
    try:
        while True:
            wake.clear()
            if run_one_job(store, index_dir):
                continue
            if store.has_pending_predictions():
                # A previous process may have stopped with a lease outstanding.
                wake.wait(_RECHECK_SECONDS)
                continue
            delay = store.next_estimation_delay()
            if delay == 0:
                continue  # A transaction queued new work between scans.
            with _LOCK:
                if not wake.is_set():
                    del _ACTIVE[key]
                    if delay is not None:
                        # Release the pool slot during backoff so an offline
                        # decision model in Dev cannot starve other workspaces.
                        timer = threading.Timer(delay, schedule_pending, args=(store, index_dir))
                        timer.daemon = True
                        _RETRIES[key] = timer
                        timer.start()
                    return
    except Exception as exc:
        # A later UI render will reschedule durable jobs on a temporary DB error.
        _LOG.error("Estimate worker paused (%s).", type(exc).__name__)
    finally:
        with _LOCK:
            if _ACTIVE.get(key) is wake:
                del _ACTIVE[key]


@contextmanager
def _heartbeat(extend):
    finished = threading.Event()
    revoked = threading.Event()

    def renew():
        while not finished.wait(_LEASE_SECONDS / 3):
            try:
                if extend():
                    continue
            except sqlite3.Error:
                # Recheck ownership before publishing a result.
                continue
            revoked.set()
            return

    heartbeat = threading.Thread(target=renew, daemon=True, name="planner-estimate-lease")
    heartbeat.start()
    try:
        yield revoked
    finally:
        finished.set()
        heartbeat.join()


def _read_documents(store, task):
    """Hydrate a snapshot from immutable uploads when extraction has not run yet."""
    if task.get("pdf_text_version") or not task.get("attachments"):
        return None, []
    pages, unreadable = [], []
    for document in task["attachments"]:
        text = extract_pdf_text(store.attachment_bytes(document["id"]))
        if text:
            pages.append(text)
        else:
            unreadable.append(document["name"])
    extracted = "\n".join(pages)
    task["pdf_text"] = extracted
    if not unreadable:
        store.cache_document_text(task["id"], extracted)
        task["pdf_text_version"] = 1
    return extracted, unreadable


def _process_relation(store, job):
    pair, token = job["pair"], job["token"]
    with _heartbeat(lambda: store.renew_relation(pair, token, lease_seconds=_LEASE_SECONDS)) as revoked:
        try:
            for task in (job["target"], job["candidate"]):
                _read_documents(store, task)
            if revoked.is_set() or not store.renew_relation(pair, token, lease_seconds=_LEASE_SECONDS):
                return
            first, second = retrieval.active_decision_order(job["target"], job["candidate"], job["examples"])
            related = retrieval.decide_related(first, second, examples=job["examples"])
        except ModelUnavailable as exc:
            _LOG.info("Pair relevance pending. %s", exc)
            related = None
        except Exception as exc:
            _LOG.error("Pair relevance failed (%s).", type(exc).__name__)
            related = None
        if not revoked.is_set():
            store.resolve_relation(pair, token, related, examples=job["examples"])


def _process(store, index_dir, job):
    task, token = job["task"], job["token"]
    task_id = task["id"]
    progress = job.get("workspace_progress")
    workspace_only = progress is not None and job["workspace_only"]
    prediction = job.get("previous_prediction")
    pdf_text = None
    with _heartbeat(lambda: store.renew_prediction(task_id, token, lease_seconds=_LEASE_SECONDS)) as revoked:
        try:
            pdf_text, unreadable = _read_documents(store, task)
            if not workspace_only:
                for source in job["history"]:
                    _read_documents(store, source)

            def renew():
                return not revoked.is_set() and store.renew_prediction(task_id, token, lease_seconds=_LEASE_SECONDS)

            if not renew():
                return
            with retrieval.estimation_lease(renew):
                if not workspace_only:
                    prediction = estimator.predict(task, job["history"], index_dir)
                if progress is not None:
                    prediction = workspace_forecast.revise(task, prediction, progress, refreshed_history=not workspace_only)
                if not workspace_only:
                    try:
                        focus_domains.review_sites(store, task, prediction, pdf_text, job.get("focus_verdicts", {}))
                    except Exception as exc:
                        # Site suggestions are optional. Preserve a completed time estimate if this review fails.
                        _LOG.warning("Website review did not finish (%s).", type(exc).__name__)
            if unreadable and not workspace_only:
                prediction["explanation"] += " No readable text was available from: " + ", ".join(unreadable) + "."
        except Exception as exc:
            # The timer and original estimate are already saved. An unexpected
            # parsing or inference error must leave a usable, truthful estimate.
            _LOG.error("Could not prepare task estimate (%s).", type(exc).__name__)
            if progress is not None:
                prediction = workspace_forecast.unavailable(progress, prediction, "The background estimate could not be completed.")
            else:
                prediction = estimator._fallback(task["estimate_minutes"], "The background estimate could not be completed.")
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
