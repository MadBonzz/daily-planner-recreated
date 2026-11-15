"""Replay the dated Dev schedules through the same store, models and timers as the dashboard."""

from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
from pathlib import Path
import json

import portalocker

from examples.estimation.activity_fixture import replay_task_at, task_layout
from examples.estimation.run import DATASET, sample_root, task_inputs, valid_id
from planner_store import PlannerStore


class _EventStore(PlannerStore):
    """Use one transaction for the timer change, fixture mapping, and replay cursor."""

    def __init__(self, owner, db, stamp):
        self.path, self.data_dir = owner.path, owner.data_dir
        self.connection = db
        self.clock = lambda: stamp

    @contextmanager
    def _db(self, *, write=False):
        yield self.connection


def schedule_exists(history_id, directory=DATASET):
    return valid_id(history_id) and (Path(directory) / "schedules" / f"{history_id}.json").is_file()


def _inputs(history_id, directory):
    if not schedule_exists(history_id, directory):
        raise ValueError("No event schedule is available for this history.")
    root = Path(directory)
    history = json.loads((root / "histories" / f"{history_id}.json").read_text(encoding="utf-8"))
    events = json.loads((root / "schedules" / f"{history_id}.json").read_text(encoding="utf-8"))
    return history, events


def initialize(store, history_id, directory, *, replay, now):
    history, events = _inputs(history_id, directory)
    calendar = {"history": history_id, "day": date.fromtimestamp(now).isoformat(),
                "opened_at": now, "cursor": 0, "replay": replay}
    store.set_setting("dev_calendar", calendar)
    store.set_setting("dev_replay_enabled", replay)
    if not replay:
        _run(store, history, events, calendar, directory, through_end=True, infer=False)


def _agenda(history, events, calendar):
    anchor = date.fromisoformat(calendar["day"])
    rows = {r["id"]: r for r in [*history["tasks"], *history["open_tasks"]]}
    layout = task_layout(history)
    starts = {}
    agenda = []
    for item in events:
        row, action = rows[item["task"]], item["action"]
        day = anchor + timedelta(days=item["day_offset"])
        stamp = (datetime.combine(day, time()) + timedelta(minutes=item["minute"])).timestamp()
        task = row["id"]
        if action in ("start", "resume"):
            starts[task] = stamp
        elif action in ("pause", "stop"):
            starts.pop(task, None)
        agenda.append({**item, "stamp": stamp, "day": day.isoformat()})
    originals = {key: [span for session in entry["sessions"] for span in session]
                 for key, entry in layout.items()}
    # Set the replay clock to the end of the observed portion of an ongoing timer.
    # In an editable snapshot the same moment should coincide with loading the sample.
    end_time = agenda[-1]["stamp"]
    for task, stamp in starts.items():
        a, b = originals[task][-1]
        end_time = max(end_time, stamp + b - a)
    if not calendar["replay"]:
        delta = calendar["opened_at"] - end_time
        # Offset today's events together to preserve open timers and avoid future timestamps.
        # Historical days keep their authored local clock times.
        for item in agenda:
            if item["day_offset"] == 0 and (starts or delta < 0):
                item["stamp"] += delta
        end_time = calendar["opened_at"]
    by_task = {}
    for item in agenda:
        if item["action"] in ("start", "resume"):
            by_task.setdefault(item["task"], []).append(item["stamp"])
    sessions = {}
    for task, times in by_task.items():
        source = layout[task]["sessions"]
        if len(times) != len(originals[task]):
            raise ValueError(f"{task}: schedule and ActivityWatch period counts differ.")
        times = iter(times)
        sessions[task] = []
        for group in source:
            periods = []
            for start, end in group:
                at = next(times)
                periods.append((at, at + end - start))
            sessions[task].append(periods)
    return rows, agenda, sessions, end_time


def _apply(owner, history, row, item, calendar, sessions, directory, infer):
    task_key = row["id"]
    stamp = item["stamp"]
    action = item["action"]
    with owner._db(write=True) as db:
        store = _EventStore(owner, db, stamp)
        if action == "create":
            fields, files = task_inputs(row, history["id"], directory)
            fields["day"] = item["day"] if row.get("status") is None else calendar["day"]
            task = store.create_task(**fields, attachments=files)
            task_id = task["id"]
            store.set_setting(f"dev_task_{task_key}", task_id)
            if task_key in sessions:
                store.set_setting(f"dev_activity_{task_id}", {
                    "history": history["id"], "task": task_key, "sessions": sessions[task_key]})
        else:
            task_id = store.get_setting(f"dev_task_{task_key}")
            if action == "finish":
                report = None
                if not infer:
                    task = store.get_task(task_id)
                    report = replay_task_at(history, row, sessions[task_key], store.periods(task_id),
                                            directory=directory, task_context=task)
                store.transition(task_id, "finish", completion_note=row.get("completion_note", ""),
                                 report=report, queue_report=infer)
            else:
                store.transition(task_id, action)
        calendar["cursor"] += 1
        calendar["clock"] = stamp
        store.set_setting("dev_calendar", calendar)


def _drain(store):
    from estimate_worker import run_one_job
    from report_worker import run_one_report

    index_dir = store.data_dir / "vectors"
    while run_one_job(store, index_dir) or run_one_report(store, True, wake_forecasts=False):
        pass


def _run(store, history, events, calendar, directory, *, through_end, infer):
    rows, agenda, sessions, end_time = _agenda(history, events, calendar)
    if infer:
        # A prior process may have exited just after recording an event.
        _drain(store)
    start = calendar["cursor"]
    end = len(events) if through_end else min(start + 1, len(events))
    for item in agenda[start:end]:
        _apply(store, history, rows[item["task"]], item, calendar, sessions, directory, infer)
        if infer:
            _drain(store)
    if end == len(events):
        calendar["clock"] = end_time
        store.set_setting("dev_calendar", calendar)


def advance(store, history_id, *, through_end=False, directory=DATASET):
    """Apply each event and resolve its forecasts before advancing the simulated clock."""
    directory = sample_root(history_id, directory)
    # Serialize two tabs advancing the same simulation. A crash releases the OS lock.
    with portalocker.Lock(str(store.data_dir / "replay.lock"), timeout=2):
        calendar = store.get_setting("dev_calendar")
        if not calendar or calendar["history"] != history_id or not calendar["replay"]:
            raise ValueError("Choose the chronological replay to advance its schedule.")
        # Only this lock holder runs replay inference. A token left by a killed
        # process cannot expire on the frozen simulation clock between actions.
        # Preserve delayed retries for model outages and the original snapshots.
        with store._db(write=True) as db:
            db.execute("UPDATE prediction_jobs SET token=NULL,lease_end=NULL WHERE token IS NOT NULL")
            db.execute("""UPDATE task_relations SET token=NULL,lease_end=NULL
                          WHERE related IS NULL AND token IS NOT NULL""")
        history, events = _inputs(history_id, directory)
        _run(store, history, events, calendar, directory, through_end=through_end, infer=True)


def progress(store, history_id, directory=DATASET):
    directory = sample_root(history_id, directory)
    calendar = store.get_setting("dev_calendar")
    if not calendar or calendar["history"] != history_id:
        return {"done": 0, "total": 0, "events": [], "last_day": None, "next_day": None}
    history, events = _inputs(history_id, directory)
    rows, agenda, _, _ = _agenda(history, events, calendar)
    done = calendar["cursor"]
    events = [{"day": date.fromtimestamp(item["stamp"]).isoformat(),
               "time": datetime.fromtimestamp(item["stamp"]).strftime("%H:%M"),
               "task": rows[item["task"]]["title"], "action": item["action"], "applied": i < done}
              for i, item in enumerate(agenda)]
    return {"events": events, "done": done, "total": len(events),
            "last_day": events[done-1]["day"] if done else None,
            "next_day": events[done]["day"] if done < len(events) else None}


def activity_preview(store, task, exclusions=(), directory=DATASET):
    ref = store.get_setting(f'dev_activity_{task["id"]}')
    if not ref:
        return []
    root = sample_root(ref["history"], directory)
    history = json.loads((root / "histories" / (ref["history"] + ".json")).read_text(encoding="utf-8"))
    row = next(r for r in [*history["tasks"], *history["open_tasks"]] if r["id"] == ref["task"])
    return replay_task_at(history, row, ref["sessions"], store.periods(task["id"]),
                          task_context=task, directory=root, exclusions=exclusions)["intervals"]
