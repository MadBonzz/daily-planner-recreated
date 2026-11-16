"""Per-task browser blocks from comparable work and explicit user choices."""

from collections import defaultdict
import json
import math
import re
from urllib.parse import urlsplit

import focus_judge
from activity import activity_key
from model_client import ModelUnavailable, planner_settings
from relevance_feedback import related_ids
from sample_retrieval import prefer_packaged


def normalize_domain(value):
    """Accept a DNS name or HTTP(S) URL and return the host consumed by the extension."""
    error = "Enter a website such as x.com or https://www.x.com."
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error)
    raw = value.strip()
    if any(c.isspace() or ord(c) < 32 for c in raw) or "\\" in raw:
        raise ValueError(error)
    try:
        url = urlsplit(raw if "://" in raw else "//" + raw)
        host = (url.hostname or "").rstrip(".").encode("idna").decode("ascii").lower().removeprefix("www.")
        # Accessing .port also rejects malformed values. The rule applies to the host at any port.
        _ = url.port
    except (ValueError, UnicodeError):
        raise ValueError(error) from None
    labels = host.split(".")
    if (url.scheme not in {"", "http", "https"} or url.username is not None or url.password is not None
            or len(host) > 253 or len(labels) < 2 or labels[-1] == "localhost"
            or not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels)
            or labels[-1].isdigit()):
        raise ValueError(error)
    return host


def _related_sources(db, target):
    known = set(related_ids(json.loads(target["prediction"])))
    for row in db.execute("SELECT source_id, related FROM relevance_choices WHERE target_id=?", (target["id"],)):
        if row["related"]:
            known.add(row["source_id"])
        else:
            known.discard(row["source_id"])
    known.discard(target["id"])
    return sorted(known)


def _blocked_by(host, domain):
    return host == domain or host.endswith("." + domain)


def _site_evidence(db, target, sources, config):
    """Collect current, sanitized evidence before making or using any domain judgment."""
    if not sources:
        return
    ids = json.dumps(sources)
    rows = db.execute("""SELECT id, title, report FROM tasks
                         WHERE status='completed' AND report IS NOT NULL
                         AND id IN (SELECT value FROM json_each(?)) ORDER BY title, id""", (ids,))
    allowed = {r["domain"] for r in db.execute("""SELECT domain FROM focus_domain_choices
               WHERE blocked=0 AND task_id IN (SELECT value FROM json_each(?))""", (ids,))}
    feedback = {(r["task_id"], r["activity_key"]): r["label"] for r in db.execute(
        """SELECT task_id,activity_key,label FROM activity_feedback WHERE task_id IN (SELECT value FROM json_each(?))""",
        (ids,))}
    setting = db.execute("SELECT value FROM settings WHERE key='exclusions'").fetchone()
    exclusions = [v.strip().casefold() for v in (json.loads(setting[0]) if setting else [])
                  if isinstance(v, str) and v.strip()]
    visits = defaultdict(dict)
    tasks = {}
    for record in rows:
        row = dict(record)
        intervals = json.loads(row.pop("report")).get("intervals", [])
        tasks[row["id"]] = row
        for event in intervals:
            seconds = event.get("seconds")
            if (event.get("excluded") or not event.get("domain") or type(seconds) not in (int, float)
                    or not math.isfinite(seconds) or seconds <= 0):
                continue
            text = " ".join(str(event.get(k) or "") for k in ("app", "title", "domain", "phone_app")).casefold()
            if any(term in text for term in exclusions):
                continue
            try:
                host = normalize_domain(event["domain"])
            except ValueError:
                continue
            label = feedback.get((row["id"], activity_key(event)), event.get("label"))
            if label == "on_task":
                allowed.add(host)
            elif label == "off_task" and not event.get("idle") and not event.get("phone_app"):
                # A combined phone/laptop interval cannot attribute the distraction to this domain.
                visit = visits[host].setdefault(row["id"], {"seconds": 0, "titles": set()})
                visit["seconds"] += seconds
                if event.get("title"):
                    visit["titles"].add(event["title"])

    target_text = " ".join(target.get(k) or "" for k in ("title", "description", "details", "pdf_text")).casefold()
    for domain, observations in sorted(visits.items()):
        if any(_blocked_by(host, domain) or _blocked_by(domain, host) for host in allowed):
            continue
        if re.search(r"(?<![\w-])" + re.escape(domain) + r"(?![\w-]|\.[\w-])", target_text):
            continue
        minutes = sum(visit["seconds"] for visit in observations.values()) / 60
        sources_here = [tasks[key] for key in observations]
        titles = set().union(*(visit["titles"] for visit in observations.values()))
        if len(sources_here) == 1:
            source = sources_here[0]
            example = " / ".join(sorted(titles)[:2])
            detail = f' The recorded window was “{example}”.' if example else ""
            reason = f'{minutes:.1f} minutes labeled distraction during one related task, “{source["title"]}”.{detail}'
        else:
            examples = " and ".join(f'“{source["title"]}”' for source in sources_here[:2])
            reason = (f"{minutes:.1f} minutes labeled distraction across {len(observations)} related tasks, "
                      f"including {examples}.")
        if config is None:
            config = planner_settings()
        yield domain, reason, focus_judge.payload(target, domain, sources_here, titles, config)


def _target(db, task_id):
    row = db.execute("""SELECT id, title, description, details, pdf_text, prediction FROM tasks
                        WHERE id=? AND archived=0""", (task_id,)).fetchone()
    if row is None:
        raise ValueError("This task is no longer available.")
    return dict(row)


def site_requests(store, task_id, source_ids, *, pdf_text=None, config=None):
    if not source_ids:
        return {}
    with store._db() as db:
        target = _target(db, task_id)
        if pdf_text is not None:
            target["pdf_text"] = pdf_text
        return {domain: request for domain, _, request in _site_evidence(db, target, source_ids, config)}


def review_sites(store, task, prediction, pdf_text, previous):
    """Publish site suggestions only while this previously saved forecast is current."""
    sources = related_ids(prediction)
    if not sources:
        return
    config = planner_settings()
    requests = site_requests(store, task["id"], sources, pdf_text=pdf_text, config=config)
    if not requests:
        return
    current = """id=? AND archived=0 AND edit_revision=? AND prediction=?
                 AND NOT EXISTS (SELECT 1 FROM prediction_jobs WHERE task_id=tasks.id)"""
    expected = json.dumps(prediction, allow_nan=False)
    args = (task["id"], task["edit_revision"], expected)
    results = {}
    for request in requests.values():
        key = focus_judge.key(request)
        with store._db() as db:
            if not db.execute("SELECT 1 FROM tasks WHERE " + current, args).fetchone():
                return
        try:
            results[key] = previous[key] if type(previous.get(key)) is bool else focus_judge.decide(request, config)
        except ModelUnavailable:
            # An unreviewed domain remains editable; a later re-estimate can retry.
            continue
    if results:
        updated = json.dumps({**prediction, "focus_verdicts": results}, allow_nan=False)
        with store._db(write=True) as db:
            db.execute("""UPDATE tasks SET prediction=?,
                       creation_prediction=CASE WHEN creation_prediction=? THEN ? ELSE creation_prediction END
                       WHERE """ + current, (updated, expected, updated, *args))


def policy_from_db(db, task_id):
    """Read the effective policy without calling a model from a UI render or browser poll."""
    task = _target(db, task_id)
    stored = json.loads(task["prediction"]).get("focus_verdicts", {})
    suggestions, pending = {}, {}
    config = planner_settings()
    for domain, reason, request in _site_evidence(db, task, _related_sources(db, task), config):
        verdict = stored.get(focus_judge.key(request))
        if type(verdict) is not bool and prefer_packaged("decision", request["model"], config):
            verdict = focus_judge.packaged(request)
        if verdict is True:
            suggestions[domain] = reason + " Kev recommends blocking it for this task."
        elif verdict is None:
            pending[domain] = reason
    choices = {r["domain"]: bool(r["blocked"]) for r in db.execute(
        "SELECT domain, blocked FROM focus_domain_choices WHERE task_id=?", (task_id,))}
    added = {d for d, enabled in choices.items() if enabled}
    removed = {d for d, enabled in choices.items() if not enabled}
    # Both a blocked parent and a blocked child can defeat an explicit unblock.
    # An explicit user addition still wins over an overlapping automatic suggestion.
    automatic = {domain for domain in suggestions if not any(
        _blocked_by(domain, allow) or _blocked_by(allow, domain) for allow in removed
    )}
    return {"domains": sorted(automatic | added), "suggestions": suggestions,
            "added": sorted(added), "removed": sorted(removed),
            "pending": {d: why for d, why in pending.items() if d not in choices}}


def policy(store, task_id):
    with store._db() as db:
        return policy_from_db(db, task_id)


def save_domains(store, task_id, selected, reviewed, *, pending=()):
    """Apply explicit additions and removals to the last displayed list, leaving new evidence live."""
    selected = {normalize_domain(v) for v in selected}
    reviewed = {normalize_domain(v) for v in reviewed}
    pending = {normalize_domain(v) for v in pending}
    edits = [(task_id, domain, 1) for domain in selected - reviewed]
    edits.extend((task_id, domain, 0) for domain in (reviewed | pending) - selected)
    with store._db(write=True) as db:
        if not db.execute("SELECT 1 FROM tasks WHERE id=? AND archived=0", (task_id,)).fetchone():
            raise ValueError("This task is no longer available.")
        db.executemany("""INSERT INTO focus_domain_choices(task_id,domain,blocked) VALUES(?,?,?)
                          ON CONFLICT(task_id,domain) DO UPDATE SET blocked=excluded.blocked""", edits)
