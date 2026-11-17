"""Per-task browser blocks from comparable work and explicit user choices."""

from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

import retrieval
import sample_retrieval
from activity import activity_key
from model_client import ModelUnavailable, planner_settings
from relevance_feedback import related_ids
from sample_retrieval import decision_hash, prefer_packaged


SITE_DECISIONS_PATH = Path(__file__).resolve().parent / "data" / "tasks" / "focus-decisions.json"

_SPECIFIC = (
    "Does the title of the observed browser tab name a specific activity or content, like playing a game, "
    "reading a story, viewing a score or shopping for an item? Answer false for an uninformative title "
    "like Home, New Tab, Sign in or a blank tab."
)
_GENERIC = (
    "Do all the observed browser tab titles consist only of generic navigation, such as Home, Dashboard, "
    "Sign in, New Tab, a bare site name, or blank? Answer false if even one title names a particular item, "
    "headline, event, game, or show. A named item need not include a verb."
)
_NEEDED = (
    "Does this current task specifically use the website in the past visit? Look at the new task goals, "
    "required resources and user instructions. Would blocking that domain prevent doing some of the new task?"
)


def _text(value, limit):
    return " ".join(str(value or "").split())[:limit]


def site_payload(target, domain, source, titles, config=None):
    """Use only task scope and observed browser evidence. IDs, timestamps and durations stay local."""
    config = planner_settings() if config is None else config
    model = config.get("PLANNER_DECISION_MODEL") or retrieval.DECISION_MODEL
    target = dict(target)
    sources = source if isinstance(source, (list, tuple)) else [source]
    previous_titles = sorted({_text(dict(row).get("title"), 180) for row in sources})
    if isinstance(titles, str):
        titles = [titles]
    tabs = sorted({clean for title in (titles or ()) if (clean := _text(title, 160))})
    # The activity report already labeled this visit off-task. Kev checks whether
    # the site is identifiable and whether blocking it would interfere with this task.
    state = {
        "observed_off_task_browser_visit": {"label": "off_task", "domain": domain, "tab_titles": tabs[:5]},
        "task_now": (_text(target.get("title"), 180) + ". " + _text(target.get("description"), 550)
                     + " " + _text(target.get("details"), 350)),
    }
    if len(sources) == 1:
        state["previous_task"] = previous_titles[0]
    else:
        state["previous_tasks"] = previous_titles[:5]
        state["related_tasks_with_off_task_visits"] = len(sources)
    # If the handout names this site/brand, retain the relevant passage. An
    # unrelated protocol, proof or assignment excerpt can overwhelm short tab evidence.
    pdf_text = target.get("pdf_text") or ""
    names = [label for label in domain.split(".")[:-1] if len(label) > 2 and label != "www"]
    if pdf_text and names:
        reference = re.search(r"(?<!\w)(?:" + "|".join(map(re.escape, names)) + r")(?!\w)", pdf_text, re.I)
        if reference:
            excerpt = pdf_text[max(0, reference.start() - 80):reference.end() + 120]
            state["site_reference_in_attachment"] = _text(excerpt, 240)
    question = ("specific", _SPECIFIC) if len(sources) == 1 else ("generic", _GENERIC)
    return {"model": model, "state": json.dumps(state, ensure_ascii=False),
            "questions": {question[0]: {"type": "noul", "instructions": question[1]},
                          "needed": {"type": "noul", "instructions": _NEEDED}}}


@lru_cache(maxsize=2)
def _site_decisions(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    if not isinstance(data, dict) or data.get("format") != 1 or not isinstance(data.get("decisions"), dict):
        return {}
    return data["decisions"]


def packaged_site_decision(request):
    """An exact packaged verdict or None. Safe to call in the bridge's read-only path."""
    digest = decision_hash(request)
    value = _site_decisions(SITE_DECISIONS_PATH).get(digest)
    if value is None:
        value = _site_decisions(SITE_DECISIONS_PATH.parent / "demo" / "models" / "sites.json").get(digest)
    return value if type(value) is bool else None


def live_site_decision(request, config=None):
    config = planner_settings() if config is None else config
    scores = retrieval.decision_scores(request, config, purpose="website blocking")
    identifiable = scores["generic"] < 0.5 if "generic" in scores else scores["specific"] >= 0.5
    return identifiable and scores["needed"] < 0.5


def _decide_site(request, config=None):
    """Use an exact public judgment or request a new decision from the local model."""
    config = planner_settings() if config is None else config
    value, _ = sample_retrieval.resolve(
        "decision", request["model"], config, lambda canonical: packaged_site_decision({**request, "model": canonical}),
        lambda: live_site_decision(request, config))
    return value


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
        yield domain, reason, site_payload(target, domain, sources_here, titles, config)


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
        key = decision_hash(request)
        with store._db() as db:
            if not db.execute("SELECT 1 FROM tasks WHERE " + current, args).fetchone():
                return
        try:
            results[key] = previous[key] if type(previous.get(key)) is bool else _decide_site(request, config)
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
        verdict = stored.get(decision_hash(request))
        if type(verdict) is not bool and prefer_packaged("decision", request["model"], config):
            verdict = packaged_site_decision(request)
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
