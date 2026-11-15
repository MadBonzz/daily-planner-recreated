"""Per-task browser blocks from comparable work and explicit user choices."""

from collections import defaultdict
import json
import re
from urllib.parse import urlsplit

from activity import activity_key
from relevance_feedback import related_ids


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


def _suggestions(db, task, sources):
    if not sources:
        return {}
    ids = json.dumps(sources)
    rows = db.execute("""SELECT id, title, report FROM tasks WHERE status='completed' AND report IS NOT NULL
                         AND id IN (SELECT value FROM json_each(?)) ORDER BY title, id""", (ids,))
    # An explicit allow on comparable work also counts against a blanket rule.
    allowed = {r["domain"] for r in db.execute("""SELECT domain FROM focus_domain_choices
               WHERE blocked=0 AND task_id IN (SELECT value FROM json_each(?))""", (ids,))}
    feedback = {(r["task_id"], r["activity_key"]): r["label"] for r in db.execute(
        """SELECT task_id,activity_key,label FROM activity_feedback WHERE task_id IN (SELECT value FROM json_each(?))""",
        (ids,))}
    setting = db.execute("SELECT value FROM settings WHERE key='exclusions'").fetchone()
    exclusions = json.loads(setting[0]) if setting else []
    excluded = [v.strip().casefold() for v in exclusions if isinstance(v, str) and v.strip()]
    repeats = defaultdict(dict)
    for row in rows:
        for event in json.loads(row["report"]).get("intervals", []):
            if event.get("excluded") or not event.get("domain") or event.get("seconds", 0) <= 0:
                continue
            text = " ".join(str(event.get(k) or "") for k in ("app", "title", "domain", "phone_app")).casefold()
            if any(term in text for term in excluded):
                continue
            try:
                host = normalize_domain(event["domain"])
            except ValueError:
                continue
            label = feedback.get((row["id"], activity_key(event)), event.get("label"))
            if label == "on_task":
                allowed.add(host)
            elif label == "off_task" and not event.get("idle") and not event.get("phone_app"):
                # One classification over a desktop and a phone cannot establish which one distracted the user.
                if row["id"] not in repeats[host]:
                    repeats[host][row["id"]] = [row["title"], 0.0]
                repeats[host][row["id"]][1] += event["seconds"]
    recurring = {domain: observations for domain, observations in repeats.items()
                 if len(observations) >= 2 and not any(_blocked_by(host, domain) or _blocked_by(domain, host) for host in allowed)}
    if not recurring:
        return {}
    # A link in the target may be part of its instructions even if it was distracting in the past.
    fields = db.execute("SELECT title, description, details, pdf_text FROM tasks WHERE id=?", (task["id"],)).fetchone()
    target_text = " ".join(value or "" for value in fields).casefold()
    suggestions = {}
    for domain, observations in sorted(recurring.items()):
        if re.search(r"(?<![\w-])" + re.escape(domain) + r"(?![\w-]|\.[\w-])", target_text):
            continue
        names = [title for title, _ in observations.values()]
        minutes = sum(seconds for _, seconds in observations.values()) / 60
        examples = " and ".join(f"“{name}”" for name in names[:2])
        suggestions[domain] = (f"{minutes:.1f} minutes labeled distraction across {len(names)} related tasks, "
                               f"including {examples}.")
    return suggestions


def policy_from_db(db, task_id):
    """Compute a read-only policy in the caller's transaction. Only the active timer should enforce it."""
    task = db.execute("SELECT id, prediction FROM tasks WHERE id=? AND archived=0", (task_id,)).fetchone()
    if task is None:
        raise ValueError("This task is no longer available.")
    suggestions = _suggestions(db, task, _related_sources(db, task))
    choices = {r["domain"]: bool(r["blocked"]) for r in db.execute(
        "SELECT domain, blocked FROM focus_domain_choices WHERE task_id=?", (task_id,))}
    added = {d for d, enabled in choices.items() if enabled}
    removed = {d for d, enabled in choices.items() if not enabled}
    return {"domains": sorted((suggestions.keys() | added) - removed), "suggestions": suggestions,
            "added": sorted(added), "removed": sorted(removed)}


def policy(store, task_id):
    with store._db() as db:
        return policy_from_db(db, task_id)


def save_domains(store, task_id, selected, reviewed):
    """Apply explicit additions and removals to the last displayed list, leaving new evidence live."""
    selected = {normalize_domain(v) for v in selected}
    reviewed = {normalize_domain(v) for v in reviewed}
    edits = [(task_id, domain, 1) for domain in selected - reviewed]
    edits.extend((task_id, domain, 0) for domain in reviewed - selected)
    with store._db(write=True) as db:
        if not db.execute("SELECT 1 FROM tasks WHERE id=? AND archived=0", (task_id,)).fetchone():
            raise ValueError("This task is no longer available.")
        db.executemany("""INSERT INTO focus_domain_choices(task_id,domain,blocked) VALUES(?,?,?)
                          ON CONFLICT(task_id,domain) DO UPDATE SET blocked=excluded.blocked""", edits)
