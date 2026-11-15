"""Ask Kev whether one observed off-task website is worth blocking for the current task."""

from functools import lru_cache
import json
from pathlib import Path
import re

from model_client import ModelUnavailable, planner_settings
import retrieval
from sample_retrieval import decision_hash as key


CACHE_PATH = Path(__file__).resolve().parent / "data" / "tasks" / "focus-decisions.json"

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


def payload(target, domain, source, titles, config=None):
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


@lru_cache(maxsize=1)
def _decisions(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    if not isinstance(data, dict) or data.get("format") != 1 or not isinstance(data.get("decisions"), dict):
        return {}
    return data["decisions"]


def packaged(request):
    """An exact packaged verdict or None. Safe to call in the bridge's read-only path."""
    value = _decisions(CACHE_PATH).get(key(request))
    return value if type(value) is bool else None


def live_decision(request, config=None):
    config = planner_settings() if config is None else config
    endpoint = config.get("PLANNER_DECISION_ENDPOINT") or "http://127.0.0.1:1234/v1/systemone"
    reply = retrieval._local_post(endpoint, request, config)
    try:
        repeated = "generic" in request["questions"]
        title_score, needed = (reply["answers"][name]["noul"]
                               for name in ("generic" if repeated else "specific", "needed"))
        if any(type(value) not in (int, float) or not 0 <= value <= 1 for value in (title_score, needed)):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelUnavailable(f"{request['model']} returned an invalid website blocking decision.") from exc
    identifiable = title_score < 0.5 if repeated else title_score >= 0.5
    return identifiable and needed < 0.5


def decide(request, config=None):
    """Use an exact public judgment or request a new decision from the local model."""
    value = packaged(request)
    if value is not None:
        return value
    return live_decision(request, config)
