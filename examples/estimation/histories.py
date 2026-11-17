"""Load the planner’s sample histories, task inputs and original attachments."""

import json
from pathlib import Path
import re

from pdf_reader import extract_pdf_text


REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "data" / "tasks"
DEMO_DATASET = DATASET / "demo"
INPUT_KEYS = ("day", "title", "description", "category", "details", "estimate_minutes")


def write_json(path, data, *, sort_keys=False):
    """Atomically replace one evaluation checkpoint after successful serialization."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False, sort_keys=sort_keys) + "\n", encoding="utf-8")
    tmp.replace(path)


def sample_root(history_id, directory=DATASET):
    """Locate a live or archived fixture without changing benchmark defaults."""
    if not valid_id(history_id):
        raise ValueError("Invalid history ID")
    root = Path(directory)
    if (root / "demo" / "histories" / f"{history_id}.json").is_file():
        return root / "demo"
    return root


def pdf_file(task, history_id, directory):
    """Resolve an authored attachment or an explicit reference to a shared sample PDF."""
    if not valid_id(history_id):
        raise ValueError("Invalid fixture history ID")
    if "pdf_path" in task:
        if Path(directory).resolve() != DEMO_DATASET.resolve():
            raise ValueError("Shared PDF references belong to the demo catalog.")
        reference = task["pdf_path"]
        if not isinstance(reference, str) or not re.fullmatch(r"pdfs/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+\.pdf", reference):
            raise ValueError("Invalid shared sample PDF reference")
        path = DATASET / reference
        if not path.resolve().is_relative_to((DATASET / "pdfs").resolve()):
            raise ValueError("Shared sample PDF is outside the fixture directory.")
        return path
    stem = task["pdf"] if isinstance(task["pdf"], str) else task["id"]
    if not valid_id(stem):
        raise ValueError("Invalid fixture PDF reference")
    return Path(directory) / "pdfs" / history_id / f"{stem}.pdf"


def documents(task, history_id, directory):
    if not task.get("pdf"):
        return []
    path = pdf_file(task, history_id, directory)
    return [(path.name, path.read_bytes())]


def task_inputs(task, history_id, directory):
    """Only pre-task fields are permitted here. Outcome, sessions and 'check' stay in the fixture."""
    files = documents(task, history_id, directory)
    fields = {key: task[key] for key in INPUT_KEYS}
    if "fixed_duration" in task:
        fields["fixed_duration"] = task["fixed_duration"]
    fields["pdf_text"] = "\n".join(extract_pdf_text(data) for _, data in files)
    return fields, files


def valid_id(value):
    return isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value)


def histories(selector, directory=DATASET):
    files = sorted((directory / "histories").glob("*.json"))
    all_histories = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    if not all_histories:
        raise ValueError("No histories found")
    ids = [history["id"] for history in all_histories]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate history ID")
    if selector is None:
        return all_histories
    matches = [h for h in all_histories if selector == h["id"] or selector == h["id"].split("-")[0]]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous history {selector!r}. Available: {', '.join(ids)}")
    return matches
