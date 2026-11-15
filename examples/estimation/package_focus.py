"""Build public Kev website verdicts, or verify coverage offline with --check.

Run with python -m examples.estimation.package_focus from a clone of the public repo.
Replays published task, PDF, ActivityWatch and retrieval fixtures in temporary workspaces.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import dev_schedule
import dev_workspace
import focus_domains
import focus_judge
import retrieval
import sample_retrieval
from examples.estimation.calibrate_retrieval import cosine
from examples.estimation.package_retrieval import inputs, public_revision, public_sources, save
from examples.estimation.run import DATASET, REPO, histories
from model_client import planner_settings
from planner_store import PlannerStore


def collect(history, store, revision, requests):
    """Replay only already-finished sources at each step. Use the production site evidence path."""
    name = history["id"]
    public_sources(history, revision)
    tasks = inputs(history)
    paths = [f"data/tasks/activitywatch/{name}.db", f"data/tasks/retrieval/{name}.json",
             *(f"data/tasks/retrieval/{name}/{key}.csv" for key in sorted(tasks))]
    published = subprocess.check_output(
        ["git", "-C", REPO, "ls-tree", "-r", "--name-only", revision, "--", *paths], text=True)
    if set(published.splitlines()) != set(paths):
        raise ValueError(f"Activity and retrieval inputs for {name} must exist in the public revision.")
    subprocess.run(["git", "-C", REPO, "diff", "--quiet", revision, "--", *paths], check=True)
    index_path = sample_retrieval.DIRECTORY / (name + ".json")
    bundle = json.loads(index_path.read_text(encoding="utf-8"))
    vectors = {key: sample_retrieval.read_vectors(index_path, key) for key in tasks}
    for key, fields in tasks.items():
        for role in sample_retrieval.ROLES:
            digest = sample_retrieval.text_hash(retrieval.embedding_input(fields, role == "query"))
            if digest != bundle["vectors"][key][role]["input_sha256"]:
                raise ValueError(f"Unexpected {role} input for {name}/{key}")

    @lru_cache(maxsize=None)
    def timing_match(target, source):
        if tasks[target].get("fixed_duration") or tasks[source].get("fixed_duration"):
            return False
        score = max(cosine(vectors[target][role], vectors[source]["document"]) for role in ("query", "document"))
        if score < retrieval.MINIMUM_THRESHOLD:
            return False
        request = retrieval.decision_payload(tasks[target], tasks[source], [], retrieval.DECISION_MODEL)
        hit = bundle["decisions"][sample_retrieval.decision_hash(request)]
        if hit["target"] != target or hit["candidate"] != source or type(hit["related"]) is not bool:
            raise ValueError(f"Unexpected packaged timing decision for {name}/{target}/{source}")
        return hit["related"]

    ids, finished, active = {}, set(), set()
    chronology = set()

    def inspect(target, selected=None):
        if selected is None:
            selected = [source for source in sorted(finished) if timing_match(target, source)]
        sources = [ids[source] for source in selected]
        evidence = focus_domains.site_requests(store, ids[target], sources, pdf_text=tasks[target]["pdf_text"],
                                                config={"PLANNER_DECISION_MODEL": retrieval.DECISION_MODEL})
        checks = {}
        for domain, request in evidence.items():
            digest = focus_judge.key(request)
            requests[digest] = request
            checks[domain] = digest
        return checks

    # Absolute dates and ephemeral database IDs never enter the Kev prompt.
    anchor = datetime(2026, 1, 15, 15, tzinfo=timezone.utc).timestamp()
    dev_schedule.initialize(store, name, DATASET, replay=True, now=anchor)
    _, events = dev_schedule._inputs(name, DATASET)
    calendar = store.get_setting("dev_calendar")
    rows, agenda, sessions, _ = dev_schedule._agenda(history, events, calendar)
    for item in agenda:
        target = item["task"]
        dev_schedule._apply(store, history, rows[target], item, calendar, sessions, DATASET, infer=False)
        if item["action"] == "create":
            ids[target] = store.get_setting(f"dev_task_{target}")
            active.add(target)
            chronology.update(inspect(target).values())
        elif item["action"] == "finish":
            active.remove(target)
            if rows[target]["actual_minutes"] > 0:
                finished.add(target)
            for other in sorted(active):
                chronology.update(inspect(other).values())

    for row in history["holdouts"]:
        ids[row["id"]] = dev_workspace.add_sample_task(store, name, row["id"], calendar["day"])["id"]
    snapshot, manual = {}, {}
    for row in history["open_tasks"] + history["holdouts"]:
        target = row["id"]
        snapshot[target] = inspect(target)
        # A site's prompt only depends on sources with visits to that site. Cover all
        # public combinations of those sources so arbitrary edits work without Kev.
        singletons = {source: checks for source in sorted(finished)
                      if (checks := inspect(target, [source]))}
        manual[target] = dict(singletons)
        for domain in sorted({domain for checks in singletons.values() for domain in checks}):
            sources = [source for source, checks in singletons.items() if domain in checks]
            for size in range(2, len(sources) + 1):
                for group in combinations(sources, size):
                    checks = inspect(target, group)
                    if domain in checks:
                        manual[target].setdefault("+".join(group), {})[domain] = checks[domain]
    return snapshot, manual, sorted(chronology)


def public_requests(revision):
    requests, snapshots, manual, chronology = {}, {}, {}, {}
    with TemporaryDirectory() as tmp:
        for history in histories(None):
            store = PlannerStore(Path(tmp) / history["id"])
            store.set_capture(False)
            name = history["id"]
            snapshots[name], manual[name], chronology[name] = collect(history, store, revision, requests)
    return requests, snapshots, manual, chronology


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify package coverage without requesting a model.")
    parser.add_argument("--public-ref", help="Published sample revision (defaults to the package revision or origin/main).")
    parser.add_argument("--checkpoint", type=Path, help="Save completed live judgments and resume an interrupted build.")
    args = parser.parse_args()
    path = focus_judge.CACHE_PATH
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    revision = public_revision(args.public_ref or old.get("public_revision", "origin/main"))
    requests, snapshots, manual, chronology = public_requests(revision)
    metadata = {"format": 1, "public_revision": revision, "model": retrieval.DECISION_MODEL,
                "cosine_floor": retrieval.MINIMUM_THRESHOLD, "snapshot_requests": snapshots,
                "manual_requests": manual, "chronological_requests": chronology}
    if args.check:
        if (any(old.get(k) != v for k, v in metadata.items()) or set(old.get("decisions", {})) != set(requests)
                or any(type(v) is not bool for v in old["decisions"].values())):
            raise ValueError("Packaged focus decisions differ from the public prompts or coverage.")
    else:
        config = {**planner_settings(), "PLANNER_DECISION_MODEL": retrieval.DECISION_MODEL}
        decisions = {k: v for k, v in old.get("decisions", {}).items() if k in requests and type(v) is bool}
        if args.checkpoint and args.checkpoint.exists():
            prior = json.loads(args.checkpoint.read_text(encoding="utf-8"))
            if prior.get("public_revision") == revision:
                decisions.update({k: v for k, v in prior["decisions"].items() if k in requests and type(v) is bool})
        ordered = sorted(requests.items(), key=lambda pair: pair[1]["state"])
        for index, (digest, request) in enumerate(ordered, 1):
            if digest in decisions:
                continue
            decisions[digest] = focus_judge.live_decision(request, config)
            if args.checkpoint:
                save(args.checkpoint, {"public_revision": revision, "decisions": decisions})
            print(f"{index}/{len(requests)} public domain judgments", flush=True)
        save(path, {**metadata, "decisions": decisions})
    count = sum(len(targets) for targets in snapshots.values())
    print(f"{len(requests)} unique judgments across the dated replay and {count} open/addable sample tasks.")


if __name__ == "__main__":
    main()
