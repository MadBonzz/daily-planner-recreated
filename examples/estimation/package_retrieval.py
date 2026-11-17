"""Build portable Qwen and Kev results for the ten original benchmark histories."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from sample_retrieval import cosine
from examples.estimation.histories import DATASET, REPO, histories, task_inputs, valid_id, write_json
import retrieval
import sample_retrieval as cache


def inputs(history):
    return {row["id"]: task_inputs(row, history["id"], DATASET)[0]
            for group in ("tasks", "open_tasks", "holdouts") for row in history[group]}


def public_revision(ref):
    remote = subprocess.check_output(["git", "-C", REPO, "config", "--get", "remote.origin.url"], text=True)
    remote = remote.strip().lower().rstrip("/").removesuffix(".git")
    if remote not in {"https://github.com/madbonzz/daily-planner-recreated",
                      "git@github.com:madbonzz/daily-planner-recreated",
                      "ssh://git@github.com/madbonzz/daily-planner-recreated"}:
        raise ValueError("Check out the published daily-planner-recreated repository before exporting sample inputs.")
    revision = subprocess.check_output(
        ["git", "-C", REPO, "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"], text=True).strip()
    subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", revision, "origin/main"], check=True)
    return revision


def public_sources(history, revision):
    """Require inputs to match blobs from the published sample tree before embedding or exporting."""
    name = history["id"]
    if not valid_id(name):
        raise ValueError("Invalid sample history ID")
    paths = {f"histories/{name}.json", f"schedules/{name}.json"}
    for group in ("tasks", "open_tasks", "holdouts"):
        for task in history[group]:
            if not valid_id(task["id"]):
                raise ValueError("Invalid sample task ID")
            pdf = task.get("pdf")
            if pdf:
                stem = pdf if isinstance(pdf, str) else task["id"]
                if not valid_id(stem):
                    raise ValueError("Invalid sample PDF reference")
                paths.add(f"pdfs/{name}/{stem}.pdf")
    files = {}
    for path in sorted(paths):
        file = DATASET / path
        if not file.resolve().is_relative_to(DATASET.resolve()) or file.is_symlink():
            raise ValueError(f"Invalid sample input path: {path}")
        committed = subprocess.check_output(["git", "-C", REPO, "show", f"{revision}:data/tasks/{path}"])
        if file.read_bytes() != committed:
            raise ValueError(f"Sample input differs from published revision {revision}: {path}")
        files[path] = hashlib.sha256(committed).hexdigest()
    if history != json.loads((DATASET / "histories" / (name + ".json")).read_text(encoding="utf-8")):
        raise ValueError("In-memory input differs from the public history.")
    return {"repository": "MadBonzz/daily-planner-recreated", "revision": revision, "files_sha256": files}


def matrix_pairs(history):
    events = json.loads((DATASET / "schedules" / (history["id"] + ".json")).read_text(encoding="utf-8"))
    active, pairs = set(), set()
    for event in events:
        task = event["task"]
        if event["action"] == "create":
            pairs.update((task, other) for other in active)
            active.add(task)
        elif event["action"] == "finish":
            active.remove(task)
    holdouts = {task["id"] for task in history["holdouts"]}
    # Either holdout can be added first, and two copies can have distinct runtime IDs.
    pairs.update((task, earlier) for task in holdouts for earlier in active | holdouts)
    return pairs


def needed_pairs(history, vectors):
    """Judge candidate pairs when the schedule could ask for them, plus holdout comparisons."""
    pairs = matrix_pairs(history)
    tasks = {row["id"]: row for group in ("tasks", "open_tasks", "holdouts") for row in history[group]}
    events = json.loads((DATASET / "schedules" / (history["id"] + ".json")).read_text(encoding="utf-8"))
    created, finished = {}, {}
    for i, e in enumerate(events):
        if e["action"] == "create":
            created[e["task"]] = i
        elif e["action"] == "finish":
            finished[e["task"]] = i
    for target, roles in vectors.items():
        if tasks[target].get("fixed_duration"):
            continue
        for candidate, prior in vectors.items():
            if (candidate == target or tasks[candidate].get("fixed_duration") or
                    finished.get(target, float("inf")) < created.get(candidate, float("inf"))):
                continue
            # Include the bare query as a control for the instruction ablation.
            if max(cosine(roles[r], prior["document"]) for r in ("query", "document")) >= retrieval.MINIMUM_THRESHOLD - 1e-6:
                pairs.add((target, candidate))
    return pairs


def empty_bundle(history, sources):
    return {"format": cache.FORMAT, "history": history["id"], "embedding_model": retrieval.EMBEDDING_MODEL,
            "decision_model": retrieval.DECISION_MODEL, "query_instruction": retrieval.QUERY_INSTRUCTION,
            "cosine_floor": retrieval.MINIMUM_THRESHOLD, "public_sources": sources, "vectors": {}, "decisions": {}}


def build_vectors(tasks, bundle, path):
    for task_id, task in tasks.items():
        vectors, hashes = {}, {}
        for role in cache.ROLES:
            text = retrieval.embedding_input(task, role == "query")
            vectors[role] = retrieval._live_vector(text, retrieval.EMBEDDING_MODEL, retrieval.planner_settings())
            hashes[role] = {"input_sha256": cache.text_hash(text)}
        cache.write_vectors(cache.csv_path(path, task_id), vectors)
        bundle["vectors"][task_id] = hashes


def build_decisions(history, tasks, bundle, path):
    vectors = {key: cache.read_vectors(path, key) for key in bundle["vectors"]}
    holdouts = {row["id"] for row in history["holdouts"]}
    pairs = sorted(needed_pairs(history, vectors), key=lambda pair: (pair[0] not in holdouts, *pair))
    for position, (target, candidate) in enumerate(pairs, 1):
        a, b = tasks[target], tasks[candidate]
        payload = retrieval.decision_payload(a, b, [], retrieval.DECISION_MODEL)
        key = cache.decision_hash(payload)
        if key in bundle["decisions"]:
            continue
        related = retrieval._live_decision(payload, retrieval.planner_settings())
        bundle["decisions"][key] = {"target": target, "candidate": candidate, "related": related}
        write_json(path, bundle, sort_keys=True)
        print(f"{history['id']} {position}/{len(pairs)} task pairs", flush=True)


def verify_index(path, history, tasks, bundle, sources, *, complete):
    expected = empty_bundle(history, sources)
    if set(bundle) != set(expected) or any(
            bundle[field] != value for field, value in expected.items() if field not in {"decisions", "vectors"}):
        raise ValueError(f"Invalid or stale fixture/model metadata in {path.name}")
    if set(bundle["vectors"]) != set(tasks):
        raise ValueError(f"Unexpected task IDs in {path.name}")
    directory = path.parent / path.stem
    if {p.name for p in directory.iterdir()} != {key + ".csv" for key in tasks}:
        raise ValueError(f"The numeric files do not match the public task IDs for {path.stem}")
    vectors = {}
    for task_id, task in tasks.items():
        hashes = {role: {"input_sha256": cache.text_hash(retrieval.embedding_input(task, role == "query"))}
                  for role in cache.ROLES}
        if bundle["vectors"][task_id] != hashes:
            raise ValueError(f"Unexpected embedding input in {path.stem}/{task_id}")
        vectors[task_id] = cache.read_vectors(path, task_id)
    pairs = set()
    for digest, row in bundle["decisions"].items():
        if set(row) != {"target", "candidate", "related"} or type(row["related"]) is not bool:
            raise ValueError("Expected two public task IDs and a boolean Kev decision.")
        target, candidate = row["target"], row["candidate"]
        if target not in tasks or candidate not in tasks:
            raise ValueError("Unknown task ID in a Kev decision.")
        payload = retrieval.decision_payload(tasks[target], tasks[candidate], [], retrieval.DECISION_MODEL)
        if cache.decision_hash(payload) != digest:
            raise ValueError(f"Stale Kev decision for {target} and {candidate}")
        pairs.add((target, candidate))
    if complete and (missing := needed_pairs(history, vectors) - pairs):
        raise ValueError(f"Missing {len(missing)} required Kev decisions for {path.stem}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage", choices=["vectors", "decisions", "verify"], required=True)
    parser.add_argument("--public-revision", default="origin/main", help="Published commit containing the sample inputs")
    args = parser.parse_args()
    revision = public_revision(args.public_revision)
    args.output.mkdir(exist_ok=True, parents=True)
    config = retrieval.planner_settings()
    if args.stage == "decisions" and (config.get("PLANNER_DECISION_MODEL") or retrieval.DECISION_MODEL) != retrieval.DECISION_MODEL:
        parser.error("Select Kev-4B to build the packaged decisions.")
    for history in histories(None):
        path = args.output / (history["id"] + ".json")
        sources = public_sources(history, revision)
        tasks = inputs(history)
        if args.stage == "vectors":
            bundle = empty_bundle(history, sources)
            build_vectors(tasks, bundle, path)
            write_json(path, bundle, sort_keys=True)
        else:
            bundle = json.loads(path.read_text(encoding="utf-8"))
            verify_index(path, history, tasks, bundle, sources, complete=args.stage == "verify")
            if args.stage == "decisions":
                build_decisions(history, tasks, bundle, path)
        print(history["id"], len(bundle["vectors"]), "tasks,", len(bundle["decisions"]), "decisions", flush=True)


if __name__ == "__main__":
    main()
