"""Build the portable Qwen and Kev results for the ten public Dev histories."""

import argparse
from itertools import combinations
import json
from pathlib import Path

from examples.estimation.calibrate_retrieval import cosine
from examples.estimation.run import DATASET, histories, task_inputs
import retrieval
import sample_retrieval as cache


def inputs(history):
    return {row["id"]: task_inputs(row, history["id"], DATASET)[0]
            for group in ("tasks", "open_tasks", "holdouts") for row in history[group]}


def matrix_pairs(history):
    events = json.loads((DATASET / "schedules" / (history["id"] + ".json")).read_text(encoding="utf-8"))
    active, pairs = set(), set()
    for event in events:
        task = event["task"]
        if event["action"] == "create":
            for other in active:
                # A finish before the matrix job can swap the unfinished task into first position.
                pairs.update(((task, other), (other, task)))
            active.add(task)
        elif event["action"] == "finish":
            active.remove(task)
    holdouts = {task["id"] for task in history["holdouts"]}
    # Users can add the same sample twice. Distinct runtime IDs still need a real pair decision.
    pairs.update((task, task) for task in holdouts)
    for first, second in combinations(active | holdouts, 2):
        pairs.update(((first, second), (second, first)))
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


def empty_bundle(history):
    return {"format": cache.FORMAT, "history": history["id"], "embedding_model": retrieval.EMBEDDING_MODEL,
            "decision_model": retrieval.DECISION_MODEL, "query_instruction": retrieval.QUERY_INSTRUCTION,
            "cosine_floor": retrieval.MINIMUM_THRESHOLD, "vectors": {}, "decisions": {}}


def save(path, bundle):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def build_vectors(tasks, bundle):
    for task_id, task in tasks.items():
        roles = {}
        for role in ("query", "document"):
            query = role == "query"
            text = retrieval.embedding_input(task, query)
            vector = retrieval._live_vector(text, retrieval.EMBEDDING_MODEL, retrieval.planner_settings())
            roles[role] = {"input_sha256": cache.text_hash(text), "f32le_base64": cache.packed_vector(vector)}
        bundle["vectors"][task_id] = roles


def build_decisions(history, tasks, bundle, path):
    vectors = {key: {role: cache.unpack_vector(v["f32le_base64"]) for role, v in roles.items()}
               for key, roles in bundle["vectors"].items()}
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
        save(path, bundle)
        print(f"{history['id']} {position}/{len(pairs)} task pairs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage", choices=["vectors", "decisions"], required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    config = retrieval.planner_settings()
    if (config.get("PLANNER_DECISION_MODEL") or retrieval.DECISION_MODEL) != retrieval.DECISION_MODEL:
        parser.error("Select Kev-4B to build the packaged decisions.")
    for history in histories(None):
        path = args.output / (history["id"] + ".json")
        tasks = inputs(history)
        if args.stage == "vectors":
            bundle = empty_bundle(history)
            build_vectors(tasks, bundle)
            save(path, bundle)
        else:
            bundle = json.loads(path.read_text(encoding="utf-8"))
            if set(bundle["vectors"]) != set(tasks):
                parser.error("Run the vector stage again after adding or removing sample tasks.")
            for task_id, task in tasks.items():
                for role in ("query", "document"):
                    expected = cache.text_hash(retrieval.embedding_input(task, role == "query"))
                    if bundle["vectors"].get(task_id, {}).get(role, {}).get("input_sha256") != expected:
                        parser.error("Run the vector stage again after changing tasks or instructions.")
            build_decisions(history, tasks, bundle, path)
        print(history["id"], len(bundle["vectors"]), "tasks,", len(bundle["decisions"]), "decisions", flush=True)


if __name__ == "__main__":
    main()
