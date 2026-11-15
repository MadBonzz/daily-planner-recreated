"""Calibrate cosine recall on five profiles and evaluate on five untouched profiles."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path

from examples.estimation.run import DATASET, histories, task_inputs
import retrieval
from model_client import planner_settings


THRESHOLDS = [value / 100 for value in range(40, 96, 5)]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True)) / math.sqrt(
        sum(x ** 2 for x in a) * sum(x ** 2 for x in b))


def metrics(rows, passes):
    counts = Counter()
    for row in rows:
        accepted = passes(row)
        counts[(row["expected"], accepted)] += 1
    tp, fp, fn, tn = (counts[key] for key in ((True, True), (False, True), (True, False), (False, False)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
            "f2": 5 * tp / (5 * tp + 4 * fn + fp) if tp + fn + fp else 0}


def inputs(history):
    tasks = history["tasks"] + [row for row in history["holdouts"] if not row.get("fixed_duration")]
    return {row["id"]: task_inputs(row, history["id"], DATASET)[0] for row in tasks}


def collect(history, vectors, *, queries=None):
    queries = vectors if queries is None else queries
    split = "calibration" if int(history["id"].split("-")[0]) % 2 else "validation"
    return [{"history": history["id"], "split": split, "query": query["id"], "candidate": row["id"],
             "expected": row["id"] in query["evaluation"]["evidence_ids"],
             "cosine": cosine(queries[query["id"]], vectors[row["id"]])}
            for query in history["holdouts"] if not query.get("fixed_duration") for row in history["tasks"]]


def summarize(rows):
    splits = {}
    for split in ("calibration", "validation"):
        subset = [row for row in rows if row["split"] == split]
        splits[split] = [{"threshold": value, **metrics(subset, lambda r: r["cosine"] >= value)}
                         for value in THRESHOLDS]
    winner = max(splits["calibration"], key=lambda row: (row["f2"], row["precision"], row["threshold"]))
    return {"selection_metric": "calibration F2, breaking ties by precision then higher threshold",
            "threshold": winner["threshold"], "scores": splits}


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def embed_all(directory, profiles):
    config = planner_settings()
    rows, input_hashes = [], {}
    for history in profiles:
        tasks = inputs(history)
        vectors = {key: retrieval.embed_task(task, config) for key, task in tasks.items()}
        if len({len(vector) for vector in vectors.values()}) != 1:
            raise ValueError("Inconsistent vector dimensions")
        queries = {row["id"]: retrieval.embed_query(tasks[row["id"]], config)
                   for row in history["holdouts"] if not row.get("fixed_duration")}
        rows.extend(collect(history, vectors, queries=queries))
        input_hashes[history["id"]] = {
            key: hashlib.sha256(retrieval.task_text(task).encode()).hexdigest() for key, task in tasks.items()}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "model": config.get("PLANNER_EMBEDDING_MODEL") or retrieval.EMBEDDING_MODEL,
              "query_instruction": retrieval.QUERY_INSTRUCTION,
              "dimensions": len(next(iter(vectors.values()))), "task_text_sha256": input_hashes,
              **summarize(rows), "pairs": rows}
    write_json(directory / "embeddings.json", result)


def judge_fingerprint(embedding_run, model):
    # Include everything that can affect a pair judgment, independent of index and worker code.
    material = {"model": model, "threshold": embedding_run["threshold"],
                "task_text_sha256": embedding_run["task_text_sha256"],
                "cases": embedding_run["pairs"],
                "pair_examples": retrieval._PAIR_EXAMPLES, "instruction": retrieval._PAIR_INSTRUCTION,
                "implementation": [inspect.getsource(f) for f in
                                   (retrieval.decide_related, retrieval._decision_record, retrieval.document_excerpt)]}
    encoded = json.dumps(material, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def judge_validation(directory, profiles):
    embedding_run = json.loads((directory / "embeddings.json").read_text(encoding="utf-8"))
    threshold = embedding_run["threshold"]
    tasks = {h["id"]: inputs(h) for h in profiles}
    rows = [r for r in embedding_run["pairs"] if r["split"] == "validation"]
    path = directory / "decisions.json"
    model = planner_settings().get("PLANNER_DECISION_MODEL") or retrieval.DECISION_MODEL
    # Preserve completed pairs on interruption. A different judge, prompt or input requires a fresh run.
    fingerprint = judge_fingerprint(embedding_run, model)
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if saved and saved.get("input_sha256") != fingerprint:
        raise ValueError("Start a new run directory after changing the model, retrieval code or embeddings")
    pairs = saved.get("pairs", [])
    done = {(r["history"], r["query"], r["candidate"]) for r in pairs}
    for row in rows:
        if row["cosine"] < threshold or (row["history"], row["query"], row["candidate"]) in done:
            continue
        profile = tasks[row["history"]]
        related = retrieval.decide_related(profile[row["query"]], profile[row["candidate"]])
        pairs.append({**row, "related": related})
        write_json(path, {"complete": False, "model": model, "input_sha256": fingerprint, "pairs": pairs})
    accepted = {(p["history"], p["query"], p["candidate"]) for p in pairs if p["related"]}
    scores = metrics(rows, lambda r: (r["history"], r["query"], r["candidate"]) in accepted)
    write_json(path, {"complete": True, "model": model, "input_sha256": fingerprint,
                      "threshold": threshold, "end_to_end_validation": scores, "pairs": pairs})


def operating_point(directory, threshold):
    """Score a stricter cutoff using the recorded embedding scores and pair decisions."""
    embedding_file = directory / "embeddings.json"
    decision_file = directory / "decisions.json"
    embeddings = json.loads(embedding_file.read_text(encoding="utf-8"))
    decisions = json.loads(decision_file.read_text(encoding="utf-8"))
    if not decisions.get("complete") or not decisions["threshold"] <= threshold <= 1:
        raise ValueError("Choose a cutoff covered by a complete decision run")

    def key(row):
        return (row["history"], row["query"], row["candidate"])

    judged = {key(row): row for row in decisions["pairs"]}
    pairs = embeddings["pairs"]
    split_scores = {}
    for split in ("calibration", "validation"):
        cohort = [p for p in pairs if p["split"] == split]
        split_scores[split] = metrics(cohort, lambda r: r["cosine"] >= threshold)

    validation = [p for p in pairs if p["split"] == "validation"]
    selected = []
    for pair in validation:
        if pair["cosine"] < threshold:
            continue
        decision = judged.get(key(pair))
        if decision is None or any(decision[k] != v for k, v in pair.items()):
            raise ValueError(f"Missing or inconsistent decision for {key(pair)}")
        selected.append(decision)
    accepted = {key(r) for r in selected if r["related"]}
    scores = metrics(validation, lambda r: key(r) in accepted)
    write_json(directory / "operating-point.json", {
        "threshold": threshold, "input_run_threshold": decisions["threshold"],
        "embedding_results_sha256": hashlib.sha256(embedding_file.read_bytes()).hexdigest(),
        "decision_results_sha256": hashlib.sha256(decision_file.read_bytes()).hexdigest(),
        "embedding_scores": split_scores, "end_to_end_validation": scores,
        "validation_candidates": selected})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="A new per-run directory under data/results/")
    parser.add_argument("--stage", choices=["embeddings", "decisions", "operating-point"], required=True)
    parser.add_argument("--threshold", type=float, help="Cosine cutoff to score using a completed run")
    args = parser.parse_args()
    if (args.stage == "operating-point") != (args.threshold is not None):
        parser.error("Supply --threshold with the operating-point stage only")
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.stage == "embeddings":
        if (args.directory / "embeddings.json").exists():
            parser.error("Choose a fresh directory to preserve an existing run")
        embed_all(args.directory, histories(None))
    elif args.stage == "decisions":
        judge_validation(args.directory, histories(None))
    else:
        operating_point(args.directory, args.threshold)


if __name__ == "__main__":
    main()
