"""Package exact missing Qwen/Kev requests for the authored Dev schedules, one resumable stage at a time."""

import argparse
from collections import Counter
from contextlib import nullcontext
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

import portalocker

import dev_schedule
import dev_workspace
import focus_domains
import focus_judge
import retrieval
import sample_retrieval as cache
import workspace_eligibility as workspace
from examples.estimation.run import DEMO_DATASET, histories, task_inputs, write_json
from examples.estimation.scoring import cosine
from planner_store import PlannerStore


MODELS = DEMO_DATASET / "models"
VECTOR_INDEX = MODELS / "retrieval.json"
CONFIG = {"PLANNER_DECISION_MODEL": retrieval.DECISION_MODEL}


def tasks_for(history):
    return {row["id"]: task_inputs(row, history["id"], DEMO_DATASET)[0]
            for group in ("tasks", "open_tasks", "holdouts") for row in history[group]}


def load(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def retrieval_bundle():
    return load(VECTOR_INDEX, {"format": cache.FORMAT, "embedding_model": retrieval.EMBEDDING_MODEL,
                               "decision_model": retrieval.DECISION_MODEL, "vectors": {}, "decisions": {}})


def vectors(tasks, *, generate, stats):
    bundle = retrieval_bundle()
    all_vectors = {}
    for task_id, task in tasks.items():
        inputs = {role: retrieval.embedding_input(task, query=role == "query") for role in cache.ROLES}
        values = {role: cache.vector_for(retrieval.EMBEDDING_MODEL, text) for role, text in inputs.items()}
        missing = [role for role, value in values.items() if value is None]
        stats["required"] += len(values)
        stats["missing"] += len(missing)
        if missing and generate:
            for role in missing:
                values[role] = retrieval._live_vector(inputs[role], retrieval.EMBEDDING_MODEL, retrieval.planner_settings())
            name = cache.text_hash(inputs["document"])
            cache.write_vectors(cache.csv_path(VECTOR_INDEX, name), values)
            bundle["vectors"][name] = {role: {"input_sha256": cache.text_hash(text)} for role, text in inputs.items()}
            write_json(VECTOR_INDEX, bundle, sort_keys=True)
            cache._indexes.cache_clear()
            print(f"Qwen {task_id} ({len(missing)} vectors)", flush=True)
        elif missing:
            continue
        all_vectors[task_id] = values
    return all_vectors


def schedule_pairs(history, tasks, vectors):
    """Enumerate first forecasts, re-estimates, active overlap, and open/addable task completion orders."""
    name = history["id"]
    events = json.loads((DEMO_DATASET / "schedules" / f"{name}.json").read_text(encoding="utf-8"))
    opened, finished = set(), set()
    requests = {}
    matrix = {}

    def pair(target, candidate, *, active):
        left, right = tasks[target], tasks[candidate]
        if active:
            left, right = retrieval.active_decision_order(left, right, model=retrieval.DECISION_MODEL)
        request = retrieval.decision_payload(left, right, [], retrieval.DECISION_MODEL)
        digest = cache.decision_hash(request)
        requests[digest] = request
        if active:
            matrix[frozenset([target, candidate])] = digest
        return digest

    @lru_cache(maxsize=None)
    def historical(target, candidate):
        if tasks[target].get("fixed_duration") or tasks[candidate].get("fixed_duration"):
            return None
        if max(cosine(vectors[target][role], vectors[candidate]["document"]) for role in cache.ROLES) < retrieval.MINIMUM_THRESHOLD - 1e-6:
            return None
        return pair(target, candidate, active=False)

    for event in events:
        target = event["task"]
        if event["action"] == "create":
            for key in sorted(opened):
                pair(target, key, active=True)
            for key in sorted(finished):
                historical(target, key)
            opened.add(target)
        elif event["action"] == "finish":
            opened.remove(target)
            finished.add(target)
            # In an editable snapshot some initial forecasts can run before cell judgment.
            for key in sorted(opened):
                historical(key, target)

    addable = {row["id"] for row in history["holdouts"]}
    for target in sorted(opened | addable):
        for key in sorted(finished | opened | addable):
            # A user can add two copies of the same stock task. Their runtime IDs differ.
            if target != key or key in addable:
                historical(target, key)
        for other in sorted(addable):
            pair(target, other, active=True)
    return requests, matrix, historical


def package_verdicts(stage, requests, *, generate, stats):
    path = VECTOR_INDEX if stage == "pairs" else MODELS / f"{stage}.json"
    bundle = (retrieval_bundle() if stage == "pairs" else
              load(path, {"format": 1, "model": retrieval.DECISION_MODEL, "decisions": {}}))
    config = {**retrieval.planner_settings(), **CONFIG}
    for key, request in sorted(requests.items()):
        stats["required"] += 1
        stored = bundle["decisions"].get(key)
        if stored is not None:
            verdict = stored.get("related") if stage == "pairs" and isinstance(stored, dict) else stored
            if type(verdict) is not bool:
                raise ValueError(f"Invalid packaged {stage} verdict for {key}")
            if generate:
                continue
        if stage == "pairs":
            hit = cache.decision_for(request)
        elif stage == "sites":
            hit = focus_judge.packaged(request)
        else:
            hit = workspace.packaged(key)
        if stored is not None and hit is not verdict:
            raise ValueError(f"The runtime did not resolve the stored {stage} verdict for {key}")
        if hit is not None:
            continue
        stats["missing"] += 1
        if not generate:
            continue
        live = {"pairs": retrieval._live_decision, "sites": focus_judge.live_decision, "workspaces": workspace.live}[stage]
        result = live(request, config)
        bundle["decisions"][key] = {"related": result} if stage == "pairs" else result
        write_json(path, bundle, sort_keys=True)
        print(f"Kev {stage} {stats['missing']} {key[:12]}", flush=True)
    cache._indexes.cache_clear()
    focus_judge._decisions.cache_clear()
    workspace._decisions.cache_clear()


def desk_requests(history):
    result = {}
    for group in ("tasks", "open_tasks", "holdouts"):
        for row in history[group]:
            request = workspace.payload({**row, "attachments": bool(row.get("pdf"))}, CONFIG)
            result[cache.decision_hash(request)] = request
    return result


def site_requests(history, tasks, matrix, historical, pairs):
    name = history["id"]
    requests, ids, active, completed = {}, {}, set(), set()

    def accepted(target, source, matrix_first):
        digest = matrix.get(frozenset((target, source))) if matrix_first else None
        if digest is None:
            digest = historical(target, source)
        if digest is None:
            return False
        value = cache.decision_for(pairs[digest])
        if value is None:
            raise ValueError("Complete the pairs stage before building website requests.")
        return value

    def inspect(target, candidates):
        if tasks[target].get("fixed_duration"):
            return {}
        sources = [ids[source] for source in sorted(candidates)]
        checks = focus_domains.site_requests(store, ids[target], sources, pdf_text=tasks[target]["pdf_text"], config=CONFIG)
        for request in checks.values():
            requests[focus_judge.key(request)] = request
        return checks

    def automatic(target):
        for prefer_matrix in (True, False):
            inspect(target, [source for source in completed
                             if not tasks[source].get("fixed_duration") and accepted(target, source, prefer_matrix)])

    with TemporaryDirectory() as tmp:
        store = PlannerStore(Path(tmp))
        now = 1768489200
        dev_schedule.initialize(store, name, DEMO_DATASET, replay=True, now=now)
        _, events = dev_schedule._inputs(name, DEMO_DATASET)
        calendar = store.get_setting("dev_calendar")
        rows, agenda, sessions, _ = dev_schedule._agenda(history, events, calendar)
        for item in agenda:
            target = item["task"]
            dev_schedule._apply(store, history, rows[target], item, calendar, sessions, DEMO_DATASET, infer=False)
            if item["action"] == "create":
                ids[target] = store.get_setting(f"dev_task_{target}")
                active.add(target)
                automatic(target)
            elif item["action"] == "finish":
                completed.add(target)
                active.remove(target)
                for other in sorted(active):
                    automatic(other)
        for row in history["holdouts"]:
            key = row["id"]
            ids[key] = dev_workspace.add_sample_task(store, name, key, calendar["day"])["id"]
            active.add(key)
        for target in sorted(active):
            automatic(target)
            # A manual edit can combine any completed sources. Each site prompt sees
            # only its own visits, so enumerate those subsets rather than entire histories.
            by_domain = {}
            for source in sorted(completed):
                for domain in inspect(target, [source]):
                    by_domain.setdefault(domain, []).append(source)
            for group in by_domain.values():
                for size in range(2, len(group) + 1):
                    for subset in combinations(group, size):
                        inspect(target, subset)
    return requests


def run_stage(stage, selected, generate):
    stats = Counter()
    for history in selected:
        name = history["id"]
        tasks = tasks_for(history)
        if stage == "vectors":
            vectors(tasks, generate=generate, stats=stats)
        elif stage == "workspaces":
            package_verdicts(stage, desk_requests(history), generate=generate, stats=stats)
        else:
            available = vectors(tasks, generate=False, stats=Counter())
            if set(available) != set(tasks):
                raise ValueError("Complete the vectors stage before resolving pairs or sites.")
            requests, matrix, historical = schedule_pairs(history, tasks, available)
            if stage == "sites":
                requests = site_requests(history, tasks, matrix, historical, requests)
            package_verdicts(stage, requests, generate=generate, stats=stats)
        print(f"{name} {stage}: {dict(stats)} cumulative", flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("vectors", "pairs", "workspaces", "sites", "verify"))
    parser.add_argument("--history", help="One demo profile ID or prefix")
    parser.add_argument("--count-only", action="store_true", help="Count required and missing requests without inference.")
    args = parser.parse_args()
    selected = histories(args.history, DEMO_DATASET)
    stages = ("vectors", "pairs", "workspaces", "sites") if args.stage == "verify" else (args.stage,)
    for stage in stages:
        generate = args.stage != "verify" and not args.count_only
        lock = nullcontext()
        if generate:
            MODELS.mkdir(exist_ok=True, parents=True)
            output = VECTOR_INDEX if stage in ("pairs", "vectors") else MODELS / f"{stage}.json"
            lock_path = Path(gettempdir()) / f"plan-smarter-package-{cache.text_hash(str(output.resolve()))}.lock"
            lock = portalocker.Lock(lock_path, timeout=0)
        with lock:
            stats = run_stage(stage, selected, generate=generate)
        if args.stage == "verify" and stats["missing"]:
            raise SystemExit(f"Missing {stats['missing']} {stage} results.")


if __name__ == "__main__":
    main()
