"""Audit forecast behavior with frozen retrieval and independent labeled cases.

python -m examples.estimation.forecast_audit --model ternary-bonsai-4b --endpoint URL --run revised TASK ...
Use "published" to replay every original task or "challenge" for all fresh cases.
The two input sets use the app's predictor and HTTP client. Only retrieval is frozen.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

import estimator
import model_client
from examples.estimation import run as fixtures
from examples.estimation.bonsai_holdouts import ObservedClient

OUT = fixtures.REPO / "data/results/component-audit/forecast"
REFERENCE = fixtures.REPO / "data/results/holdouts/2026-11-15-bonsai-direct-050"
MODELS = {"ternary-bonsai-4b": "bonsai-4b", "ternary-bonsai-2-9b": "bonsai-9b"}
ESTIMATOR_SHA = hashlib.sha256(Path(estimator.__file__).read_bytes()).hexdigest()


def scenario(name, description, budget, previous, *, direction=None, warnings=(), fixed=False,
             expect_low_confidence=False, evidence=None):
    target = {"id": name, "title": name.replace("-", " ").capitalize(), "description": description,
              "category": "Personal projects", "details": "", "estimate_minutes": budget, "fixed_duration": fixed}
    rows = []
    for i, (work, estimate, actual, note) in enumerate(previous, 1):
        rows.append({**target, "id": f"prior-{i}", "title": work, "description": work,
                     "estimate_minutes": estimate, "actual_seconds": actual * 60, "completion_note": note})
    rubric = {"direction": direction, "expected_sources": list(warnings),
              "expected_evidence": [r["id"] for r in rows] if evidence is None else evidence,
              "low_confidence": expect_low_confidence, "exhaustive_warning_labels": True}
    # This component can receive a user-edited comparison set. Force that set
    # deliberately in the independent scenarios without inventing cosine scores.
    target["relevance_overrides"] = {r["id"]: True for r in rows}
    return target, rows, rubric


def challenge():
    # Identical model-visible inputs in both arms except historical durations
    # and their derived arithmetic. Target IDs and private labels stay outside
    # the request body so neither arm is identifiable by a label.
    def counterfactual(name, actuals, expected):
        target, history, rubric = scenario(name,
            "Run the weekly indexing check on the same 20 images: load the manifest, index all images, "
            "compare checksums and record the result. The same workstation and script are available.", 40,
            [(title, 40, actual, "") for title, actual in zip(
                ["Run the Monday indexing and checksum check on 20 images",
                 "Run the Tuesday indexing and checksum check on 20 images",
                 "Run the Wednesday indexing and checksum check on 20 images"], actuals, strict=True)],
            direction=expected)
        target["title"] = "Run the weekly indexing check on 20 images"
        return target, history, rubric

    return [
        scenario("dark-frame-series", "Collect the next week's 20 dark frames with the same camera and cooling "
                 "controller. Use last week's acquisition and export procedure.", 40, [
                     ("Collect and export 20 dark frames on this camera", 40, 59,
                      "Cooling controller lost its USB pairing twice; re-pairing delayed capture."),
                     ("Acquire the weekly batch of 20 dark frames", 40, 60,
                      "Had to re-pair the cooling controller before the camera stabilized.")],
                 direction="up", warnings=["prior-1", "prior-2"]),
        scenario("shelve-next-cart", "Shelve one standard 40-book returns cart using the same shelf order and scanner.", 40, [
                     ("Scan and shelve a standard 40-book returns cart", 40, 26, "All books shelved successfully."),
                     ("Scan and shelve another 40-book returns cart", 40, 27, "Completed the cart as planned.")],
                 direction="down"),
        scenario("export-only", "Export the existing approved two-slide deck. The copyedit, layout and client review "
                 "are already complete. This is a single PDF export in the same editor, usually around 5 minutes.", 20, [
                     ("Copyedit, lay out, review and export the two-slide deck. Export takes 5 minutes.",
                      20, 39, "Client arrived late to the review.")], direction="down"),
        scenario("first-cabinet-check", "Figure out what is causing the noise in the cabinet. I haven't examined it yet.", 30,
                 [], expect_low_confidence=True),
        scenario("mail-registration-link", "Email the approved registration link using the existing 12-address "
                 "mailing list. Check delivery. No printing, copying or physical distribution is needed.", 15, [
                     ("Email the registration link to 12 members and check delivery", 15, 12, "All messages delivered."),
                     ("Print and distribute registration flyers around campus", 15, 52, "Printer jammed repeatedly.")],
                 direction="down", evidence=["prior-1"]),
        counterfactual("same-indexing-slower", (58, 60, 62), "up"),
        counterfactual("same-indexing-faster", (22, 24, 26), "down"),
        counterfactual("same-indexing-variable", (25, 60, 95), None),
        scenario("reserve-theatre-slot", "Attend the scheduled 50-minute screening.", 50, [], fixed=True),
    ]


@lru_cache(maxsize=None)
def past(history_id):
    return fixtures.historical_rows(fixtures.histories(history_id)[0], fixtures.DATASET)


def published():
    result = []
    for history in fixtures.histories(None):
        for task in history["holdouts"]:
            name = task["id"]
            saved = json.loads((REFERENCE / "bonsai-9b/cases" / f"{name}.json").read_text())
            selected = saved["related_tasks"]
            by_id = {r["id"]: r for r in past(history["id"])}
            target = {"id": name, **fixtures.task_inputs(task, history["id"], fixtures.DATASET)[0]}
            labels = task["evaluation"]
            band = labels["expected_ratio_range"]
            direction = ("up" if band[0] > 1 else "down" if band[1] < 1 else "near") if band else None
            rubric = {"direction": direction, "expected_band": band, "expected_evidence": labels["evidence_ids"],
                      "expected_sources": labels.get("expected_caution", {}).get("evidence_ids"),
                      "low_confidence": False}
            result.append((target, [dict(by_id[c["task_id"]], saved_cosine=c["score"]) for c in selected], rubric))
    return result


def evaluate(case, destination, model):
    task, rows, rubric = case
    path = destination / (task["id"] + ".json")
    case_digest = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
    if path.is_file():
        old = json.loads(path.read_text())
        if (old.get("case_sha256"), old.get("estimator_sha256"), old.get("model")) != (case_digest, ESTIMATOR_SHA, model):
            raise RuntimeError(f"Inputs or estimator changed since {path}; choose a fresh run name.")
        return
    receipts = []
    real_chat = model_client.chat_json

    def record(messages, schema, **options):
        payload = {"messages": messages, "schema": schema, "options": options, "model": model}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        receipt = {"request_sha256": digest, "max_tokens": options["max_tokens"]}
        receipts.append(receipt)
        start = time.monotonic()
        try:
            endpoint, _ = model_client.model_settings()
            with patch.object(model_client, "build_opener", lambda *h: ObservedClient(build_opener(*h), receipt, endpoint)):
                reply = real_chat(messages, schema, **options)
            if receipt.get("response_model") != model:
                raise RuntimeError(f"Unexpected responding model: {receipt.get('response_model')}")
            receipt.pop("content", None)
            receipt["response"] = reply
            return reply
        except model_client.ModelUnavailable as error:
            receipt["error"] = str(error)
            receipt["invalid_reply"] = isinstance(error, model_client.ModelResponseInvalid)
            raise
        finally:
            receipt["seconds"] = round(time.monotonic() - start, 2)

    with TemporaryDirectory() as tmp, patch.object(estimator, "chat_json", record), patch.object(
        estimator, "_search", return_value=[(r, r.get("saved_cosine")) for r in rows]), patch.object(estimator.retrieval, "decide_related",
                                                                                  return_value=True):
        prediction = estimator.predict(task, rows, Path(tmp))
    if receipts and all("error" in r and not r["invalid_reply"] for r in receipts):
        raise RuntimeError(f"No model result for {task['id']}: {receipts}")
    result = {"task": task["id"], "model": model, "time_utc": datetime.now(timezone.utc).isoformat(),
              "estimator_sha256": ESTIMATOR_SHA, "case_sha256": case_digest,
              "frozen_candidates": [r["id"] for r in rows], "budget": task["estimate_minutes"],
              "fixed": task.get("fixed_duration", False), "rubric": rubric, "receipts": receipts,
              "prediction": prediction}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    tmp_path.replace(path)
    print(task["id"], prediction["point_min"], prediction["provenance"]["decision"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("cases", nargs="+")
    args = parser.parse_args()
    if urlsplit(args.endpoint).hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("Use a loopback endpoint serving the requested local model.")
    os.environ.update(PLANNER_MODEL_ENDPOINT=args.endpoint, PLANNER_MODEL=args.model,
                      PLANNER_MODEL_TIMEOUT_SECONDS="3600")
    with build_opener(ProxyHandler({})).open(args.endpoint.removesuffix("/chat/completions") + "/models") as response:
        if args.model not in {record["id"] for record in json.load(response)["data"]}:
            parser.error("Endpoint does not serve requested model")
    groups = {"challenge": challenge(), "published": published()}
    pool = {case[0]["id"]: case for group in groups.values() for case in group}
    for name in args.cases:
        if name not in groups and name not in pool:
            parser.error(f"Unknown case {name}")
        for case in groups[name] if name in groups else [pool[name]]:
            evaluate(case, OUT / args.run / MODELS[args.model], args.model)


if __name__ == "__main__":
    main()
