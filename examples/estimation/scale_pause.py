"""Run or score independent workspace-pause conversations against the configured production forecaster."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from contextvars import ContextVar
import hashlib
import json
import math
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

import model_client
import workspace_forecast as forecast
from examples.estimation.audit_activity_progress import binary
from examples.estimation.forecast_audit import ObservedClient
from model_client import ModelContextExceeded, ModelResponseInvalid, ModelUnavailable, chat_json, model_settings


ROOT = Path(__file__).resolve().parents[2] / "data/results/component-scale/pause"
MODELS = {"4b": "ternary-bonsai-4b", "9b": "ternary-bonsai-2-9b"}
PROMPT_HASH = "756d8805e6fdd2763a54f9addfbec5943b4e7d1ef86f70f02a3e5c127d9da979"
CASE_HASH = "65affbaf966b52f81cf462c8934798a51fc39fd8347ce2b24ff106243af0b633"
REFERENCE_HASH = "1cb4428ec37d755d27157f081109d5527e53770559317f283a5f5eddb70b6187"
calls = ContextVar("pause_audit_calls")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(name, digest=None):
    data = (ROOT / name).read_bytes()
    if digest and sha(data) != digest:
        raise ValueError(f"Different frozen {name}")
    return json.loads(data)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def traced(messages, schema, **kwargs):
    attempt = {"request_sha256": sha(json.dumps([messages, schema, kwargs], sort_keys=True).encode())}
    calls.get().append(attempt)
    started = time.monotonic()
    try:
        result = chat_json(messages, schema, **kwargs)
    except ModelUnavailable as exc:
        cause = exc.__cause__
        if isinstance(exc, ModelContextExceeded):
            kind = "context"
        elif isinstance(exc, ModelResponseInvalid):
            kind = "model response"
        elif isinstance(cause, HTTPError):
            kind = "http"
            attempt["http_status"] = cause.code
        elif isinstance(cause, (OSError, URLError)):
            kind = "timeout" if isinstance(cause, TimeoutError) or isinstance(getattr(cause, "reason", None), TimeoutError) else "network"
        else:
            kind = "configuration"
        attempt.update(error_type=type(exc).__name__, error=str(exc), failure=kind)
        raise
    finally:
        attempt["seconds"] = round(time.monotonic() - started, 3)
    if attempt["response_model"] != model_settings()[1]:
        raise RuntimeError(f"Unexpected responding checkpoint: {attempt['response_model']}")
    attempt.pop("content", None)
    attempt["reply"] = result
    return result


def assess_case(record, model, output, retry_transport):
    path = output / (record["id"] + ".json")
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["model"] != model or existing["prompt_sha256"] != PROMPT_HASH or existing["input_sha256"] != CASE_HASH:
            raise ValueError(f"Incompatible receipt: {path}")
        if not (retry_transport and existing["transport_error"]):
            return
        # Retain the original transport failure as a receipt outside the scored run.
        old = output.parent / (output.name + "-transport") / path.name
        save(old, existing)
    task = record["task"]
    minutes = task["estimate_minutes"]
    baseline = {"point_min": minutes, "low_min": max(1, round(minutes * 0.7)), "high_min": math.ceil(minutes * 1.5),
                "explanation": "Initial estimate for completing the stated scope."}
    elapsed = record["elapsed_minutes"]
    fresh = record["new_message_start"]
    if fresh:
        prior_elapsed = max(1, elapsed - 5)
        baseline.update(basis="Workspace progress forecast", workspace_update={
            "status": "adjusted", "elapsed_minutes": prior_elapsed, "remaining_min": max(0, minutes - prior_elapsed),
            "evidence": [{"kind": "completed", "quote": record["messages"][0]["content"], "document": ""}]})
    snapshot = {"paused_at": 1842000000 + elapsed * 60, "elapsed_seconds": elapsed * 60,
                "new_message_start": fresh, "messages": record["messages"]}
    token = calls.set([])
    try:
        result = forecast.revise(task, baseline, snapshot)
        trace = calls.get()
    finally:
        calls.reset(token)
    update = result["workspace_update"]
    provenance = update["provenance"]
    assert provenance["model"] == model and len(trace) in (1, 2), (provenance, trace)
    transport = any(c.get("failure") in {"timeout", "network", "http"} for c in trace)
    save(path, {"id": record["id"], "model": model, "endpoint": model_settings()[0], "prompt_sha256": PROMPT_HASH, "input_sha256": CASE_HASH,
                "status": update["status"], "transport_error": transport, "attempts": trace,
                "initial_minutes": [baseline[k] for k in ("low_min", "point_min", "high_min")],
                "final_minutes": [result[k] for k in ("low_min", "point_min", "high_min")], "update": update})
    print(record["id"], update["status"], flush=True)


def run(short, workers, retry_transport):
    endpoint, model = model_settings()
    if model != MODELS[short] or urlsplit(endpoint).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(f"Expected local endpoint for {MODELS[short]}, got {model} at {urlsplit(endpoint).hostname}")
    if sha(forecast._INSTRUCTIONS.encode()) != PROMPT_HASH:
        raise ValueError("This benchmark expects the published self-report prompt.")
    if not 1 <= workers <= 3:
        raise ValueError("Use between one and three workers per endpoint.")
    records = read("conversations.json", CASE_HASH)
    build_opener = model_client.build_opener
    # The observer uses the current thread's request. Installing it once avoids
    # swapping global HTTP handlers mid-request across parallel conversations.
    model_client.build_opener = lambda *handlers: ObservedClient(build_opener(*handlers), calls.get()[-1], endpoint)
    forecast.chat_json = traced
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda c: assess_case(c, model, ROOT / "runs" / short, retry_transport), records))
    finally:
        forecast.chat_json = chat_json
        model_client.build_opener = build_opener


def three_way(supported, finished):
    return "finished" if supported and finished else "open" if supported else "keep"


def score(short):
    records = read("conversations.json", CASE_HASH)
    labels = read("reference.json", REFERENCE_HASH)
    rows = []
    for case in records:
        receipt = read(Path("runs") / short / (case["id"] + ".json"))
        if (receipt["model"], receipt["id"], receipt["prompt_sha256"], receipt["input_sha256"]) != (
                MODELS[short], case["id"], PROMPT_HASH, CASE_HASH):
            raise ValueError(f"Unexpected receipt identity for {short} / {case['id']}")
        for attempt in receipt["attempts"]:
            if "reply" in attempt and attempt.get("response_model") != MODELS[short]:
                raise ValueError(f"Wrong responding model for {case['id']}")
        expected = labels[case["id"]]
        update, status = receipt["update"], receipt["status"]
        adjusted = status == "adjusted"
        fresh = [m["content"] for m in case["messages"][case["new_message_start"]:] if m["role"] == "user"]
        quotes = [e["quote"] for e in update["evidence"]]
        grounded = any(q in turn and any(a.casefold() in q.casefold() for a in expected["evidence_anchors"])
                       for q in quotes for turn in fresh)
        times = receipt["final_minutes"]
        remaining = update["remaining_min"]
        elapsed = case["elapsed_minutes"]
        stated = expected["stated_remaining_min"]
        initial = receipt["attempts"][0].get("reply", {}).get("assessment")
        invariants = {
            "ordered_range": times[0] >= 0 and times == sorted(times),
            "retained_range_if_no_update": adjusted or times == receipt["initial_minutes"],
            "evidence_for_adjustment": not adjusted or (all(any(q in turn["content"] for turn in case["messages"]
                if turn["role"] == "user") for q in quotes) and any(q in turn for q in quotes for turn in fresh)),
            "elapsed_plus_remaining": not adjusted or (remaining is not None and remaining + elapsed == times[1]),
        }
        zero_ok = (remaining == 0 if expected["outcome"] == "complete" else remaining > 0) if adjusted else None
        numeric_range = times[0] - elapsed <= stated <= times[2] - elapsed if stated is not None and adjusted else None
        rows.append({"id": case["id"], "family": case["family"], "gold": expected["outcome"],
                     "expected": expected["expected_supported"], "adjusted": adjusted, "status": status,
                     "attempts": len(receipt["attempts"]), "initial_reply": initial,
                     "transport_error": receipt["transport_error"], "grounded_anchor": grounded if adjusted else None,
                     "zero_or_positive_as_expected": zero_ok, "stated_remaining_in_range": numeric_range,
                     "stated_remaining": stated, "remaining": remaining, "invariants": invariants,
                     "reference_state": three_way(expected["expected_supported"], expected["outcome"] == "complete"),
                     "final_state": three_way(adjusted, remaining == 0)})
    def confusion(rows):
        return binary([r["expected"] for r in rows], [r["adjusted"] for r in rows])
    output = {"model": MODELS[short], "reference_sha256": REFERENCE_HASH, "input_sha256": CASE_HASH,
              "count": len(rows), "decision": confusion(rows),
              "status_counts": dict(Counter(r["status"] for r in rows)),
              "attempt_counts": dict(sorted(Counter(r["attempts"] for r in rows).items())),
              "initial_decision": binary(
                  [r["expected"] for r in rows if r["initial_reply"]],
                  [r["initial_reply"] == "supported" for r in rows if r["initial_reply"]]),
              "initial_unparseable": sum(r["initial_reply"] is None for r in rows),
              "final_states": {state: dict(Counter(r["final_state"] for r in rows if r["reference_state"] == state))
                               for state in ("keep", "open", "finished")},
              "invariant_failures": [r["id"] for r in rows if not all(r["invariants"].values())],
              "transport_errors": [r["id"] for r in rows if r["transport_error"]],
              "by_family": {f: confusion([r for r in rows if r["family"] == f]) for f in sorted({r["family"] for r in rows})},
              "by_label": {l: {"n": len(group), "updated": sum(r["adjusted"] for r in group)}
                           for l in sorted({r["gold"] for r in rows}) if (group := [r for r in rows if r["gold"] == l])},
              "cases": rows}
    save(ROOT / (short + "-scores.json"), output)
    render_report()
    print(model_summary(output))


def pct(number, denominator):
    return f"{100 * number / denominator:.1f}%" if denominator else "—"


def render_report():
    if not all((ROOT / f"{m}-scores.json").exists() for m in MODELS):
        return
    scores = {m: read(f"{m}-scores.json") for m in MODELS}
    lines = ["# Paired workspace-pause results", "", "Each checkpoint received the same 48 frozen conversations. "
             "The positive class is a supported forecast revision. TP, FP, FN and TN refer to decisions "
             "returned to the application.", "",
             "| Model | TP | FP | FN | TN | Precision | Recall | Specificity | Three-way correct |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m, score in scores.items():
        d = score["decision"]
        tp, fp, fn, tn = (d[k] for k in ("TP", "FP", "FN", "TN"))
        correct = sum(r["reference_state"] == r["final_state"] for r in score["cases"])
        lines.append(f"| {m.upper()} | {tp} | {fp} | {fn} | {tn} | {pct(tp,tp+fp)} | "
                     f"{pct(tp,tp+fn)} | {pct(tn,tn+fp)} | {correct}/48 |")
    paired = Counter()
    for left, right in zip(scores["4b"]["cases"], scores["9b"]["cases"], strict=True):
        assert left["id"] == right["id"]
        paired[(left["reference_state"] == left["final_state"], right["reference_state"] == right["final_state"])] += 1
    lines.extend(["", f"On the three-way decision, both models were correct on {paired[True, True]} cases. "
                  f"Only 4B was correct on {paired[True, False]}, only 9B on {paired[False, True]}, and both "
                  f"missed {paired[False, False]}."])
    lines += ["", "## Model decisions and validation", "",
              "| Model | First-reply TP/FP/FN/TN | First replies without valid JSON | Cases retried | "
              "Direct abstentions | Unavailable final assessments | Transport failures |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for m, score in scores.items():
        first = "/".join(str(score["initial_decision"][k]) for k in ("TP", "FP", "FN", "TN"))
        status = score["status_counts"]
        retries = sum(r["attempts"] > 1 for r in score["cases"])
        lines.append(f"| {m.upper()} | {first} | {score['initial_unparseable']} | {retries} | "
                     f"{status.get('insufficient_evidence',0)} | {status.get('unavailable',0)} | "
                     f"{len(score['transport_errors'])} |")
    lines += ["", "The first reply precedes semantic validation. The final decision can differ if a citation "
              "is stale, a quote cannot be verified, the explanation is missing, or the minutes contradict the "
              "claimed progress. Each unavailable receipt includes the app's reason.", "",
              "## When work was still open", "",
              "| Model | Gold state | Kept forecast | Updated with work left | Updated to zero remaining |",
              "| --- | --- | ---: | ---: | ---: |"]
    for m, score in scores.items():
        for state, counts in score["final_states"].items():
            lines.append(f"| {m.upper()} | {state} | {counts.get('keep',0)} | {counts.get('open',0)} | "
                         f"{counts.get('finished',0)} |")
    lines += ["", "## Evidence and remaining time", "",
              "| Model | Accepted updates with a fresh cite containing a reference phrase | "
              "Supported conversations with correct open/finished state and a reference phrase | "
              "Stated remaining time covered by model interval | Invariant failures |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for m, score in scores.items():
        rows = score["cases"]
        anchors = sum(r["grounded_anchor"] is True for r in rows)
        accepted = sum(r["adjusted"] for r in rows)
        supported = sum(r["grounded_anchor"] is True and r["reference_state"] == r["final_state"] for r in rows)
        covered = sum(r["stated_remaining_in_range"] is True for r in rows)
        lines.append(f"| {m.upper()} | {anchors}/{accepted} | {supported}/24 | {covered}/3 | "
                     f"{len(score['invariant_failures'])}/48 |")
    lines += ["", "## Variation across task families", "",
              "Each family contains three positive and three negative decisions. Counts below are "
              "TP / FP / FN / TN. Family differences describe these particular tasks.", "",
              "| Family | 4B | 9B |", "| --- | --- | --- |"]
    for family in scores["4b"]["by_family"]:
        cells = ["/".join(str(scores[m]["by_family"][family][k]) for k in ("TP","FP","FN","TN")) for m in MODELS]
        lines.append(f"| {family} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Individual errors", "",
              "These rows include every failed three-way decision, every accepted update missing a frozen "
              "evidence phrase and every missed explicit duration. The evidence-phrase test is deliberately "
              "literal. A shorter valid quote may warrant manual review.", "",
              "| Case and receipt | Model | Expected | App decision | Additional check |",
              "| --- | --- | --- | --- | --- |"]
    count = 0
    for m, score in scores.items():
        for r in score["cases"]:
            extra = []
            if r["grounded_anchor"] is False:
                extra.append("missing reference phrase")
            if r["stated_remaining"] is not None and r["stated_remaining_in_range"] is not True:
                extra.append(f"missed stated {r['stated_remaining']} minutes")
            if r["reference_state"] == r["final_state"] and not extra:
                continue
            count += 1
            app = r["final_state"] if r["adjusted"] else r["status"].replace("_"," ")
            lines.append(f"| [{r['id']}](runs/{m}/{r['id']}.json) | {m.upper()} | {r['reference_state']} | "
                         f"{app} | {', '.join(extra) or '—'} |")
    if not count:
        lines.append("| All cases passed these checks | Both | — | — | — |")
    lines += ["", "The [protocol](README.md) describes the frozen inputs, the blind review, thresholds "
              "and limitations. Both the per-case receipts and machine-readable score sheets are in this directory.", ""]
    (ROOT / "REPORT.md").write_text("\n".join(lines))


def model_summary(output):
    return f"{output['model']} {output['count']} cases: {output['decision']}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "score"])
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retry-transport", action="store_true")
    args = parser.parse_args()
    if args.action == "run":
        run(args.model, args.workers, args.retry_transport)
    else:
        score(args.model)
