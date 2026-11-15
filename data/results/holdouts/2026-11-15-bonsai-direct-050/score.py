"""Recompute the paired comparison from committed case records. No model or dependencies required."""

from collections import Counter
import json
from pathlib import Path
from statistics import mean, median


HERE = Path(__file__).resolve().parent
COHERE = HERE.parent / "2026-11-15-qwen-query-instruction"
ARMS = ("cohere", "bonsai-4b", "bonsai-9b")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases(directory):
    rows = [read(path) for path in directory.glob("*.json")]
    cases = {(r["history"], r["task"]): r for r in rows}
    if len(cases) != 20 or len(cases) != len(rows):
        raise ValueError(f"Expected 20 unique cases in {directory}")
    return cases


def route(row):
    if row["fixed_duration"] and row["model_status"] == "not_called":
        return "scheduled"
    if row["model_status"] in {"ok", "ok_after_retry"} and row["provenance"].get("minutes_by") == "model":
        return "model"
    if row["provenance"].get("decision") == "fallback":
        return "fallback"
    raise ValueError(f"Unrecognized forecast status on {row['task']}: {row['model_status']}")


def mae(rows, field="planner"):
    return round(mean(abs(r[field] - r["actual"]) for r in rows), 2) if rows else None


def quality(rows):
    warnings = [r for r in rows if r["evaluation"].get("expected_caution")]
    evidence = [r for r in rows if r["evaluation"]["evidence_ids"]]
    hits = sum(bool(r["evaluation"]["selected_expected_ids"]) for r in evidence)
    covered = sum(r["range"][0] <= r["actual"] <= r["range"][1] for r in rows)
    return {
        "cases": len(rows), "mae_minutes": mae(rows), "user_mae_minutes": mae(rows, "original"),
        "mean_signed_error_minutes": round(mean(r["planner"] - r["actual"] for r in rows), 2) if rows else None,
        "user_estimate_adjustment": {
            "increased": sum(r["planner"] > r["original"] for r in rows),
            "unchanged": sum(r["planner"] == r["original"] for r in rows),
            "decreased": sum(r["planner"] < r["original"] for r in rows),
        },
        "range_hits": covered, "mean_range_width_minutes": (
            round(mean(r["range"][1] - r["range"][0] for r in rows), 2) if rows else None),
        "expected_warning_hits": sum(bool(r["evaluation"]["caution_cites_expected_source"]) for r in warnings),
        "expected_warning_cases": len(warnings), "timing_evidence_hits": hits,
        "cases_with_expected_timing_evidence": len(evidence), "cases_with_warnings": sum(bool(r["cautions"]) for r in rows),
        "cases_with_unknown_warning_sources": sum(bool(r["evaluation"]["unknown_caution_source_ids"]) for r in rows),
        "routes": dict(Counter(route(r) for r in rows)),
    }


def paired(rows, comparison, field="planner"):
    delta = [abs(r["planner"] - r["actual"]) - abs(ref[field] - ref["actual"])
             for r, ref in zip(rows, comparison, strict=True)]
    return {"closer": sum(v < 0 for v in delta), "tied": delta.count(0), "further": sum(v > 0 for v in delta)}


def forecast_seconds(arm, row, hosted_receipts):
    if arm == "cohere":
        # Per-case wall times in the reference include replays of identical provider
        # requests. Look up the original observed HTTP attempt durations, including retries.
        return sum(a["seconds"] for call in row["forecast_calls"]
                   for a in hosted_receipts[call["request_id"]]["attempts"])
    return row["wall_model_seconds"]


def latency(rows, arm, hosted_receipts):
    seconds = [forecast_seconds(arm, r, hosted_receipts) for r in rows]
    return {"cases": len(seconds), "mean_seconds": round(mean(seconds), 3) if seconds else None,
            "median_seconds": round(median(seconds), 3) if seconds else None}


def compared_cases():
    cases = {arm: load_cases(COHERE / "instructed" if arm == "cohere" else HERE / arm / "cases") for arm in ARMS}
    for arm in ARMS[1:]:
        if cases[arm].keys() != cases["cohere"].keys():
            raise ValueError(f"{arm} case IDs differ from the Cohere reference")
        for key, row in cases[arm].items():
            ref = cases["cohere"][key]
            same = all(row[name] == ref[name] for name in ("actual", "original", "fixed_duration", "title"))
            selected = [x["task_id"] for x in row["related_tasks"]]
            expected = [x["task_id"] for x in ref["related_tasks"]]
            if not same or row["retrieval_error"] or selected != expected:
                raise ValueError(f"{arm} inputs or ordered selections differ for {key}")
    return cases


def score():
    cases = compared_cases()
    ids = sorted(cases["cohere"])
    cohorts = {"all": ids, "flexible": [k for k in ids if not cases["cohere"][k]["fixed_duration"]],
               "retrieval_validation": [k for k in ids if int(k[0].split("-")[0]) % 2 == 0],
               "model_forecasts_in_all_three": [k for k in ids if all(route(cases[a][k]) == "model" for a in ARMS)]}
    flexible = cohorts["flexible"]
    cohorts["with_selected_history"] = [k for k in flexible if cases["cohere"][k]["related_tasks"]]
    cohorts["without_selected_history"] = [k for k in flexible if not cases["cohere"][k]["related_tasks"]]
    output = {"cohorts": {}, "model_authored_or_fallback": {}, "forecast_latency": {}, "provider_attempts": {}}
    receipts = read(COHERE / "requests.json")
    for cohort, keys in cohorts.items():
        scored = {}
        for arm in ARMS:
            rows = [cases[arm][k] for k in keys]
            refs = [cases["cohere"][k] for k in keys]
            scored[arm] = {**quality(rows), "versus_user": paired(rows, refs, "original"),
                           "versus_cohere": paired(rows, refs)}
        output["cohorts"][cohort] = scored
    for arm in ARMS:
        out, times = {}, {}
        for status in ("model", "fallback"):
            subset = [k for k in ids if route(cases[arm][k]) == status]
            rows = [cases[arm][k] for k in subset]
            refs = [cases["cohere"][k] for k in subset]
            out[status] = {**quality(rows), "cohere_mae_minutes": mae(refs), "case_ids": [k[1] for k in subset]}
            times[status] = latency(rows, arm, receipts)
        times["all_attempted"] = latency([cases[arm][k] for k in cohorts["flexible"]], arm, receipts)
        output["model_authored_or_fallback"][arm] = out
        output["forecast_latency"][arm] = times
        attempted = [cases[arm][k] for k in flexible]
        if arm == "cohere":
            output["provider_attempts"][arm] = {
                "total": sum(len(receipts[call["request_id"]]["attempts"]) for r in attempted for call in r["forecast_calls"])
            }
        else:
            reasons = Counter(call.get("finish_reason") or "no response" for r in attempted for call in r["http_receipts"])
            output["provider_attempts"][arm] = {
                "total": sum(reasons.values()), "finish_reasons": dict(reasons),
                "cases_with_multiple_attempts": [r["task"] for r in attempted if len(r["http_receipts"]) > 1],
            }
    return output


if __name__ == "__main__":
    print(json.dumps(score(), indent=2, allow_nan=False))
