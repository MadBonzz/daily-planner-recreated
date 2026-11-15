"""Score published or fresh forecast receipts against task-level behavioral labels."""
from collections import Counter
import json
import re
from statistics import median

from examples.estimation.forecast_audit import OUT, REFERENCE, challenge


# Diagnostic terminology proxies written against the published warning rubrics.
# A source citation alone cannot establish that the advice addresses the obstacle.
WARNING_TOPIC = {
    "mara-rooms": r"late|delay|wait|punctual|arrival|join.{0,12}time",
    "bump-crossing": r"disconnect|reconnect|usb|bridge",
    "triangle-video": r"recommend|unrelated|distract|autoplay|sidebar",
    "bridge-proof": r"failure schedule|separate.{0,15}(schedule|trace)|counterexample",
    "patch-tool": r"minimi[sz]|reduc.{0,18}(fixture|regression)|small.{0,15}(fixture|reproducer)",
    "dark-frame-series": r"pair|reconnect|connect|link|USB",
}


def decision(counter, actual, predicted):
    counter[("t" if actual == predicted else "f") + ("p" if predicted else "n")] += 1


def binary(counts):
    def rate(num, den):
        return round(num / den, 3) if den else None
    tp, fp, tn, fn = [counts[k] for k in ("tp", "fp", "tn", "fn")]
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": rate(tp, tp + fp),
            "recall": rate(tp, tp + fn)}


def published_baseline(model):
    for path in sorted((REFERENCE / model / "cases").glob("*.json")):
        row = json.loads(path.read_text())
        ev = row["evaluation"]
        band = ev["expected_ratio_range"]
        yield {"task": row["task"], "fixed": row["fixed_duration"], "budget": row["original"],
               "receipts": row.get("http_receipts", []),
               "frozen_candidates": [c["task_id"] for c in row["related_tasks"]],
               "rubric": {"direction": ("up" if band[0] > 1 else "down" if band[1] < 1 else "near") if band else None,
                          "expected_sources": ev.get("expected_caution", {}).get("evidence_ids"),
                          "expected_evidence": ev["evidence_ids"], "expected_band": band},
               "prediction": {"point_min": row["planner"], "low_min": row["range"][0], "high_min": row["range"][1],
                              "confidence": row["confidence"], "evidence": row["evidence"], "cautions": row["cautions"],
                              "provenance": row["provenance"]}}


def score(rows):
    dirs = {label: Counter() for label in ("up", "near", "down")}
    evidence, warning, topical_warning, warning_detection = Counter(), Counter(), Counter(), Counter()
    summary = Counter()
    by_confidence = {name: {"labeled": 0, "correct_direction": 0} for name in ("low", "medium", "high")}
    response_seconds, task_seconds = [], []
    retries, truncated = 0, 0
    details = {}
    for row in rows:
        name, rule, p = row["task"], row["rubric"], row["prediction"]
        calls = row.get("receipts", [])
        if calls:
            seconds = [c["seconds"] for c in calls]
            response_seconds.extend(seconds)
            task_seconds.append(sum(seconds))
            retries += len(calls) - 1
            truncated += sum(c.get("finish_reason") == "length" for c in calls)
        summary["total"] += 1
        direct = p["provenance"]["decision"] == "direct"
        summary["direct"] += direct
        if row["fixed"]:
            summary["fixed_without_call"] += p["provenance"]["decision"] == "fixed" and not row.get("receipts")
            continue
        if not direct:
            summary["fallback"] += 1
            continue
        band = rule.get("expected_band")
        ratio = p["point_min"] / row["budget"]
        direction = "up" if ratio > 1.05 else "down" if ratio < 0.95 else "near"
        if rule["direction"]:
            level = by_confidence[p["confidence"]]
            level["labeled"] += 1
            level["correct_direction"] += direction == rule["direction"]
            for label, table in dirs.items():
                decision(table, rule["direction"] == label, direction == label)
        if band:
            summary["labeled_bands"] += 1
            summary["inside_rubric_band"] += band[0] <= ratio <= band[1]
        candidates = set(row["frozen_candidates"])
        cited = {e["task_id"] for e in p["evidence"]}
        expected = set(rule["expected_evidence"])
        for task_id in candidates:
            decision(evidence, task_id in expected, task_id in cited)
        cautions = {s["task_id"] for c in p["cautions"] for s in c["sources"]}
        sources = rule["expected_sources"]
        if sources is not None:
            # The published rubric specifies acceptable alternative sources for one
            # warning. Absence of a warning label in the original rubric is ungraded.
            if sources and not candidates.intersection(sources):
                summary["warning_source_unavailable"] += 1
            else:
                decision(warning, bool(sources), bool(set(sources) & cautions) if sources else bool(cautions))
                pattern = WARNING_TOPIC.get(name)
                hits = [bool(set(sources).intersection(s["task_id"] for s in c["sources"])
                             and pattern and re.search(pattern, c["advice"], re.I)) for c in p["cautions"]]
                topical = any(hits)
                decision(topical_warning, bool(sources), topical if sources else bool(cautions))
                if rule.get("exhaustive_warning_labels"):
                    # Count each expected obstacle once. A missed obstacle and
                    # an unrelated warning on that same task are both errors.
                    warning_detection["tp"] += topical
                    warning_detection["fn"] += bool(sources) and not topical
                    warning_detection["fp"] += False in hits
                    warning_detection["tn"] += not sources and not hits
                if sources and cautions and not set(sources) & cautions:
                    summary["warnings_citing_only_wrong_source"] += 1
        summary["valid_interval"] += 0 < p["low_min"] <= p["point_min"] <= p["high_min"] and p["low_min"] < p["high_min"]
        if rule.get("low_confidence"):
            summary["cold_low_confidence"] += p.get("confidence") == "low"
            summary["cold_underspecified"] += 1
            description = p.get("explanation", "") + " " + p.get("confidence_reason", "")
            summary["explicit_no_timing"] += bool(re.search(
                r"(no|lack|absence|unavailable|without).{0,48}(compar|histor|prior|previous|past|timing)|"
                r"(compar|histor|timing).{0,30}(unavailable|absent)", description, re.I))
        details[name] = {"budget": row["budget"], "forecast": p["point_min"], "expected_direction": rule["direction"],
                         "direction": direction, "cited": sorted(cited), "warnings": sorted(cautions),
                         "interval_width_min": p["high_min"] - p["low_min"], "confidence": p.get("confidence")}
    cf = {}
    slow, fast = (details.get("same-indexing-" + label) for label in ("slower", "faster"))
    if slow and fast:
        cf = {"slower_history_point": slow["forecast"], "faster_history_point": fast["forecast"],
              "order_correct": slow["forecast"] > fast["forecast"],
              "both_directions_correct": slow["direction"] == "up" and fast["direction"] == "down"}
    spread = details.get("same-indexing-variable")
    if slow and spread:
        confidence = {"low": 0, "medium": 1, "high": 2}
        cf.update(tight_width=slow["interval_width_min"], variable_width=spread["interval_width_min"],
                  wider_for_variable=spread["interval_width_min"] > slow["interval_width_min"],
                  tight_confidence=slow["confidence"], variable_confidence=spread["confidence"],
                  lower_confidence_for_variable=confidence[spread["confidence"]] < confidence[slow["confidence"]])
    timing = {"model_calls": len(response_seconds), "retries": retries, "length_truncations": truncated,
              "median_call_seconds": round(median(response_seconds), 2) if response_seconds else None,
              "median_model_seconds_per_task": round(median(task_seconds), 2) if task_seconds else None,
              "calls_over_default_90_seconds": sum(t > 90 for t in response_seconds)}
    return {"counts": dict(summary), "matched_counterfactual": cf, "direction_by_confidence": by_confidence,
            "wall_clock_including_queue": timing, "direction_one_vs_rest": {k: binary(v) for k, v in dirs.items()},
            "timing_citations_given_candidates": binary(evidence),
            "warning_cases_given_reachable_source_or_explicit_negative": binary(warning),
            "warning_topic_and_source_cases": binary(topical_warning),
            "warning_detection_exhaustively_labeled": binary(warning_detection), "cases": details}


def main():
    result = {}
    fresh_ids = {c[0]["id"] for c in challenge()}
    for short in ("bonsai-4b", "bonsai-9b"):
        old = list(published_baseline(short))
        result[short] = {"original_full": score(old)}
        for group in sorted(p for p in OUT.iterdir() if p.is_dir()):
            rows = [json.loads(p.read_text()) for p in sorted((group / short).glob("*.json"))]
            if not rows:
                continue
            existing = [r for r in rows if r["task"] not in fresh_ids]
            names = {r["task"] for r in existing}
            result[short][group.name] = {
                "paired_original": score([r for r in old if r["task"] in names]),
                "replayed": score(existing), "fresh": score([r for r in rows if r["task"] in fresh_ids])}
    (OUT / "scores.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
