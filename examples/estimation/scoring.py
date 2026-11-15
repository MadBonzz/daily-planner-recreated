"""Grade predictions after inference against declared patterns and realized effort."""

from collections import Counter
from math import isfinite, sqrt
from statistics import mean


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True)) / sqrt(
        sum(x ** 2 for x in a) * sum(x ** 2 for x in b))


def pair_metrics(rows, passes):
    counts = Counter()
    for row in rows:
        accepted = passes(row)
        counts[(row["expected"], accepted)] += 1
    tp, fp, fn, tn = (counts[key] for key in ((True, True), (False, True), (True, False), (False, False)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
            "f2": 5 * tp / (5 * tp + 4 * fn + fp) if tp + fn + fp else 0}


def validate_rubric(rubric, past_ids):
    if not isinstance(rubric, dict) or rubric.get("kind") not in {"patterned", "control"}:
        raise ValueError("An evaluation needs a patterned or control kind")
    if not isinstance(rubric.get("rationale"), str) or not rubric["rationale"].strip():
        raise ValueError("An evaluation needs its private rationale")
    if "expected_ratio_range" not in rubric:
        raise ValueError("Declare an expected_ratio_range or null")
    band = rubric["expected_ratio_range"]
    if band is None and rubric["kind"] == "patterned":
        raise ValueError("A patterned case needs an expected_ratio_range")
    if band is not None and (
        not isinstance(band, list) or len(band) != 2
        or any(type(v) not in (int, float) or not isfinite(v) for v in band)
        or not 0 < band[0] <= band[1]
    ):
        raise ValueError("Invalid expected_ratio_range")
    items = [rubric]
    if "expected_caution" in rubric:
        caution = rubric["expected_caution"]
        if (not isinstance(caution, dict) or not isinstance(caution.get("rationale"), str)
                or not caution["rationale"].strip() or not caution.get("evidence_ids")):
            raise ValueError("A declared caution needs rationale and historical evidence IDs")
        items.append(caution)
    for item in items:
        ids = item.get("evidence_ids")
        if not isinstance(ids, list) or any(not isinstance(i, str) or i not in past_ids for i in ids):
            raise ValueError("Evaluation evidence_ids must refer to this person's earlier tasks")
        if len(set(ids)) != len(ids):
            raise ValueError("Evaluation evidence IDs must be unique")


def grade(rubric, prediction, user_minutes, actual_minutes, past_ids):
    """Review-only values never enter retrieval, the model prompt, or the store."""
    band = rubric["expected_ratio_range"]
    point_ratio = prediction["point_min"] / user_minutes
    actual_ratio = actual_minutes / user_minutes
    selected = {e["task_id"] for e in prediction.get("evidence", [])}
    sources = {source["task_id"] for item in prediction.get("cautions", []) for source in item.get("sources", [])}
    expected = rubric.get("expected_caution")
    return {
        **rubric,
        "predicted_ratio": point_ratio,
        "realized_ratio": actual_ratio,
        "prediction_in_band": band[0] <= point_ratio <= band[1] if band else None,
        "realization_in_band": band[0] <= actual_ratio <= band[1] if band else None,
        "selected_expected_ids": sorted(selected.intersection(rubric["evidence_ids"])),
        "selected_unlisted_ids": sorted(selected.difference(rubric["evidence_ids"])),
        "caution_cites_expected_source": bool(sources.intersection(expected["evidence_ids"])) if expected else None,
        "unknown_caution_source_ids": sorted(sources.difference(past_ids)),
    }


def rubric_summary(scores):
    scored = [row for row in scores if row.get("evaluation")]
    if not scored:
        return {}
    patterned = [r for r in scored if r["evaluation"]["kind"] == "patterned"]
    checks = [r["evaluation"] for r in scored]
    cautions = [g for g in checks if g.get("expected_caution")]
    sourced = [g for g in checks if g["evidence_ids"]]
    patterns = {}
    for method in ("original", "category", "planner", "actual"):
        hits = 0
        for row in patterned:
            lo, hi = row["evaluation"]["expected_ratio_range"]
            hits += lo <= row[method] / row["original"] <= hi
        patterns[method] = hits
    model_patterned = [r for r in patterned if not r.get("fixed_duration")]
    return {
        "rubric_cases": len(scored),
        "fixed_duration_cases": sum(bool(r.get("fixed_duration")) for r in scored),
        "model_eligible_cases": sum(not r.get("fixed_duration") for r in scored),
        "model_eligible_patterned_cases": len(model_patterned),
        "model_eligible_patterned_hits": sum(r["evaluation"]["prediction_in_band"] for r in model_patterned),
        "patterned_cases": len(patterned),
        "patterned_cases_in_declared_band": patterns,
        "control_cases": sum(g["kind"] == "control" for g in checks),
        "cases_with_declared_comparables": len(sourced),
        "cases_selecting_a_declared_comparable": sum(bool(g["selected_expected_ids"]) for g in sourced),
        "expected_caution_cases": len(cautions),
        "cautions_citing_expected_source": sum(g["caution_cites_expected_source"] for g in cautions),
        "cases_with_unknown_caution_sources": sum(bool(g["unknown_caution_source_ids"]) for g in checks),
        "cases_with_cautions": sum(bool(r.get("cautions")) for r in scored),
        "mae_on_patterned_minutes": round(mean(abs(r["planner"] - r["actual"]) for r in patterned), 2) if patterned else None,
    }
