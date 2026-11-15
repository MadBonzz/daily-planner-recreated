"""Score published or fresh forecast receipts against task-level behavioral labels."""
from collections import Counter
import json
import re
from statistics import median

from examples.estimation.forecast_audit import OUT, REFERENCE, challenge, confidence_cases


# Diagnostic terminology proxies written against the published warning rubrics.
# A source citation alone cannot establish that the advice addresses the obstacle.
WARNING_TOPIC = {
    "mara-rooms": r"late|delay|wait|punctual|arrival|join.{0,12}time|(?:joined|arrived).{0,45}after",
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


def copies_unsupported_example(warning):
    """Diagnostic for the exact demonstration sentence observed in the development run."""
    advice = " ".join(re.findall(r"[a-z]+", warning["advice"].lower()))
    phrase = "test the pairing before the run"
    sources = " ".join(s["observation"] for s in warning["sources"])
    return phrase in advice and not re.search(r"\b(?:pair(?:ed|ing)?|re-?pair(?:ed|ing)?|bluetooth)\b", sources, re.I)


def uncertainty_ordering(cases):
    confidence_rank = {"low": 0, "medium": 1, "high": 2}
    results = {}
    for label, tight_id, broad_id in [
        ("indexing", "same-indexing-slower", "same-indexing-variable"),
        ("supply count", "supply-count-tight", "supply-count-variable"),
    ]:
        tight, broad = cases.get(tight_id), cases.get(broad_id)
        if not tight or not broad:
            continue
        low_width, high_width = tight["interval_width_min"], broad["interval_width_min"]
        first, second = tight["confidence"], broad["confidence"]
        results[label] = {"tight_width": low_width, "variable_width": high_width,
                          "tight_confidence": first, "variable_confidence": second,
                          "width_ordering": "inversion" if high_width < low_width else "tie" if low_width == high_width else "wider",
                          "confidence_ordering": ("inversion" if confidence_rank[second] > confidence_rank[first] else
                                                  "tie" if first == second else "lower"),
                          "tight_point": tight["forecast"], "variable_point": broad["forecast"]}
    return results


def score(rows):
    dirs = {label: Counter() for label in ("up", "near", "down")}
    evidence, warning, topical_warning, warning_detection = Counter(), Counter(), Counter(), Counter()
    grounded_warning = Counter()
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
        candidates = set(row["supplied_candidates"] if "supplied_candidates" in row else row["frozen_candidates"])
        cited = {e["task_id"] for e in p["evidence"]}
        expected = set(rule["expected_evidence"])
        for task_id in candidates:
            decision(evidence, task_id in expected, task_id in cited)
        cautions = {s["task_id"] for c in p["cautions"] for s in c["sources"]}
        warnings = p["cautions"]
        leaks = [copies_unsupported_example(c) for c in warnings]
        summary["warnings_emitted"] += len(warnings)
        summary["unsupported_copied_example_warnings"] += sum(leaks)
        summary["cases_with_copied_example"] += any(leaks)
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
                clean_hits = [hit and not leak for hit, leak in zip(hits, leaks, strict=True)]
                decision(grounded_warning, bool(sources), any(clean_hits) if sources else bool(cautions))
                if rule.get("exhaustive_warning_labels"):
                    if not sources:
                        offered = any("warnings" in c.get("response", {}) for c in calls)
                        summary["negative_cases_with_observations" if offered else "negative_cases_without_observations"] += 1
                    # Count each expected obstacle once. A missed obstacle and
                    # an unrelated warning on that same task are both errors.
                    warning_detection["tp"] += any(clean_hits)
                    warning_detection["fn"] += bool(sources) and not any(clean_hits)
                    warning_detection["fp"] += False in clean_hits
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
                         "interval_width_min": p["high_min"] - p["low_min"], "confidence": p.get("confidence"),
                         "supplied": sorted(candidates), "unsupported_example_copies": sum(leaks)}
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
    return {"counts": dict(summary), "matched_counterfactual": cf, "uncertainty_pairs": uncertainty_ordering(details),
            "direction_by_confidence": by_confidence,
            "wall_clock_including_queue": timing, "direction_one_vs_rest": {k: binary(v) for k, v in dirs.items()},
            "timing_citations_given_candidates": binary(evidence),
            "warning_cases_given_reachable_source_or_explicit_negative": binary(warning),
            "warning_topic_and_source_cases": binary(topical_warning),
            "warning_topic_source_without_example_leak": binary(grounded_warning),
            "warning_detection_exhaustively_labeled": binary(warning_detection), "cases": details}


def write_tables(data):
    def rate(value):
        return "—" if value is None else f"{value:.0%}"

    def fraction(s, a, b):
        return f"{s['counts'].get(a, 0)}/{s['counts'].get(b, 0)}"

    def direction(s):
        d = s["direction_by_confidence"].values()
        return f"{sum(x['correct_direction'] for x in d)}/{sum(x['labeled'] for x in d)}"

    lines = ["# Forecast measurements", "",
             "Direct outputs alone receive model-quality credit. A scheduled event is counted separately and uses zero calls. "
             "Direction credit compares the change from the user's budget against the behavioral label.",
             "", "## Published cases", "",
             "| Model and version | Cases present | Direct | Fallback | Correct direction | Within labeled band | Citation precision / recall |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    groups = []
    for model, runs in data.items():
        groups.append((model, "original", runs["original_full"]))
        if "revised" in runs:
            groups.append((model, "revised", runs["revised"]["replayed"]))
    for model, stage, s in groups:
        c = s["counts"]
        e = s["timing_citations_given_candidates"]
        lines.append(f"| {model} {stage} | {c.get('total',0)} | {c.get('direct',0)} | {c.get('fallback',0)} | "
                     f"{direction(s)} | {fraction(s,'inside_rubric_band','labeled_bands')} | "
                     f"{rate(e['precision'])} / {rate(e['recall'])} |")
    lines += ["", "| Model and version | Increase TP / FP / FN | Decrease TP / FP / FN | Near budget TP / FP / FN | "
              "Citation TP / FP / FN | Reachable warning source hit / labeled |", "| --- | --- | --- | --- | --- | --- |"]
    for model, stage, s in groups:
        d = s["direction_one_vs_rest"]
        e = s["timing_citations_given_candidates"]
        w = s["warning_cases_given_reachable_source_or_explicit_negative"]
        counts = [f"{v['tp']} / {v['fp']} / {v['fn']}" for v in (d["up"], d["down"], d["near"], e)]
        lines.append(f"| {model} {stage} | " + " | ".join(counts) + f" | {w['tp']}/{w['tp']+w['fn']} |")
    lines += ["", "The following rows restrict each within-model comparison to cases with a direct forecast in both versions. "
              "A case with no direction label remains eligible for other checks.", "",
              "| Model | Common direct cases | Old → revised correct direction | Old → revised in behavioral band |",
              "| --- | ---: | --- | --- |"]
    for model, runs in data.items():
        if "revised" in runs:
            a = runs["revised"]["matched_direct_original"]
            b = runs["revised"]["matched_direct_current"]
            lines.append(f"| {model} | {a['counts'].get('direct',0)} | {direction(a)} → {direction(b)} | "
                         f"{fraction(a,'inside_rubric_band','labeled_bands')} → {fraction(b,'inside_rubric_band','labeled_bands')} |")
    lines += ["", "## Fresh cases", "",
              "| Model | Direct | Correct direction | Evidence TP / FP / FN | Warning TP / FP / FN | Cold low confidence | "
              "Explicitly acknowledged missing timing |", "| --- | ---: | ---: | --- | --- | --- | --- |"]
    for model, runs in data.items():
        if "revised" not in runs:
            continue
        s = runs["revised"]["fresh"]
        e = s["timing_citations_given_candidates"]
        w = s["warning_detection_exhaustively_labeled"]
        lines.append(f"| {model} | {s['counts'].get('direct',0)} | {direction(s)} | {e['tp']} / {e['fp']} / {e['fn']} | "
                     f"{w['tp']} / {w['fp']} / {w['fn']} | {fraction(s,'cold_low_confidence','cold_underspecified')} | "
                     f"{fraction(s,'explicit_no_timing','cold_underspecified')} |")
    lines += ["", "Only the historical durations and derived arithmetic change across these requests. All three new tasks "
              "have a 40-minute budget. The first and third histories also share a mean and median of 60.", "",
              "| Model | Slower history point | Faster history point | Both directions correct | Tight interval width | "
              "Variable interval width | Tight / variable confidence |", "| --- | ---: | ---: | --- | ---: | ---: | --- |"]
    for model, runs in data.items():
        cf = runs.get("revised", {}).get("fresh", {}).get("matched_counterfactual", {})
        if cf:
            lines.append(f"| {model} | {cf['slower_history_point']} | {cf['faster_history_point']} | "
                         f"{'yes' if cf['both_directions_correct'] else 'no'} | {cf.get('tight_width','—')} | "
                         f"{cf.get('variable_width','—')} | {cf.get('tight_confidence','—')} / {cf.get('variable_confidence','—')} |")
    lines += ["", "## One-sentence confidence trial", "",
              "The two arms differ only by the confidence instruction. Each pair preserves the same task and historical "
              "mean and median. Ties are permitted. A narrower interval or higher confidence on the more variable history "
              "is an inversion. Indexing is the development pair, and supply counting checks transfer to a fresh task.",
              "", "| Model and arm | Pair | Width, tight → variable | Width ordering | Confidence, tight → variable | "
              "Confidence ordering |", "| --- | --- | --- | --- | --- | --- |"]
    for model, runs in data.items():
        for stage in ("confidence-control", "confidence-trial"):
            scores = runs.get(stage, {}).get("fresh", {}).get("uncertainty_pairs", {})
            for pair, values in scores.items():
                lines.append(f"| {model} {stage} | {pair} | {values['tight_width']} → {values['variable_width']} | "
                             f"{values['width_ordering']} | {values['tight_confidence']} → {values['variable_confidence']} | "
                             f"{values['confidence_ordering']} |")
    lines += ["", "## Integrated retrieval and forecasting", "",
              "Both forecast versions use the same revised forecast prompt. The integrated arm searches the full history through "
              "production Qdrant and Kev, using shipped model results. The frozen arm receives the previously selected records.",
              "", "| Model and case | Frozen supplied → cited | Integrated supplied → cited | Budget | Frozen point → integrated | "
              "Expected direction |", "| --- | --- | --- | ---: | ---: | --- |"]
    def ids(values):
        return ", ".join(values) or "none"
    for model, runs in data.items():
        for task, pair in runs.get("integrated_vs_frozen", {}).items():
            old, new = pair["frozen"], pair["integrated"]
            lines.append(f"| {model} {task} | {ids(old['supplied'])} → {ids(old['cited'])} | "
                         f"{ids(new['supplied'])} → {ids(new['cited'])} | {old['budget']} | "
                         f"{old['forecast']} → {new['forecast']} | {old['expected_direction']} |")
    lines += ["", "## Runtime", "",
              "Shared CPU wall time includes queueing. The solo check uses the median-sized archived case, patch-tool, after "
              "other clients drain. One solo request cannot establish a general latency distribution.", "",
              "| Model and workload | Calls | Retry calls | Truncations | Median call seconds | Calls over 90 seconds |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for model, runs in data.items():
        items = [("original published", runs["original_full"])]
        if "revised" in runs:
            items += [("revised published", runs["revised"]["replayed"]), ("revised fresh", runs["revised"]["fresh"])]
        if "solo" in runs:
            items.append(("solo patch-tool", runs["solo"]["replayed"]))
        for name, s in items:
            t = s["wall_clock_including_queue"]
            lines.append(f"| {model} {name} | {t['model_calls']} | {t['retries']} | {t['length_truncations']} | "
                         f"{t['median_call_seconds']} | {t['calls_over_default_90_seconds']} |")
    lines += ["", "The following latency comparison checks primary HTTP payload hashes before comparing timings. "
              "The server is warm. Context reuse and generated length can differ.", "",
              "| Model | Shared → solo wall seconds | Shared → solo generated tokens by attempt | Shared → solo cached input tokens by attempt |",
              "| --- | ---: | --- | --- |"]
    for model, runs in data.items():
        if "same_request_latency" in runs:
            stages = runs["same_request_latency"]
            shared, solo = stages["shared"], stages["solo"]
            lines.append(f"| {model} | {shared['seconds']} → {solo['seconds']} | "
                         f"{shared['completion_tokens']} → {solo['completion_tokens']} | "
                         f"{shared['cached_prompt_tokens']} → {solo['cached_prompt_tokens']} |")
    (OUT / "metrics.md").write_text("\n".join(lines) + "\n")


def main():
    result = {}
    fresh_ids = {c[0]["id"] for c in challenge() + confidence_cases()}
    for short in ("bonsai-4b", "bonsai-9b"):
        old = list(published_baseline(short))
        result[short] = {"original_full": score(old)}
        for group in sorted(p for p in OUT.iterdir() if p.is_dir()):
            rows = [json.loads(p.read_text()) for p in sorted((group / short).glob("*.json"))]
            if not rows:
                continue
            existing = [r for r in rows if r["task"] not in fresh_ids]
            names = {r["task"] for r in existing}
            common = {r["task"] for r in old if r["prediction"]["provenance"]["decision"] == "direct"} & {
                r["task"] for r in existing if r["prediction"]["provenance"]["decision"] == "direct"}
            result[short][group.name] = {
                "matched_direct_original": score([r for r in old if r["task"] in common]),
                "matched_direct_current": score([r for r in existing if r["task"] in common]),
                "paired_original": score([r for r in old if r["task"] in names]),
                "replayed": score(existing), "fresh": score([r for r in rows if r["task"] in fresh_ids])}
        solo_path = OUT / "solo" / short / "patch-tool.json"
        shared_path = OUT / "revised" / short / "patch-tool.json"
        if solo_path.exists() and shared_path.exists():
            solo = json.loads(solo_path.read_text())
            shared = json.loads(shared_path.read_text())
            if solo["receipts"][0]["payload_sha256"] != shared["receipts"][0]["payload_sha256"]:
                raise ValueError("Solo timing does not use the exact same primary forecast request.")
            result[short]["same_request_latency"] = {
                context: {"seconds": round(sum(c["seconds"] for c in row["receipts"]), 2),
                          "attempts": len(row["receipts"]),
                          "completion_tokens": [c.get("usage", {}).get("completion_tokens") for c in row["receipts"]],
                          "cached_prompt_tokens": [c.get("timings", {}).get("cache_n") for c in row["receipts"]],
                          "point": row["prediction"]["point_min"]}
                for context, row in (("shared", shared), ("solo", solo))}
        if "integrated" in result[short] and "revised" in result[short]:
            integrated = result[short]["integrated"]["replayed"]["cases"]
            frozen = result[short]["revised"]["replayed"]["cases"]
            result[short]["integrated_vs_frozen"] = {
                task: {"integrated": row, "frozen": frozen[task]} for task, row in integrated.items() if task in frozen}
    (OUT / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    write_tables(result)


if __name__ == "__main__":
    main()
