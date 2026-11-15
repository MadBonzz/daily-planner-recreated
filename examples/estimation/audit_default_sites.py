"""Compare default browser suggestions on unique editable and addable stock cards across both corpora."""

import argparse
from collections import Counter
from datetime import date, datetime, time
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import dev_workspace
import estimate_worker
import estimator
import focus_domains
import focus_judge
import retrieval
import sample_retrieval
from examples.estimation.run import DATASET, DEMO_DATASET, histories, write_json
from model_client import ModelUnavailable
from relevance_feedback import related_ids


DAY = date(2026, 11, 15)


def source_observations(store, target, source_id, sample_ids):
    """Obtain the same sanitized single-source evidence that the policy accepts."""
    result = {}
    source = store.get_task(source_id)
    for domain, request in focus_domains.site_requests(store, target["id"], [source_id]).items():
        state = json.loads(request["state"])
        seconds = sum(event["seconds"] for event in source["report"]["intervals"]
                      if event.get("domain") and not event.get("excluded") and not event.get("idle")
                      and not event.get("phone_app") and event.get("label") == "off_task"
                      and focus_domains.normalize_domain(event["domain"]) == domain)
        if seconds <= 0:
            raise ValueError(f"No recorded time for suggested evidence from {source['title']}")
        result[domain] = {"id": sample_ids[source_id], "title": source["title"], "seconds": round(seconds, 3),
                          "tab_titles": state["observed_off_task_browser_visit"]["tab_titles"]}
    return result


def profile(history):
    name = history["id"]
    missing = {}
    planned = len(history["open_tasks"]) + len(history["holdouts"])

    def no_local(endpoint, body, config):
        role = "Qwen"
        if "state" in body:
            state = json.loads(body["state"])
            role = "Kev website" if "observed_off_task_browser_visit" in state else "Kev task pair"
        missing[sample_retrieval.decision_hash(body)] = role
        raise ModelUnavailable("Requested input has no bundled model result.")

    def no_forecast(*args, **kwargs):
        # The website pipeline consumes prediction.related_tasks, assembled before the time model call.
        raise ModelUnavailable("Time forecast omitted from website comparison.")

    with TemporaryDirectory() as working, patch.object(retrieval, "_local_post", no_local), \
            patch.object(estimator, "chat_json", no_forecast):
        stamp = datetime.combine(DAY, time(15)).timestamp()
        with patch.object(dev_workspace, "time", SimpleNamespace(time=lambda: stamp)):
            store = dev_workspace.open_dev_workspace(working, name)
        store.clock = lambda: stamp

        def selection_incomplete():
            return any(role != "Kev website" for role in missing.values())

        def drain():
            # A pending website verdict cannot alter later comparison selection.
            # An unavailable timing pair can, so stop before measuring on incomplete history.
            while not selection_incomplete() and estimate_worker.run_one_job(store, store.data_dir / "vectors"):
                pass

        sample_ids = {store.get_setting(f"dev_task_{r['id']}"): r["id"]
                      for r in [*history["tasks"], *history["open_tasks"]]}
        open_ids = {r["id"] for r in history["open_tasks"]}
        target_ids = [store.get_setting(f"dev_task_{row['id']}") for row in history["open_tasks"]]
        drain()
        for row in history["holdouts"]:
            if selection_incomplete():
                break
            task = dev_workspace.add_sample_task(store, name, row["id"], DAY)
            sample_ids[task["id"]] = row["id"]
            target_ids.append(task["id"])
            drain()
        result = {"profile": name, "expected_cards": planned}
        if missing:
            return {**result, "status": "pending", "missing_model_inputs": dict(sorted(missing.items()))}
        if store.has_estimation_work() or len(target_ids) != planned:
            raise ValueError(f"Unfinished forecast queue for {name}")
        rows = []
        for target_id in target_ids:
            target = store.get_task(target_id)
            policy = focus_domains.policy(store, target_id)
            if policy["added"] or policy["removed"] or policy["pending"]:
                raise ValueError(f"{name}/{target['title']}: policy contains manual choices or missing judgments.")
            related = related_ids(target["prediction"])
            requests = focus_domains.site_requests(store, target_id, related)
            domains = {}
            for host in sorted(policy["suggestions"]):
                request = requests[host]
                if focus_judge.packaged(request) is not True:
                    raise ValueError(f"Missing affirmative packaged Kev judgment for {name}/{host}")
                domains[host] = {"request_sha256": focus_judge.key(request), "sources": []}
            if domains:
                for source_id in sorted(related, key=sample_ids.__getitem__):
                    for host, observation in source_observations(store, target, source_id, sample_ids).items():
                        if host in domains:
                            domains[host]["sources"].append(observation)
            for host, item in domains.items():
                if not item["sources"]:
                    raise ValueError(f"Unable to attribute {name}/{target['title']}/{host}")
            declined = sorted(requests.keys() - domains.keys())
            if any(focus_judge.packaged(requests[host]) is not False for host in declined):
                raise ValueError(f"A declined site has no matching Kev result for {name}/{target['title']}")
            key = sample_ids[target_id]
            rows.append({"id": key, "kind": "open" if key in open_ids else "addable",
                         "title": target["title"], "selected_timing_comparisons": len(related),
                         "suggestions": domains, "kev_declined_domains": declined})
        return {**result, "status": "verified", "cards": rows}


def totals(profiles):
    hosts = Counter(host for p in profiles for card in p.get("cards", []) for host in card["suggestions"])
    checked = [card for p in profiles for card in p.get("cards", [])]
    count = len(checked)
    any_site = sum(bool(row["suggestions"]) for row in checked)
    by_kind = {}
    for kind in ("open", "addable"):
        cards = [row for row in checked if row["kind"] == kind]
        by_kind[kind] = {"cards": len(cards), "cards_with_suggestions": sum(bool(row["suggestions"]) for row in cards),
                         "task_domain_pairs": sum(len(row["suggestions"]) for row in cards)}
    return {"profiles_verified": sum(row["status"] == "verified" for row in profiles),
            "expected_cards": sum(row["expected_cards"] for row in profiles), "verified_cards": count,
            "cards_with_suggestions": any_site, "share_of_verified_cards": round(any_site / count, 4) if count else None,
            "task_domain_pairs": sum(hosts.values()), "distinct_domains": len(hosts),
            "cards_with_site_evidence_sent_to_kev": sum(bool(row["suggestions"] or row["kev_declined_domains"])
                                                       for row in checked),
            "declined_task_domain_pairs": sum(len(row["kev_declined_domains"]) for row in checked),
            "domain_counts": dict(sorted(hosts.items())), "by_card_kind": by_kind}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = {"method": "Open a fresh dated editable snapshot with authored ActivityWatch labels. Drain initial work, then add each stock example once and drain after each addition. Read the effective policy using the production estimator selection, worker, Qwen/Kev packages and default choices. The blocker reads the screened related_tasks list before the forecast model selects minutes or cites comparisons. This audit runs that path with time forecasting disabled.",
            "units": "A task-domain pair is one suggested domain on one actionable card. A source contributes seconds from off-task desktop browser intervals after filtering excluded events, idle time and simultaneous phone activity. Counts describe the default recommendations available before a browser session starts.",
            "anchor_day": DAY.isoformat(), "cosine_floor": retrieval.MINIMUM_THRESHOLD,
            "embedding_model": retrieval.EMBEDDING_MODEL, "decision_model": retrieval.DECISION_MODEL,
            "command": "python -m examples.estimation.audit_default_sites data/results/demo-histories/default-sites.json",
            "cohorts": {}}
    settings = {"PLANNER_EMBEDDING_MODEL": retrieval.EMBEDDING_MODEL, "PLANNER_DECISION_MODEL": retrieval.DECISION_MODEL,
                "PLANNER_RELEVANCE_THRESHOLD": str(retrieval.MINIMUM_THRESHOLD)}
    with patch.dict(os.environ, settings):
        for cohort, directory in (("benchmark", DATASET), ("demo", DEMO_DATASET)):
            profiles = [profile(h) for h in histories(None, directory)]
            data["cohorts"][cohort] = {"totals": totals(profiles), "profiles": profiles}
            print(cohort, json.dumps(totals(profiles)), flush=True)
    archive = {p["profile"]: p for p in data["cohorts"]["benchmark"]["profiles"]}
    origins = json.loads((DEMO_DATASET / "sources.json").read_text(encoding="utf-8"))
    pairs = []
    for new in data["cohorts"]["demo"]["profiles"]:
        original_ids = sorted({source["history"] for source in origins[new["profile"]].values()})
        row = {"demo": new["profile"], "benchmark_profiles": original_ids, "status": new["status"]}
        for key, group in (("benchmark", [archive[name] for name in original_ids]), ("current", [new])):
            if all(p["status"] == "verified" for p in group):
                counts = totals(group)
                row[key] = {k: counts[k] for k in ("verified_cards", "cards_with_suggestions",
                                                    "task_domain_pairs", "distinct_domains")}
        pairs.append(row)
    data["matched_profile_comparison"] = pairs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, data)


if __name__ == "__main__":
    main()
