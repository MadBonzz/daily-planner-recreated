"""Judge the same public pairs with and without explicit named-section excerpts."""

import argparse
import json
from pathlib import Path

import retrieval
import sample_retrieval
from pdf_reader import assigned_section_text
from examples.estimation.audit_retrieval import FROZEN, frozen_request
from examples.estimation.calibrate_retrieval import write_json
from examples.estimation.package_retrieval import inputs
from examples.estimation.run import histories


def run(output):
    rows = json.loads(output.read_text()) if output.exists() else {}
    settings = retrieval.planner_settings()
    reference = json.loads(FROZEN.read_text())
    for history in histories(None):
        name = history["id"]
        tasks = inputs(history)
        scoped = {key: {**task, "pdf_text": assigned_section_text(task)}
                  for key, task in tasks.items()
                  if assigned_section_text(task) != (task.get("pdf_text") or "").strip()}
        if not scoped:
            continue
        completed = {r["id"] for r in history["tasks"]}
        holdouts = {r["id"]: r["evaluation"]["evidence_ids"] for r in history["holdouts"]}
        for pair, previous in reference["packaged_baseline"][name].items():
            target, candidate = pair.split("/")
            if target not in scoped and candidate not in scoped:
                continue
            frozen_request(reference, name, target, candidate, False)  # Verify the recorded input fingerprint.
            body = retrieval.decision_payload(scoped.get(target, tasks[target]),
                                              scoped.get(candidate, tasks[candidate]), (), retrieval.DECISION_MODEL)
            key = sample_retrieval.decision_hash(body)
            if key == previous["sha256"] or key in rows:
                continue
            reply = retrieval._local_post(settings["PLANNER_DECISION_ENDPOINT"], body, settings)
            score = reply["answers"]["relevant"]["noul"]
            if type(score) not in (int, float) or not 0 <= score <= 1:
                raise ValueError("Kev returned an invalid relevance probability")
            expected = candidate in holdouts[target] if target in holdouts and candidate in completed else None
            rows[key] = dict(history=name, target=target, candidate=candidate, old_hash=previous["sha256"],
                             previous=previous["related"], scoped_score=score, related=score >= .5, expected=expected)
            write_json(output, rows)
            print(target, candidate, previous["related"], score, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
