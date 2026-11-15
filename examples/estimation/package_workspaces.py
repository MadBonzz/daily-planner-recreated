"""Checkpoint genuine Kev desk-applicability decisions for the actionable sample tasks."""

import json

from examples.estimation.run import histories
from model_client import planner_settings
from sample_retrieval import decision_hash
from workspace_eligibility import PACK, live, payload


def main():
    config = planner_settings()
    records = json.loads(PACK.read_text(encoding="utf-8"))["decisions"] if PACK.exists() else {}
    seen = set()
    for history in histories(None):
        for row in [*history.get("open_tasks", []), *history["holdouts"]]:
            request = payload({**row, "attachments": bool(row.get("pdf"))}, config)
            key = decision_hash(request)
            seen.add(key)
            if key in records:
                continue
            records[key] = live(request, config)
            temp = PACK.with_suffix(".tmp")
            temp.write_text(json.dumps({"model": request["model"], "decisions": records},
                                       indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temp.replace(PACK)
            print(history["id"], row["id"], records[key], flush=True)

    # Drop old versions only after every task has a decision for the current prompt.
    remaining = {key: records[key] for key in sorted(seen)}
    PACK.write_text(json.dumps({"model": config.get("PLANNER_DECISION_MODEL") or "kev-4b",
                                "decisions": remaining}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
