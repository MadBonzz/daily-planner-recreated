"""Build Kev desk decisions for actionable tasks in the original benchmark."""

import argparse
import json
from pathlib import Path

from examples.estimation.run import histories, write_json
from model_client import planner_settings
from retrieval import DECISION_MODEL
from sample_retrieval import decision_hash
from workspace_eligibility import PACK, live, payload


def build(config, *, check=False, checkpoint=None):
    published = json.loads(PACK.read_text(encoding="utf-8"))["decisions"] if PACK.exists() else {}
    known = dict(published)
    if checkpoint and checkpoint.exists():
        known.update(json.loads(checkpoint.read_text(encoding="utf-8"))["decisions"])
    active = {}
    model = config.get("PLANNER_DECISION_MODEL") or DECISION_MODEL
    for history in histories(None):
        for row in [*history.get("open_tasks", []), *history["holdouts"]]:
            request = payload({**row, "attachments": bool(row.get("pdf"))}, config)
            key = decision_hash(request)
            if type(known.get(key)) is not bool:
                if check:
                    raise ValueError(f"Missing workspace verdict for {history['id']}/{row['id']}")
                known[key] = live(request, config)
                if checkpoint:
                    write_json(checkpoint, {"model": model, "decisions": known}, sort_keys=True)
                print(history["id"], row["id"], known[key], flush=True)
            active[key] = known[key]
    if check:
        if published != active:
            raise ValueError("Packaged workspace decisions do not match the current task inputs.")
    else:
        # Publish one complete set. Existing checkouts can keep serving their prior
        # exact matches while fresh local judgments are collected in the checkpoint.
        write_json(PACK, {"model": model, "decisions": active}, sort_keys=True)
    print(f"{len(active)} actionable sample tasks have current workspace decisions.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    opts = parser.add_mutually_exclusive_group()
    opts.add_argument("--checkpoint", type=Path, help="Resume in a separate file before replacing the entire package.")
    opts.add_argument("--check", action="store_true", help="Check coverage without loading a model.")
    args = parser.parse_args()
    if args.checkpoint and args.checkpoint.resolve() == PACK.resolve():
        parser.error("Use a separate file for the resumable checkpoint.")
    build(planner_settings(), check=args.check, checkpoint=args.checkpoint)
