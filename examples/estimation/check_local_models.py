"""Check the three configured local model endpoints using synthetic planner inputs."""

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import estimator
import retrieval
from model_client import ModelUnavailable, model_settings, planner_settings


TASK = {
    "title": "Collate the boathouse launch checklist",
    "description": "Check the printed launch checklist against the eight laminated copies.",
    "estimate_minutes": 20,
}
PRIOR = {
    "title": "Check the cabin safety checklists",
    "description": "Compare eight laminated checklists with the signed master copy.",
}


def check(component, config):
    if component == "embedding":
        vector = retrieval.embed_query(TASK, config)
        model = config.get("PLANNER_EMBEDDING_MODEL") or retrieval.EMBEDDING_MODEL
        return f"{model} returned {len(vector)} dimensions."
    if component == "decision":
        related = retrieval.decide_related(TASK, PRIOR)
        model = config.get("PLANNER_DECISION_MODEL") or retrieval.DECISION_MODEL
        return f"{model} classified the example pair as {'related' if related else 'unrelated'}."

    endpoint, model = model_settings(config)
    if urlsplit(endpoint).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ModelUnavailable("Set PLANNER_MODEL_ENDPOINT to your local chat endpoint for this check.")
    with TemporaryDirectory() as directory:
        result = estimator.predict(TASK, [], Path(directory))
    if result["provenance"]["decision"] != "direct":
        raise ModelUnavailable(result["model_error"])
    return (f"{model} returned a validated forecast of {result['point_min']} min "
            f"({result['low_min']}–{result['high_min']} min, {result['confidence']} confidence).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("all", "embedding", "decision", "forecast"), default="all")
    component = parser.parse_args().component
    components = ("embedding", "decision", "forecast") if component == "all" else (component,)
    config = planner_settings()
    failed = False
    for component in components:
        try:
            print(f"{component}: {check(component, config)}", flush=True)
        except (ModelUnavailable, ValueError) as error:
            print(f"{component}: {error}", file=sys.stderr, flush=True)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
