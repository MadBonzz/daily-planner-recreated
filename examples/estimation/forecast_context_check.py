"""Exercise an over-context, user-edited history through the real forecast entry point."""
import argparse
import json
from io import BytesIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import build_opener
from unittest.mock import patch

import estimator
import model_client
from examples.estimation.forecast_audit import MODELS, OUT


def check(model, endpoint):
    target = {"id": "target", "title": "Prepare the weekly instrument log", "description":
              "Check the sensor calibration and write the weekly log.", "details": "", "category": "Instrument lab",
              "estimate_minutes": 35}
    # Repeating the same PDF would tokenize too cheaply on some models. Include
    # distinct, meaningful measurement identifiers across all user-selected records.
    history = []
    for i in range(80):
        excerpt = " ".join(f"Sensor S{i:02} batch B{j:02} needs zero-point, offset and drift checks before export."
                           for j in range(16))
        history.append({**target, "id": f"past-{i:02}", "title": f"Inspect sensor batch {i:02}",
                        "description": f"Calibrate batch {i:02}, record deviations, validate drift and archive the log.",
                        "pdf_text": excerpt, "actual_seconds": (39 + i % 3) * 60})
    target["relevance_overrides"] = {r["id"]: True for r in history}
    receipts = []
    real = build_opener

    class Observe:
        def __init__(self, opener):
            self.opener = opener

        def open(self, request, **kwargs):
            receipt = {"bytes_sent": len(request.data)}
            receipts.append(receipt)
            try:
                response = self.opener.open(request, **kwargs)
            except HTTPError as error:
                receipt["http_status"] = error.code
                body = error.read(65536)
                try:
                    detail = json.loads(body).get("error")
                except (ValueError, UnicodeError):
                    detail = None
                if isinstance(detail, dict):
                    receipt["server_error"] = {k: str(v)[:250] for k, v in detail.items()
                                               if k in {"type", "code", "message", "n_prompt_tokens", "n_ctx"}}
                raise HTTPError(error.url, error.code, error.msg, error.headers, BytesIO(body)) from error
            receipt["http_status"] = response.status
            return response

    os.environ.update(PLANNER_MODEL=model, PLANNER_MODEL_ENDPOINT=endpoint, PLANNER_MODEL_TIMEOUT_SECONDS="900")
    with TemporaryDirectory() as tmp, patch.object(model_client, "build_opener", lambda *h: Observe(real(*h))):
        prediction = estimator.predict(target, history, Path(tmp))
    if not receipts or not all("http_status" in r for r in receipts):
        raise RuntimeError(f"Model endpoint did not return a response: {receipts}")
    chosen = prediction["related_tasks"]
    return {"model": model, "number_explicitly_selected": len(history), "receipts": receipts,
            "retained_selection": {r["id"] for r in history} == {r["task_id"] for r in chosen}
                                  and all(r["source"] == "user" for r in chosen),
            "provenance": prediction["provenance"], "explanation": prediction["explanation"],
            "model_error": prediction.get("model_error"), "retryable": bool(prediction.get("retryable"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args()
    if urlsplit(args.endpoint).hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("Use the local model server's loopback endpoint.")
    output = OUT / ("context-" + MODELS[args.model] + ".json")
    result = check(args.model, args.endpoint)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
