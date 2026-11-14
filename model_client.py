"""Schema-constrained chat with the locally loaded Ternary-Bonsai-4B model."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from jsonschema import ValidationError, validate


MODEL_ID = "ternary-bonsai-4b"
API_URL = "http://127.0.0.1:1234/v1/chat/completions"


class ModelUnavailable(RuntimeError):
    """The configured local model could not produce a usable response."""


class _LocalOnly(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def chat_json(messages, schema, *, max_tokens=512):
    """Return validated JSON, or raise ModelUnavailable so callers can fall back."""
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "planner_response", "strict": True, "schema": schema},
        },
    }
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Personal activity and document excerpts must stay on loopback even on
    # machines with a system-wide web proxy installed.
    client = build_opener(ProxyHandler({}), _LocalOnly())
    try:
        with client.open(request, timeout=90) as response:
            completion = json.load(response)
        choice = completion["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ModelUnavailable("Local model reached its response limit.")
        result = json.loads(choice["message"]["content"])
        validate(result, schema)
        return result
    except HTTPError as exc:
        raise ModelUnavailable(f"Local LM Studio returned HTTP {exc.code}.") from exc
    except (OSError, URLError, KeyError, IndexError, ValueError, TypeError, ValidationError) as exc:
        raise ModelUnavailable("Local LM Studio is unavailable or returned invalid JSON.") from exc
