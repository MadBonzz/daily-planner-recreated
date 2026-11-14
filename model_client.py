"""Validated chat completions from LM Studio or a configured compatible endpoint."""

import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from dotenv import dotenv_values
from jsonschema import ValidationError, validate


MODEL_ID = "ternary-bonsai-4b"
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
ENV_FILE = Path(__file__).with_name(".env")
COHERE_HOSTS = {"api.cohere.ai", "api.cohere.com"}


class ModelUnavailable(RuntimeError):
    """The configured endpoint could not produce a usable response."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _configuration():
    # Reload on every request so editing the two settings takes effect immediately.
    config = {**dotenv_values(ENV_FILE, interpolate=False), **os.environ}
    endpoint = config.get("PLANNER_MODEL_ENDPOINT") or API_URL
    model = config.get("PLANNER_MODEL") or MODEL_ID
    return endpoint, model, config.get("COHERE_API") or config.get("COHERE_API_KEY")


def model_settings():
    """The active endpoint and model ID, without credentials."""
    endpoint, model, _ = _configuration()
    return endpoint, model


def chat_json(messages, schema, *, max_tokens=512):
    """Return locally validated JSON, or raise ModelUnavailable for the planner fallback."""
    endpoint, model, key = _configuration()
    headers = {"Content-Type": "application/json"}
    try:
        url = urlsplit(endpoint)
        if not url.hostname or url.scheme not in {"https", "http"} or url.username or url.password:
            raise ValueError
        cohere = url.hostname in COHERE_HOSTS
        if cohere and (url.scheme != "https" or url.port not in {None, 443}):
            raise ValueError
    except ValueError:
        raise ModelUnavailable("Invalid PLANNER_MODEL_ENDPOINT. Use the chat completions URL.") from None

    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if cohere:
        if model in {"c4ai-aya-expanse-8b", "c4ai-aya-vision-8b"}:
            raise ModelUnavailable(f"Cohere retired {model} on April 4, 2026. Choose a hosted model in .env.")
        if not key:
            raise ModelUnavailable("Set COHERE_API in .env to use the Cohere endpoint.")
        headers["Authorization"] = f"Bearer {key}"
    if cohere and (model.startswith("c4ai-aya-") or model.startswith("tiny-aya-")):
        # Aya supports ordinary chat. Provide its response contract in the prompt.
        instruction = "Return only one JSON object matching this schema:\n" + json.dumps(schema)
        conversation = list(messages)
        if conversation and conversation[0].get("role") in {"system", "developer"}:
            first = conversation.pop(0)
            instruction = first["content"] + "\n\n" + instruction
        payload["messages"] = [{"role": "system", "content": instruction}, *conversation]
    elif cohere:
        payload["response_format"] = {"type": "json_object", "schema": schema}
    else:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "planner_response", "strict": True, "schema": schema},
        }

    request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    # Only the explicitly configured endpoint receives the prompt. Loopback
    # traffic bypasses system web proxies; hosted requests honor HTTPS_PROXY.
    handlers = [_NoRedirect()]
    if url.hostname in {"localhost", "127.0.0.1", "::1"}:
        handlers.append(ProxyHandler({}))
    client = build_opener(*handlers)
    try:
        with client.open(request, timeout=90) as response:
            completion = json.load(response)
        choice = completion["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ModelUnavailable(f"{model} reached its response length limit.")
        content = choice["message"]["content"].strip()
        if cohere and content.startswith(("```json\n", "```\n")) and content.endswith("```"):
            content = content.split("\n", 1)[1][:-3].strip()
        result = json.loads(content)
        validate(result, schema)
        return result
    except HTTPError as exc:
        raise ModelUnavailable(f"{url.hostname} returned HTTP {exc.code} for {model}.") from exc
    except (OSError, KeyError, IndexError, ValueError, TypeError, AttributeError, ValidationError) as exc:
        raise ModelUnavailable(f"{model} is unavailable or returned invalid JSON.") from exc
