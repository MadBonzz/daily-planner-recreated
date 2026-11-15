"""Validated chat completions from LM Studio or a configured compatible endpoint."""

import json
import math
import os
import shlex
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

try:
    from dotenv import dotenv_values
except ModuleNotFoundError as exc:
    if exc.name != "dotenv":
        raise
    dotenv_values = None

from jsonschema import ValidationError, validate


MODEL_ID = "ternary-bonsai-4b"
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
ENV_FILE = Path(__file__).with_name(".env")
COHERE_HOSTS = {"api.cohere.ai", "api.cohere.com"}
SETTING_NAMES = {
    "PLANNER_MODEL_ENDPOINT", "PLANNER_MODEL", "COHERE_API", "COHERE_API_KEY",
    "PLANNER_EMBEDDING_MODEL", "PLANNER_EMBEDDING_ENDPOINT", "PLANNER_DECISION_MODEL",
    "PLANNER_DECISION_ENDPOINT", "PLANNER_RELEVANCE_THRESHOLD",
    "PLANNER_MODEL_TIMEOUT_SECONDS", "PLANNER_ACTIVITYWATCH_HOSTNAME",
    "PLANNER_ACTIVITYWATCH_DB", "PLANNER_ACTIVITYWATCH_PHONE_DB",
}


class ModelUnavailable(RuntimeError):
    """The configured endpoint could not produce a usable response."""


class ModelContextExceeded(ModelUnavailable):
    """The provider rejected an oversized input before producing a completion."""


class ModelResponseInvalid(ModelUnavailable):
    """A completed HTTP response did not supply JSON matching the requested contract."""


class ModelResponseTruncated(ModelResponseInvalid):
    """The endpoint explicitly reported that generation exhausted the response budget."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _file_settings():
    if dotenv_values is not None:
        return dotenv_values(ENV_FILE, interpolate=False, encoding="utf-8-sig")
    # Existing installations can read the planner's simple one-line settings
    # before the user updates their environment from requirements.txt.
    if not ENV_FILE.is_file():
        return {}
    result = {}
    for line in ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        name = name.strip()
        if not sep or name not in SETTING_NAMES:
            continue
        try:
            lexer = shlex.shlex(value, posix=True)
            lexer.whitespace_split = True
            if name in {"PLANNER_ACTIVITYWATCH_DB", "PLANNER_ACTIVITYWATCH_PHONE_DB"}:
                lexer.escape = ""  # Keep Windows path separators in legacy .env files.
            parts = list(lexer)
        except ValueError:
            raise ModelUnavailable(f"Check the quotes around {name} in .env.") from None
        result[name] = " ".join(parts)
    return result


def planner_settings():
    # Reload on every request so editing the two settings takes effect immediately.
    return {**_file_settings(), **os.environ}


def model_settings(config=None):
    """The active endpoint and model ID, without credentials."""
    if config is None:
        config = planner_settings()
    return config.get("PLANNER_MODEL_ENDPOINT") or API_URL, config.get("PLANNER_MODEL") or MODEL_ID


_forecast_config = ContextVar("forecast_config", default=None)


@contextmanager
def forecast_endpoint():
    """Pin settings for one forecast so its saved attribution matches all its calls."""
    config = planner_settings()
    endpoint, model = model_settings(config)
    try:
        host = urlsplit(endpoint).hostname
    except ValueError:
        host = None
    provider = "Cohere" if host in COHERE_HOSTS else host
    if host in {"127.0.0.1", "localhost", "::1"}:
        provider = "Local endpoint"
    token = _forecast_config.set(config)
    try:
        yield {"provider": provider, "model": model}
    finally:
        _forecast_config.reset(token)


def _canonical_integers(value, schema):
    """JSON Schema integers can arrive as 1.0. Only coerce declared integer fields after validation."""
    if type(value) is float and schema.get("type") == "integer":
        return int(value)
    if isinstance(value, dict) and "properties" in schema:
        props = schema["properties"]
        return {key: _canonical_integers(item, props[key]) if key in props else item
                for key, item in value.items()}
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [_canonical_integers(item, schema["items"]) for item in value]
    return value


def chat_json(messages, schema, *, max_tokens=512):
    """Return locally validated JSON, or raise ModelUnavailable for the planner fallback."""
    config = _forecast_config.get() or planner_settings()
    endpoint, model = model_settings(config)
    try:
        timeout = float(config.get("PLANNER_MODEL_TIMEOUT_SECONDS") or 90)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError
    except ValueError:
        raise ModelUnavailable("PLANNER_MODEL_TIMEOUT_SECONDS must be a positive finite number.") from None
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
        key = config.get("COHERE_API") or config.get("COHERE_API_KEY")
        if not key:
            raise ModelUnavailable("Set COHERE_API in .env to use the Cohere endpoint.")
        headers["Authorization"] = f"Bearer {key}"
    if cohere and model.startswith(("c4ai-aya-", "tiny-aya-")):
        # Aya supports ordinary chat. Provide its response contract in the prompt.
        instruction = "Return only one JSON object matching this schema:\n" + json.dumps(schema)
        conversation = list(messages)
        if conversation and conversation[0].get("role") in {"system", "developer"}:
            first = conversation.pop(0)
            instruction = first["content"] + "\n\n" + instruction
        payload["messages"] = [{"role": "system", "content": instruction}, *conversation]
    else:
        # Both LM Studio and Cohere's OpenAI compatibility endpoint accept this envelope.
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
        with client.open(request, timeout=timeout) as response:
            completion = json.load(response)
        choice = completion["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ModelResponseTruncated(f"{model} reached its response length limit.")
        content = choice["message"]["content"].strip()
        if cohere and content.startswith(("```json\n", "```\n")) and content.endswith("```"):
            content = content.split("\n", 1)[1][:-3].strip()
        result = json.loads(content)
        validate(result, schema)
        return _canonical_integers(result, schema)
    except HTTPError as exc:
        if exc.code in (400, 413):
            try:
                body = json.loads(exc.read(16384))
                detail = body.get("error") if isinstance(body, dict) else None
            except (OSError, ValueError):
                detail = None
            if isinstance(detail, dict) and any(detail.get(k) in ("exceed_context_size_error", "context_length_exceeded")
                                               for k in ("type", "code")):
                count, limit = detail.get("n_prompt_tokens"), detail.get("n_ctx")
                sizes = (f" ({count:,} input tokens, {limit:,} available)"
                         if type(count) is int and type(limit) is int and count > 0 and limit > 0 else "")
                raise ModelContextExceeded(f"{model} rejected the input because it exceeds the context window{sizes}.") from exc
        raise ModelUnavailable(f"{url.hostname} returned HTTP {exc.code} for {model}.") from exc
    except OSError as exc:
        raise ModelUnavailable(f"{model} is unavailable.") from exc
    except ValidationError as exc:
        # ValidationError.message can contain the entire response. Limit the
        # retry hint to a field path and failed rule, so it cannot echo task text.
        field = ".".join(map(str, exc.absolute_path)) or "response"
        raise ModelResponseInvalid(f"{model} returned an invalid {field} field ({exc.validator}).") from exc
    except (KeyError, IndexError, ValueError, TypeError, AttributeError) as exc:
        raise ModelResponseInvalid(f"{model} returned an incomplete or invalid JSON object.") from exc
