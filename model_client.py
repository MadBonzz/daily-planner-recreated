"""Validated chat completions from LM Studio or a configured compatible endpoint."""

import json
import math
import os
import re
import shlex
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

import local_model_runtime
from local_model_runtime import ModelUnavailable, NoRedirect as _NoRedirect

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
    "PLANNER_SINGLE_MODEL", "PLANNER_MODEL_KEY", "PLANNER_DECISION_MODEL_KEY", "PLANNER_EMBEDDING_MODEL_KEY",
    "PLANNER_CHAT_CONTEXT_LENGTH", "PLANNER_DECISION_CONTEXT_LENGTH", "PLANNER_EMBEDDING_CONTEXT_LENGTH",
}


class ModelContextExceeded(ModelUnavailable):
    """The provider rejected an oversized input before producing a completion."""


class ModelResponseInvalid(ModelUnavailable):
    """A completed HTTP response did not supply JSON matching the requested contract."""


class ModelResponseTruncated(ModelResponseInvalid):
    """The endpoint explicitly reported that generation exhausted the response budget."""


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


def _final_json(message, model):
    """Decode one answer from the assistant's public text, ignoring reasoning blocks."""
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part["text"] for part in content if isinstance(part, dict)
                          and part.get("type") in {"text", "output_text"} and isinstance(part.get("text"), str))
    if not isinstance(content, str) or not content.strip():
        if message.get("reasoning_content") or message.get("reasoning"):
            raise ModelResponseInvalid(f"{model} returned reasoning without a final JSON answer.")
        raise ModelResponseInvalid(f"{model} returned no final answer text.")
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Bonsai templates can prefill the opening think tag and emit only its close.
    # Match a thinking prefix or a delimiter on its own line, outside the answer.
    if content.startswith("<think>"):
        _, close, content = content.partition("</think>")
        if not close:
            raise ModelResponseInvalid(f"{model} returned an unfinished thinking block without a final JSON answer.")
    else:
        close = re.search(r"(?m)^\s*</think>\s*|</think>\s*(?=```|[\[{]|$)", content)
        if close:
            content = content[close.end():]
    content = content.strip()
    # A local runtime may wrap the answer in a Markdown fence or a short preamble.
    # Decode from the first opening delimiter only: a nested object in malformed
    # output, or one of several proposed answers, is not a completed response.
    start = re.search(r"[\[{]", content)
    if not start or "<think>" in content[:start.start()]:
        raise ModelResponseInvalid(f"{model} returned no final JSON object.")
    try:
        result, end = json.JSONDecoder().raw_decode(content, start.start())
    except json.JSONDecodeError as exc:
        raise ModelResponseInvalid(f"{model} returned incomplete or malformed final JSON.") from exc
    if any(char in content[end:] for char in "{}[]") or "</think>" in content[end:]:
        raise ModelResponseInvalid(f"{model} returned ambiguous text around the final JSON answer.")
    return result


def _post_completion(client, endpoint, headers, payload, timeout, model):
    host = urlsplit(endpoint).hostname
    bonsai = "reasoning_effort" in payload
    while True:
        request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with client.open(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            with exc:
                if exc.code not in (400, 413, 422):
                    raise ModelUnavailable(f"{host} returned HTTP {exc.code} for {model}.") from exc
                try:
                    body = json.loads(exc.read(16384))
                except (OSError, ValueError):
                    body = {}
            problem = body.get("error", body) if isinstance(body, dict) else {}
            if not isinstance(problem, dict):
                problem = {}
            message = str(problem.get("message", "")).casefold()
            param = problem.get("param")
            if any(problem.get(k) in ("exceed_context_size_error", "context_length_exceeded") for k in ("type", "code")):
                count, limit = problem.get("n_prompt_tokens"), problem.get("n_ctx")
                sizes = (f" ({count:,} input tokens, {limit:,} available)"
                         if type(count) is int and type(limit) is int and count > 0 and limit > 0 else "")
                raise ModelContextExceeded(f"{model} rejected the input because it exceeds the context window{sizes}.") from exc
            if "reasoning_effort" in payload and exc.code in (400, 422):
                # A parameter rejected by HTTP validation has not started a generation.
                if param == "reasoning_effort" or ("reasoning_effort" in message and any(word in message for word in
                        ("unsupported", "not supported", "unknown", "unrecognized", "invalid", "unexpected"))):
                    del payload["reasoning_effort"]
                    continue
            # Provider messages can echo private input. Surface only the error category.
            if param == "response_format" or "response_format" in message or "json schema" in message:
                if bonsai and "response_format" in payload and exc.code in (400, 422):
                    raise ModelResponseInvalid(f"{model} rejected the requested JSON schema.") from exc
                if host in COHERE_HOSTS and exc.code == 400:
                    raise ModelUnavailable(f"{model} rejected the requested response format (HTTP 400).") from exc
            raise ModelUnavailable(f"{host} returned HTTP {exc.code} for {model}.") from exc


def chat_json(messages, schema, *, plain_local_retry=False):
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

    local = url.hostname in {"localhost", "127.0.0.1", "::1"}
    bonsai = local and any("bonsai-2-" in name.casefold() for name in (model, config.get("PLANNER_MODEL_KEY") or ""))
    payload = {"model": model, "messages": messages, "temperature": 0}
    if cohere:
        if model in {"c4ai-aya-expanse-8b", "c4ai-aya-vision-8b"}:
            raise ModelUnavailable(f"Cohere retired {model} on April 4, 2026. Choose a hosted model in .env.")
        key = config.get("COHERE_API") or config.get("COHERE_API_KEY")
        if not key:
            raise ModelUnavailable("Set COHERE_API in .env to use the Cohere endpoint.")
        headers["Authorization"] = f"Bearer {key}"
    aya = cohere and model.startswith(("c4ai-aya-", "tiny-aya-"))
    if bonsai or aya:
        # Prompt the same contract when the runtime uses grammar-constrained or ordinary chat.
        instruction = "Return only one JSON object matching this schema:\n" + json.dumps(schema)
        conversation = list(messages)
        if conversation and conversation[0].get("role") in {"system", "developer"}:
            first = conversation.pop(0)
            instruction = first["content"] + "\n\n" + instruction
        payload["messages"] = [{"role": "system", "content": instruction}, *conversation]
    if bonsai:
        payload.update(temperature=0.7, top_p=0.8, reasoning_effort="none")
    if not aya and not (bonsai and plain_local_retry):
        # Both LM Studio and Cohere's OpenAI compatibility endpoint accept this envelope.
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "planner_response", "strict": True, "schema": schema},
        }

    # Only the explicitly configured endpoint receives the prompt. Loopback
    # traffic bypasses system web proxies; hosted requests honor HTTPS_PROXY.
    handlers = [_NoRedirect()]
    if local:
        handlers.append(ProxyHandler({}))
    client = build_opener(*handlers)
    try:
        with local_model_runtime.managed_model(endpoint, model, config, kind="llm", timeout=timeout) as instance:
            payload["model"] = instance
            completion = _post_completion(client, endpoint, headers, payload, timeout, model)
        choice = completion["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ModelResponseTruncated(f"The endpoint stopped {model} at its response length limit.")
        if local:
            result = _final_json(choice["message"], model)
        else:
            content = choice["message"]["content"].strip()
            if cohere and content.startswith(("```json\n", "```\n")) and content.endswith("```"):
                content = content.split("\n", 1)[1][:-3].strip()
            result = json.loads(content)
        validate(result, schema)
        return _canonical_integers(result, schema)
    except OSError as exc:
        # urllib can surface a read timeout directly or wrap a connection timeout in URLError.reason.
        if isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError):
            raise ModelUnavailable(f"The request to {model} timed out after {timeout:g} seconds. "
                                   "Retry when the endpoint is less busy or increase PLANNER_MODEL_TIMEOUT_SECONDS.") from exc
        raise ModelUnavailable(f"{model} is unavailable.") from exc
    except ValidationError as exc:
        # ValidationError.message can contain the entire response. Limit the
        # retry hint to a field path and failed rule, so it cannot echo task text.
        field = ".".join(map(str, exc.absolute_path)) or "response"
        raise ModelResponseInvalid(f"{model} returned an invalid {field} field ({exc.validator}).") from exc
    except (KeyError, IndexError, ValueError, TypeError, AttributeError) as exc:
        raise ModelResponseInvalid(f"{model} returned an incomplete or invalid JSON object.") from exc
