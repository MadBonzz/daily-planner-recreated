"""Validated chat completions from LM Studio or a configured compatible endpoint."""

import codecs
import json
import math
import os
import re
import shlex
from contextlib import contextmanager
from contextvars import ContextVar
from http.client import IncompleteRead
from io import StringIO
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
    "PLANNER_SINGLE_MODEL", "PLANNER_PREFER_PRECOMPUTED", "PLANNER_MODEL_KEY", "PLANNER_DECISION_MODEL_KEY", "PLANNER_EMBEDDING_MODEL_KEY",
    "PLANNER_CHAT_CONTEXT_LENGTH", "PLANNER_DECISION_CONTEXT_LENGTH", "PLANNER_EMBEDDING_CONTEXT_LENGTH",
}


class ModelContextExceeded(ModelUnavailable):
    """The provider rejected an oversized input before producing a completion."""


class ModelResponseInvalid(ModelUnavailable):
    """A completed HTTP response did not supply JSON matching the requested contract."""


class ModelResponseTruncated(ModelResponseInvalid):
    """The endpoint explicitly reported that generation exhausted the response budget."""


def _file_settings():
    try:
        raw = ENV_FILE.read_bytes()
    except FileNotFoundError:
        return {}
    encoding = "utf-16" if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else "utf-8-sig"
    contents = raw.decode(encoding)
    if dotenv_values is not None:
        return dotenv_values(stream=StringIO(contents), interpolate=False)
    # Older installations may not yet have python-dotenv.
    result = {}
    for line in contents.splitlines():
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
    endpoint = config.get("PLANNER_MODEL_ENDPOINT") or API_URL
    if model := config.get("PLANNER_MODEL"):
        return endpoint, model
    try:
        hosted = urlsplit(endpoint).hostname in COHERE_HOSTS
    except ValueError:
        hosted = False  # chat_json reports malformed endpoint URLs.
    return endpoint, "command-r7b-12-2024" if hosted else MODEL_ID


_forecast_config = ContextVar("forecast_config", default=None)


def _local_bonsai_names(config):
    endpoint, name = model_settings(config)
    if urlsplit(endpoint).hostname not in {"localhost", "127.0.0.1", "::1"}:
        return ()
    return tuple(value.casefold() for value in (name, config.get("PLANNER_MODEL_KEY"))
                 if value and "bonsai-2-" in value.casefold())


def using_local_bonsai():
    return bool(_local_bonsai_names(_forecast_config.get() or planner_settings()))


def plain_chat_reply(generate, messages, schema):
    """Use the existing repair attempt with the original evidence and a complete final answer."""
    correction = ("Answer in the final assistant message with one complete JSON object. "
                  "Fill the required fields using the supplied data. Return only that object.")
    first, *rest = messages
    repaired = [{**first, "content": first["content"] + "\n" + correction}, *rest]
    if using_local_bonsai():
        return generate(repaired, schema, plain_local_retry=True)
    return generate(repaired, schema)


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
    if not isinstance(message, dict):
        raise ModelResponseInvalid(f"The chat endpoint returned a choice without an assistant message for {model}.")
    content = message.get("content")
    if message.get("refusal") or (isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "refusal" for part in content)):
        raise ModelResponseInvalid(f"{model} returned a refusal instead of a final JSON answer.")
    if isinstance(content, dict):
        return content
    if content is None and isinstance(message.get("parsed"), dict):
        return message["parsed"]
    if isinstance(content, list):
        content = "".join(part["text"] for part in content if isinstance(part, dict)
                          and part.get("type") in ("text", "output_text") and isinstance(part.get("text"), str))
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


def _schema_prompt(messages, schema):
    instruction = "Return only one JSON object matching this schema:\n" + json.dumps(schema)
    conversation = list(messages)
    if conversation and conversation[0].get("role") in {"system", "developer"}:
        first = conversation.pop(0)["content"]
        if not first.endswith(instruction):
            first += "\n\n" + instruction
        instruction = first
    return [{"role": "system", "content": instruction}, *conversation]


def _post_completion(client, endpoint, headers, payload, timeout, model):
    host = urlsplit(endpoint).hostname
    while True:
        request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with client.open(request, timeout=timeout) as response:
                return json.load(response)
        except IncompleteRead:
            raise ModelResponseInvalid(f"The chat endpoint closed the HTTP response before it was complete for {model}.") from None
        except ValueError:
            raise ModelResponseInvalid(f"The chat endpoint returned an HTTP response that could not be decoded as JSON for {model}.") from None
        except HTTPError as exc:
            with exc:
                if host in COHERE_HOSTS and exc.code in (401, 403, 429):
                    hint = {401: "Check COHERE_API in the repository-root .env.",
                            403: "Check that your Cohere key has access to this model.",
                            429: "Cohere rate-limited this request. Retry after your account limit resets."}[exc.code]
                    raise ModelUnavailable(f"{host} returned HTTP {exc.code} for {model}. {hint}") from exc
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
                if "response_format" in payload and exc.code in (400, 422):
                    # The server rejected the constraint before starting inference.
                    # Send the same contract in the prompt and validate it locally.
                    schema = payload.pop("response_format")["json_schema"]["schema"]
                    payload["messages"] = _schema_prompt(payload["messages"], schema)
                    continue
                raise ModelUnavailable(f"{model} rejected the requested response format (HTTP {exc.code}).") from exc
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
    bonsai = _local_bonsai_names(config)
    payload = {"model": model, "messages": messages, "temperature": 0}
    if cohere:
        if model in {"c4ai-aya-expanse-8b", "c4ai-aya-vision-8b"}:
            raise ModelUnavailable(f"Cohere retired {model} on April 4, 2026. Choose a hosted model in .env.")
        key = config.get("COHERE_API") or config.get("COHERE_API_KEY")
        if not key:
            raise ModelUnavailable("Set COHERE_API in .env to use the Cohere endpoint.")
        headers["Authorization"] = f"Bearer {key}"
    aya = cohere and model.startswith(("c4ai-aya-", "tiny-aya-"))
    local_command_r = local and any(
        re.search(r"(?:^|/)(?:c4ai-)?command-r(?:7b|[-+]|$)", name.casefold())
        for name in (model, config.get("PLANNER_MODEL_KEY") or "")
    )
    if local_command_r:
        # The native Command R template can append its turn delimiter after a
        # valid JSON answer. Stop there before the server parses the response.
        payload["stop"] = ["<|END_OF_TURN_TOKEN|>"]
    if bonsai or aya or local_command_r:
        # Prompt the same contract when the runtime uses grammar-constrained or ordinary chat.
        payload["messages"] = _schema_prompt(messages, schema)
    if bonsai:
        payload.update(temperature=0.7, top_p=0.8, reasoning_effort="none")
    # On Bonsai 2 MLX grammar constraints can consume the answer in the reasoning
    # channel. Its system message already contains the full schema for local validation.
    if not aya and not (bonsai and (plain_local_retry or any("mlx" in name for name in bonsai))):
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
        if not isinstance(completion, dict):
            raise ModelResponseInvalid(f"The chat endpoint returned JSON without a chat completion object for {model}.")
        if completion.get("error") is not None:
            raise ModelResponseInvalid(f"The chat endpoint returned an error object instead of a completion for {model}.")
        choices = completion.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelResponseInvalid(f"The chat endpoint returned a completion without a nonempty choices array for {model}.")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ModelResponseInvalid(f"The chat endpoint returned an invalid first choice for {model}.")
        if choice.get("finish_reason") == "content_filter":
            raise ModelResponseInvalid(f"The chat endpoint filtered the response for {model}.")
        try:
            result = _final_json(choice.get("message"), model)
            validate(result, schema)
        except (ModelResponseInvalid, ValidationError) as exc:
            if choice.get("finish_reason") == "length":
                raise ModelResponseTruncated(f"The endpoint stopped {model} at its response length limit.") from exc
            raise
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
        if exc.validator == "required" and isinstance(exc.instance, dict):
            missing = [name for name in exc.validator_value if name not in exc.instance]
            raise ModelResponseInvalid(f"{model} omitted required fields: {', '.join(missing)}.") from exc
        field = ".".join(map(str, exc.absolute_path)) or "response"
        raise ModelResponseInvalid(f"{model} returned an invalid {field} field ({exc.validator}).") from exc
    except (KeyError, IndexError, ValueError, TypeError, AttributeError) as exc:
        raise ModelResponseInvalid(f"{model} returned an incomplete or invalid JSON object.") from exc
