"""Serialize planner requests to a single resident model on a local LM Studio server."""

import json
import os
import stat
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import portalocker


class ModelUnavailable(RuntimeError):
    """The configured endpoint could not produce a usable response."""


class _ManagementError(ModelUnavailable):
    def __init__(self, url, status, error):
        message = error.get("message") if isinstance(error, dict) else error
        self.detail = message if isinstance(message, str) else ""
        self.code = error.get("code") if isinstance(error, dict) else None
        self.status = status
        super().__init__(f"LM Studio returned HTTP {status} at {url}. {self.detail[:400]}".rstrip())


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_ROLES = {
    "embedding": ("PLANNER_EMBEDDING_MODEL_KEY", "PLANNER_EMBEDDING_CONTEXT_LENGTH", 2048),
    "decision": ("PLANNER_DECISION_MODEL_KEY", "PLANNER_DECISION_CONTEXT_LENGTH", 4096),
    "llm": ("PLANNER_MODEL_KEY", "PLANNER_CHAT_CONTEXT_LENGTH", 8192),
}
_active = threading.local()


def is_managed(endpoint, config):
    if config.get("PLANNER_SINGLE_MODEL") != "1":
        return False
    url = urlsplit(endpoint)
    return (url.scheme in {"http", "https"} and url.hostname in {"127.0.0.1", "::1", "localhost"}
            and url.username is None and url.password is None)


def _api(client, url, timeout, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with client.open(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        # These management endpoints receive only a model key, instance ID and load settings.
        try:
            error = json.load(exc).get("error", {})
        except (ValueError, AttributeError, OSError):
            error = None
        finally:
            exc.close()
        raise _ManagementError(url, exc.code, error) from exc
    except (OSError, ValueError) as exc:
        raise ModelUnavailable(f"Could not read the local LM Studio model management response at {url}: {exc}") from exc


@contextmanager
def _port_lock(port):
    # A persistent filename lets independent UI, estimator and report processes
    # lock the same inode. Closing the handle releases the lock even after a crash.
    uid = os.getuid() if hasattr(os, "getuid") else None
    name = "daily-planner-runtime" if uid is None else f"daily-planner-runtime-{uid}"
    directory = Path(tempfile.gettempdir()) / name
    try:
        directory.mkdir(mode=0o700, exist_ok=True)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or (uid is not None and
                (info.st_uid != uid or info.st_mode & 0o077)):
            raise OSError(f"Expected a private lock directory at {directory}")
        handle = (directory / f"{port}.lock").open("a+b")
    except OSError as exc:
        raise ModelUnavailable(f"Could not open the local model lock: {exc}") from exc
    with handle:
        try:
            portalocker.lock(handle, portalocker.LOCK_EX)
        except (portalocker.exceptions.LockException, OSError) as exc:
            raise ModelUnavailable(f"Could not lock the local model server on port {port}.") from exc
        yield


def _installed_model(models, requested, explicit_key, kind):
    candidates = [m for m in models if m.get("type") == kind]
    matches = [m for m in candidates if m["key"] == (explicit_key or requested)]
    if not matches and not explicit_key:
        matches = [m for m in candidates if any(i["id"] == requested for i in m["loaded_instances"])]
    if not matches and not explicit_key:
        # Live IDs also distinguish local results from packaged values in planner caches.
        def stable_name(name):
            return name.rsplit("/", 1)[-1].casefold().removeprefix("text-embedding-").removesuffix("-gguf")
        wanted = stable_name(requested.removesuffix("-live"))
        matches = [m for m in candidates if stable_name(m["key"]) == wanted]
    if len(matches) != 1:
        setting = _ROLES[kind][0]
        reason = "multiple installed models match" if matches else "no installed model matches"
        raise ModelUnavailable(f"LM Studio: {reason} {explicit_key or requested!r} ({kind}). "
                               f"Set {setting} to the exact key from /api/v1/models.")
    return matches[0]


def _prepare(origin, model, key, kind, context, timeout):
    client = build_opener(ProxyHandler({}), NoRedirect())
    url = origin + "/api/v1/models"
    inventory = _api(client, url, timeout)
    try:
        models = inventory["models"]
        if not isinstance(models, list):
            raise TypeError
        # Validate the inventory before changing it. Silently skipping a malformed
        # entry could leave an unknown model resident while a new one loads.
        for entry in models:
            if not isinstance(entry["key"], str) or not isinstance(entry["loaded_instances"], list):
                raise TypeError
            for instance in entry["loaded_instances"]:
                if not isinstance(instance["id"], str) or not instance["id"]:
                    raise TypeError
        target = _installed_model(models, model, key, kind)
        ready = []
        for instance in target["loaded_instances"]:
            settings = instance.get("config") or {}
            if settings.get("context_length") == context and (kind != "llm" or settings.get("parallel", 1) == 1):
                ready.append(instance["id"])
        keep = model if model in ready else next(iter(ready), None)
        resident = {i["id"] for entry in models for i in entry["loaded_instances"]}
    except (KeyError, TypeError, AttributeError) as exc:
        raise ModelUnavailable(f"LM Studio returned an invalid model inventory at {url}.") from exc

    for instance_id in sorted(resident - {keep}):
        _api(client, url + "/unload", timeout, {"instance_id": instance_id})
    if keep is not None:
        return keep
    settings = {"model": target["key"], "context_length": context}
    if kind == "llm":
        settings["parallel"] = 1
    try:
        loaded = _api(client, url + "/load", timeout, settings)
    except _ManagementError as exc:
        detail = exc.detail.casefold()
        if not (kind == "llm" and exc.status == 400 and "parallel" in detail and
                (exc.code == "unrecognized_keys" or any(word in detail for word in
                 ("unrecognized", "unrecognised", "unsupported", "unknown", "not supported", "unexpected")))):
            raise
        # Older native APIs can still configure the context but lack this LLM option.
        del settings["parallel"]
        loaded = _api(client, url + "/load", timeout, settings)
    if not isinstance(loaded, dict) or not isinstance(loaded.get("instance_id"), str) or not loaded["instance_id"]:
        raise ModelUnavailable(f"LM Studio did not return a loaded instance ID for {target['key']}.")
    return loaded["instance_id"]


@contextmanager
def managed_model(endpoint, model, config, *, kind, timeout):
    """Hold the server lock through inference and yield the actual API instance ID."""
    if not is_managed(endpoint, config):
        yield model
        return
    url = urlsplit(endpoint)
    origin = f"{url.scheme}://{url.netloc}"
    port = url.port or (443 if url.scheme == "https" else 80)
    key_setting, context_setting, default_context = _ROLES[kind]
    try:
        context = int(config.get(context_setting) or default_context)
        if context < 1:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ModelUnavailable(f"{context_setting} must be a positive integer.") from exc
    key = config.get(key_setting)
    request = (kind, model, key, context)
    if not hasattr(_active, "servers"):
        _active.servers = {}
    if port in _active.servers:
        previous, instance = _active.servers[port]
        if request != previous:
            raise ModelUnavailable("Cannot switch local models while this thread has an active inference request.")
        yield instance
        return

    with _port_lock(port):
        instance = _prepare(origin, model, key, kind, context, timeout)
        _active.servers[port] = request, instance
        try:
            yield instance
        finally:
            del _active.servers[port]
