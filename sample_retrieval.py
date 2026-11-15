"""Load exact model results for the public samples. Qwen vectors are ordinary numeric CSVs."""

import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from fallback_provenance import record_packaged_fallback
from local_model_runtime import ModelUnavailable

DIRECTORY = Path(__file__).resolve().parent / "data" / "tasks" / "retrieval"
FORMAT = 2
DIMENSIONS = 1024
ROLES = ("document", "query")
QWEN = "qwen3-embedding-0.6b"
KEV = "kev-4b"
_MODELS = {
    "embedding": (QWEN, "PLANNER_EMBEDDING_MODEL_KEY",
                  {QWEN, "text-embedding-qwen3-embedding-0.6b", "qwen3-embedding-0.6b-gguf",
                   "qwen/qwen3-embedding-0.6b-gguf", "qwen/qwen3-embedding-0.6b"}),
    "decision": (KEV, "PLANNER_DECISION_MODEL_KEY", {KEV, "kev-4b-gguf", "ggml-org/kev-4b-gguf", "ggml-org/kev-4b"}),
}
_NUMBER = re.compile(r"-?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?", re.ASCII)


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def decision_hash(payload):
    return text_hash(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def packaged_model(role, model, config):
    """Only compare requests against the archived weights for that component."""
    canonical, key_setting, installed_keys = _MODELS[role]
    physical = config.get(key_setting)
    if physical and physical.casefold() not in installed_keys:
        return None
    if physical and config.get("PLANNER_SINGLE_MODEL") == "1":
        # Native management maps the logical alias to this specific installed key.
        return canonical
    if model.casefold() in installed_keys or model == canonical + "-live":
        return canonical
    return None


def prefer_packaged(role, model, config):
    use_saved = config.get("PLANNER_SINGLE_MODEL") != "1" or config.get("PLANNER_PREFER_PRECOMPUTED") == "1"
    return use_saved and packaged_model(role, model, config) == model


def resolve(role, model, config, lookup, infer):
    """Return the result and source. A live failure may reuse only a matched input."""
    canonical = packaged_model(role, model, config)
    package_first = prefer_packaged(role, model, config)
    if package_first:
        hit = lookup(canonical)
        if hit is not None:  # A false Kev decision is a complete result.
            return hit, f"Packaged {canonical} {role}"
    try:
        return infer(), model
    except ModelUnavailable:
        if canonical is not None and not package_first:
            try:
                hit = lookup(canonical)
            except (OSError, ValueError):
                # An unreadable or invalid bundled file cannot satisfy this request.
                hit = None
            if hit is not None:
                record_packaged_fallback(role)
                return hit, f"Packaged {canonical} {role} (local model unavailable)"
        raise


def csv_path(index_path, task_id):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", task_id):
        raise ValueError("Invalid sample task ID")
    return index_path.parent / index_path.stem / (task_id + ".csv")


def _check_vector(values):
    if len(values) != DIMENSIONS or not all(math.isfinite(x) and abs(x) <= 1 for x in values):
        raise ValueError("Expected 1,024 finite Qwen components in [-1, 1].")
    if not math.isclose(math.hypot(*values), 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError("Expected a unit-normalized Qwen embedding.")


def write_vectors(path, vectors):
    """Emit canonical decimal representations of the model's binary32 components."""
    columns = []
    for role in ROLES:
        values = vectors[role]
        _check_vector(values)
        # Preserve the model's single precision; no rounding for display or additional quantization.
        columns.append(struct.unpack(f"<{DIMENSIONS}f", struct.pack(f"<{DIMENSIONS}f", *values)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(ROLES)
        writer.writerows(zip(*columns, strict=True))
    temp.replace(path)
    _columns.cache_clear()


@lru_cache(maxsize=160)
def _columns(path):
    columns = [[], []]
    with path.open(encoding="ascii", newline="") as file:
        reader = csv.reader(file, strict=True)
        if next(reader, []) != list(ROLES):
            raise ValueError(f"Expected document,query CSV header in {path.name}")
        for row in reader:
            if (len(columns[0]) >= DIMENSIONS or len(row) != 2 or
                    not all(_NUMBER.fullmatch(field) for field in row)):
                raise ValueError(f"Invalid numeric embedding CSV in {path.name}")
            for column, field in zip(columns, row, strict=True):
                value = float(field)
                if (not math.isfinite(value) or abs(value) > 1 or repr(value) != field or
                        struct.unpack("<f", struct.pack("<f", value))[0] != value):
                    raise ValueError(f"Invalid canonical binary32 component in {path.name}")
                column.append(value)
    for column in columns:
        _check_vector(column)
    return dict(zip(ROLES, columns, strict=True))


def read_vectors(index_path, task_id):
    return _columns(csv_path(index_path, task_id))


@lru_cache(maxsize=1)
def _indexes():
    vectors, decisions = {}, {}
    paths = sorted(DIRECTORY.glob("*.json"))
    demo = DIRECTORY.parent / "demo" / "models" / "retrieval.json"
    if demo.exists():
        paths.append(demo)
    for path in paths:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        if bundle["format"] != FORMAT:
            continue
        model = bundle["embedding_model"]
        for task_id, record in bundle["vectors"].items():
            for role in ROLES:
                vectors[model, record[role]["input_sha256"]] = (csv_path(path, task_id), role)
        for key, row in bundle["decisions"].items():
            if type(row["related"]) is not bool:
                raise ValueError("Expected a boolean Kev decision.")
            decisions[key] = row["related"]
    return vectors, decisions


def vector_for(model, text):
    hit = _indexes()[0].get((model, text_hash(text)))
    if hit is None:
        return None
    path, role = hit
    return list(_columns(path)[role])


def decision_for(payload):
    return _indexes()[1].get(decision_hash(payload))
