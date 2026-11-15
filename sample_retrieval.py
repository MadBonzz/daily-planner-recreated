"""Load packaged results of exact local model requests for the public sample tasks."""

import base64
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import struct

DIRECTORY = Path(__file__).resolve().parent / "data" / "tasks" / "retrieval"
FORMAT = 1


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def decision_hash(payload):
    return text_hash(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


@lru_cache(maxsize=1)
def _indexes():
    vectors, decisions = {}, {}
    for path in sorted(DIRECTORY.glob("*.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        if bundle["format"] != FORMAT:
            continue
        model = bundle["embedding_model"]
        for task in bundle["vectors"].values():
            for vector in task.values():
                vectors[model, vector["input_sha256"]] = vector["f32le_base64"]
        for key, row in bundle["decisions"].items():
            decisions[key] = row["related"]
    return vectors, decisions


def packed_vector(vector):
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


def unpack_vector(encoded):
    data = base64.b64decode(encoded, validate=True)
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def vector_for(model, text):
    encoded = _indexes()[0].get((model, text_hash(text)))
    return unpack_vector(encoded) if encoded is not None else None


def decision_for(payload):
    return _indexes()[1].get(decision_hash(payload))
