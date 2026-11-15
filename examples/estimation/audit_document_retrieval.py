"""Run controlled document perturbations through upload, parsing, Qdrant, Qwen and Kev.

Run with python -m examples.estimation.audit_document_retrieval OUTPUT.json.
Only the four model-specific endpoints/settings used by the app are relevant.
"""
import argparse
import hashlib
import json
import math
import tempfile
from unittest.mock import patch
import textwrap
from io import BytesIO
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import estimate_worker
import retrieval
from planner_store import PlannerStore
from model_client import ModelUnavailable
from pdf_reader import extract_pdf_text


# Each handout contains two independent deliverables, much like a combined lab assignment.
PACKETS = {
    "development": (
        ("kernel", "Implement the segmented prefix sum", "Write a CUDA warp-shuffle kernel for segmented prefix sums. Check tile boundaries against the CPU reference, collect Nsight throughput on three input shapes, and submit the kernel and performance table."),
        ("interview", "Interview the pantry volunteer", "Conduct a recorded interview with a pantry volunteer about grocery pickup. Transcribe the recording, code five recurring themes, and submit an anonymized evidence table.")
    ),
    "validation": (
        ("circuits", "Characterize the amplifier", "Wire an op-amp breadboard for the specified gain. Calibrate the oscilloscope probes, sweep a signal generator across ten frequencies and plot the measured gain and phase response."),
        ("translate", "Translate the oral history", "Translate the archival oral-history recording from Hindi to English. Resolve dialect terms using the glossary, align timestamps to the transcript, and submit a bilingual annotated passage.")
    )
}
GENERIC = ("Complete the attached exercise", "Finish the assigned work in the attachment.")
INTRO = ("Course packet. Different students are assigned independent sections of this packet. "
         "Consult the task description to determine the assigned section. Keep a copy for your records.")


def pdf(pages):
    writer = PdfWriter()
    for paragraph in pages:
        page = writer.add_blank_page(595, 842)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        lines = ["BT /F1 10 Tf 40 795 Td 14 TL"]
        for line in textwrap.wrap(paragraph, width=94):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            lines.append(f"({escaped}) Tj T*")
        stream = DecodedStreamObject()
        stream.set_data(("\n".join(lines) + "\nET").encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def metrics(rows, key):
    tp = sum(r["expected"] and r[key] for r in rows)
    fp = sum(not r["expected"] and r[key] for r in rows)
    fn = sum(r["expected"] and not r[key] for r in rows)
    tn = sum(not r["expected"] and not r[key] for r in rows)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None)


def evaluate(store, directory, variant, parts, vectors, prepare):
    packet = pdf([INTRO, parts[0][2], parts[1][2]])
    singles = [pdf([s[2]]) for s in parts]

    def upload(title, description, details, data, category="Assignments", filename="assignment.pdf"):
        row = store.create_task(title=title, description=description, category=category, details=details,
                                attachments=[(filename, data)] if data else [],
                                day="2026-11-15", estimate_minutes=45)
        _, unreadable = estimate_worker._read_documents(store, row)
        row["unreadable"] = unreadable
        return row

    def scoped(i, data, **kw):
        return upload(parts[i][1], parts[i][2], "Complete only my assigned deliverable.", data, **kw)

    # Both historical submissions used the same full course packet. Their declared work differs.
    sources = [scoped(i, packet) for i in range(2)]
    queries = []
    for i, (_, title, desc) in enumerate(parts):
        queries.extend([
            (i, "change-only-pdf", upload(*GENERIC, "", singles[i])),
            (i, "change-prose-keep-pdf", scoped(i, packet)),
            (i, "remove-pdf", scoped(i, None)),
            (i, "category-only", scoped(i, packet, category="Personal")),
            (i, "rename-file-only", scoped(i, packet, filename="reader-copy.pdf")),
            (i, "details-only", upload(*GENERIC, title + ". " + desc, packet)),
            (i, "page-reference-only", upload(*GENERIC, f"Only do page {i+2}.", packet)),
        ])
    pairs, vector_cache, decisions = [], {}, {}
    config = retrieval.planner_settings()
    def embedding(row, query):
        key = (retrieval.task_text(row), query)
        if key not in vector_cache:
            vector_cache[key] = retrieval.embed_task(row, config, query=query)
        return vector_cache[key]

    if prepare:
        for row in [*sources, *(query[2] for query in queries)]:
            for role in (False, True):
                content = retrieval.embedding_input(row, role)
                key = retrieval.sample_retrieval.text_hash(content)
                if key not in vectors:
                    vectors[key] = embedding(row, role)
        return []

    for expected_index, change, target in queries:
        candidates = {r["id"]: float(s) for r, s in retrieval.search(target, sources, directory)}
        for index, source in enumerate(sources):
            payload = retrieval.decision_payload(target, source, [], retrieval.DECISION_MODEL)
            # Score Kev even for gate misses so that its errors can be separated from Qwen's.
            key = retrieval.sample_retrieval.decision_hash(payload)
            if key not in decisions:
                decisions[key] = retrieval.decide_related(target, source)
            q, d = embedding(target, True), embedding(source, False)
            cosine = sum(a*b for a,b in zip(q,d))/(math.hypot(*q)*math.hypot(*d))
            admitted = source["id"] in candidates
            pairs.append(dict(split=variant, change=change, query_scope=parts[expected_index][0],
                              source=parts[index][0], expected=index == expected_index,
                              instructed_cosine=round(cosine,6), retrieved=admitted, kev=decisions[key],
                              selected=admitted and decisions[key], pdf_sha256=hashlib.sha256(
                                  store.attachment_bytes(target["attachments"][0]["id"])).hexdigest()
                              if target.get("attachments") else None))
    return pairs


def index_mutations(path, vectors):
    """Exercise replacement under the same Qdrant ID, using real recorded Qwen vectors."""
    state = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)/"vectors"
        base = {"id": "same-task", "title": GENERIC[0], "description": GENERIC[1],
                "category": "Study", "estimate_minutes": 45, "status": "completed"}
        first_doc, second_doc = [extract_pdf_text(pdf([text])) for _, _, text in PACKETS["development"]]
        first = {**base, "pdf_text": first_doc}
        replacement = {**base, "pdf_text": second_doc}
        def read():
            with retrieval._index(root) as client:
                rows, _ = client.scroll(retrieval.COLLECTION, with_vectors=True)
            assert len(rows) == 1
            return rows[0]
        def digest(vector):
            return hashlib.sha256(json.dumps(vector).encode()).hexdigest()
        def cached(model, content):
            assert model == retrieval.EMBEDDING_MODEL
            return vectors[retrieval.sample_retrieval.text_hash(content)]
        with patch.object(retrieval.sample_retrieval, "vector_for", cached):
            retrieval.index_task(root, first)
            original = read()
            # A failed embedding for the edited document cannot return the stale match.
            lookup = retrieval.embed_task
            def unavailable(row, config=None, query=False):
                if not query and row["id"] == base["id"]:
                    raise ModelUnavailable("Injected embedding outage")
                return lookup(row, config, query)
            errors = set()
            with patch.object(retrieval, "embed_task", unavailable):
                hits = retrieval.search({**first, "id": "new-query"}, [replacement], root, errors=errors)
            assert not hits and errors
            state.append({"step": "embedding outage during replacement", "stale_hits": len(hits), "errors": sorted(errors)})
            retrieval.index_task(root, replacement)
            edited = read()
            assert original.id == edited.id and digest(original.vector) != digest(edited.vector)
            assert edited.payload["content_hash"] == retrieval._payload(replacement)["content_hash"]
            state.append({"step": "replace PDF, same ID and metadata", "before_vector": digest(original.vector),
                          "after_vector": digest(edited.vector), "same_point_id": True})
            metadata = {**replacement, "actual_seconds": 4200}
            with patch.object(retrieval, "embed_task", side_effect=AssertionError("unnecessary inference")):
                retrieval.index_task(root, metadata)
            updated = read()
            assert digest(updated.vector) == digest(edited.vector) and updated.payload["actual_seconds"] == 4200
            state.append({"step": "change timing only", "actual_seconds": updated.payload["actual_seconds"],
                          "unchanged_vector": True, "embedding_calls": 0})
            retrieval.forget_task(root, base["id"])
            with retrieval._index(root) as client:
                count = client.count(retrieval.COLLECTION).count
            assert count == 0
            state.append({"step": "forget task", "remaining_points": count})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--split", choices=PACKETS, action="append")
    p.add_argument("--vectors", type=Path, help="Exact-input cache of real Qwen vectors for two-phase CPU evaluation")
    p.add_argument("--prepare-vectors", action="store_true", help="Populate --vectors and exit before judging")
    p.add_argument("--index-mutations", action="store_true", help="Check Qdrant replacement, outage, timing edit and purge using --vectors")
    args = p.parse_args()
    if args.prepare_vectors and not args.vectors:
        p.error("--prepare-vectors requires --vectors")
    vectors = json.loads(args.vectors.read_text()) if args.vectors and args.vectors.exists() else {}
    if args.index_mutations:
        if not vectors:
            p.error("--index-mutations requires recorded --vectors")
        index_mutations(args.output, vectors)
        return
    def cached(model, content):
        if model != retrieval.EMBEDDING_MODEL:
            raise ValueError("Unexpected embedding model")
        return vectors[retrieval.sample_retrieval.text_hash(content)]
    records = []
    for split in args.split or PACKETS:
        with tempfile.TemporaryDirectory() as tmp:
            store = PlannerStore(Path(tmp)/"store")
            if args.vectors and not args.prepare_vectors:
                with patch.object(retrieval.sample_retrieval, "vector_for", cached):
                    records.extend(evaluate(store, Path(tmp)/"index", split, PACKETS[split], vectors, False))
            else:
                records.extend(evaluate(store, Path(tmp)/"index", split, PACKETS[split], vectors, args.prepare_vectors))
        if args.prepare_vectors:
            args.vectors.write_text(json.dumps(vectors))
            print(split, len(vectors), "distinct vectors", flush=True)
            continue
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"pairs": records, "metrics": {
            key: metrics(records, key) for key in ["retrieved","kev","selected"]}}, indent=2)+"\n")
        print(split, len(records), flush=True)


if __name__ == "__main__":
    main()
