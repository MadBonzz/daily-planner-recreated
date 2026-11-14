"""Local PDF text extraction and bounded passages for the model."""

import io
import re

from pypdf import PdfReader
from pypdf.errors import PyPdfError


_READ_ERRORS = (PyPdfError, OSError, ValueError, TypeError, KeyError, IndexError,
                RecursionError, NotImplementedError)
_SCOPE = re.compile(r"\b(?:implement|derive|prove|compute|design|debug|benchmark|evaluate|"
                    r"measure|compare|analy[sz]e|explain|deliverables?|submit|report|"
                    r"experiment|results?|theorem|proof|methods?)\b", re.I)
_STOP = frozenset("with from into have work through this that assignment paper task read "
                  "about doing complete section pages problem should".split())


def extract_pdf_text(data: bytes) -> str:
    """Extract every readable page locally. Keep page labels for later reference."""
    if not data:
        return ""
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            return ""
        pages = []
        for number, page in enumerate(reader.pages, 1):
            try:
                text = " ".join((page.extract_text() or "").split())
            except _READ_ERRORS:
                continue
            if text:
                pages.append(f"Page {number}: {text}")
        return "\n".join(pages)
    except _READ_ERRORS:
        return ""


def _terms(text):
    return {s for s in re.findall(r"[^\W_]+", text.casefold()) if len(s) > 2 and s not in _STOP}


def document_excerpt(task, limit):
    """Select passages by task relevance and coverage; preserve source page order."""
    text = (task.get("pdf_text") or "").strip()
    if len(text) <= limit:
        return text
    query = " ".join(task.get(k) or "" for k in ("title", "description", "details"))
    terms = _terms(query)
    passages = []
    # Retain page/attachment boundaries, including legacy stored page excerpts.
    for line in text.splitlines():
        header, sep, body = line.partition(": ")
        if not sep or not re.fullmatch(r"Page \d+", header):
            header, body = "", line
        for match in re.finditer(r".{1,205}(?:\s+|$)|\S{1,205}", body):
            snippet = match.group().strip()
            if not snippet:
                continue
            overlap = len(terms & _terms(snippet))
            score = 2 * overlap + min(3, len(_SCOPE.findall(snippet)))
            passages.append((score, f"{header}: {snippet}" if header else snippet))
    if not passages:
        return ""
    slots = min(len(passages), max(1, limit // 225))
    # Reserve the strongest match anywhere. Fill the remaining slots from
    # different sections so a long preamble or a repeated topic cannot dominate.
    best = max(range(len(passages)), key=lambda i: passages[i][0])
    chosen = {best}
    if slots > 1:
        for i in range(slots):
            start, end = i * len(passages) // slots, (i + 1) * len(passages) // slots
            if start <= best < end:
                continue
            chosen.add(max(range(start, end), key=lambda j: passages[j][0]))
    return "\n".join(passages[i][1] for i in sorted(chosen))[:limit]
