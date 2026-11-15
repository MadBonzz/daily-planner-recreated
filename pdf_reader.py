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


_PAGE_PREFIX = r"(?:pages?\s+|p{1,2}\.\s*)"
_PAGE_JOIN = r"(?:[-–—/&]|,\s*(?:and\b|or\b)?|\b(?:to|through|and|or)\b)"
_PAGE_SCOPE = re.compile(r"\b" + _PAGE_PREFIX + r"(\d+(?:\s*" + _PAGE_JOIN
                         + r"\s*(?:" + _PAGE_PREFIX + r")?\d+)*)", re.I)

# A mention such as "not only page 2" asks for a broader search. Removing other
# pages here would change the assigned work before the model ever sees it.
_BROAD_SCOPE = re.compile(r"\b(?:not\s+(?:only|just|(?:limited|restricted)\s+to)|"
                          r"(?:more\s+than|beyond)\s+(?:only|just))\b", re.I)


def requested_pages(question):
    # References used as a starting point for a search elsewhere are not a page restriction.
    if _BROAD_SCOPE.search(question) or re.search(r"\b(?:elsewhere|where else|onward)\b|"
                 r"\b(?:other than|except|excluding|outside|before|after)\s+(?:for\s+)?(?:page|p\.)",
                 question, re.I):
        return []
    ranges = []
    for scope in _PAGE_SCOPE.finditer(question):
        for first, last in re.findall(r"(\d+)(?:\s*(?:[-–—]|to|through)\s*(\d+))?", scope[1], re.I):
            ranges.append((int(first), int(last or first)))
    return ranges


def _assigned_pages(task, text):
    """When a single handout is explicitly scoped, use the assigned pages in compact model contexts."""
    scope = " ".join(task.get(k) or "" for k in ("title", "description", "details"))
    if len(task.get("attachments") or ()) > 1 or _BROAD_SCOPE.search(scope):
        return text
    restriction = re.compile(r"\b(?:only|just|assigned|limited to|restricted to)\s+"
                            r"(?:(?:read|do|use|cover|summarize|complete|review|the)\s+){0,3}"
                            r"$", re.I)
    ranges = []
    for page_ref in _PAGE_SCOPE.finditer(scope):
        if (restriction.search(scope[:page_ref.start()]) or
                re.match(r"\s+only\b", scope[page_ref.end():], re.I)):
            ranges.extend(requested_pages(page_ref.group()))
    if ranges and not requested_pages(scope):
        return text
    # Some assignments paste the selected exercise verbatim instead of giving
    # its page number. Only narrow on that signal when the user limits the scope.
    limited = bool(re.search(r"\b(?:only|just|assigned)\b", scope, re.I))
    if not ranges and not limited:
        return text
    pages = []
    seen = set()
    normalized_scope = " ".join(scope.casefold().split())
    for line in text.splitlines():
        match = re.fullmatch(r"Page (\d+): (.+)", line)
        if not match or int(match[1]) in seen:
            return text  # Legacy text or multiple attachments with overlapping page numbers.
        number = int(match[1])
        seen.add(number)
        if ranges:
            if any(first <= number <= last for first, last in ranges):
                pages.append(line)
        elif len(_terms(match[2])) >= 8 and " ".join(match[2].casefold().split()) in normalized_scope:
            pages.append(line)
    if pages:
        return "\n".join(pages)
    return "The requested PDF pages have no extracted text." if ranges else text


_SECTION_ID = r"(?:[ivxlcdm]+|[a-z]|[0-9]+(?:\.[0-9]+)*)"
_SECTION = rf"(part|task|section|exercise|problem)\s+({_SECTION_ID})\b"
_ONLY_BEFORE = re.compile(r"\bonly\s+(?:(?:do|read|complete|use)\s+)?" + _SECTION, re.I)
_ONLY_AFTER = re.compile(r"\b" + _SECTION + r"\s+only\b", re.I)
# "Only Part I and II" and "Part I through Part III only" assign multiple
# sections even though the word "only" touches just one of them.
_SECTION_LIST = re.compile(
    r"\b" + _SECTION + r"\s*(?:,\s*(?:and\s+|or\s+)?|[-–—/&]|\b(?:and|or|through|to)\b)\s*"
    r"(?:(?:part|task|section|exercise|problem)\s+)?" + _SECTION_ID + r"\b", re.I)


def assigned_section_text(task):
    """Locate an expressly assigned section in a packet with named sibling headings."""
    raw = (task.get("pdf_text") or "").strip()
    scope = " ".join(task.get(k) or "" for k in ("title", "description", "details"))
    requested = {tuple(part.casefold() for part in m.groups()) for pattern in (_ONLY_BEFORE, _ONLY_AFTER)
                 for m in pattern.finditer(scope)}
    if (len(requested) != 1 or len(task.get("attachments") or ()) > 1
            or _BROAD_SCOPE.search(scope) or _SECTION_LIST.search(scope)):
        return raw
    kind, label = next(iter(requested))
    heading = re.compile(r"\b" + kind + r"\s+(" + _SECTION_ID + r")\s*(?:[·:–—]|\s-\s)", re.I)
    current = None
    headings, lines, seen_pages = [], [], set()
    for line in raw.splitlines():
        page = re.fullmatch(r"Page (\d+): (.*)", line)
        if not page or page[1] in seen_pages:
            return raw
        seen_pages.add(page[1])
        body = page[2]
        start = 0
        for mark in heading.finditer(body):
            if current == label and mark.start() > start:
                lines.append(f"Page {page[1]}: {body[start:mark.start()].strip()}")
            current, start = mark[1].casefold(), mark.start()
            headings.append(current)
        if current == label and body[start:].strip():
            lines.append(f"Page {page[1]}: {body[start:].strip()}")
    # A table of contents can repeat all headings. Preserve general passage
    # selection when the target cannot be localized unambiguously.
    return "\n".join(lines) if len(set(headings)) > 1 and headings.count(label) == 1 and lines else raw


def document_excerpt(task, limit):
    """Select passages by task relevance and coverage; preserve source page order."""
    text = (task.get("pdf_text") or "").strip()
    text = _assigned_pages(task, text)
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
