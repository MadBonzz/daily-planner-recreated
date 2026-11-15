"""Original task uploads and the document types the planner can read."""

from pathlib import Path

from pdf_reader import extract_pdf_text


def upload_info(name, data):
    name = Path(name.replace("\\", "/")).name.strip()
    if not name or not isinstance(data, bytes):
        raise ValueError("Choose a named file to attach.")
    if b"%PDF-" in data[:1024]:
        return name, "pdf"
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        raise ValueError(f"{name} is not a valid PDF attachment.")
    if suffix in {".txt", ".md", ".csv", ".tsv", ".json", ".py", ".log"}:
        return name, "text"
    return name, "file"


def document_text(kind, data):
    if kind == "pdf":
        return extract_pdf_text(data)
    if kind == "text":
        try:
            return data.decode("utf-8-sig").strip()
        except UnicodeDecodeError:
            return ""
    return ""
