"""Rebuild the synthetic attachments from the published JSON (requires reportlab)."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

ROOT = Path(__file__).resolve().parent
# The collection includes classroom handouts, lab briefs and professional work orders.
# The document type is independent of the estimator's task descriptions and ground truth.
SERIES = {
    "01": ("Distributed systems seminar", "Problem sheet", "#294E69"),
    "02": ("Software migration lab", "Lab brief", "#265C55"),
    "03": ("Research reading group", "Reading packet", "#5A476C"),
    "04": ("GPU systems lab", "Engineering brief", "#2B5162"),
    "05": ("Design studio", "Creative brief", "#874B41"),
    "06": ("Data operations", "Analysis brief", "#2C5768"),
    "07": ("Editorial desk", "Editorial brief", "#684958"),
    "08": ("Facilities workshop", "Work order", "#785638"),
    "09": ("Teaching studio", "Lesson brief", "#4C5A35"),
    "10": ("Community office", "Task brief", "#574E6B"),
}


def markup(text):
    return escape(text).replace("\n", "<br/>")


def make_pdf(doc, history_id, task_id, destination):
    series, kind, accent = SERIES[history_id[:2]]
    accent = colors.HexColor(accent)
    ink = colors.HexColor("#202B34")
    muted = colors.HexColor("#55626B")
    heading = ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=13,
                             leading=17, textColor=ink, spaceBefore=16, spaceAfter=5,
                             keepWithNext=True)
    body = ParagraphStyle("Body", fontName="Times-Roman", fontSize=12, leading=17,
                          textColor=ink, spaceAfter=5)
    title = ParagraphStyle("Title", fontName="Times-Bold", fontSize=26,
                           leading=30, textColor=ink, spaceAfter=9)
    kind_style = ParagraphStyle("Type", fontName="Helvetica-Bold", fontSize=9,
                                leading=13, textColor=accent, spaceAfter=6)

    def decorate(canvas, template):
        width, height = LETTER
        canvas.saveState()
        canvas.setFillColor(accent)
        canvas.rect(50, height - 46, 24, 3, fill=True, stroke=False)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(82, height - 48, series)
        canvas.setStrokeColor(colors.HexColor("#CDD3D6"))
        canvas.line(50, 52, width - 50, 52)
        canvas.setFillColor(muted)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(50, 36, "Synthetic test document  /  Daily Planner")
        canvas.drawRightString(width - 50, 36, str(template.page))
        canvas.restoreState()

    destination.parent.mkdir(parents=True, exist_ok=True)
    output = SimpleDocTemplate(
        str(destination), pagesize=LETTER, title=doc["title"], author="Daily Planner fixtures",
        subject=f"Synthetic testing attachment, {history_id}/{task_id}",
        leftMargin=48, rightMargin=48, topMargin=68, bottomMargin=64,
        pageCompression=1, invariant=1,
    )
    story = [Paragraph(escape(kind.upper()), kind_style), Paragraph(markup(doc["title"]), title)]
    for index, section in enumerate(doc["sections"]):
        if index and section.get("page_break_before"):
            story.extend([PageBreak(), Paragraph(markup(doc["title"]), kind_style)])
        story.append(Paragraph(markup(section["heading"]), heading))
        for paragraph in section["body"].split("\n\n"):
            if paragraph.strip():
                story.append(Paragraph(markup(paragraph.strip()), body))
    output.build(story, onFirstPage=decorate, onLaterPages=decorate)


def main():
    written = 0
    for file in sorted((ROOT / "histories").glob("*.json")):
        history = json.loads(file.read_text(encoding="utf-8"))
        for task in history["tasks"] + history["holdouts"]:
            if doc := task.get("pdf"):
                target = ROOT / "pdfs" / history["id"] / f"{task['id']}.pdf"
                make_pdf(doc, history["id"], task["id"], target)
                written += 1
    print(f"Rendered {written} fixture PDFs.")


if __name__ == "__main__":
    main()
