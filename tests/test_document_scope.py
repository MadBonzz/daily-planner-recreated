import pytest

from pdf_reader import document_excerpt
from retrieval import embedding_input


TEXT = "Page 1: Assemble a frequency sweep.\nPage 2: Translate an interview.\nPage 3: Annotate dialect terms."


@pytest.mark.parametrize("details, expected", [
    ("Only do page 2.", [2]), ("Read only pp. 2–3.", [2, 3]),
    ("Just review pages 1 and 3.", [1, 3]), ("Read pp. 2–3 only.", [2, 3]),
    ("Skip page 1. Only use page 2.", [2]),
    ("Draft a two-page report.", [1, 2, 3]),
    ("Review page 2 and find contradictory details elsewhere.", [1, 2, 3]),
])
def test_judge_excerpt_follows_assigned_pages_but_embedding_keeps_full_upload(details, expected):
    task = {"title": "Complete the handout", "details": details, "pdf_text": TEXT}
    snippet = document_excerpt(task, 1000)
    assert [n for n in (1, 2, 3) if f"Page {n}:" in snippet] == expected
    assert TEXT in embedding_input(task, query=True)


def test_unknown_page_and_ambiguous_attachment_numbering():
    task = {"details": "Only do page 7.", "pdf_text": TEXT}
    assert document_excerpt(task, 1000) == "The requested PDF pages have no extracted text."
    task.update(details="Only do page 2.", attachments=[{"name": "a.pdf"}, {"name": "b.pdf"}])
    assert document_excerpt(task, 1000) == TEXT


def test_late_scoped_page_survives_a_long_rubric():
    body = "\n".join(f"Page {i}: " + "Explain the report and submit the experiment. " * 70 for i in range(1, 13))
    body += "\nPage 13: Wire the oscilloscope probes and verify their calibration."
    row = {"description": "Only read page 13.", "pdf_text": body}
    assert document_excerpt(row, 900).startswith("Page 13: Wire")
    assert "Page 1:" in embedding_input(row)


def test_limited_scope_with_verbatim_exercise_and_an_unrelated_section():
    exercise = "Wire the probe and sweep the gain across ten frequencies before plotting phase and amplitude."
    body = f"Page 1: {exercise}\nPage 2: Translate an interview from Hindi into English."
    row = {"title": "Only complete my assigned exercise", "description": exercise, "pdf_text": body}
    assert document_excerpt(row, 1000) == f"Page 1: {exercise}"
    assert body in embedding_input(row)
    assert document_excerpt({**row, "title": "Work through both sections"}, 1000) == body


def test_named_part_on_shared_page_and_across_page_breaks():
    from pdf_reader import assigned_section_text
    text = ("Page 1: Overview. Part I · Translate the report. Part II · Wire a circuit.\n"
            "Page 2: Measure the gain and plot it. Part III · Book a room.")
    task = {"title": "Circuit assignment", "details": "Complete part II only.", "pdf_text": text}
    assert assigned_section_text(task) == "Page 1: Part II · Wire a circuit.\nPage 2: Measure the gain and plot it."
    assert assigned_section_text({**task, "details": "Review all parts."}) == text
    assert assigned_section_text({**task, "details": "Only part IV."}) == text
    duplicate = "Page 0: Part II · Contents entry.\n" + text
    assert assigned_section_text({**task, "pdf_text": duplicate}) == duplicate


def test_named_scope_reaches_both_judges_while_embedding_keeps_other_parts():
    from retrieval import _decision_record
    from estimator import _prompt_record
    task = {"description": "Explain the saved traces. Part II only.",
            "pdf_text": "Page 1: Part I · Implement the decoder. Part II · Fill the replay table."}
    assert "Implement the decoder" not in _decision_record(task)["document_excerpt"]
    assert "Fill the replay table" in _decision_record(task)["document_excerpt"]
    assert "Implement the decoder" not in _prompt_record(task, target=True)["document_excerpt"]
    assert "Fill the replay table" in _prompt_record(task)["document_excerpt"]
    assert "Implement the decoder" in embedding_input(task)


@pytest.mark.parametrize("instruction", [
    "Read not only page 2 but also the other chapters.",
    "Use more than just page 2.",
    "The discussion is not restricted to page 2.",
])
def test_broader_scope_keeps_other_pages(instruction):
    from pdf_reader import requested_pages
    assert document_excerpt({"details": instruction, "pdf_text": TEXT}, 1000) == TEXT
    assert not requested_pages(instruction)


def test_separate_sentence_with_incidental_page_does_not_expand_assignment():
    task = {"details": "Only use page 2. Page 3 lists the office hours.", "pdf_text": TEXT}
    assert document_excerpt(task, 1000) == "Page 2: Translate an interview."


def test_not_only_part_does_not_truncate_a_shared_handout():
    from pdf_reader import assigned_section_text
    raw = "Page 1: Part I · Write code. Part II · Write prose."
    task = {"details": "Do not only do part I. Complete the whole packet.", "pdf_text": raw}
    assert assigned_section_text(task) == raw
