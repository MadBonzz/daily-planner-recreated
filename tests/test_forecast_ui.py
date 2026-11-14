from streamlit.testing.v1 import AppTest


def show_card(prediction, **settings):
    task = {"prediction": prediction, "creation_prediction": prediction, "estimate_updating": False,
            "fixed_duration": False, "estimate_minutes": 20, "actual_seconds": 45,
            "status": "paused", "id": "target", **settings}
    app = AppTest.from_string(f"from app import estimate_section\nestimate_section(None, {task!r})").run()
    assert not app.exception
    return app


def budget(**details):
    return dict(point_min=25, low_min=17, high_min=35, confidence="low", **details)


def prose(app):
    return "\n".join([element.value for element in [*app.caption, *app.markdown]]
                     + [element.proto.body for element in app.get("html")])


def test_model_card_identifies_saved_model_and_quotes_supported_reasoning():
    card = show_card(budget(
        basis="Model forecast",
        provenance={"provider": "Cohere", "model": "command-r7b-12-2024", "decision": "direct", "minutes_by": "model"},
        model_rationale="Printing the flyer and walking to the entrance add time.",
        confidence_reason="The previous flyer used a different print queue.",
        evidence=[{"title": "Print a flyer", "estimate_min": 15, "actual_min": 21, "ratio": 1.4,
                   "reason": "Both tasks use colour stock.", "reason_source": "model", "source_quote": "colour stock"}],
        cautions=[{"advice": "Check the print queue before leaving.",
                   "sources": [{"title": "Print a flyer", "origin": "Completion note"}]}],
    ))
    copy = prose(card)
    for expected in ("Cohere · command-r7b-12-2024", "The model supplied the minutes", "walking to the entrance",
                     "a different print queue", "Both tasks use colour stock", "Check the print queue",
                     "Print a flyer · Completion note"):
        assert expected in copy
    assert [button.label for button in card.button] == ["Re-estimate"]


def test_legacy_card_identifies_hybrid_and_does_not_attribute_compact_placeholder_to_model():
    card = show_card(budget(
        basis="Model-selected tasks", evidence=[{"title": "Print a flyer", "estimate_min": 15, "actual_min": 21,
                    "ratio": 1.4, "reason": "The model selected this task in the compact comparison."}]))
    copy = prose(card)
    assert "Plan Smarter calculated the minutes" in copy and "Model identity was not recorded" in copy
    assert "No model rationale was saved" in copy and "No planning caution was saved" in copy
    assert "compact comparison" not in copy


def test_reestimating_hides_stale_explanation_and_completed_and_fixed_tasks_have_no_reestimate():
    old = budget(basis="User estimate", explanation="old explanation")
    pending = show_card(old, estimate_updating=True)
    assert "Previous forecast" in prose(pending) and "Updating estimate" in prose(pending)
    assert not pending.expander and not pending.button and "old explanation" not in prose(pending)
    for settings in ({"status": "completed"}, {"fixed_duration": True}):
        assert not show_card(old, **settings).button
