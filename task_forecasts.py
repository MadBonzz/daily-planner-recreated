"""Shared helpers for saved planner forecasts."""


def scheduled_prediction(minutes):
    value = float(minutes)
    minutes = int(value) if value.is_integer() else value
    return {
        "point_min": minutes, "low_min": minutes, "high_min": minutes,
        "basis": "Scheduled duration", "confidence": "fixed",
        "explanation": f"The scheduled duration is {minutes} minutes.",
        "evidence": [], "cautions": [],
    }


def can_retry_forecast(prediction):
    """Recognize both current failures and fallback forecasts saved by earlier releases."""
    if "retryable" in prediction:
        return prediction["retryable"]
    explanation = prediction.get("explanation", "")
    return (explanation.startswith(("Model comparison unavailable.", "The background estimate could not be completed."))
            or explanation.startswith("No timed personal history. Model estimate unavailable.")
            or explanation.startswith("No timed personal history. The model returned an invalid estimate."))
