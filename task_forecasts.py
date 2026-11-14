"""Forecasts for tasks whose duration is set by a schedule."""


def scheduled_prediction(minutes):
    value = float(minutes)
    minutes = int(value) if value.is_integer() else value
    return {
        "point_min": minutes, "low_min": minutes, "high_min": minutes,
        "basis": "Scheduled duration", "confidence": "fixed",
        "explanation": f"The scheduled duration is {minutes} minutes.",
        "evidence": [], "cautions": [],
    }
