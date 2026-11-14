"""Shared helpers for saved planner forecasts."""


def scheduled_prediction(minutes):
    value = float(minutes)
    minutes = int(value) if value.is_integer() else value
    return {
        "point_min": minutes, "low_min": minutes, "high_min": minutes,
        "basis": "Scheduled duration", "confidence": "fixed",
        "explanation": f"The scheduled duration is {minutes} minutes.",
        "provenance": {"provider": None, "model": None, "decision": "fixed", "minutes_by": "schedule"},
        "evidence": [], "cautions": [],
    }
