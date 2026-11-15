"""Read-only planning for a selected day. Time arithmetic stays local."""
from datetime import date, datetime, time, timedelta
from math import ceil, floor
import json
import re

from model_client import ModelUnavailable


def planning_context(tasks, selected_day, now):
    """Project only visible unfinished work. Reports, attachments and activity logs stay out."""
    open_tasks = []
    for t in tasks:
        if t.get("archived") or t["status"] == "completed":
            continue
        prediction = t.get("prediction") or {}
        pending = prediction.get("status") == "pending" or not prediction.get("point_min")
        point = t.get("estimate_minutes") if pending else prediction["point_min"]
        spent = max(0, t.get("actual_seconds", 0)) / 60
        low, high = (None, None) if pending else (prediction.get("low_min"), prediction.get("high_min"))
        left = max(0, ceil(point - spent)) if point is not None and point > spent else None
        open_tasks.append({
            "id": t["id"], "title": t["title"], "description": (t.get("description") or "")[:250],
            "details": (t.get("details") or "")[:250], "category": t.get("category") or "Uncategorized",
            "day": t["day"], "status": t["status"], "fixed_duration": bool(t.get("fixed_duration")),
            "estimate_source": "provisional user estimate" if pending else prediction.get("basis") or "planner estimate",
            "confidence": prediction.get("confidence"), "estimated_total_minutes": point,
            "recorded_minutes": round(spent, 2), "remaining_minutes": left,
            "remaining_low": max(0, floor(low - spent)) if low is not None else None,
            "remaining_high": max(0, ceil(high - spent)) if high is not None else None,
        })
    return {"selected_day": selected_day.isoformat(), "now": now, "tasks": open_tasks}


_NUMBER = r"(?:\d+(?:\.\d+)?|half an?|an?|one|two|three|four)"
_UNIT = r"(?:hours?|hrs?|h|minutes?|mins?|m)"
_AMOUNT = rf"({_NUMBER})[\s-]*({_UNIT})(?=\d|\b)"
_DURATION = re.compile(r"\b(?:i (?:only )?have|i(?:'ve| have) got|i can (?:spend|work)(?: for)?|"
                       r"available for|free for|work for|only have|within(?: the next)?|with|in(?: the next)?|"
                       r"for the next|budget(?: of)?|(?:have|got) another)\s+" + _AMOUNT, re.I)
_DURATION_LEFT = re.compile(r"\b" + _AMOUNT + r"(?=\s+(?:available|free|left|remaining)\b)", re.I)
_CUTOFF = re.compile(r"\b(?:before|by|until|till|stop(?:ping)? at|finish(?:ing)? at|work to)\s+"
                     r"(midnight|noon|(?:[01]?\d|2[0-3])(?::[0-5]\d)?\s*(?:[ap]\.?m\.?)?)\b", re.I)


def plan_window(question, snap, previous=None):
    now = snap["now"]
    selected = date.fromisoformat(snap["selected_day"])
    day = selected
    explicit = re.search(r"\b\d{4}-\d{2}-\d{2}\b", question)
    q = question.casefold()
    # A recommendation to defer work to tomorrow concerns today's (or the selected day's) tasks.
    moving = re.search(r"\b(?:postpone|postponing|defer|push|move|shift|put off|leave|delay|reschedule)\b", q)
    if "today" in q:
        day = now.date()
    elif "tomorrow" in q and not moving:
        day = now.date() + timedelta(days=1)
    elif "yesterday" in q:
        day = now.date() - timedelta(days=1)
    elif explicit and not moving:
        try:
            day = date.fromisoformat(explicit.group())
        except ValueError:
            return {"error": "Which valid calendar date should I use for the plan?"}
    previous = previous or {}
    if (previous.get("selected_day") == snap["selected_day"] and previous.get("day")
            and not any(word in q for word in ("today", "tomorrow", "yesterday")) and not explicit):
        day = date.fromisoformat(previous["day"])
    # timestamp() resolves the machine's local timezone and DST transition at each wall time.
    start = now if day == now.date() else datetime.combine(day, time.min).astimezone()
    midnight = datetime.combine(day + timedelta(days=1), time.min).astimezone()
    end = midnight
    end_kind = "local midnight"
    budget = None
    duration = _DURATION.search(question) or _DURATION_LEFT.search(question)
    constraint = _CUTOFF.search(question)
    if constraint:
        value = constraint[1].lower().replace(" ", "").replace(".", "")
        if value == "noon":
            hour, minute = 12, 0
        elif value == "midnight":
            hour, minute = 24, 0
        else:
            match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?([ap]m)?", value)
            hour, minute = int(match[1]), int(match[2] or 0)
            if not match[3] and hour <= 12 and (not match[2] or len(match[1]) == 1):
                return {"error": f"Does {hour} mean {hour} AM or {hour} PM in your local time?"}
            if match[3]:
                if not 1 <= hour <= 12:
                    return {"error": "What local clock time should the plan end?"}
                hour = hour % 12 + 12 * (match[3] == "pm")
        if hour < 24:
            end = datetime.combine(day, time(hour, minute)).astimezone()
        end_kind = "your stated cutoff"
    elif not duration and re.search(r"\b(?:before|until|by)\s+(?:my |the )?(?:dinner|bedtime|lunch|class|meeting|exam|work)\b", q):
        return {"error": "What local time is that commitment, or how many free minutes do you have before it?"}
    if duration:
        numbers = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "half a": .5, "half an": .5}
        number = numbers.get(duration[1].lower())
        if number is None:
            number = float(duration[1])
        hours = duration[2][0].lower() == "h"
        budget = floor(number * (60 if hours else 1))
        if hours:
            extra = re.match(r"(?:\s+and)?\s*(\d{1,2})[\s-]*(?:minutes?|mins?|m)\b", question[duration.end():], re.I)
            if extra:
                budget += int(extra[1])
            elif re.match(r"\s+and a half\b", question[duration.end():], re.I):
                budget += 30
    if previous.get("day") == day.isoformat() and not constraint and not duration:
        if previous.get("end_kind") == "your stated cutoff":
            end = datetime.fromisoformat(previous["end"])
            end_kind = previous["end_kind"]
        if previous.get("budget_minutes") is not None:
            budget = previous["budget_minutes"]
            if previous.get("start") and day == now.date():
                elapsed = max(0, (start.timestamp() - datetime.fromisoformat(previous["start"]).timestamp()) / 60)
                budget = max(0, ceil(budget - elapsed))
    calendar = 0 if day < now.date() else max(0, floor((end.timestamp() - start.timestamp()) / 60))
    return {"selected_day": snap["selected_day"], "day": day.isoformat(), "local_now": now.isoformat(timespec="minutes"),
            "start": start.isoformat(timespec="minutes"), "end": end.isoformat(timespec="minutes"),
            "end_kind": end_kind, "calendar_minutes": calendar, "budget_minutes": budget,
            "available_minutes": min(calendar, budget) if budget is not None else calendar,
            "past": day < now.date(), "future": day > now.date()}


_ORDER_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["order"],
                 "properties": {"order": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                     "required": ["id", "reason"], "properties": {"id": {"type": "integer"}, "reason": {"type": "string"}}}}}}


def _work_amount(t):
    if t["remaining_minutes"] is None:
        return f"{t['recorded_minutes']:g} min recorded; remaining effort unknown"
    amount = f"{t['remaining_minutes']} min remaining"
    if t["remaining_high"] is not None and t["remaining_low"] is not None:
        amount += f" (range {t['remaining_low']}–{t['remaining_high']})"
    if t["estimate_source"] == "provisional user estimate":
        amount += " from your provisional budget"
    return amount


def answer_plan(question, snap, history, past_metrics, chat_json, still_current=None):
    prior = next((t.get("plan_window") for t in reversed(history) if t.get("plan_window")), None)
    window = plan_window(question, snap, prior)
    if "error" in window:
        return {"answer": window["error"], "answer_source": "local clarification"}
    include_earlier = re.search(r"\b(backlog|overdue|carry.?over|all (?:open|unfinished|pending))\b", question, re.I)
    def visible(t):
        if t["day"] == window["day"]:
            return True
        if include_earlier and t["day"] < window["day"]:
            return True
        return t["status"] == "running" and window["day"] == snap["now"].date().isoformat()
    tasks = [t for t in snap["tasks"] if visible(t)]
    backlog = sum(t["day"] < window["day"] and not visible(t) for t in snap["tasks"])
    backlog_note = f" {backlog} unfinished task(s) were planned earlier; ask to include the backlog." if backlog else ""
    base = {"plan_window": window, "tool": "plan_remaining", "answer_source": "local status"}
    day = window["day"]
    if window["past"]:
        return {**base, "answer": f"{day} has ended. It has {len(tasks)} unfinished tasks. Choose today or a future plan date to schedule remaining work."}
    if not tasks:
        return {**base, "answer": f"There are no unfinished tasks on the plan for {day}." + backlog_note}
    rows = [{"id": i, **{k: v for k, v in t.items() if k != "id"}} for i, t in enumerate(tasks, 1)]
    remaining = window["available_minutes"]
    earlier = [{"question": t["question"], "answer": t.get("answer", "")[:650]} for t in history[-3:]]
    data = {"question": question, "window": window, "unfinished_tasks": rows, "previous_turns": earlier,
            "completed_task_accuracy_by_category": past_metrics}
    if still_current and not still_current():
        return {"answer": "Your records changed. Ask again using the current data."}
    try:
        reply = chat_json([{"role": "system", "content":
            "Help the user decide which work to do now or defer. Return every supplied task ID once in your recommended sequence. Prefer explicit "
            "requirements in the question/task text, time-sensitive dependencies, work already in progress and "
            "achievable scope. Give one short reason per task based only on the supplied data. Do not invent "
            "deadlines, commitments, priorities or free time. If no deadline is given, do not imply one. "
            "Use a previous answer or historical errors only when relevant. Python checks remaining minutes "
            "and the cumulative fit. Do not claim you changed, moved, scheduled or completed anything."},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False)}], _ORDER_SCHEMA, max_tokens=900)
    except ModelUnavailable as exc:
        return {**base, "answer": f"The model could not prioritize this plan. {exc}"}
    order = {}
    for item in reply["order"]:
        i = item["id"]
        if 1 <= i <= len(rows):
            order.setdefault(i, item["reason"].strip())
    got_model_priority = bool(order)
    for r in rows:
        order.setdefault(r["id"], "")
    cutoff = datetime.fromisoformat(window["end"]).strftime("%H:%M")
    if window["budget_minutes"] is not None:
        intro = f"You gave a {window['budget_minutes']}-minute work budget. Up to {remaining} min fit before {cutoff}."
    elif window["future"]:
        intro = f"The calendar window for {day} ends at {cutoff}. How many minutes will you actually have free?"
    else:
        intro = f"From {snap['now']:%H:%M} to {window['end_kind']} ({cutoff}), {remaining} clock minutes remain. Your free time may be shorter."
    fit_label = "Fits by point estimate" if window["budget_minutes"] is not None else "Fits if the clock time is free"
    sections = {fit_label: [], "Consider deferring or splitting": [], "Confirm remaining work": []}
    audit = []
    upper_total = 0
    all_have_range = True
    for i, reason in order.items():
        r = rows[i-1]
        effort = r["remaining_minutes"]
        if effort is None or r["status"] == "awaiting_finish":
            group = "Confirm remaining work"
        elif effort <= remaining:
            group = fit_label
            remaining -= effort
            if r["remaining_high"] is None:
                all_have_range = False
            else:
                upper_total += r["remaining_high"]
        else:
            group = "Consider deferring or splitting"
        if not reason and group == "Consider deferring or splitting":
            reason = f"This needs {effort} min against {remaining} min left in the window."
        elif not reason and group == "Confirm remaining work":
            reason = "Confirm completion or update the remaining estimate first."
        deadline = next((piece.strip() for field in ("description", "details", "title")
                         for piece in re.split(r"[.!?]\s+", r[field])
                         if re.search(r"\b(?:due|deadline|submit (?:before|by)|hand-in|must finish by)\b",
                                      piece, re.I)), None)
        line = f"**{r['title']}** — {_work_amount(r)}."
        if reason:
            line += " " + reason
        if deadline and group == "Consider deferring or splitting":
            line += f' Check the stated timing before postponing. “{deadline[:170]}”.'
        sections[group].append("- " + line)
        audit.append({"Task": r["title"], "Status": r["status"], "Total estimate (min)": r["estimated_total_minutes"],
                      "Recorded (min)": r["recorded_minutes"], "Remaining (min)": effort,
                      "Range (min)": f"{r['remaining_low']}–{r['remaining_high']}" if r["remaining_high"] is not None else "",
                      "Estimate source": r["estimate_source"], "Suggestion": group})
    if upper_total > window["available_minutes"]:
        intro += f" The upper estimates of the fitting tasks add up to {upper_total} min, so the plan can overrun."
    elif sections[fit_label] and not all_have_range:
        intro += " At least one fitting task has no planning range yet."
    text = intro + backlog_note + "\n\n" + "\n\n".join("**" + name + "**\n\n" + "\n".join(items)
                                          for name, items in sections.items() if items)
    if not any(re.search(r"\b(due|deadline|must|priority|urgent|hand-in|submit before)\b", text, re.I)
               for text in [question, *(r["details"] + " " + r["description"] for r in rows)]):
        text += "\n\nShare any hard deadlines or priorities that should change this order."
    return {**base, "answer": text, "answer_source": "model" if got_model_priority else "local order",
            "plan_rows": audit}
