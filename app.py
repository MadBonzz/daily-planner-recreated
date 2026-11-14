"""Plan Smarter UI. Run with python -m streamlit run app.py."""

from datetime import date, datetime, timedelta
from html import escape
import os
from pathlib import Path
import sqlite3
import time

import streamlit as st

import activity
import estimator
from dev_workspace import dev_activity_report, history_choices, open_dev_workspace
from estimate_worker import schedule_pending
from planner_store import PlannerStore
from pdf_reader import extract_pdf_text
from model_client import model_settings
from urllib.parse import urlsplit

st.set_page_config(page_title="Plan Smarter", page_icon="🗓️", layout="centered")
DATA_ROOT = Path(os.environ.get("PLANNER_DATA_DIR", Path(__file__).parent / "data"))
STATUS = {"ready": "Ready", "running": "Running", "paused": "Paused",
          "awaiting_finish": "Session stopped", "completed": "Completed"}

STYLES = """
<style>
[data-testid="stMainBlockContainer"] {max-width: 970px; padding-top: 3.5rem; padding-bottom: 4rem}
[data-testid="stHeader"] {background: transparent}
h1 {letter-spacing: -.055em}
h2,h3 {letter-spacing: -.033em}
[data-testid="stVerticalBlockBorderWrapper"] {background: #fff; border-radius: 16px}
[data-testid="stDialog"] [data-testid="stWidgetLabelRequired"] {font-size:0}
[data-testid="stDialog"] [data-testid="stWidgetLabelRequired"]::after {content:"*";color:#b42318;font-size:1rem;font-weight:700}
.metric-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));margin:.35rem 0 1.1rem;
 background:white;border:1px solid #d8deea;border-radius:14px;padding:1.05rem 0}
.metric-grid div {padding:0 clamp(.6rem,2.7vw,1.7rem);border-left:1px solid #d8deea}
.metric-grid div:first-child {border-left:0}
.metric-grid small {display:block;color:#526174;font-size:.83rem}
.metric-grid strong {display:block;font-size:clamp(1.2rem,3vw,1.9rem);font-weight:650;letter-spacing:-.04em}
.task-head {display:flex;align-items:baseline;justify-content:space-between;gap:1rem}
.task-head h3 {font-size:1.15rem;margin:0;padding:0;overflow-wrap:anywhere;line-height:1.4}
.task-head span {color:#506073;font-size:.82rem;white-space:nowrap;font-weight:600}
.task-head span.running {color:#234cce}.task-head span.completed {color:#287053}
.task-copy {color:#425266;margin:.3rem 0 .25rem;white-space:pre-wrap;overflow-wrap:anywhere}
.empty-plan {background:white;border:1px dashed #b9c4d5;padding:2.6rem 1rem;text-align:center;border-radius:16px}
@media (max-width:640px) {
 .st-key-plan_navigation [data-testid="stHorizontalBlock"] {display:grid;grid-template-columns:4.5rem minmax(0,1fr) 2.8rem;gap:.45rem}
 .st-key-plan_navigation [data-testid="stColumn"] {width:100%;min-width:0;flex:none}
 .st-key-plan_navigation [data-testid="stColumn"]:last-child {grid-column:span 2}
 .st-key-plan_header [data-testid="stHorizontalBlock"] {display:flex;flex-wrap:nowrap;align-items:center}
 .st-key-plan_header [data-testid="stColumn"]:first-child {width:auto;flex:1;min-width:0}
 .st-key-plan_header [data-testid="stColumn"]:last-child {width:6.5rem;flex:none;min-width:0}
}
.empty-plan strong {display:block;font-size:1.18rem}.empty-plan p {color:#546477;margin:.35rem 0 0}
</style>
"""
st.html(STYLES)


def duration(seconds):
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02}m"
    return f"{minutes}m {secs:02}s" if minutes else f"{secs}s"


def minutes(value):
    return f"{float(value):g} min"


def show_error(error):
    st.error(str(error))


def notice(message):
    st.session_state["notice"] = message


def sync_index(store, index_dir, task):
    try:
        estimator.index_current_task(index_dir, store, task["id"])
    except (OSError, ValueError, RuntimeError):
        notice("Task saved. The planner will retry indexing it when it searches completed work.")


def forecast_source(prediction):
    provenance = prediction.get("provenance") or {}
    decision = provenance.get("decision")
    model = " · ".join(str(value) for value in (provenance.get("provider"), provenance.get("model")) if value)
    if not decision:
        basis = prediction.get("basis", "Saved forecast")
        if basis == "Model-selected tasks":
            return "A model selected past tasks. Plan Smarter calculated the minutes. Model identity was not recorded."
        return f"{basis} · Model identity was not recorded"
    role = {
        "direct": "Model forecast",
        "matches": "Selected comparable tasks",
        "no_matches": "Found no comparable tasks",
        "fallback": "Model estimate unavailable" if model else "Local fallback",
        "unused": "No model call",
        "fixed": "Scheduled duration",
    }.get(decision, "Forecast")
    return f"{role} · {model}" if model else role


def show_prediction(prediction):
    st.markdown(f"**{minutes(prediction['point_min'])} suggested** · "
                f"{prediction['low_min']:g}–{prediction['high_min']:g} min planning range")
    provenance = prediction.get("provenance") or {}
    minutes_by = provenance.get("minutes_by")
    calculation = {
        "model": "The model supplied the minutes, planning range and confidence rating.",
        "history": "Plan Smarter calculated the minutes from previous durations and your estimate.",
        "user": "Plan Smarter used your own estimate as the point estimate.",
        "schedule": "The event uses its scheduled duration.",
    }.get(minutes_by)
    decision = provenance.get("decision")
    matched = decision in {"direct", "matches"} or (not decision and prediction.get("basis") == "Model-selected tasks")
    reasons = []
    if matched:
        for row in prediction.get("evidence", []):
            reason = row.get("model_rationale") or row.get("reason")
            if (reason and row.get("reason_source", "model") == "model"
                    and reason != "The model selected this task in the compact comparison."):
                reasons.append((row["title"], reason, row.get("source_quote")))
    rationale = prediction.get("model_rationale")
    explanation = prediction.get("explanation", "")
    # Older direct forecasts saved the provider's words after this exact prefix.
    old_direct = "No timed personal history. This estimate is based on the task scope. "
    if not provenance and prediction.get("basis") == "Model estimate from task scope" and explanation.startswith(old_direct):
        rationale = explanation.removeprefix(old_direct)
    if rationale or reasons:
        st.markdown("**Model rationale**")
        if rationale and all(reason != rationale for _, reason, _ in reasons):
            st.html(f'<p class="task-copy">{escape(rationale)}</p>')
        for title, reason, quote in reasons:
            st.html(f'<p class="task-copy"><strong>{escape(title)}</strong> · {escape(reason)}</p>')
            if quote:
                st.html(f'<small>From the task · “{escape(quote)}”</small>')
    elif matched:
        st.caption("No model rationale was saved for this forecast.")

    if calculation:
        st.html(f'<p class="task-copy">{escape(calculation)}</p>')
    if explanation and (not rationale or rationale not in explanation):
        st.html(f'<p class="task-copy">{escape(explanation)}</p>')
    st.markdown(f"**{prediction['confidence'].capitalize()} confidence**")
    confidence_reason = prediction.get("confidence_reason")
    if confidence_reason:
        st.html(f'<p class="task-copy">{escape(confidence_reason)}</p>')
    elif not provenance:
        ratios = [row["ratio"] for row in prediction.get("evidence", []) if row.get("ratio") is not None]
        if ratios:
            st.caption(f"The {len(ratios)} saved comparison{'s' if len(ratios) != 1 else ''} had actual-to-estimated "
                       f"ratios from {min(ratios):.2f}× to {max(ratios):.2f}×.")
        st.caption("This saved forecast did not record the reason for its confidence rating.")


@st.dialog("Add task", width="large")
def new_task(store, day):
    with st.form("new_task_form"):
        title = st.text_input("Task name", required=True, placeholder="Read the optimization paper")
        description = st.text_area("Description", required=True, placeholder="What do you want to finish?", height=86)
        left, right = st.columns(2)
        estimate = left.number_input("Duration (minutes)", min_value=1, value=None, step=1,
                                     required=True, placeholder="e.g. 30")
        category = right.selectbox("Category", store.list_categories(), index=None, required=True,
                                   placeholder="Choose a category",
                                   help="Add categories in Planner settings.")
        fixed = st.checkbox("Fixed duration", help="Use the scheduled length exactly, such as for a screening. Leave this off for meetings that can run over.")
        details = st.text_area("More details (optional)", height=80,
                               placeholder="Which sections matter? How familiar are you with this work?")
        files = st.file_uploader("Attach PDFs (optional)", type=["pdf"], accept_multiple_files=True,
                                 help="Readable text from the documents can inform the estimate.")
        submitted = st.form_submit_button("Create task", type="primary")
    if not submitted:
        return
    missing = [label for label, value in (("task name", title.strip()), ("description", description.strip()),
                                          ("duration in minutes", estimate), ("category", category))
               if not value]
    if missing:
        st.error("Fill in the required fields: " + ", ".join(missing) + ".")
        return
    attachments = [(file.name, file.getvalue()) for file in files]
    try:
        store.create_task(title=title.strip(), description=description.strip(), details=details.strip(),
                          category=category, estimate_minutes=estimate, day=day, fixed_duration=fixed,
                          attachments=attachments, prediction=None)
    except (ValueError, sqlite3.Error, OSError) as error:
        show_error(error)
        return
    st.rerun()


def perform(store, task, action, **kwargs):
    try:
        return store.transition(task["id"], action, **kwargs)
    except (ValueError, sqlite3.Error) as error:
        show_error(error)
        return None


def task_report(store, task, dev):
    with st.spinner("Reviewing recorded activity…"):
        if dev:
            return dev_activity_report(store, task)
        return activity.build_report(task, store.periods(task["id"]),
                                     exclusions=store.get_setting("exclusions", []),
                                     disabled_periods=store.disabled_periods(task["id"]))


@st.dialog("Have you finished this task?")
def stop_confirmation(store, index_dir, task, dev):
    st.html(f'<strong>{escape(task["title"])}</strong>')
    st.write(f"Your timer stopped at {duration(task['actual_seconds'])} of recorded work.")
    note = st.text_area("Completion note (optional)", key=f"finish-note-{task['id']}",
                        placeholder="What affected how long this took?")
    yes, no = st.columns(2)
    if yes.button("Yes, finish task", type="primary", width="stretch"):
        # Stop has closed the timer. Finalize the note, report, actual and comparison
        # history together, so a new forecast sees this task's activity immediately.
        parsed = None
        try:
            if task["fixed_duration"] and task["attachments"] and not task["pdf_text"]:
                # Fixed events have no prediction worker. Read attachments here,
                # after the timer stops, for classification and future comparisons.
                parsed = "\n".join(extract_pdf_text(store.attachment_bytes(doc["id"]))
                                   for doc in task["attachments"])
                task = {**task, "pdf_text": parsed}
            report = task_report(store, task, dev)
        except (ValueError, RuntimeError, OSError) as error:
            report = None
            notice(f"Task finished. Activity review can be retried on its card. {error}")
        saved = perform(store, task, "finish", completion_note=note, report=report, pdf_text=parsed)
        if saved:
            if saved["prediction"].get("status") != "pending":
                sync_index(store, index_dir, saved)
            st.rerun()
    if no.button("No, continue later", width="stretch"):
        if perform(store, task, "continue_later"):
            st.rerun()


def timeline(intervals):
    if not intervals:
        st.caption("No intervals to show.")
        return
    rows = []
    spans_days = len({datetime.fromtimestamp(i["start"]).date() for i in intervals}) > 1
    for event in intervals:
        start = datetime.fromtimestamp(event["start"]).strftime("%b %d %H:%M:%S" if spans_days else "%H:%M:%S")
        kind = "Excluded" if event["excluded"] else event.get("app") or "Unknown"
        row = {"Start": start, "Length": duration(event["seconds"]), "App": kind,
               "Window or tab": event.get("title", ""), "Domain": event.get("domain", "")}
        if event.get("phone_app"):
            row["Phone app"] = event["phone_app"]
        if "label" in event:
            row["Assessment"] = event["label"].replace("_", "-")
            row["Reason"] = event.get("reason", "")
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")


def show_report(report):
    st.html(f'<p class="task-copy">{escape(report["summary"])}</p>')
    st.caption(f"{report['switches']} recorded activity switches. Labels estimate task relevance.")
    for warning in report.get("warnings", []):
        st.caption(warning)
    timeline(report["intervals"])


@st.fragment(run_every="1s")
def running_clock(store, task_id):
    task = store.get_task(task_id)
    if task["status"] != "running":
        st.rerun()
    st.markdown(f"**{duration(task['actual_seconds'])}** recorded")


def live_activity(store, task):
    seconds = store.get_setting("poll_seconds", 5)

    @st.fragment(run_every=seconds)
    def preview():
        current = store.get_task(task["id"])
        if current["status"] != "running":
            st.rerun()
        if not store.get_setting("capture_enabled", True):
            st.caption("Activity capture is paused.")
            return
        now = time.time()
        result = activity.live_preview(max(current["started_at"], now - 120), now,
                                       exclusions=store.get_setting("exclusions", []),
                                       disabled_periods=store.disabled_periods(task["id"]))
        for warning in result["warnings"]:
            st.caption(warning)
        timeline(result["intervals"])
    preview()


@st.dialog("Delete task")
def delete_task(store, index_dir, task):
    st.text(f'Delete “{task["title"]}” and its stored sessions, PDFs, and report?')
    if st.button("Delete task permanently", type="primary"):
        try:
            store.delete_task(task["id"])
            estimator.forget_task(index_dir, task["id"])
        except (ValueError, RuntimeError, OSError, sqlite3.Error) as error:
            show_error(error)
            return
        st.rerun()


@st.dialog("Planner settings", width="large")
def settings(store, index_dir, dev):
    endpoint, model = model_settings()
    st.subheader("Model connection")
    st.text(model)
    try:
        host = urlsplit(endpoint).hostname or ""
    except ValueError:
        host = ""
    if host in {"127.0.0.1", "localhost", "::1"}:
        st.caption("Inference runs on this computer.")
    else:
        st.text(f"Configured host: {host}" if host else "Check the configured endpoint in .env.")
        st.caption("This provider receives the task descriptions, document excerpts, and filtered activity used for inference.")
    st.subheader("Categories")
    st.caption(" · ".join(store.list_categories()))
    with st.form("add_category", clear_on_submit=True):
        category_name = st.text_input("New category name", placeholder="e.g. Lab meetings")
        if st.form_submit_button("Add category"):
            try:
                store.add_category(category_name)
            except ValueError as error:
                st.error(str(error))
            else:
                st.rerun(scope="fragment")
    st.subheader("Activity capture")
    if dev:
        st.caption("Dev uses synthetic activity from the loaded history.")
    else:
        captured = store.get_setting("capture_enabled", True)
        enabled = st.toggle("Read ActivityWatch during timed work", value=captured)
        if enabled != captured:
            store.set_capture(enabled)
        st.caption("Pausing here affects the planner. ActivityWatch manages its own watchers and raw logs.")
    with st.form("privacy_settings"):
        exclusions = st.text_area("Exclude apps, window titles or domains",
                                  value="\n".join(store.get_setting("exclusions", [])),
                                  help="One text match per line, case-insensitive. Applies to newly generated reports and previews.")
        interval = st.number_input("Live activity refresh (seconds)", min_value=1, max_value=60,
                                   value=store.get_setting("poll_seconds", 5))
        if st.form_submit_button("Save settings", type="primary"):
            store.set_setting("exclusions", [s.strip() for s in exclusions.splitlines() if s.strip()])
            store.set_setting("poll_seconds", interval)
            st.rerun()
    st.divider()
    st.subheader("Delete planner data")
    which = "example" if dev else "personal"
    confirm = st.checkbox(f"Delete my {which} tasks, PDFs, reports and estimate history.")
    if st.button("Delete planner data", disabled=not confirm):
        try:
            store.clear_data()
            estimator.clear_index(index_dir)
        except (OSError, RuntimeError, sqlite3.Error) as error:
            show_error(error)
            return
        st.rerun()


def move_day(days):
    st.session_state["plan_day"] += timedelta(days=days)


def select_day(day):
    st.session_state["plan_day"] = day


def estimate_pending(task):
    return task["estimate_updating"]


def estimate_section(store, task):
    prediction = task["prediction"]
    initial_pending = prediction.get("status") == "pending"
    fixed = task["fixed_duration"]
    label = "Scheduled" if fixed else "Your estimate"
    values = [f"{label} {minutes(task['estimate_minutes'])}"]
    if initial_pending:
        values.append("Estimating task time…")
    elif "point_min" in prediction and not fixed:
        budget_label = "Previous forecast" if estimate_pending(task) else "Planner"
        values.append(f"{budget_label} {minutes(prediction['point_min'])} ({prediction['low_min']:g}–"
                      f"{prediction['high_min']:g} min, {prediction['confidence']} confidence)")
    if estimate_pending(task) and not initial_pending:
        values.append("Updating estimate…")
    if task["status"] != "running":
        values.append(f"Recorded {duration(task['actual_seconds'])}")
    st.caption("  ·  ".join(values))
    if initial_pending or estimate_pending(task) or fixed:
        return
    if task["status"] != "completed":
        if st.button("Re-estimate", key=f"estimate-retry-{task['id']}",
                     help="Use the current model, task details and completed history for another forecast."):
            store.retry_estimate(task["id"])
            st.rerun()
    if not prediction or "point_min" not in prediction:
        return
    st.caption(forecast_source(prediction))
    cautions = prediction.get("cautions", [])
    if cautions:
        # Keep one observation visible while the full evidence is collapsed.
        first = cautions[0]
        source = next(iter(first.get("sources", [])), {})
        preview = first.get("advice") or source.get("observation", "")
        st.html(f'<p class="task-copy"><strong>Watch for</strong> · {escape(preview)}</p>')
    with st.expander("Why this estimate?"):
        show_prediction(prediction)
        st.markdown("**What to watch for**")
        if cautions:
            for caution in cautions:
                st.html(f'<p class="task-copy">{escape(caution["advice"])}</p>')
                for source in caution.get("sources", []):
                    citation = f'{escape(source["title"])} · {escape(source["origin"])}'
                    observation = source.get("observation")
                    if observation and observation not in caution["advice"]:
                        citation += f' · “{escape(observation)}”'
                    st.html(f'<small>{citation}</small>')
        elif prediction.get("provenance"):
            st.caption("No specific obstacle or distraction was identified in related task history.")
        else:
            st.caption("No planning caution was saved with this forecast.")
        evidence = prediction.get("evidence", [])
        if evidence:
            st.dataframe([{"Completed task": row["title"], "Your estimate (min)": row["estimate_min"],
                           "Actual (min)": round(row["actual_min"], 2),
                           "Actual ÷ estimate": f"{row['ratio']:.2f}×"}
                          for row in evidence], hide_index=True, width="stretch")
        original = task["creation_prediction"]
        if original and original != prediction and "point_min" in original:
            st.caption(f"At creation · {minutes(original['point_min'])} "
                       f"({original['low_min']:g}–{original['high_min']:g} min).")


def live_estimate_section(store, task, index_dir):
    # Refresh only this part of the card. A finished forecast must not close an open dialog.
    @st.fragment(run_every="1s", key=f"estimate-{task['id']}")
    def refresh():
        if estimate_pending(task):
            try:
                fresh = store.get_task(task["id"])
            except ValueError:
                st.rerun()
            task.update(fresh)
            if estimate_pending(fresh):
                schedule_pending(store, index_dir)
        estimate_section(store, task)
    refresh()


def daily_totals(tasks):
    original = sum(t["estimate_minutes"] for t in tasks)
    budget = sum(t["prediction"].get("point_min", t["estimate_minutes"]) for t in tasks)
    done = sum(t["status"] == "completed" for t in tasks)
    budget_label = "Estimating…" if any(t["prediction"].get("status") == "pending" for t in tasks) else minutes(budget)
    st.html('<div class="metric-grid">' + "".join(
        f"<div><small>{label}</small><strong>{value}</strong></div>" for label, value in
        (("Your estimates", minutes(original)), ("Planner budget", budget_label),
         ("Tasks complete", f"{done} / {len(tasks)}"))) + '</div>')


def live_daily_totals(store, day, tasks):
    @st.fragment(run_every="1s")
    def refresh():
        if any(estimate_pending(t) for t in tasks):
            tasks[:] = store.list_tasks(day)
        daily_totals(tasks)
    refresh()


def render_task(store, index_dir, task, dev):
    key = task["id"]
    status = task["status"]
    with st.container(border=True):
        st.html(f'<div class="task-head"><h3>{escape(task["title"])}</h3>'
                f'<span class="{status}">{STATUS[status]}</span></div>'
                f'<p class="task-copy">{escape(task["description"])}</p>')
        if estimate_pending(task):
            live_estimate_section(store, task, index_dir)
        else:
            estimate_section(store, task)
        if task["category"]:
            st.html(f'<small>{escape(task["category"])}</small>')
        if status == "running":
            running_clock(store, key)
        if status == "running" and time.time() - task["started_at"] >= 8 * 3600:
            start = datetime.fromtimestamp(task["started_at"])
            with st.expander(f"Timer running since {start:%b %d at %H:%M} · correct the end time"):
                with st.form(f"correct-stop-{key}"):
                    end_day = st.date_input("End date", value=date.today(), min_value=start.date(),
                                            max_value=date.today(), key=f"end-day-{key}")
                    end_time = st.time_input("End time", value=datetime.now().time(), step=60,
                                             key=f"end-time-{key}")
                    if st.form_submit_button("Stop at corrected time"):
                        end_ts = datetime.combine(end_day, end_time).timestamp()
                        if perform(store, task, "stop", ended_at=end_ts):
                            st.session_state["ask_finish"] = key
                            st.rerun()
        if status in ("running", "paused"):
            left, right = st.columns(2)
            action = "pause" if status == "running" else "resume"
            if left.button(action.title(), key=f"{action}-{key}", width="stretch"):
                if perform(store, task, action):
                    st.rerun()
            if right.button("Stop", key=f"stop-{key}", width="stretch"):
                if perform(store, task, "stop"):
                    st.session_state["ask_finish"] = key
                    st.rerun()
        elif status == "ready":
            if st.button("Start", key=f"start-{key}", type="primary"):
                if perform(store, task, "start"):
                    st.rerun()
        elif status == "awaiting_finish":
            if st.button("Confirm completion", key=f"confirm-{key}", type="primary"):
                stop_confirmation(store, index_dir, task, dev)
        if status == "running" and not dev:
            if st.toggle("Live activity", key=f"live-{key}", help="Preview the last two minutes."):
                live_activity(store, task)
        if status == "completed":
            if task["completion_note"]:
                st.caption("Completion note")
                st.text(task["completion_note"])
            with st.expander("Activity report", expanded=True):
                if task["report"]:
                    show_report(task["report"])
                else:
                    st.caption("An activity report has not been generated for this task.")
                if st.button("Rebuild report" if task["report"] else "Generate report", key=f"report-{key}"):
                    report = task_report(store, task, dev)
                    store.save_report(key, report)
                    st.rerun()
        with st.expander("Task details"):
            if task["details"]:
                st.text(task["details"])
            for item in task["attachments"]:
                st.download_button("Download PDF", store.attachment_bytes(item["id"]), item["name"],
                                   mime="application/pdf", key=f"pdf-{item['id']}", on_click="ignore")
                st.text(item["name"])
            if status != "completed":
                with st.form(f"reschedule-{key}"):
                    new_day = st.date_input("Scheduled day", value=date.fromisoformat(task["day"]), key=f"date-{key}")
                    if st.form_submit_button("Move task"):
                        store.reschedule(key, new_day)
                        st.rerun()
            if st.button("Delete task", key=f"delete-{key}"):
                delete_task(store, index_dir, task)


def main():
    st.session_state.setdefault("plan_day", date.today())
    title, options = st.container(key="plan_header").columns([4, 1], vertical_alignment="center")
    title.title("Plan Smarter")
    with options.popover("Options", width="stretch"):
        open_settings = st.button("Planner settings")
    mode = st.segmented_control("Mode", ["Demo", "Dev"], default="Demo", required=True,
                                key="planner_mode", on_change=select_day, args=(date.today(),),
                                help="Demo is your planner. Dev loads editable sample histories.")
    dev = mode == "Dev"
    if dev:
        choices = history_choices()
        selected = st.selectbox("Sample history", list(choices), format_func=choices.get,
                                key="dev_history", on_change=select_day, args=(date.today(),))
        store = open_dev_workspace(DATA_ROOT, selected)
        with st.expander("History details"):
            st.caption("This workspace contains synthetic tasks. Your edits are saved separately for each sample.")
            past = store.list_tasks()
            st.dataframe([{"Date": t["day"], "Task": t["title"], "Status": STATUS[t["status"]],
                           "Duration (min)": t["estimate_minutes"], "Recorded (min)": round(t["actual_seconds"] / 60, 1)}
                          for t in past], hide_index=True, width="stretch")
            available_days = sorted({date.fromisoformat(t["day"]) for t in past}, reverse=True)
            if available_days:
                date_column, jump = st.columns([3, 1], vertical_alignment="bottom")
                target = date_column.selectbox("Task date", available_days, key=f"dev-date-{selected}")
                jump.button("Open date", on_click=select_day, args=(target,), width="stretch")
            if st.button("Reset sample history", help="Restore the original tasks and discard edits to this sample."):
                open_dev_workspace(DATA_ROOT, selected, reset=True)
                select_day(date.today())
                st.session_state.pop("ask_finish", None)
                st.rerun()
    else:
        store = PlannerStore(DATA_ROOT / "live")
    index_dir = store.data_dir / "vectors"
    if store.has_pending_predictions():
        schedule_pending(store, index_dir)
    if open_settings:
        settings(store, index_dir, dev)
    if "notice" in st.session_state:
        st.warning(st.session_state.pop("notice"))
    controls = st.container(key="plan_navigation").columns([1, 2, 1, 1, 2], vertical_alignment="bottom")
    controls[0].button("←", key="prev", help="Previous day", on_click=move_day, args=(-1,), width="stretch")
    day = controls[1].date_input("Plan date", key="plan_day", format="MM/DD/YYYY",
                                 label_visibility="collapsed")
    controls[2].button("→", key="next", help="Next day", on_click=move_day, args=(1,), width="stretch")
    controls[3].button("Today", on_click=select_day, args=(date.today(),), width="stretch")
    if controls[4].button("＋ Add task", type="primary", width="stretch"):
        new_task(store, day)
    tasks = store.list_tasks(day)
    if any(estimate_pending(t) for t in tasks):
        live_daily_totals(store, day, tasks.copy())
    else:
        daily_totals(tasks)
    if running := store.running_task():
        if running["day"] != day.isoformat():
            st.text(f'Timer running on {running["day"]}: {running["title"]}')
            st.button("View running task", on_click=select_day, args=(date.fromisoformat(running["day"]),))
    if pending := store.pending_task():
        if pending["day"] != day.isoformat():
            st.text(f'A stopped session on {pending["day"]} is waiting for your decision.')
            st.button("View stopped task", on_click=select_day, args=(date.fromisoformat(pending["day"]),))
    st.subheader(day.strftime("%A, %B %d").replace(" 0", " "))
    if not tasks:
        st.html('<div class="empty-plan"><strong>No tasks planned</strong>'
                '<p>Add a task to plan and track your work for this day.</p></div>')
    for task in sorted(tasks, key=lambda t: (t["status"] == "completed", t["created_at"])):
        render_task(store, index_dir, task, dev)
    if key := st.session_state.pop("ask_finish", None):
        stopped = store.get_task(key)
        if stopped["status"] == "awaiting_finish":
            stop_confirmation(store, index_dir, stopped, dev)


if __name__ == "__main__":
    main()
