"""Plan Smarter UI. Run with python -m streamlit run app.py."""

from collections import Counter
from datetime import date, datetime, timedelta
from html import escape
import os
from pathlib import Path
import sqlite3
import time

import streamlit as st

import activity
import estimator
from dev_workspace import add_sample_task, history_choices, open_dev_workspace, sample_task_choices
from estimate_worker import schedule_pending
from planner_store import PlannerStore
from relevance_feedback import related_ids
from planner_analytics import render_analytics
from report_worker import schedule_reports
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
[class*="st-key-task_header_"] [data-testid="stHorizontalBlock"] {flex-wrap:nowrap}
[class*="st-key-task_header_"] [data-testid="stColumn"]:first-child {flex:1;min-width:0}
[class*="st-key-task_header_"] [data-testid="stColumn"]:last-child {flex:0 0 2.5rem;min-width:2.5rem}
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
    if prediction.get("timing_context") and not prediction.get("evidence"):
        context = prediction["timing_context"]
        if minutes_by == "model":
            context += " This estimate uses the task description and budget without a personal timing comparison."
        st.caption(context)
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


def entered_minutes(task, *, optional):
    hint = (f"Leave blank to keep {duration(task['actual_seconds'])}." if task["actual_seconds"]
            else "Leave blank if you don’t know how much time you spent.")
    return st.number_input("Total time spent (minutes)" + (" · optional" if optional else ""),
                           min_value=0.0, value=None, step=1.0, placeholder="e.g. 30",
                           help=hint if optional else "Enter the total across all sessions.")


def close_time_dialog():
    st.session_state.pop("timer_dialog", None)


@st.dialog("Edit time spent", on_dismiss=close_time_dialog)
def edit_time(store, task, dev):
    st.write(task["title"])
    st.caption(f"Current total · {duration(task['actual_seconds'])}")
    with st.form(f"time-form-{task['id']}"):
        value = entered_minutes(task, optional=False)
        st.caption("A higher total adds time without ActivityWatch logs. A lower total removes the most recent minutes first.")
        submitted = st.form_submit_button("Save time", type="primary")
    if submitted:
        if value is None:
            st.error("Enter how many minutes you spent.")
        elif perform(store, task, "adjust_time", spent_seconds=value * 60):
            close_time_dialog()
            if task["status"] == "completed":
                schedule_reports(store, dev)
            st.rerun()


@st.dialog("Have you finished this task?", on_dismiss=close_time_dialog)
def stop_confirmation(store, index_dir, task, dev):
    stopped = task["status"] == "awaiting_finish"
    st.html(f'<strong>{escape(task["title"])}</strong>')
    if task["actual_seconds"]:
        st.write(f"Time spent so far · {duration(task['actual_seconds'])}")
    value = entered_minutes(task, optional=True)
    spent = None if value is None else value * 60
    note = st.text_area("Completion note (optional)", key=f"finish-note-{task['id']}",
                        placeholder="What affected how long this took?")
    yes, no = st.columns(2)
    if yes.button("Yes, finish task", type="primary", width="stretch"):
        if perform(store, task, "finish", completion_note=note, spent_seconds=spent, queue_report=True):
            close_time_dialog()
            schedule_reports(store, dev)
            st.rerun()
    if no.button("No, continue later" if stopped else "Cancel", width="stretch"):
        if not stopped or perform(store, task, "continue_later", spent_seconds=spent):
            close_time_dialog()
            st.rerun()


def timeline(intervals, *, show_date=False):
    if not intervals:
        st.caption("No intervals to show.")
        return
    rows = []
    spans_days = show_date or len({datetime.fromtimestamp(i["start"]).date() for i in intervals}) > 1
    for event in intervals:
        start = datetime.fromtimestamp(event["start"]).strftime("%b %d %H:%M:%S" if spans_days else "%H:%M:%S")
        kind = "Excluded" if event["excluded"] else event.get("app", "")
        row = {"Start": start, "Length": duration(event["seconds"]), "Device": activity.device_label(event),
               "Laptop app": kind,
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
    st.caption(f"{report['switches']} recorded activity switches.")
    for warning in report.get("warnings", []):
        st.caption(warning)
    with st.expander("Activity timeline"):
        timeline(report["intervals"])


def review_activity(store, task, dev):
    groups = activity.review_groups(task["report"])
    if not groups:
        return
    choices = {"on_task": "Work", "off_task": "Distraction", "uncertain": "Uncertain"}
    by_name = {name: label for label, name in choices.items()}
    rows = []
    for group in groups.values():
        labels = group["labels"]
        rows.append({"Device": activity.device_label(group), "Laptop app": group["app"], "Site": group["domain"], "Window": group["title"],
                     "Phone app": group["phone_app"], "Laptop idle": group["idle"],
                     "Time": duration(group["seconds"]), "Source": ", ".join(sorted(group["sources"])),
                     "Classification": choices[next(iter(labels))] if len(labels) == 1 else "Mixed"})
    with st.form(f'review-{task["id"]}'):
        st.caption("Review how each app, site or window was used. Saved changes inform future reports and estimates.")
        edited = st.data_editor(rows, hide_index=True, width="stretch", key=f'labels-{task["id"]}',
                                disabled=[key for key in rows[0] if key != "Classification"],
                                column_config={"Classification": st.column_config.SelectboxColumn(
                                    options=[*by_name, "Mixed"], required=True)})
        if st.form_submit_button("Save classifications and regenerate report"):
            changes = {key: by_name[row["Classification"]]
                       for key, original, row in zip(groups, rows, edited)
                       if row["Classification"] in by_name and original["Classification"] != row["Classification"]}
            if changes:
                try:
                    store.request_report(task["id"], changes)
                    schedule_reports(store, dev)
                    st.rerun()
                except (ValueError, sqlite3.Error) as error:
                    show_error(error)
            else:
                st.caption("Choose a different classification to save a correction.")


def activity_section(store, task, dev):
    task_id = task["id"]
    job = store.report_job(task_id)
    if job and not job["error"]:
        @st.fragment(run_every="2s", key=f"report-progress-{task_id}")
        def progress():
            current = store.report_job(task_id)
            if not current or current["error"]:
                st.rerun()
            st.info("Generating activity report…")
            schedule_reports(store, dev)

        progress()
        return
    if job:
        st.warning(job["error"])
    task = store.get_task(task_id)
    if task["report"]:
        show_report(task["report"])
        review_activity(store, task, dev)
    if st.button("Rebuild report" if task["report"] else "Generate report", key=f"report-{task_id}"):
        store.request_report(task_id)
        schedule_reports(store, dev)
        st.rerun()


@st.fragment(run_every="1s")
def running_clock(store, task_id):
    task = store.get_task(task_id)
    if task["status"] != "running":
        st.rerun()
    st.markdown(f"**{duration(task['actual_seconds'])}** spent")
    if task["actual_seconds"] >= 2 * 60 * task["estimate_minutes"]:
        st.warning(f"Time spent has reached twice your {minutes(task['estimate_minutes'])} estimate. Still working?")
        if st.button("Stop and review time", key=f"overdue-{task_id}"):
            if perform(store, task, "stop"):
                st.session_state["timer_dialog"] = ("finish", task_id)
                st.rerun()


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


def purge_vectors(store, index_dir):
    for task_id in store.pending_vector_purges():
        try:
            estimator.forget_task(index_dir, task_id)
            store.confirm_vector_purge(task_id)
        except (OSError, RuntimeError, ValueError):
            st.warning("The task is excluded from estimates. Plan Smarter will retry removing its saved vector.")
            return


def close_delete_dialog():
    st.session_state.pop("delete_target", None)


@st.dialog("Delete task", on_dismiss=close_delete_dialog)
def delete_task(store, index_dir, task):
    st.text(f'Remove “{task["title"]}” from the planner?')
    forget = st.checkbox("Also delete this task’s data from memory", value=False, key=f"forget-{task['id']}")
    st.caption("This erases the task, attachments, tracked sessions, activity report and learned feedback."
               if forget else "Its history stays available for estimates and analytics.")
    yes, no = st.columns(2)
    if no.button("No", width="stretch"):
        close_delete_dialog()
        st.rerun()
    if yes.button("Yes, delete", type="primary", width="stretch"):
        try:
            if forget:
                store.delete_task(task["id"])
                purge_vectors(store, index_dir)
            else:
                store.archive_task(task["id"])
        except (ValueError, RuntimeError, OSError, sqlite3.Error) as error:
            show_error(error)
            return
        close_delete_dialog()
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
    archived = {t["id"]: f'{t["day"]} · {t["title"]}' for t in store.archived_tasks()}
    if archived:
        st.subheader("Tasks removed from the planner")
        selected = st.selectbox("Saved in memory", archived.keys(), format_func=archived.get)
        forget = st.checkbox("Permanently erase this task and its data", key=f"erase-saved-{selected}")
        if st.button("Erase selected task", disabled=not forget):
            store.delete_task(selected)
            purge_vectors(store, index_dir)
            st.rerun()
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


def forecast_may_change(task):
    return estimate_pending(task) or task.get("awaiting_related_update", False)


def comparison_summary(prediction):
    used = list(prediction.get("evidence", []))
    for caution in prediction.get("cautions", []):
        used.extend(caution.get("sources", []))
    if "related_tasks" in prediction:
        selected = prediction["related_tasks"]
        titles = " · ".join(escape(row["title"]) for row in selected) or "None"
        st.html(f"<small><strong>Selected comparisons</strong> · {titles}</small>")
    else:
        st.caption("This earlier forecast saved only the comparisons used for timing.")
    if "used_task_ids" in prediction:
        by_id = {row["task_id"]: row for row in [*prediction.get("related_tasks", []), *used] if row.get("task_id")}
        used = [by_id[task_id] for task_id in prediction["used_task_ids"] if task_id in by_id]
    names = {row.get("task_id", row["title"]): row["title"] for row in used}
    citations = " · ".join(escape(title) for title in names.values()) or "None"
    st.html(f"<small><strong>Used in this forecast</strong> · {citations}</small>")


@st.dialog("Edit relevant tasks", width="large")
def edit_comparisons(store, task, options):
    baseline = related_ids(task["prediction"])
    labels = {row["id"]: f'{row["title"]} · {row["category"]} · {row["day"]}' for row in options}
    # Distinguish same-day tasks with identical names and categories in the picker.
    duplicates = {label for label, count in Counter(labels.values()).items() if count > 1}
    for task_id, label in labels.items():
        if label in duplicates:
            labels[task_id] = f"{label} · {task_id[:8]}"
    st.text(task["title"])
    st.caption("Choose completed tasks with recorded time. Your additions and removals also guide future forecasts.")
    with st.form(f"compare-{task['id']}"):
        selected = st.multiselect("Use these comparisons", options=list(labels), format_func=labels.get,
                                  default=[task_id for task_id in baseline if task_id in labels],
                                  placeholder="Choose completed tasks")
        if st.form_submit_button("Save and re-estimate", type="primary"):
            try:
                store.revise_related_tasks(task["id"], selected, previous_ids=baseline, reviewed_ids=list(labels))
            except (ValueError, sqlite3.Error) as error:
                show_error(error)
            else:
                st.rerun()


def estimate_section(store, task, *, read_only=False):
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
    if read_only or task["status"] != "running":
        if task.get("time_edited_at") is not None:
            values.append(f"Time spent {duration(task['actual_seconds'])} (user-adjusted)")
        elif task["status"] == "completed" and not task["actual_seconds"]:
            values.append("Time spent unknown")
        else:
            values.append(f"Recorded {duration(task['actual_seconds'])}")
    st.caption("  ·  ".join(values))
    if initial_pending or estimate_pending(task) or fixed:
        return
    has_forecast = "point_min" in prediction
    if has_forecast:
        st.caption(forecast_source(prediction))
        comparison_summary(prediction)
    if prediction.get("retrieval_error"):
        st.warning(prediction["retrieval_error"])
    if not read_only and task["status"] != "completed":
        with st.container(horizontal=True):
            if st.button("Re-estimate", key=f"estimate-retry-{task['id']}",
                         help="Use the current model, task details and completed history for another forecast."):
                store.retry_estimate(task["id"])
                st.rerun()
            if has_forecast and st.button("Edit relevant tasks", key=f"estimate-edit-{task['id']}"):
                edit_comparisons(store, task, store.comparison_options(task["id"]))
    if not has_forecast:
        return
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


def live_estimate_section(store, task):
    # Refresh only this part of the card. A finished forecast must not close an open dialog.
    @st.fragment(run_every="2s", key=f"estimate-{task['id']}")
    def refresh():
        if forecast_may_change(task):
            try:
                fresh = store.get_task(task["id"])
            except ValueError:
                st.rerun()
            task.update(fresh)
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
    @st.fragment(run_every="2s")
    def refresh():
        if any(forecast_may_change(t) for t in tasks):
            tasks[:] = store.list_tasks(day)
        daily_totals(tasks)
    refresh()


def replay_forecasts(store, task_id):
    rows = []
    for entry in store.forecast_timeline(task_id):
        forecast = entry["prediction"]
        rows.append({"As of": datetime.fromtimestamp(entry["at"]).strftime("%b %d %Y %H:%M"),
                     "Forecast": "Initial" if entry["initial"] else "Re-estimate",
                     "Estimate (min)": forecast.get("point_min"),
                     "Range (min)": f'{forecast["low_min"]:g}–{forecast["high_min"]:g}',
                     "Confidence": forecast["confidence"], "Source": forecast_source(forecast),
                     "Tasks used": ", ".join(row["title"] for row in forecast.get("evidence", [])) or "None"})
    if rows:
        with st.expander("Forecast history"):
            st.dataframe(rows, hide_index=True, width="stretch")


def render_task(store, index_dir, task, dev, *, replay=False):
    key = task["id"]
    status = task["status"]
    with st.container(border=True):
        header, trash = st.container(key=f"task_header_{key}").columns([14, 1], vertical_alignment="center")
        header.html(f'<div class="task-head"><h3>{escape(task["title"])}</h3>'
                    f'<span class="{status}">{STATUS[status]}</span></div>')
        if not replay and trash.button(":material/delete:", type="tertiary", key=f"delete-{key}", help=f'Delete {task["title"]}'):
            st.session_state["delete_target"] = key
        if not replay and st.session_state.get("delete_target") == key:
            delete_task(store, index_dir, task)
        st.html(f'<p class="task-copy">{escape(task["description"])}</p>')
        if not replay and forecast_may_change(task):
            live_estimate_section(store, task)
        else:
            estimate_section(store, task, read_only=replay)
        if replay:
            replay_forecasts(store, key)
        if task["category"]:
            st.html(f'<small>{escape(task["category"])}</small>')
        if not replay and status == "running":
            running_clock(store, key)
        if not replay and status == "running" and time.time() - task["started_at"] >= 8 * 3600:
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
                            st.session_state["timer_dialog"] = ("finish", key)
                            st.rerun()
        if not replay and status in ("running", "paused"):
            left, right = st.columns(2)
            action = "pause" if status == "running" else "resume"
            if left.button(action.title(), key=f"{action}-{key}", width="stretch"):
                if perform(store, task, action):
                    st.rerun()
            if right.button("Stop", key=f"stop-{key}", width="stretch"):
                if perform(store, task, "stop"):
                    st.session_state["timer_dialog"] = ("finish", key)
                    st.rerun()
        elif not replay and status == "ready":
            if st.button("Start", key=f"start-{key}", type="primary"):
                if perform(store, task, "start"):
                    st.rerun()
        elif not replay and status == "awaiting_finish":
            if st.button("Confirm completion", key=f"confirm-{key}", type="primary"):
                st.session_state["timer_dialog"] = ("finish", key)
        if not replay and status in ("ready", "paused", "completed"):
            controls = st.columns(2)
            if controls[0].button("Edit time spent", key=f"edit-time-{key}"):
                st.session_state["timer_dialog"] = ("edit", key)
            if status != "completed" and controls[1].button("Mark complete", key=f"finish-{key}"):
                st.session_state["timer_dialog"] = ("finish", key)
        if not replay:
            dialog = st.session_state.get("timer_dialog")
            if dialog == ("edit", key) and status in ("ready", "paused", "completed"):
                edit_time(store, task, dev)
            elif dialog == ("finish", key) and status in ("ready", "paused", "awaiting_finish"):
                stop_confirmation(store, index_dir, task, dev)
        if status == "running" and not dev:
            if st.toggle("Live activity", key=f"live-{key}", help="Preview the last two minutes."):
                live_activity(store, task)
        if dev and task["actual_seconds"] > 0:
            if st.toggle("ActivityWatch log", key=f"sample-log-{key}"):
                from dev_schedule import activity_preview

                intervals = activity_preview(store, task, store.get_setting("exclusions", []))
                timeline([{key: value for key, value in interval.items() if key not in ("label", "reason")}
                          for interval in intervals], show_date=True)
        if status == "completed":
            if task["completion_note"]:
                st.caption("Completion note")
                st.text(task["completion_note"])
            with st.expander("Activity report", expanded=True):
                if not replay:
                    activity_section(store, task, dev)
                elif task["report"]:
                    show_report(task["report"])
                else:
                    st.caption("No activity report was recorded.")
        with st.expander("Task details"):
            if task["details"]:
                st.text(task["details"])
            for item in task["attachments"]:
                st.download_button("Download PDF", store.attachment_bytes(item["id"]), item["name"],
                                   mime="application/pdf", key=f"pdf-{item['id']}", on_click="ignore")
                st.text(item["name"])
            if not replay and status != "completed":
                with st.form(f"reschedule-{key}"):
                    new_day = st.date_input("Scheduled day", value=date.fromisoformat(task["day"]), key=f"date-{key}")
                    if st.form_submit_button("Move task"):
                        store.reschedule(key, new_day)
                        st.rerun()


def replay_controls(store, history_id):
    from dev_schedule import advance, progress

    state = progress(store, history_id)
    profile = str(store.path)
    focus_day = state["last_day"] or state["next_day"]
    if profile != st.session_state.get("replay_profile") and focus_day:
        select_day(date.fromisoformat(focus_day))
    st.session_state["replay_profile"] = profile
    st.progress(state["done"] / max(1, state["total"]),
                text=f'{state["done"]} of {state["total"]} scheduled actions applied')
    if state["done"] < state["total"]:
        upcoming = state["events"][state["done"]]
        st.caption(f'Next · {upcoming["day"]} {upcoming["time"]} · {upcoming["action"]} · {upcoming["task"]}')
        step, run = st.columns(2)
        single = step.button("Apply next action", key="replay-step", type="primary", width="stretch")
        all_actions = run.button("Run to end", key="replay-run", width="stretch")
        if single or all_actions:
            with st.spinner("Applying scheduled actions and processing model jobs…"):
                advance(store, history_id, through_end=all_actions)
            applied = progress(store, history_id)["last_day"]
            if applied:
                select_day(date.fromisoformat(applied))
            st.rerun()
    with st.expander("Replay schedule"):
        st.caption("The schedule controls task creation, timers and completion in this workspace.")
        st.dataframe([{"Date": e["day"], "Time": e["time"], "Task": e["task"], "Action": e["action"],
                       "Status": "Applied" if e["applied"] else "Upcoming"} for e in state["events"]],
                     hide_index=True, width="stretch")
        days = sorted({date.fromisoformat(e["day"]) for e in state["events"]})
        if days:
            day, jump = st.columns([3, 1], vertical_alignment="bottom")
            selected = day.selectbox("Schedule date", days, key=f"replay-date-{history_id}")
            jump.button("Open date", key="replay-jump", on_click=select_day, args=(selected,), width="stretch")
        if st.button("Restart replay", key="replay-reset", help="Start this schedule again from its first action."):
            open_dev_workspace(DATA_ROOT, history_id, reset=True, replay=True)
            close_time_dialog()
            st.rerun()


def main():
    st.session_state.setdefault("plan_day", date.today())
    title, options = st.container(key="plan_header").columns([4, 1], vertical_alignment="center")
    title.title("Plan Smarter")
    mode = st.segmented_control("Mode", ["Demo", "Dev"], default="Demo", required=True,
                                key="planner_mode", on_change=select_day, args=(date.today(),),
                                help="Demo is your planner. Dev loads editable sample histories.")
    dev = mode == "Dev"
    replay = False
    if dev:
        choices = history_choices()
        selected = st.selectbox("Sample history", list(choices), format_func=choices.get,
                                key="dev_history", on_change=select_day, args=(date.today(),))
        view = st.radio("History view", ["Editable history", "Chronological replay"], horizontal=True,
                        key="dev_history_view", on_change=select_day, args=(date.today(),))
        replay = view == "Chronological replay"
        store = open_dev_workspace(DATA_ROOT, selected, replay=replay)
        if replay:
            replay_controls(store, selected)
        else:
            st.session_state.pop("replay_profile", None)
            with st.expander("History details"):
                st.caption("Your edits are saved separately for each sample.")
                if not store.get_setting("dev_calendar"):
                    st.caption("Reset this saved sample to load the dated task schedule and activity logs.")
                past = store.list_tasks()
                st.dataframe([{"Date": t["day"], "Task": t["title"], "Status": STATUS[t["status"]],
                               "Duration (min)": t["estimate_minutes"], "Recorded (min)": round(t["actual_seconds"] / 60, 1)}
                              for t in past], hide_index=True, width="stretch")
                available_days = sorted({date.fromisoformat(t["day"]) for t in past}, reverse=True)
                if available_days:
                    date_column, jump = st.columns([3, 1], vertical_alignment="bottom")
                    target = date_column.selectbox("Task date", available_days, key=f"dev-date-{selected}")
                    jump.button("Open date", on_click=select_day, args=(target,), width="stretch")
                samples = sample_task_choices(selected)
                if samples:
                    choice, add = st.columns([3, 1], vertical_alignment="bottom")
                    sample_id = choice.selectbox("Try a sample task", list(samples),
                                                 format_func=lambda key: samples[key]["title"], key=f"sample-task-{selected}",
                                                 help="Add the task and its documents to the selected plan date. Its final estimate uses your configured forecast model.")
                    if add.button("Add sample task", width="stretch"):
                        add_sample_task(store, selected, sample_id, st.session_state["plan_day"])
                        st.rerun()
                if st.button("Reset sample history", help="Restore the original tasks and discard edits to this sample."):
                    open_dev_workspace(DATA_ROOT, selected, reset=True)
                    select_day(date.today())
                    close_time_dialog()
                    st.rerun()
    else:
        store = PlannerStore(DATA_ROOT / "live")
    index_dir = store.data_dir / "vectors"
    if not replay:
        schedule_reports(store, dev)
        purge_vectors(store, index_dir)
        if store.has_estimation_work():
            schedule_pending(store, index_dir)
        with options.popover("Options", width="stretch"):
            open_settings = st.button("Planner settings")
        if open_settings:
            settings(store, index_dir, dev)
    if "notice" in st.session_state:
        st.warning(st.session_state.pop("notice"))
    page = st.segmented_control("View", ["Daily plan", "Analytics & logs"], default="Daily plan",
                                required=True, key="planner_view")
    if page == "Analytics & logs":
        render_analytics(store)
        return
    controls = st.container(key="plan_navigation").columns([1, 2, 1, 1, 2], vertical_alignment="bottom")
    controls[0].button("←", key="prev", help="Previous day", on_click=move_day, args=(-1,), width="stretch")
    day = controls[1].date_input("Plan date", key="plan_day", format="MM/DD/YYYY",
                                 label_visibility="collapsed")
    controls[2].button("→", key="next", help="Next day", on_click=move_day, args=(1,), width="stretch")
    controls[3].button("Today", on_click=select_day, args=(date.today(),), width="stretch")
    if not replay and controls[4].button("＋ Add task", type="primary", width="stretch"):
        new_task(store, day)
    tasks = store.list_tasks(day)
    if not replay and any(forecast_may_change(t) for t in tasks):
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
    other = store.worked_on(day) if dev else []
    if not tasks and not other:
        st.html('<div class="empty-plan"><strong>No tasks planned</strong>'
                + ("<p>Use the replay schedule to browse or advance to a task.</p>" if replay else
                   "<p>Add a task to plan and track your work for this day.</p>") + "</div>")
    for task in sorted(tasks, key=lambda t: (t["status"] == "completed", t["created_at"])):
        render_task(store, index_dir, task, dev, replay=replay)
    if other:
        st.subheader("Also worked on this day" if tasks else "Worked on this day")
        for task in other:
            render_task(store, index_dir, task, dev, replay=replay)



if __name__ == "__main__":
    main()
