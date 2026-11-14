"""Daily planner UI. Run with python -m streamlit run app.py."""

from datetime import date, datetime, timedelta
from html import escape
import os
from pathlib import Path
import sqlite3
import time

import streamlit as st

import activity
import estimator
from planner_store import PlannerStore
from model_client import model_settings
from urllib.parse import urlsplit

st.set_page_config(page_title="Daily planner", page_icon="🗓️", layout="centered")
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


def sync_index(index_dir, task):
    try:
        estimator.index_task(index_dir, task)
    except (OSError, ValueError, RuntimeError):
        notice("Task saved. The planner will retry indexing it when it searches completed work.")


def show_prediction(prediction):
    if not prediction or "point_min" not in prediction:
        st.caption("No planner prediction was recorded for this example.")
        return
    st.markdown(f"**{minutes(prediction['point_min'])} suggested** · "
                f"{prediction['low_min']}–{prediction['high_min']} min planning range · "
                f"{prediction['confidence']} confidence")
    # The estimate itself is frozen when the task is created.
    st.html(f'<p class="task-copy">{escape(prediction.get("explanation", ""))}</p>')
    evidence = prediction.get("evidence", [])
    if evidence:
        st.dataframe([{"Completed task": row["title"], "Your estimate (min)": row["estimate_min"],
                       "Actual (min)": round(row["actual_min"], 2),
                       "Correction": f"{row['ratio']:.2f}×", "Why it matches": row["reason"]}
                      for row in evidence], hide_index=True, width="stretch")


@st.dialog("Add task", width="large")
def new_task(store, index_dir, day):
    with st.form("new_task_form"):
        title = st.text_input("Task name", placeholder="Read the optimization paper")
        description = st.text_area("Description", placeholder="What do you want to finish?", height=86)
        left, right = st.columns(2)
        estimate = left.number_input("Your estimate (minutes)", min_value=1, value=30, step=1)
        category = right.text_input("Category (optional)", placeholder="Reading")
        details = st.text_area("More details (optional)", height=80,
                               placeholder="Which sections matter? How familiar are you with this work?")
        files = st.file_uploader("Attach PDFs (optional)", type=["pdf"], accept_multiple_files=True,
                                 help="Readable text from the documents can inform the estimate.")
        submitted = st.form_submit_button("Create task", type="primary")
    if not submitted:
        return
    if not title.strip() or not description.strip():
        st.error("Enter a task name and description.")
        return
    attachments = [(file.name, file.getvalue()) for file in files]
    unreadable = []
    excerpts = []
    with st.spinner("Preparing your estimate…"):
        for name, data in attachments:
            excerpt = estimator.extract_pdf_text(data)
            if excerpt:
                excerpts.append(excerpt)
            else:
                unreadable.append(name)
        fields = dict(title=title.strip(), description=description.strip(), details=details.strip(),
                      category=category.strip(), pdf_text="\n".join(excerpts),
                      estimate_minutes=estimate, day=day)
        try:
            prediction = estimator.predict(fields, store.completed_tasks(), index_dir)
            task = store.create_task(**fields, prediction=prediction, attachments=attachments)
        except (ValueError, sqlite3.Error, OSError) as error:
            show_error(error)
            return
    sync_index(index_dir, task)
    if unreadable:
        notice("Task saved. These PDFs were attached but yielded no readable text: " + ", ".join(unreadable))
    st.rerun()


def perform(store, task, action, **kwargs):
    try:
        return store.transition(task["id"], action, **kwargs)
    except (ValueError, sqlite3.Error) as error:
        show_error(error)
        return None


def generate_report(store, task, demo):
    with st.spinner("Reviewing recorded activity…"):
        report = activity.build_report(task, store.periods(task["id"]),
                                       exclusions=store.get_setting("exclusions", []),
                                       disabled_periods=store.disabled_periods(task["id"]),
                                       capture=not demo)
        store.save_report(task["id"], report)


@st.dialog("Have you finished this task?")
def stop_confirmation(store, index_dir, task, demo):
    st.html(f'<strong>{escape(task["title"])}</strong>')
    st.write(f"Your timer stopped at {duration(task['actual_seconds'])} of recorded work.")
    yes, no = st.columns(2)
    if yes.button("Yes, finish task", type="primary", width="stretch"):
        saved = perform(store, task, "finish")
        if saved:
            sync_index(index_dir, saved)
            generate_report(store, saved, demo)
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
            estimator.forget_task(index_dir, task["id"])
            store.delete_task(task["id"])
        except (ValueError, RuntimeError, OSError, sqlite3.Error) as error:
            show_error(error)
            return
        st.rerun()


@st.dialog("Planner settings", width="large")
def settings(store, index_dir, demo):
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
    st.subheader("Activity capture")
    if demo:
        st.caption("Example mode uses a separate history and skips live activity collection.")
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
    which = "example" if demo else "personal"
    confirm = st.checkbox(f"Delete my {which} tasks, PDFs, reports and estimate history.")
    if st.button("Delete planner data", disabled=not confirm):
        try:
            estimator.clear_index(index_dir)
            store.clear_data()
        except (OSError, RuntimeError, sqlite3.Error) as error:
            show_error(error)
            return
        st.rerun()


def move_day(days):
    st.session_state["plan_day"] += timedelta(days=days)


def select_day(day):
    st.session_state["plan_day"] = day


def render_task(store, index_dir, task, demo):
    key = task["id"]
    status = task["status"]
    with st.container(border=True):
        st.html(f'<div class="task-head"><h3>{escape(task["title"])}</h3>'
                f'<span class="{status}">{STATUS[status]}</span></div>'
                f'<p class="task-copy">{escape(task["description"])}</p>')
        point = task["prediction"].get("point_min")
        values = [f"Your estimate {minutes(task['estimate_minutes'])}"]
        if point is not None:
            pred = task["prediction"]
            values.append(f"Planner {minutes(point)} ({pred['low_min']:g}–{pred['high_min']:g} min, "
                          f"{pred['confidence']} confidence)")
        if status != "running":
            values.append(f"Recorded {duration(task['actual_seconds'])}")
        st.caption("  ·  ".join(values))
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
                stop_confirmation(store, index_dir, task, demo)
        with st.expander("Why this estimate?"):
            show_prediction(task["prediction"])
        if status == "running" and not demo:
            if st.toggle("Live activity", key=f"live-{key}", help="Preview the last two minutes."):
                live_activity(store, task)
        if status == "completed":
            with st.expander("Activity report", expanded=True):
                if task["report"]:
                    show_report(task["report"])
                else:
                    st.caption("An activity report has not been generated for this task.")
                if st.button("Rebuild report" if task["report"] else "Generate report", key=f"report-{key}"):
                    generate_report(store, task, demo)
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
    title.title("Daily planner")
    with options.popover("Options", width="stretch"):
        mode = st.radio("Workspace", ["My planner", "Example history"], key="data_mode")
        open_settings = st.button("Planner settings")
    demo = mode == "Example history"
    data_dir = DATA_ROOT / ("demo" if demo else "live")
    store = PlannerStore(data_dir)
    index_dir = data_dir / "vectors"
    if open_settings:
        settings(store, index_dir, demo)
    if demo:
        st.info("Example workspace · All records here are separate from your personal history.")
        if not store.get_setting("example_history_seeded", False):
            if st.button("Load example history"):
                store.seed_demo()
                st.rerun()
        else:
            with st.expander("Example history"):
                st.dataframe([{"Date": t["day"], "Completed task": t["title"],
                               "Estimated (min)": t["estimate_minutes"],
                               "Actual (min)": round(t["actual_seconds"] / 60, 2)}
                              for t in store.completed_tasks()], width="stretch", hide_index=True)
    if "notice" in st.session_state:
        st.warning(st.session_state.pop("notice"))
    controls = st.container(key="plan_navigation").columns([1, 2, 1, 1, 2], vertical_alignment="bottom")
    controls[0].button("←", key="prev", help="Previous day", on_click=move_day, args=(-1,), width="stretch")
    day = controls[1].date_input("Plan date", key="plan_day", format="MM/DD/YYYY",
                                 label_visibility="collapsed")
    controls[2].button("→", key="next", help="Next day", on_click=move_day, args=(1,), width="stretch")
    controls[3].button("Today", on_click=select_day, args=(date.today(),), width="stretch")
    if controls[4].button("＋ Add task", type="primary", width="stretch"):
        new_task(store, index_dir, day)
    tasks = store.list_tasks(day)
    original = sum(t["estimate_minutes"] for t in tasks)
    budget = sum(t["prediction"].get("point_min", t["estimate_minutes"]) for t in tasks)
    done = sum(t["status"] == "completed" for t in tasks)
    st.html('<div class="metric-grid">' + "".join(
        f"<div><small>{label}</small><strong>{value}</strong></div>" for label, value in
        (("Your estimates", minutes(original)), ("Planner budget", minutes(budget)),
         ("Tasks complete", f"{done} / {len(tasks)}"))) + '</div>')
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
        render_task(store, index_dir, task, demo)
    if key := st.session_state.pop("ask_finish", None):
        stopped = store.get_task(key)
        if stopped["status"] == "awaiting_finish":
            stop_confirmation(store, index_dir, stopped, demo)


if __name__ == "__main__":
    main()
