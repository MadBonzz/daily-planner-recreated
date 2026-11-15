"""Daily review controls used by the personal Streamlit workspace."""

from datetime import date, datetime, time, timedelta
import sys

import streamlit as st

import daily_review


def review_settings(store):
    config = daily_review.preferences(store)
    st.subheader("Daily review reminder")
    failed = store.get_setting("daily_review_desktop_error", {})
    if config["desktop"] and failed:
        st.warning(f"Windows reminder on {failed['day']} failed. {failed['message']}")
    zone = datetime.now().astimezone().tzname()
    with st.form("daily-review-settings"):
        enabled = st.checkbox("Remind me to review my day", value=config["enabled"])
        at = st.time_input("Reminder time", value=time.fromisoformat(config["time"]), step=60,
                           help=f"Uses the local clock of the computer running Plan Smarter ({zone}).")
        desktop = False
        if sys.platform == "win32":
            desktop = st.checkbox("Notify my Windows desktop even when the planner is closed", value=config["desktop"],
                                  help="Uses Windows Task Scheduler under your signed-in account.")
        if st.form_submit_button("Save review reminder"):
            warning = daily_review.configure(store, at.strftime("%H:%M"), enabled, desktop)
            if warning:
                st.warning(warning + " You can still review your day in the planner.")
            else:
                st.success("Review reminder saved.")


@st.fragment(run_every="30s")
def review_notice(store):
    plan = daily_review.due(store)
    if not plan:
        return
    daily_review.deliver(store, lambda p: st.toast(daily_review.reminder_text(p), icon=":material/event_note:"),
                         desktop=False)
    with st.container(border=True):
        st.markdown(f"**Daily review · {date.fromisoformat(plan['day']):%A, %B %d}**")
        st.caption(f"{plan['done']} of {plan['total']} tasks complete")
        with st.container(horizontal=True):
            if st.button("Review this day", key="review-open"):
                daily_review.acknowledge(store, plan["day"])
                st.session_state["review_target_day"] = plan["day"]
                st.rerun()
            if st.button("Dismiss reminder", key="review-dismiss"):
                daily_review.acknowledge(store, plan["day"])
                st.rerun(scope="app")


def postpone_action(store, task):
    key = task["id"]
    day = date.fromisoformat(task["day"])
    today = date.fromtimestamp(store.clock())
    first = max(day + timedelta(days=1), today)
    suggested = max(first, today + timedelta(days=1))

    def move(target):
        try:
            store.postpone(key, target, previous_day=task["day"])
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["postponed_to"] = target
            st.rerun()

    with st.popover("Postpone", icon=":material/event_repeat:", key=f"postpone-{key}"):
        if task["status"] in ("running", "awaiting_finish"):
            st.caption("Postponing pauses this task and keeps the time already recorded.")
        if st.button("Tomorrow" if suggested == today + timedelta(days=1) else "Next day",
                     key=f"postpone-next-{key}"):
            move(suggested)
        with st.form(f"postpone-form-{key}"):
            chosen = st.date_input("Later plan date", value=suggested, min_value=first, key=f"postpone-date-{key}")
            if st.form_submit_button("Postpone to selected date"):
                move(chosen)
