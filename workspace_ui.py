"""Streamlit reader/writer page opened by a native new-tab link."""

from uuid import UUID
from urllib.parse import urlencode, urlsplit, urlunsplit

import streamlit as st

from dev_workspace import history_choices, open_dev_workspace
from estimate_worker import schedule_pending
from model_client import ModelUnavailable
from pdf_reader import extract_pdf_text
from planner_store import PlannerStore
import task_workspace as desk


def _start(store, token):
    try:
        desk.start_workspace(store, token)
    except ValueError as exc:
        st.session_state["notice"] = str(exc)


def _button(store, task, choice, sample):
    if choice["eligible"] is None:
        if st.button("Retry workspace check", key="retry-desk-" + task["id"], help=choice["source"]):
            desk.availability(store, task, retry=True)
            st.rerun()
        return
    if not choice["eligible"]:
        return
    query = {"workspace": choice["token"]}
    if sample:
        query["sample"] = sample
    here = urlsplit(st.context.url or "")
    destination = urlunsplit((here.scheme, here.netloc, here.path, urlencode(query), ""))
    label = {"ready": "Start with workspace", "paused": "Resume with workspace", "running": "Open workspace"}
    st.link_button(label[task["status"]], destination, key="open-workspace-" + task["id"],
                   on_click=_start,
                   args=(store, choice["token"]), icon=":material/open_in_new:",
                   help="Open this task’s reader or writing desk in a new tab.")


def offer_workspace(store, task, sample=None):
    decision = desk.availability(store, task)
    if decision is not None:
        _button(store, task, decision, sample)
        return

    @st.fragment(run_every="2s")
    def check():
        if desk.availability(store, task) is not None:
            st.rerun()
        st.caption("Checking workspace availability…")
    check()


def _chat(store, task, token, filename, text, draft, doc_id):
    identity = token + ":" + doc_id
    try:
        history = desk.chat_history(store, token, doc_id)
    except ValueError as exc:
        st.warning(str(exc))
        return
    st.subheader("Ask about this document" if filename else "Task chat")
    if filename and not text:
        st.info("This PDF has no readable text for chat. You can still use the reader.")
        return
    feed = st.container(height=470, border=False)
    def bubble(row):
        with feed.chat_message(row["role"]):
            st.text(row["content"])
            if row.get("model"):
                st.caption(row["model"])
            for source in row.get("sources", []):
                st.text(f'Page {source["page"]} · “{source["quote"]}”')
    for row in history:
        bubble(row)
    question = st.chat_input("Ask a question", key="desk-question:" + identity)
    if question:
        try:
            turn_id = desk.begin_question(store, token, doc_id, question)
        except ValueError as exc:
            feed.warning(str(exc))
            return
        previous = [{"role": row["role"], "content": row["content"]} for row in history]
        turn = {"role": "user", "content": question}
        bubble(turn)
        try:
            with st.spinner("Reading the relevant passages…" if filename else "Thinking through the task…"):
                reply = desk.answer(task, question, document=filename, text=text, draft=draft, dialogue=previous)
            desk.finish_question(store, token, doc_id, turn_id, reply)
        except (ModelUnavailable, ValueError) as exc:
            feed.warning(str(exc))
        else:
            turn = {"role": "assistant", "content": reply["answer"], "sources": reply["sources"],
                    "model": " · ".join(v for v in (reply["provider"], reply["model"]) if v)}
            bubble(turn)


def pause_summary(task):
    """Show the state of the last chat-based estimate on either task surface."""
    if task.get("workspace_progress_pending"):
        st.caption("Updating estimate from workspace chat…")
        return
    prediction = task["prediction"]
    update = prediction.get("workspace_update")
    if not update:
        return
    if update["status"] == "adjusted":
        st.caption(f"At the last workspace pause · {round(update['elapsed_minutes'], 1):g} min spent · "
                   f"{update['remaining_min']:g} min remaining · "
                   f"{prediction['point_min']:g} min total (before chat review {update['previous_total_min']:g} min)")
    elif update["status"] == "insufficient_evidence":
        st.caption(f"The workspace chat gave too little evidence to revise the {prediction['point_min']:g} min estimate.")
    else:
        st.caption(f"Workspace estimate unavailable. The current forecast is {prediction['point_min']:g} min.")


def pause_evidence(update, *, show_reason=True):
    if show_reason and update.get("explanation"):
        st.text(update["explanation"])
    support = update.get("last_adjusted") or update
    if support is not update and support.get("evidence"):
        st.caption("Workspace evidence behind the current forecast")
    for item in support.get("evidence", []):
        document = item.get("document")
        label = {"completed": "Completed", "remaining": "Remaining", "blocker": "Blocker",
                 "task_finished": "Reported finished"}.get(item.get("kind"), "Progress")
        st.text(f'{label} · “{item["quote"]}”' + (f" · {document}" if document else ""))


def display(data_root, *, confirm_stop):
    token = st.query_params.get("workspace", "")
    sample = st.query_params.get("sample", "")
    try:
        UUID(token)
        if sample and sample not in history_choices():
            raise ValueError
        store = open_dev_workspace(data_root, sample) if sample else PlannerStore(data_root / "live")
        saved = desk.open_workspace(store, token)
        if saved is None:
            raise ValueError
    except (ValueError, FileNotFoundError):
        st.error("This task workspace is unavailable.")
        return
    task, draft = saved
    if store.has_estimation_work():
        schedule_pending(store, store.data_dir / "vectors")
    st.html("<style>[data-testid=stMainBlockContainer]{max-width:100%;padding:2.5rem clamp(1rem,3vw,3rem)}</style>")
    st.title(task["title"])
    st.write(task["description"])

    @st.fragment(run_every="3s")
    def clock():
        current = desk.open_workspace(store, token)
        if not current:
            st.rerun()
        state = current[0]
        status = state["status"]
        st.caption(f"Timer · {status.replace('_', ' ')} · {state['actual_seconds']/60:.1f} min spent")
        pause_summary(state)
        update = state["prediction"].get("workspace_update")
        if update and not state.get("workspace_progress_pending"):
            with st.expander("Why this workspace estimate?"):
                if update["status"] == "adjusted":
                    forecast = state["prediction"]
                    st.caption(f"{forecast['low_min']:g}–{forecast['high_min']:g} min total · "
                               f"{forecast['confidence']} confidence")
                pause_evidence(update)
                source = update.get("provenance") or {}
                if source.get("model"):
                    st.caption(" · ".join(v for v in (source.get("provider"), source["model"]) if v))
                if update["status"] == "unavailable" and status == "paused" and st.button("Retry workspace estimate"):
                    store.retry_estimate(task["id"])
                    st.rerun()
        actions = st.container(horizontal=True)
        if status in ("ready", "paused") and actions.button("Start timer" if status == "ready" else "Resume timer"):
            _start(store, token)
            st.rerun()
        if status == "running" and actions.button("Pause timer", help="Pause and review progress from this session’s workspace chat."):
            store.transition(task["id"], "pause")
            st.rerun()
        if status in ("running", "paused") and actions.button("Stop timer"):
            store.transition(task["id"], "stop")
            st.session_state["timer_dialog"] = ("finish", task["id"])
            st.rerun()
        if status == "awaiting_finish" and actions.button("Confirm completion"):
            st.session_state["timer_dialog"] = ("finish", task["id"])
            st.rerun()
    clock()
    if task["status"] == "awaiting_finish" and st.session_state.get("timer_dialog") == ("finish", task["id"]):
        confirm_stop(store, store.data_dir/"vectors", task, bool(sample))
    reader, side = st.columns([7, 3], gap="medium")
    filename = None
    text = ""
    doc_id = "task"
    with reader:
        files = {item["id"]: item["name"] for item in task["attachments"]}
        if files:
            doc_id = (st.selectbox("Document", list(files), format_func=files.get, key="desk-document:" + token)
                      if len(files) > 1 else next(iter(files)))
            filename = files[doc_id]
            data = store.attachment_bytes(doc_id)
            cache = "desk-text:" + token + ":" + doc_id
            if cache not in st.session_state:
                st.session_state[cache] = extract_pdf_text(data)
            text = st.session_state[cache]
            st.pdf(data, height=780, key="desk-pdf:" + token + ":" + doc_id, alt=filename)
        else:
            if task["details"]:
                st.write(task["details"])
            version_key = "desk-draft-base:" + token
            st.session_state.setdefault(version_key, draft)
            with st.form("desk-write:" + token):
                current_draft = st.text_area("Draft", value=draft, height=440, key="desk-draft:" + token)
                if st.form_submit_button("Save draft"):
                    try:
                        desk.save_draft(store, token, current_draft, previous=st.session_state[version_key])
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        draft = current_draft
                        st.session_state[version_key] = draft
                        st.toast("Draft saved")
    with side:
        _chat(store, task, token, filename, text, draft, doc_id)
