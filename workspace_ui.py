"""Streamlit reader/writer page opened by a native new-tab link."""

from uuid import UUID
from urllib.parse import urlencode, urlsplit, urlunsplit

import streamlit as st

from dev_workspace import history_choices, open_dev_workspace
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
                   on_click=_start if task["status"] != "running" else "ignore",
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


def _chat(task, token, filename, text, draft, doc_id):
    identity = token + ":" + doc_id
    key = "desk-chat:" + identity
    history = st.session_state.setdefault(key, [])
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
        previous = [{"role": row["role"], "content": row["content"]} for row in history]
        turn = {"role": "user", "content": question}
        history.append(turn)
        bubble(turn)
        try:
            with st.spinner("Reading the relevant passages…" if filename else "Thinking through the task…"):
                reply = desk.answer(task, question, document=filename, text=text, draft=draft, dialogue=previous)
        except ModelUnavailable as exc:
            feed.warning(str(exc))
        else:
            turn = {"role": "assistant", "content": reply["answer"], "sources": reply["sources"],
                    "model": " · ".join(v for v in (reply["provider"], reply["model"]) if v)}
            history.append(turn)
            bubble(turn)


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
        actions = st.container(horizontal=True)
        if status in ("ready", "paused") and actions.button("Start timer" if status == "ready" else "Resume timer"):
            _start(store, token)
            st.rerun()
        if status == "running" and actions.button("Pause timer"):
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
        _chat(task, token, filename, text, draft, doc_id)
