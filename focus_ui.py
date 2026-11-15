"""Task website choices and browser pairing for the Streamlit dashboard."""

import hashlib
import sqlite3

import streamlit as st

import focus_bridge
import focus_domains


def start_if_paired(data_root):
    try:
        focus_bridge.ensure_listener(data_root)
    except (OSError, sqlite3.Error) as error:
        st.warning(f"Browser blocking could not connect on 127.0.0.1:{focus_bridge.PORT}. {error}")


def website_settings(data_root, store, *, dev):
    st.subheader("Browser blocking")
    config = focus_bridge.binding(data_root)
    current = store.path.resolve().relative_to(data_root.resolve()).as_posix()
    label = "this editable Dev history" if dev else "your personal planner"
    if config:
        parts = config["workspace"].split("/")
        bound = f"Dev history {parts[1]}" if parts[0] == "dev" else "Personal planner"
        st.text(f"Bound workspace · {bound}")
    if not config or config["workspace"] != current:
        if st.button(f"Enable browser blocking for {label}"):
            try:
                focus_bridge.bind(data_root, store.path)
                focus_bridge.ensure_listener(data_root)
                st.rerun()
            except (ValueError, OSError, sqlite3.Error) as exc:
                st.error(str(exc))
                return
    if not config:
        st.caption("Pair a Chrome or Edge extension to block a task’s selected sites while its timer runs.")
        return
    st.caption("In chrome://extensions or edge://extensions, enable Developer mode, select Load unpacked, "
               "and choose the browser_extension folder from this repository. Open the extension and paste this code.")
    st.text_input("Browser pairing code", value=config["token"], type="password", disabled=True,
                  help="Reveal the code to copy it. Pair each browser profile you use.")
    @st.fragment(run_every="4s")
    def connections():
        latest = focus_bridge.binding(data_root)
        if not latest:
            return
        state = focus_bridge.snapshot(data_root, latest)
        count, in_sync = focus_bridge.recent_browsers(data_root, state["revision"])
        st.caption(f"{in_sync} of {count} connected browser profiles have applied the current list "
                   f"({len(state['domains'])} sites).")
    connections()
    if st.button("Disconnect all browsers and release sites"):
        focus_bridge.disable(data_root)
        st.rerun()


def site_editor(store, task):
    key = task["id"]
    with st.expander("Websites to block during this task"):
        policy = focus_domains.policy(store, key)
        if policy["suggestions"]:
            for d, evidence in policy["suggestions"].items():
                st.text(f"{d} · {evidence}")
        else:
            st.caption("No recurring distracting site was found in related tasks. You can add a site yourself.")
        domains = policy["domains"]
        options = sorted(set(domains) | set(policy["suggestions"]) | set(policy["removed"]))
        version = hashlib.sha256("\n".join(domains).encode()).hexdigest()[:12]
        with st.form(f"site-list-{key}-{version}"):
            selected = st.multiselect("Blocked websites", options, default=domains, accept_new_options=True,
                                      key=f"site-options-{key}-{version}", placeholder="Add a domain, such as x.com")
            st.caption("Paired browsers block these domains and their subdomains while this task is running.")
            if st.form_submit_button("Save websites"):
                try:
                    focus_domains.save_domains(store, key, selected, reviewed=domains)
                    st.rerun()
                except (ValueError, sqlite3.Error) as error:
                    st.error(str(error))


def running_status(data_root, store, task):
    """Acknowledge actual browser rule installation for the active task."""
    if not focus_bridge.binding(data_root):
        return

    @st.fragment(run_every="4s", key="focus-status-" + task["id"])
    def status():
        config = focus_bridge.binding(data_root)
        if not config or (data_root / config["workspace"]).resolve() != store.path.resolve():
            return
        state = focus_bridge.snapshot(data_root, config)
        if not state["domains"]:
            st.caption("Browser blocking · no websites selected")
            return
        count, applied = focus_bridge.recent_browsers(data_root, state["revision"])
        sites = ", ".join(state["domains"][:4])
        if len(state["domains"]) > 4:
            sites += f" and {len(state['domains']) - 4} more"
        if applied:
            st.caption(f"Browser blocking · {sites} · applied in {applied} of {count} connected profiles")
        else:
            st.caption(f"Waiting for the browser extension to confirm blocking · {sites}")
    status()
