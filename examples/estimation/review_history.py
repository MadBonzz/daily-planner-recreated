"""Record what the live dashboard shows for an editable sample in a temporary planner."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.request import urlopen

from dev_workspace import history_choices, open_dev_workspace
from model_client import forecast_endpoint


@contextmanager
def dashboard(root):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with (root / "streamlit.log").open("w") as log:
        app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py",
                                "--server.address", "127.0.0.1", "--server.port", str(port),
                                "--server.headless", "true"], env={**os.environ, "PLANNER_DATA_DIR": str(root)},
                               stdout=log, stderr=log)
        try:
            for _ in range(120):
                if app.poll() is not None:
                    raise RuntimeError((root / "streamlit.log").read_text())
                try:
                    with urlopen(url + "/_stcore/health", timeout=1) as response:
                        if response.read() == b"ok":
                            break
                except OSError:
                    pass
                time.sleep(.25)
            else:
                raise TimeoutError("The planner did not start.")
            yield url
        finally:
            app.terminate()
            app.wait(timeout=30)


def card(page, title):
    return page.get_by_role("heading", name=title, exact=True).locator(
        "xpath=ancestor::*[@data-testid='stVerticalBlock'][.//p[@class='task-copy']][1]")


def snapshot(page, store, day):
    # Wait for the worker and the dashboard separately so stale cards stay observable.
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        rows = store.list_tasks(day)
        if all(not (r["estimate_updating"] or r["awaiting_related_update"]) for r in rows):
            break
        page.wait_for_timeout(1000)
    page.wait_for_timeout(3000)
    rows = store.list_tasks(day)
    cards = {}
    for row in rows:
        view = card(page, row["title"])
        forecast = row["prediction"]
        cards[row["title"]] = {
            "state": row["status"], "recorded_minutes": round(row["actual_seconds"] / 60, 2),
            "user_minutes": row["estimate_minutes"], "point": forecast.get("point_min"),
            "range": [forecast.get("low_min"), forecast.get("high_min")],
            "source": (forecast.get("provenance") or {}).get("minutes_by"),
            "reason": forecast.get("model_rationale") or forecast.get("explanation"),
            "selected": [r["title"] for r in forecast.get("related_tasks", [])],
            "used": [r["title"] for r in forecast.get("evidence", [])],
            "errors": [forecast[k] for k in ("model_error", "retrieval_error") if forecast.get(k)],
            "pending": bool(row["estimate_updating"] or row["awaiting_related_update"]),
            "display": view.locator('[data-testid="stCaptionContainer"]').first.inner_text(),
        }
    return {"dashboard": page.locator(".metric-grid").inner_text(), "cards": cards}


def finish(page, title, minutes):
    view = card(page, title)
    status = view.locator(".task-head span").inner_text()
    transitions = [status]
    if status == "Ready":
        view.get_by_role("button", name="Start", exact=True).click()
        view.get_by_text("Running", exact=True).wait_for()
        transitions.append("Running")
    if transitions[-1] == "Running":
        view.get_by_role("button", name="Pause", exact=True).click()
        view.get_by_text("Paused", exact=True).wait_for()
        transitions.append("Paused")
    view.get_by_role("button", name="Mark complete", exact=True).click()
    dialog = page.get_by_role("dialog")
    dialog.get_by_role("spinbutton").fill(str(minutes))
    dialog.get_by_role("button", name="Yes, finish task").click()
    view.get_by_text("Completed", exact=True).wait_for()
    return [*transitions, "Completed"]


def review(args, page, url, root, result):
    page.goto(url)
    page.get_by_text("Dev", exact=True).click()
    name = history_choices()[args.history]
    picker = page.get_by_role("combobox", name="Sample history")
    picker.click()
    page.get_by_role("option", name=name, exact=True).click()
    store = open_dev_workspace(root, args.history)
    day = store.get_setting("dev_calendar")["day"]
    page.get_by_role("heading", name=store.list_tasks(day)[0]["title"], exact=True).wait_for()
    previous = {}
    def record(label):
        nonlocal previous
        observed = snapshot(page, store, day)
        current = observed.pop("cards")
        changes = {title: state for title, state in current.items() if previous.get(title) != state}
        result["stages"].append({"action": label, **observed, "changed_cards": changes})
        previous = current
        print(label, observed["dashboard"].replace("\n", " "), flush=True)
    record("Open editable history")
    for title, minutes in args.finish:
        states = finish(page, title, float(minutes))
        record(f"{title}, {minutes} minutes: " + " → ".join(states))
    page.get_by_text("Analytics & logs", exact=True).click()
    composer = page.get_by_placeholder("Ask about your plan, add a task, or attach a file")
    for question in args.ask:
        messages = page.locator('[data-testid="stChatMessage"]').filter(
            has=page.locator('[data-testid="stChatMessageAvatarAssistant"]'))
        previous_count = messages.count()
        composer.fill(question)
        composer.press("Enter")
        reply = messages.nth(previous_count)
        reply.locator('[data-testid="stMarkdownContainer"] p').first.wait_for(timeout=600_000)
        result["chat"].append({"question": question, "reply": reply.inner_text()})


def main():
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", choices=history_choices())
    parser.add_argument("--finish", nargs=2, action="append", default=[], metavar=("TITLE", "MINUTES"))
    parser.add_argument("--ask", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with forecast_endpoint() as endpoint:
        result = {"history": args.history, "forecast_endpoint": endpoint, "stages": [], "chat": []}
    try:
        with TemporaryDirectory() as tmp, dashboard(Path(tmp)) as url, sync_playwright() as driver:
            browser = driver.chromium.launch(executable_path=os.environ.get("PLANNER_REVIEW_CHROMIUM"))
            try:
                review(args, browser.new_page(), url, Path(tmp), result)
            finally:
                browser.close()
    except Exception as exc:
        result["interrupted"] = str(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
