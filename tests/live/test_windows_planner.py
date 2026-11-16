"""Exercise a clean install via a real browser and the configured model API."""
from contextlib import contextmanager
from datetime import date, timedelta
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import pytest

from planner_store import PlannerStore


ROOT = Path(__file__).resolve().parents[2]
QUESTION = ("This is my assignment pdf. can you create a new distributed computing assignment, "
            "I would estimate it at aroun 200 minutes, the task is around solving questions "
            "related to Paxos, Consensus protocols etc")


@contextmanager
def running_planner(data):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with (data / "streamlit.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py",
                                    "--server.address", "127.0.0.1", "--server.port", str(port),
                                    "--server.headless", "true"], cwd=ROOT, stdout=log, stderr=log,
                                   env={**os.environ, "PLANNER_DATA_DIR": str(data),
                                        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false"})
        try:
            for _ in range(150):
                if process.poll() is not None:
                    pytest.fail(f"Streamlit exited at launch with {process.returncode}")
                try:
                    with urlopen(url + "/_stcore/health", timeout=1) as response:
                        if response.read() == b"ok":
                            break
                except OSError:
                    pass
                time.sleep(.2)
            else:
                pytest.fail("Streamlit did not become healthy")
            yield url
        finally:
            process.terminate()
            process.wait(timeout=15)


@pytest.mark.skipif(os.getenv("PLANNER_TEST_LIVE_BROWSER") != "1", reason="Requires Chromium and a model API")
def test_assignment_from_browser_and_saved_history(tmp_path):
    from playwright.sync_api import expect, sync_playwright

    pdf = (ROOT / "data/tasks/pdfs/01-replica-proofs/bridge-proof.pdf").read_bytes()
    with sync_playwright() as driver:
        browser = driver.chromium.launch(executable_path=os.getenv("PLANNER_TEST_CHROMIUM"))
        page = browser.new_page()
        assistants = page.locator('[data-testid="stChatMessage"]').filter(
            has=page.locator('[data-testid="stChatMessageAvatarAssistant"]'))
        composer = page.get_by_placeholder("Ask about your plan, add a task, or attach a file")

        def say(question, expected):
            previous_turns = assistants.count()
            composer.fill(question)
            composer.press("Enter")
            expect(assistants).to_have_count(previous_turns + 1, timeout=240_000)
            expect(assistants.last).to_contain_text(expected)
            expect(page.locator('[data-testid="stException"]')).to_have_count(0)

        with running_planner(tmp_path) as url:
            page.goto(url)
            expect(composer).to_be_visible(timeout=30_000)
            page.locator('[data-testid="stChatInput"] input[type="file"]').set_input_files(
                {"name": "HW_3_1.pdf", "mimeType": "application/pdf", "buffer": pdf})
            expect(page.locator('[data-testid="stChatInput"]')).to_contain_text("HW_3_1.pdf")
            say(QUESTION, re.compile(r"Added .+200-minute estimate\. Attached HW_3_1\.pdf", re.S))
            store = PlannerStore(tmp_path / "live")
            task, = store.list_tasks()
            task_id = task["id"]
            assert task["estimate_minutes"] == 200 and store.attachment_bytes(task["attachments"][0]["id"]) == pdf
            expect(page.get_by_text("Model forecast", exact=False).first).to_be_visible(timeout=240_000)
            assert store.get_task(task_id)["prediction"]["provenance"]["minutes_by"] == "model"
            say("Edit that task and change its description to Prove Paxos safety", "Updated")
            assert store.get_task(task_id)["description"] == "Prove Paxos safety"
            say("Postpone that task to tomorrow", "Postponed")
            assert store.get_task(task_id)["day"] == (date.today() + timedelta(days=1)).isoformat()

        with running_planner(tmp_path) as url:
            page.goto(url)
            page.get_by_role("button", name="→", exact=True).click()
            expect(page.get_by_text("Prove Paxos safety", exact=True)).to_be_visible()
            page.get_by_role("button", name="Start", exact=True).click()
            page.get_by_role("button", name="Pause", exact=True).click()
            page.get_by_role("button", name="Mark complete").click()
            dialog = page.get_by_role("dialog")
            dialog.get_by_role("spinbutton").fill("210")
            dialog.get_by_role("button", name="Yes, finish task").click()
            expect(page.get_by_text("Completed", exact=True)).to_be_visible(timeout=20_000)
            saved, = store.list_tasks()
            assert saved["id"] == task_id and saved["status"] == "completed" and saved["actual_seconds"] == 210 * 60
            assert store.attachment_bytes(saved["attachments"][0]["id"]) == pdf
            page.get_by_text("Dev", exact=True).click()
            expect(page.get_by_text("Minimize the marker-word failure", exact=True).first).to_be_visible(timeout=30_000)
            page.get_by_text("Analytics & logs", exact=True).click()
            say("which websites do I use the most when I get distracted?", "Websites ranked by recorded off-task time")
            expect(assistants.last).to_contain_text(re.compile(r"[\w.-]+\.[a-z]{2,} — [\d.]+ min"))
            say("what are my biggest distractions when working?", "off-task time")
        browser.close()
