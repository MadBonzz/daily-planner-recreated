"""Open the live planner and its PDF tab, then ask a grounded question over HTTP."""

from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.estimation.run import DATASET, histories, task_inputs
from planner_store import PlannerStore


def run():
    profile = histories("01")[0]
    case = next(r for r in profile["holdouts"] if r["id"] == "bridge-proof")
    fields, files = task_inputs(case, profile["id"], DATASET)
    second_file = next(p for p in (DATASET/"pdfs").rglob("*.pdf") if p.name != files[0][0])
    received = []
    class Model(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.loads(request["messages"][1]["content"])
            received.append(body)
            first = body["passages"][0]
            answer = {"answer": "The supplied handout describes the recovery problem.", "supported": True,
                      "sources": [{"page": first["page"], "quote": first["text"][:45]}]}
            data = json.dumps({"choices": [{"finish_reason": "stop", "message":{"content": json.dumps(answer)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass
    fake = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            store = PlannerStore(Path(tmp)/"live")
            fields.update(day=date.today().isoformat(), fixed_duration=True)
            task = store.create_task(**fields, attachments=[*files, ("Second document.pdf",second_file.read_bytes())])
            with socket.socket() as port:
                port.bind(("127.0.0.1",0))
                number = port.getsockname()[1]
            url = f"http://127.0.0.1:{number}"
            config = dict(os.environ, PLANNER_DATA_DIR=tmp, PLANNER_MODEL_ENDPOINT=f"http://127.0.0.1:{fake.server_port}/v1/chat/completions",
                          PLANNER_MODEL="browser-test", STREAMLIT_BROWSER_GATHER_USAGE_STATS="false")
            with (Path(tmp)/"streamlit.log").open("w") as log:
                app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(number),
                                        "--server.address", "127.0.0.1", "--server.headless", "true"],
                                       cwd=ROOT, env=config, stdout=log, stderr=log)
                try:
                    for _ in range(80):
                        try:
                            with urlopen(url+"/_stcore/health", timeout=1) as health:
                                if health.status == 200:
                                    break
                        except OSError:
                            pass
                        time.sleep(.2)
                    with sync_playwright() as browser_driver:
                        browser = browser_driver.chromium.launch(executable_path=os.environ.get("PLANNER_TEST_CHROMIUM"),
                                                                 headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
                        try:
                            planner = browser.new_page(viewport={"width": 1460, "height": 1000})
                            planner.goto(url)
                            with planner.expect_popup() as opened:
                                planner.get_by_role("link",name="Start with workspace").click(timeout=20000)
                            reader = opened.value
                            expect(reader.get_by_role("heading",name=case["title"])).to_be_visible(timeout=20000)
                            expect(planner.get_by_role("button", name="Pause", exact=True)).to_be_visible(timeout=20000)
                            assert store.running_task()["id"] == task["id"]
                            # The official PDF viewer renders pages into canvases, with its own scroll/zoom controls.
                            expect(reader.locator("canvas").first).to_be_visible(timeout=25000)
                            reader.get_by_placeholder("Ask a question").fill("Which problem is described?")
                            reader.get_by_placeholder("Ask a question").press("Enter")
                            expect(reader.get_by_text("The supplied handout describes the recovery problem.",exact=True)).to_be_visible()
                            assert len(received) == 1 and received[0]["document"] == files[0][0]
                            assert received[0]["passages"] and "completion_note" not in received[0]["task"]
                            reader.get_by_role("combobox",name="Document").click()
                            reader.get_by_role("option",name="Second document.pdf").click()
                            expect(reader.get_by_text("The supplied handout describes the recovery problem.",exact=True)).not_to_be_visible()
                            planner.get_by_role("button", name="Pause", exact=True).click()
                            reader.reload()
                            expect(reader.get_by_role("button",name="Resume timer")).to_be_visible(timeout=15000)
                            assert store.running_task() is None
                            print("New PDF tab, running timer, rendered pages, scoped model chat, attachment switching and pause/reload passed.")
                        finally:
                            browser.close()
                finally:
                    app.terminate()
                    app.wait(15)
    finally:
        fake.shutdown()
        fake.server_close()


if __name__ == "__main__":
    run()
