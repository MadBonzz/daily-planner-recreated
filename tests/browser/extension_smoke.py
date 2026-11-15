"""Exercise real Chromium session rules on local test pages with a disposable browser profile."""

from contextlib import contextmanager
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Thread

from playwright.sync_api import Error, expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import focus_bridge
from focus_domains import save_domains
from planner_store import PlannerStore


@contextmanager
def test_site():
    paths = []

    class Page(BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b"<title>Test destination</title><h1>Test destination</h1>")

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Page) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_port, paths
        finally:
            server.shutdown()
            thread.join()


def check_policy(context, store, task, site, visits):
    token = focus_bridge.bind(store.data_dir.parent, store.path)["token"]
    focus_bridge.ensure_listener(store.data_dir.parent)
    worker = (context.service_workers or [context.wait_for_event("serviceworker")])[0]
    worker.evaluate("(token) => chrome.storage.local.set({pairToken: token})", token)
    target = context.new_page()
    target.goto(f"http://focus-blocked.example.com:{site}/already-open")
    expect(target.get_by_role("heading", name="Test destination")).to_be_visible()
    store.transition(task, "start")
    # The open tab must be covered as well as newly requested pages.
    target.wait_for_url("chrome-extension://*/blocked.html*", timeout=45_000)
    other = context.new_page()
    other.goto(f"http://focus-allowed.example.com:{site}/allowed")
    expect(other.get_by_role("heading", name="Test destination")).to_be_visible()
    blocked_url = f"http://focus-blocked.example.com:{site}/new-navigation"
    try:
        other.goto(blocked_url)
    except Error as exc:
        if not any(reason in str(exc) for reason in ("ERR_BLOCKED_BY_CLIENT", "ERR_ABORTED", "interrupted by another navigation")):
            raise
    assert "/new-navigation" not in visits, "A new navigation reached the blocked test site"
    store.transition(task, "pause")
    target.wait_for_url(f"http://focus-blocked.example.com:{site}/already-open", timeout=45_000)
    # The extension may also be restoring the tab used for the failed navigation.
    released = context.new_page()
    released.goto(blocked_url)
    expect(released.get_by_role("heading", name="Test destination")).to_be_visible()
    assert "/new-navigation" in visits, "Pausing did not release the original rule"


def main():
    extension = ROOT / "browser_extension"
    with TemporaryDirectory() as tmp, test_site() as (site, visits), sync_playwright() as p:
        root = Path(tmp)
        store = PlannerStore(root / "live")
        task = store.create_task(title="Fix the parser", description="Trace the missing delimiter",
                                 category="Programming", estimate_minutes=30, day=date.today(),
                                 prediction={})["id"]
        save_domains(store, task, selected=["focus-blocked.example.com"], reviewed=[])
        context = p.chromium.launch_persistent_context(
            root / "test-browser", channel="chromium", headless=True,
            args=[f"--disable-extensions-except={extension}", f"--load-extension={extension}",
                  "--no-proxy-server", "--host-resolver-rules=MAP *.example.com 127.0.0.1"],
        )
        try:
            check_policy(context, store, task, site, visits)
        finally:
            context.close()
            for server in focus_bridge.LISTENERS.values():
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    main()
