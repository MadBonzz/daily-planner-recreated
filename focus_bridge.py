"""Share the running task's domain rules with a paired browser on this computer."""

from contextlib import closing
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import sqlite3
import threading
import time
from urllib.parse import parse_qs, urlsplit


HOST = "127.0.0.1"
PORT = 47843
POLL_SECONDS = 16
LISTENERS = {}
LOCK = threading.Lock()
SCHEMA = """
CREATE TABLE IF NOT EXISTS binding (slot INTEGER PRIMARY KEY CHECK(slot=1), workspace TEXT NOT NULL, token TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS browsers (client TEXT PRIMARY KEY, seen REAL NOT NULL, applied TEXT NOT NULL);
"""


def _path(root):
    return Path(root) / "browser-pairing.sqlite3"


def binding(root):
    if not _path(root).exists():
        return None
    with closing(sqlite3.connect(_path(root), timeout=2)) as db:
        row = db.execute("SELECT workspace, token FROM binding WHERE slot=1").fetchone()
    return {"workspace": row[0], "token": row[1]} if row else None


def _workspace(root, relative):
    base = Path(root).resolve()
    path = (base / relative).resolve()
    parts = path.relative_to(base).parts
    if parts == ("live", "planner.sqlite3"):
        return path
    if (len(parts) in (3, 4) and parts[0] == "dev" and parts[1] != "replays"
            and parts[-1] == "planner.sqlite3" and (len(parts) == 3 or parts[2].startswith("generation-"))):
        return path
    raise ValueError("Choose the personal planner or an editable Dev history.")


def bind(root, workspace):
    relative = Path(workspace).resolve().relative_to(Path(root).resolve()).as_posix()
    if not _workspace(root, relative).is_file():
        raise ValueError("Open this workspace before pairing the browser.")
    _path(root).touch(mode=0o600, exist_ok=True)
    with closing(sqlite3.connect(_path(root), timeout=3)) as db, db:
        db.executescript(SCHEMA)
        db.execute("""INSERT INTO binding VALUES (1,?,?) ON CONFLICT(slot) DO UPDATE SET workspace=excluded.workspace""",
                   (relative, secrets.token_urlsafe(32)))
    return binding(root)


def disable(root):
    if _path(root).is_file():
        with closing(sqlite3.connect(_path(root), timeout=2)) as db, db:
            db.execute("DELETE FROM binding")
            db.execute("DELETE FROM browsers")


def active_workspace(root, relative):
    path = _workspace(root, relative)
    parts = path.relative_to(Path(root).resolve()).parts
    if parts[0] == "dev":
        registry = Path(root).resolve() / "dev" / "profiles.sqlite3"
        with closing(sqlite3.connect(registry.as_uri() + "?mode=ro", uri=True, timeout=1)) as db:
            selected = db.execute("SELECT generation FROM profiles WHERE id=?", (parts[1],)).fetchone()
        expected = parts[2] if len(parts) == 4 else None
        if not selected or selected[0] != expected:
            raise ValueError("This Dev snapshot has been reset. Pair the current workspace in Planner settings.")
    return path


def snapshot(root, config):
    """Read actual committed timer state without creating, migrating or modifying a planner."""
    from focus_domains import policy_from_db

    domains = []
    try:
        path = active_workspace(root, config["workspace"])
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            replay = db.execute("SELECT value FROM settings WHERE key='dev_replay_enabled'").fetchone()
            running = db.execute("SELECT id FROM tasks WHERE archived=0 AND status='running'").fetchone()
            if running and not (replay and json.loads(replay[0])):
                domains = sorted(set(policy_from_db(db, running["id"])["domains"]))
    except (OSError, ValueError, sqlite3.Error):
        # A reset/removed/locked workspace cannot authorize blocking.
        domains = []
    revision = hashlib.sha256(json.dumps(domains, separators=(",", ":")).encode()).hexdigest()
    return {"domains": domains, "revision": revision}


def recent_browsers(root, current_revision):
    if not _path(root).is_file():
        return 0, 0
    with closing(sqlite3.connect(_path(root), timeout=2)) as db:
        rows = db.execute("SELECT applied FROM browsers WHERE seen>?", (time.time() - 45,)).fetchall()
    return len(rows), sum(row[0] == current_revision for row in rows)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _reply(self, status, payload=None):
        body = json.dumps(payload or {}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (ConnectionError, TimeoutError):
            pass

    def do_GET(self):
        parts = urlsplit(self.path)
        origin = self.headers.get("Origin")
        if (self.headers.get("Host") != f"{HOST}:{self.server.server_port}"
                or (origin and not re.fullmatch(r"chrome-extension://[a-p]{32}", origin))
                or parts.path != "/v1/sites"):
            return self._reply(403)
        auth = self.headers.get("Authorization", "")
        client = self.headers.get("X-Planner-Browser", "")
        if not re.fullmatch(r"[a-zA-Z0-9-]{8,80}", client) or not auth.startswith("Bearer "):
            return self._reply(403)
        after = parse_qs(parts.query).get("after", [""])[0]
        if after and not re.fullmatch(r"[a-f0-9]{64}", after):
            return self._reply(400)
        root = self.server.root
        deadline = time.monotonic() + self.server.poll_seconds
        recorded = False
        while True:
            config = binding(root)
            if not config or not hmac.compare_digest(auth.encode(), ("Bearer " + config["token"]).encode()):
                return self._reply(403)
            if not recorded:
                with closing(sqlite3.connect(_path(root), timeout=2)) as db, db:
                    db.execute("INSERT OR REPLACE INTO browsers(client,seen,applied) VALUES (?,?,?)",
                               (client, time.time(), after))
                recorded = True
            policy = snapshot(root, config)
            if policy["revision"] != after or time.monotonic() >= deadline:
                return self._reply(200, policy)
            time.sleep(0.6)


def ensure_listener(root, *, port=PORT):
    """Use a single loopback listener across Streamlit sessions for an enabled data directory."""
    if not binding(root):
        return
    address = (Path(root).resolve(), port)
    with LOCK:
        if address in LISTENERS:
            return
        server = ThreadingHTTPServer((HOST, port), _Handler)
        server.root = address[0]
        server.poll_seconds = POLL_SECONDS
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, name="plan-smarter-browser-guard", daemon=True)
        thread.start()
        LISTENERS[address] = server
