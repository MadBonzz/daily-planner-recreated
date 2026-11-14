"""Read ActivityWatch aw-server-rust SQLite stores without modifying watcher data."""

from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import socket
import sqlite3
from urllib.parse import unquote

from model_client import planner_settings


NANOSECONDS = 1_000_000_000
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _nanoseconds(value):
    """Translate the collector's UTC instants to the Rust database's epoch nanoseconds."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, datetime):
        if value.utcoffset() is None:
            raise ValueError("ActivityWatch timestamps require a timezone")
        return ((value - EPOCH) // timedelta(microseconds=1)) * 1000
    return int(Decimal(str(value)) * NANOSECONDS)


class SQLiteActivitySource:
    """Serve the collector's bucket list and bounded event reads from a local database."""

    def __init__(self, path, hostname=None):
        self.path = Path(path).expanduser()
        self.hostname = hostname or socket.gethostname()

    def get(self, path, **query):
        # mode=ro prevents writes and accidental creation of a missing database.
        # Keep SQLite's normal locking so a running watcher's committed WAL is visible.
        uri = self.path.resolve().as_uri() + "?mode=ro"
        try:
            with closing(sqlite3.connect(uri, uri=True, timeout=1)) as conn:
                conn.execute("PRAGMA trusted_schema=OFF")
                if path == "buckets/":
                    return {
                        name: {"id": name, "type": kind, "client": client, "hostname": host, "created": created}
                        for name, kind, client, host, created in conn.execute(
                            "SELECT name, type, client, hostname, created FROM buckets ORDER BY id"
                        )
                    }
                prefix, name, suffix = path.split("/")
                if prefix != "buckets" or suffix != "events" or not {"start", "end"} <= query.keys():
                    raise ValueError("A bucket and a bounded time window are required")
                start, end = _nanoseconds(query["start"]), _nanoseconds(query["end"])
                if end <= start:
                    return []
                # A heartbeat can start before the timer. Use both ends and let
                # the normal collector clip to each running period and privacy gap.
                rows = conn.execute(
                    """SELECT e.id, e.starttime, e.endtime, e.data
                       FROM buckets b JOIN events e ON b.id = e.bucketrow
                       WHERE b.name = ? AND e.starttime < ? AND e.endtime > ?
                         AND e.endtime > e.starttime
                       ORDER BY e.starttime DESC, e.id DESC""",
                    (unquote(name), end, start),
                )
                events = []
                for event_id, at, until, raw in rows:
                    try:
                        data = json.loads(raw)
                    except (ValueError, TypeError):
                        data = None  # Let the collector report an invalid event.
                    events.append({"id": event_id, "timestamp": at / NANOSECONDS,
                                   "duration": (until - at) / NANOSECONDS, "data": data})
                return events
        except (sqlite3.Error, OverflowError) as exc:
            raise OSError("Could not read the ActivityWatch SQLite database") from exc


def windows_database_candidates(config=None):
    """Find local and synced stores in the configured user's known Windows directories."""
    if config is None:
        config = planner_settings()
    found = []
    local = config.get("LOCALAPPDATA")
    if local:
        found.append(Path(local) / "activitywatch" / "aw-server-rust" / "sqlite.db")
    profile = config.get("USERPROFILE")
    if profile:
        downloads = Path(profile) / "Downloads"
        roots = sorted(downloads.glob("syncthing-windows-*/syncthing-windows-*/activity-watch-one-plus"))
        for root in roots:
            # Syncthing trees can put the store directly below a UUID or under
            # both a device/host directory and a UUID. Do not scan unrelated downloads.
            for pattern in ("test.db", "*/test.db", "*/*/test.db"):
                found.extend(sorted(root.glob(pattern)))
    return tuple(dict.fromkeys(path for path in found if path.is_file()))


def configured_desktop_source(config=None):
    """Pick the local desktop store, or a synced desktop copy if that is all we have."""
    from activity_phone import is_android_window

    if config is None:
        config = planner_settings()
    path = config.get("PLANNER_ACTIVITYWATCH_DB")
    host = config.get("PLANNER_ACTIVITYWATCH_HOSTNAME")
    preferred = host or socket.gethostname()
    candidates = (path,) if path else windows_database_candidates(config)
    for candidate in candidates:
        source = SQLiteActivitySource(candidate, hostname=preferred)
        try:
            buckets = source.get("buckets/")
        except (ValueError, OSError):
            if path:
                return source  # Surface an explicit path error via the collector.
            continue
        desktops = {str(meta["hostname"]) for key, meta in buckets.items()
                    if meta["type"] == "currentwindow" and not is_android_window(key, meta)}
        if any(h.split(".")[0].casefold() == preferred.split(".")[0].casefold() for h in desktops):
            return source
        if not host and len(desktops) == 1:
            source.hostname = desktops.pop()
            return source
        if path:
            return source
    return None
