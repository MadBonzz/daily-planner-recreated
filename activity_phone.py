"""Read the Android foreground-app stream from a locally synced ActivityWatch database."""

from pathlib import Path
from urllib.parse import quote

from activity_db import SQLiteActivitySource, windows_database_candidates
from model_client import planner_settings


def is_android_window(name, metadata):
    watcher = str(metadata.get("client", "")).casefold()
    bucket = str(name).casefold()
    return metadata.get("type") == "currentwindow" and (
        watcher.startswith("aw-watcher-android") or bucket.startswith("aw-watcher-android")
    )


class AndroidSource:
    def __init__(self, store, buckets):
        self.store = store
        self.buckets = buckets

    def events(self, start, end):
        from activity import _iso, _stamp

        entries = []
        for bucket in self.buckets:
            path = "buckets/" + quote(bucket, safe="") + "/events"
            for row in self.store.get(path, start=_iso(start), end=_iso(end)):
                at = _stamp(row["timestamp"])
                until = at + _stamp(row["duration"])
                if until > start and at < end and until > at and isinstance(row["data"], dict):
                    entries.append((at, until, "phone", bucket, row["data"]))
        return entries


def configured_phone_source(config=None, *, desktop_path=None):
    config = planner_settings() if config is None else config
    override = config.get("PLANNER_ACTIVITYWATCH_PHONE_DB")
    if override:
        candidates = (Path(override).expanduser(),)
    else:
        # A desktop DB can retain an old copy of a phone bucket. Syncthing
        # updates the separate device store. Prefer that copy when available.
        candidates = [path for path in windows_database_candidates(config)
                      if desktop_path is None or path.resolve() != Path(desktop_path).resolve()]
        candidates.sort(key=lambda path: path.name.casefold() != "test.db")
    for path in candidates:
        try:
            source = SQLiteActivitySource(path)
            buckets = [name for name, metadata in source.get("buckets/").items()
                       if is_android_window(name, metadata)]
        except (OSError, ValueError):
            if override:
                raise
            continue
        if buckets:
            return AndroidSource(source, buckets)
    if override:
        raise ValueError("The configured phone database has no Android foreground watcher.")
    return None
