"""Read the Android foreground-app stream from a locally synced ActivityWatch database."""

from pathlib import Path

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

    def events(self, start, end, warnings=None):
        from activity import _read_events

        return _read_events(self.store.get, [(bucket, "phone") for bucket in self.buckets],
                            start, end, warnings if warnings is not None else [])


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
