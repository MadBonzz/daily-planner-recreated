"""Replay synthetic ActivityWatch SQLite stores for the planner and estimator.

The generator uses only the historical tasks in a profile. Watcher buckets contain
ordinary device events. Timer boundaries and synthetic relevance annotations are
separate, so the same export can also be replayed with the real model classifier.
"""

import argparse
from collections import defaultdict
from datetime import date, datetime, time, timezone
import hashlib
from functools import partial
import json
import math
from pathlib import Path
import sqlite3
from contextlib import closing
from urllib.parse import unquote

import activity


EXCLUDED_MARKER = 'private-fixture.example'


def _seconds(minutes):
    if type(minutes) not in (int, float) or not math.isfinite(minutes) or minutes <= 0:
        raise ValueError('Expected a positive duration in minutes')
    return round(minutes * 60, 6)


def _sessions_by_day(history):
    days = defaultdict(list)
    for task in history['tasks']:
        sessions = task.get('activity')
        if sessions is None:
            # Older histories include a timer but no watcher observations.
            sessions = [[{'minutes': length, 'label': 'uncertain', 'idle': None}]
                        for length in task['sessions_minutes']]
        if len(sessions) != len(task['sessions_minutes']):
            raise ValueError(f"{task['id']}: activity must cover every timer session")
        for index, (slices, length) in enumerate(zip(sessions, task['sessions_minutes'])):
            if not slices or not math.isclose(sum(_seconds(s['minutes']) for s in slices),
                                              _seconds(length), rel_tol=0, abs_tol=1e-5):
                raise ValueError(f"{task['id']}: activity duration differs from session {index + 1}")
            days[task['day']].append((index, task, slices))
    return days


def _render(history, *, render_events=True):
    hostname = 'demo-' + history['id']
    buckets = {}
    agenda = {}

    def emit(watcher, kind, at, until, data):
        key = f'{watcher}_{hostname}'
        bucket = buckets.setdefault(key, {'id': key, 'created': activity._iso(at), 'name': '',
                                         'type': kind, 'client': watcher, 'hostname': hostname,
                                         'events': []})
        events = bucket['events']
        # Watchers heartbeat independently. Unchanged AFK state, for example,
        # can span many foreground switches and task boundaries.
        if events and events[-1]['data'] == data:
            previous = events[-1]
            left = activity._stamp(previous['timestamp'])
            if abs(left + previous['duration'] - at) < 1e-5:
                previous['duration'] = round(until - left, 6)
                return
        events.append({'id': len(events) + 1, 'timestamp': activity._iso(at),
                       'duration': round(until - at, 6), 'data': data})

    def watcher_slice(scene, at, end):
        if not render_events:
            return
        app, title = scene.get('app', ''), scene.get('title', '')
        domain = scene.get('domain', '')
        browser = activity._browser(app)
        if scene.get('excluded'):
            if browser:
                domain = EXCLUDED_MARKER
            else:
                title = f'{EXCLUDED_MARKER} — {title}'
        if app or title:
            emit('aw-watcher-window', 'currentwindow', at, end, {'app': app, 'title': title})
        if browser and domain:
            emit('aw-watcher-web-' + browser, 'web.tab.current', at, end,
                 {'url': 'https://' + domain + '/', 'title': title,
                  'audible': scene.get('audible', False), 'incognito': scene.get('incognito', False)})
        elif scene.get('incognito'):
            raise ValueError('A synthetic incognito interval needs a browser app and domain')
        idle = scene.get('idle', False)
        if idle is not None:
            emit('aw-watcher-afk', 'afkstatus', at, end, {'status': 'afk' if idle else 'not-afk'})

    def between(at, until):
        # These events are part of the exported device day but outside the timer.
        watcher_slice({'app': 'Calendar', 'title': 'Today', 'idle': False}, at, min(at + 60, until))
        if until > at + 60:
            watcher_slice({'app': 'Calendar', 'title': 'Today', 'idle': True}, at + 60, until)

    timed_tasks = [*history['tasks'], *(r for r in history.get('open_tasks', []) if r.get('sessions_minutes'))]
    for day, day_sessions in sorted(_sessions_by_day({'tasks': timed_tasks}).items()):
        clock = datetime.combine(date.fromisoformat(day), time(8), timezone.utc).timestamp()
        # When two tasks share a date, the user can switch to the other task
        # before returning for the next session of the first.
        for index, (_, task, slices) in enumerate(sorted(day_sessions, key=lambda entry: entry[0])):
            if index:
                between(clock, clock + 15 * 60)
                clock += 15 * 60
            item = agenda.setdefault(task['id'], {'sessions': [], 'annotations': []})
            periods = []
            item['sessions'].append(periods)
            start = clock
            for offset, scene in enumerate(slices):
                label = scene.get('label', 'uncertain')
                if label not in activity.LABELS:
                    raise ValueError(f"{task['id']}: unsupported annotation {label!r}")
                end = clock + _seconds(scene['minutes'])
                watcher_slice(scene, clock, end)
                item['annotations'].append((clock, end, label))
                clock = end
                if 'pause_after_minutes' in scene:
                    if offset == len(slices) - 1:
                        raise ValueError(f"{task['id']}: pause belongs between two slices")
                    periods.append((start, clock))
                    wake = clock + _seconds(scene['pause_after_minutes'])
                    between(clock, wake)
                    start = clock = wake
            periods.append((start, clock))
    return {'buckets': buckets}, agenda


def export_history(history):
    """Create a /api/0/export-compatible synthetic file for one device."""
    return _render(history)[0]


class ExportSource:
    """Read the two bounded ActivityWatch REST resources used by the collector."""

    def __init__(self, export, hostname):
        self.buckets = export['buckets']
        self.hostname = hostname

    def get(self, path, **query):
        if path == 'buckets/':
            return {key: {k: v for k, v in bucket.items() if k != 'events'}
                    for key, bucket in self.buckets.items()}
        prefix, key, suffix = path.split('/')
        if prefix != 'buckets' or suffix != 'events':
            raise ValueError('Unknown ActivityWatch resource')
        start, end = activity._stamp(query['start']), activity._stamp(query['end'])
        rows = self.buckets[unquote(key)]['events']
        return sorted((r for r in rows if activity._stamp(r['timestamp']) < end
                       and activity._stamp(r['timestamp']) + r['duration'] > start),
                      key=lambda r: r['timestamp'], reverse=True)


def _review(row, *, annotations):
    labels = {label for start, end, label in annotations if start < row['end'] and end > row['start']}
    label = labels.pop() if len(labels) == 1 else 'uncertain'
    return label, 'Synthetic fixture annotation for this observed foreground activity.'


def _report(task, source, periods, annotations, exclusions, use_model=False, *, examples=(), overrides=None):
    reviewed = None if use_model else partial(_review, annotations=annotations)
    if isinstance(exclusions, str):
        exclusions = exclusions.splitlines()
    report = activity.build_report(task, periods, [EXCLUDED_MARKER, *exclusions],
                                   source=source, reviewed_label=reviewed, examples=examples, overrides=overrides)
    report['classification_source'] = 'model' if use_model else 'synthetic_annotations'
    return report


def _replay_source(history, directory, export=None):
    """Packaged samples read their original DB. Inline authored scenes support old profiles."""
    fingerprint = history.get('activitywatch_sha256')
    raw, agenda = _render(history, render_events=not fingerprint)
    hostname = 'demo-' + history['id']
    if fingerprint:
        from activity_db import SQLiteActivitySource
        path = Path(directory) / 'activitywatch' / (history['id'] + '.db')
        if hashlib.sha256(path.read_bytes()).hexdigest() != fingerprint:
            raise ValueError(f'{path.name}: database does not match this history')
        return SQLiteActivitySource(path, hostname=hostname), agenda
    if export is not None and export != raw:
        raise ValueError('Export differs from the authored device scenes. Regenerate it.')
    return ExportSource(raw, hostname), agenda


def replay_history(history, *, directory=None, export=None, use_model=False, tasks_by_id=None):
    """Classify recorded windows for completed tasks. Open tasks and holdouts are excluded.

    Historical annotations give deterministic estimator-only scores. use_model=True
    sends only the sanitised intervals and supplied pre-task context to the classifier.
    """
    source, agenda = _replay_source(history, directory, export)
    results = {}
    for row in history['tasks']:
        task_id = row['id']
        item = agenda[task_id]
        sessions = item['sessions']
        periods = [period for session in sessions for period in session]
        context = tasks_by_id[task_id] if tasks_by_id is not None else row
        report = _report(context, source, periods, item['annotations'], [], use_model)
        results[task_id] = {'sessions': sessions, 'periods': periods, 'report': report}
    return results


def task_layout(history):
    """Seed calendars and annotations without regenerating, reading, or classifying events."""
    _, agenda = _render(history, render_events=False)
    return agenda


class ShiftedSource:
    """Query a seeded interval at its original DB time; map only its intersection back."""

    def __init__(self, source, mappings):
        self.source = source
        self.hostname = source.hostname
        self.mappings = mappings

    def get(self, path, **query):
        if path == 'buckets/':
            return self.source.get(path)
        start, end = activity._stamp(query['start']), activity._stamp(query['end'])
        events = []
        for src_start, src_end, delta in self.mappings:
            left, right = max(src_start, start - delta), min(src_end, end - delta)
            if left >= right:
                continue
            for row in self.source.get(path, start=activity._iso(left), end=activity._iso(right)):
                at = activity._stamp(row['timestamp'])
                a, b = max(at, left), min(at + row['duration'], right)
                if a < b:
                    events.append({**row, 'timestamp': activity._iso(a + delta), 'duration': b - a})
        return sorted(events, key=lambda row: row['timestamp'], reverse=True)


def replay_task_at(history, row, destination_sessions, periods, *, directory=None, exclusions=(),
                   use_model=False, task_context=None, examples=(), overrides=None):
    """Move bounded seed observations to Dev's clock. Later user activity stays unobserved."""
    if isinstance(history, str):  # Backward compatibility with inline scenes.
        history = {'id': history, 'tasks': [row]}
    source, agenda = _replay_source(history, directory)
    item = agenda[row['id']]
    original = item['sessions']
    if [len(s) for s in original] != [len(s) for s in destination_sessions]:
        raise ValueError('Seeded timer session/period counts differ from the activity fixture')
    mappings = []
    for src_session, dest_session in zip(original, destination_sessions):
        for (start, end), (dest_start, dest_end) in zip(src_session, dest_session):
            duration = min(end - start, dest_end - dest_start)
            if duration > 0:
                mappings.append((start, start + duration, dest_start - start))

    annotations = [(max(a, left) + delta, min(b, right) + delta, label)
                   for a, b, label in item['annotations']
                   for left, right, delta in mappings if left < b and right > a]
    return _report(task_context or row, ShiftedSource(source, mappings), periods, annotations,
                   exclusions, use_model, examples=examples, overrides=overrides)


def write_database(export, path):
    """Write native AW v6 bucket/event records from an authored ActivityWatch export."""
    # Make the output reproducible and avoid overwriting an existing real watcher DB.
    with Path(path).open('xb'):
        pass
    try:
        with closing(sqlite3.connect(path)) as db, db:
            db.executescript("""
                PRAGMA user_version = 6;
                PRAGMA foreign_keys = ON;
                CREATE TABLE buckets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL,
                    type TEXT NOT NULL, client TEXT NOT NULL, hostname TEXT NOT NULL,
                    created TEXT NOT NULL, data_deprecated TEXT DEFAULT '{}', data TEXT NOT NULL DEFAULT '{}');
                CREATE TABLE events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, bucketrow INTEGER NOT NULL,
                    starttime INTEGER NOT NULL, endtime INTEGER NOT NULL, data TEXT NOT NULL,
                    FOREIGN KEY (bucketrow) REFERENCES buckets(id));
                CREATE TABLE key_value (key TEXT PRIMARY KEY, value TEXT, last_modified NUMBER NOT NULL);
                CREATE INDEX bucket_id_index ON buckets(id);
                CREATE INDEX events_bucketrow_endtime_starttime_index ON events(bucketrow, endtime, starttime DESC);
                CREATE INDEX events_bucketrow_starttime_endtime_index ON events(bucketrow, starttime DESC, endtime);
            """)
            for key, bucket in sorted(export['buckets'].items()):
                created = datetime.fromtimestamp(activity._stamp(bucket['created']), timezone.utc)
                bucket_id = db.execute(
                    'INSERT INTO buckets(name,type,client,hostname,created) VALUES (?,?,?,?,?)',
                    (key, bucket['type'], bucket['client'], bucket['hostname'],
                     created.isoformat(timespec='microseconds'))).lastrowid
                for event in bucket['events']:
                    # API events have a start and a duration. AW's SQLite store uses two Unix nanosecond endpoints.
                    start = round(activity._stamp(event['timestamp']) * 1_000_000) * 1000
                    end = start + round(event['duration'] * 1_000_000_000)
                    db.execute('INSERT INTO events(bucketrow,starttime,endtime,data) VALUES (?,?,?,?)',
                               (bucket_id, start, end, json.dumps(event['data'], ensure_ascii=False)))
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description='Write a new AW v6 SQLite fixture from an ActivityWatch JSON export.')
    parser.add_argument('export', type=Path)
    parser.add_argument('database', type=Path, help='New .db path')
    args = parser.parse_args()
    write_database(json.loads(args.export.read_text(encoding='utf-8')), args.database)


if __name__ == '__main__':
    main()
