"""Map packaged ActivityWatch records and authored task sessions onto the planner clock."""

from collections import defaultdict
from datetime import date, datetime, time, timezone
import hashlib
from functools import partial
import math
from pathlib import Path

import activity
from activity_db import SQLiteActivitySource


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
        starts = task.get('session_starts')
        if starts is not None and len(starts) != len(sessions):
            raise ValueError(f"{task['id']}: each session needs its recorded start time")
        for index, (slices, length) in enumerate(zip(sessions, task['sessions_minutes'])):
            if not slices or not math.isclose(sum(_seconds(s['minutes']) for s in slices),
                                              _seconds(length), rel_tol=0, abs_tol=1e-5):
                raise ValueError(f"{task['id']}: activity duration differs from session {index + 1}")
            when = datetime.fromisoformat(starts[index]) if starts else None
            stamp = activity._stamp(when) if when else None
            day = when.date().isoformat() if when else task['day']
            days[day].append((stamp if when else index, stamp, task, slices))
    return days


def task_layout(history):
    """Return the original timer periods and reviewed intervals for each task."""
    agenda = {}
    timed_tasks = [*history['tasks'], *(r for r in history.get('open_tasks', []) if r.get('sessions_minutes'))]
    for day, day_sessions in sorted(_sessions_by_day({'tasks': timed_tasks}).items()):
        clock = datetime.combine(date.fromisoformat(day), time(8), timezone.utc).timestamp()
        # When two tasks share a date, the user can switch to the other task
        # before returning for the next session of the first.
        for index, (_, stamp, task, slices) in enumerate(sorted(day_sessions, key=lambda entry: entry[0])):
            if stamp is not None:
                clock = stamp
            elif index:
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
                item['annotations'].append((clock, end, label, scene.get('review_reason')))
                clock = end
                if 'pause_after_minutes' in scene:
                    if offset == len(slices) - 1:
                        raise ValueError(f"{task['id']}: pause belongs between two slices")
                    periods.append((start, clock))
                    clock += _seconds(scene['pause_after_minutes'])
                    start = clock
            periods.append((start, clock))
    return agenda


def _review(row, *, annotations):
    matched = [(label, reason) for start, end, label, reason in annotations
               if start < row['end'] - 1e-6 and end > row['start'] + 1e-6]
    labels = {label for label, _ in matched}
    if len(labels) != 1:
        return 'uncertain', 'The observed interval has no single reviewed classification.'
    reasons = {reason for _, reason in matched if reason}
    reason = reasons.pop() if len(reasons) == 1 else 'Synthetic fixture annotation for this observed foreground activity.'
    return labels.pop(), reason


def _report(task, source, periods, annotations, exclusions, use_model=False, *, examples=(), overrides=None):
    reviewed = None if use_model else partial(_review, annotations=annotations)
    if isinstance(exclusions, str):
        exclusions = exclusions.splitlines()
    report = activity.build_report(task, periods, [EXCLUDED_MARKER, *exclusions],
                                   source=source, reviewed_label=reviewed, examples=examples, overrides=overrides)
    report['classification_source'] = 'model' if use_model else 'synthetic_annotations'
    return report


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


def replay_task_at(history, row, destination_sessions, periods, *, directory, exclusions=(),
                   use_model=False, task_context=None, examples=(), overrides=None):
    """Move bounded seed observations to Dev's clock. Later user activity stays unobserved."""
    path = Path(directory) / 'activitywatch' / (history['id'] + '.db')
    if hashlib.sha256(path.read_bytes()).hexdigest() != history['activitywatch_sha256']:
        raise ValueError(f'{path.name}: database does not match this history')
    source = SQLiteActivitySource(path, hostname='demo-' + history['id'])
    item = task_layout(history)[row['id']]
    original = item['sessions']
    if [len(s) for s in original] != [len(s) for s in destination_sessions]:
        raise ValueError('Seeded timer session/period counts differ from the activity fixture')
    mappings = []
    for src_session, dest_session in zip(original, destination_sessions):
        for (start, end), (dest_start, dest_end) in zip(src_session, dest_session):
            duration = min(end - start, dest_end - dest_start)
            if duration > 0:
                mappings.append((start, start + duration, dest_start - start))

    annotations = [(max(a, left) + delta, min(b, right) + delta, label, reason)
                   for a, b, label, reason in item['annotations']
                   for left, right, delta in mappings if left < b and right > a]
    return _report(task_context or row, ShiftedSource(source, mappings), periods, annotations,
                   exclusions, use_model, examples=examples, overrides=overrides)
