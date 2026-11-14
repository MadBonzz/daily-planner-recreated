"""Render synthetic ActivityWatch exports and replay completed planner sessions.

The generator uses only the historical tasks in a profile. Watcher buckets contain
ordinary device events. Timer boundaries and synthetic relevance annotations are
separate, so the same export can also be replayed with the real model classifier.
"""

import argparse
from collections import defaultdict
from datetime import date, datetime, time, timezone
from functools import partial
import json
import math
from pathlib import Path
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


def _render(history):
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

    for day, day_sessions in sorted(_sessions_by_day(history).items()):
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


def _report(task, source, periods, annotations, exclusions, use_model=False):
    reviewed = None if use_model else partial(_review, annotations=annotations)
    if isinstance(exclusions, str):
        exclusions = exclusions.splitlines()
    report = activity.build_report(task, periods, [EXCLUDED_MARKER, *exclusions],
                                   source=source, reviewed_label=reviewed)
    report['classification_source'] = 'model' if use_model else 'synthetic_annotations'
    return report


def replay_history(history, *, directory=None, export=None, use_model=False):
    """Rebuild historical activity reviews, preserving pauses and session grouping.

    By default visible activity uses the author's fixture annotation for a
    repeatable benchmark. use_model=True classifies the redacted intervals with
    the configured production provider. No held-out activity is inspected.
    """
    generated, agenda = _render(history)
    if export is None and directory is not None:
        path = Path(directory) / 'activitywatch' / (history['id'] + '.json')
        if path.is_file():
            export = json.loads(path.read_text(encoding='utf-8'))
    if export is not None and not use_model and export != generated:
        raise ValueError('Export differs from annotated history. Regenerate it or use model classification.')
    source = ExportSource(generated if export is None else export, 'demo-' + history['id'])
    by_id = {task['id']: task for task in history['tasks']}
    results = {}
    for task_id, item in agenda.items():
        sessions = item['sessions']
        periods = [period for session in sessions for period in session]
        report = _report(by_id[task_id], source, periods, item['annotations'], [], use_model)
        results[task_id] = {'sessions': sessions, 'periods': periods, 'report': report}
    return results


def replay_task_at(history_id, row, destination_sessions, periods, *, exclusions=()):
    """Move an authored task's device traces to a DEV workspace's seed clock.

    Only original observations within each seeded period can be transplanted.
    Later user-created running periods and extensions have no synthetic evidence.
    Rebuilding after a corrected stop clips to the actual timer periods.
    """
    raw, agenda = _render({'id': history_id, 'tasks': [row]})
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

    def transplant(start, end):
        for left, right, delta in mappings:
            if left < end and right > start:
                yield max(start, left) + delta, min(end, right) + delta

    shifted_annotations = [(a, b, label) for start, end, label in item['annotations']
                           for a, b in transplant(start, end)]
    for bucket in raw['buckets'].values():
        shifted = []
        for event in bucket['events']:
            at = activity._stamp(event['timestamp'])
            for a, b in transplant(at, at + event['duration']):
                shifted.append({'id': len(shifted) + 1, 'timestamp': activity._iso(a),
                                'duration': b - a, 'data': event['data']})
        bucket['events'] = shifted
    source = ExportSource(raw, 'demo-' + history_id)
    return _report(row, source, periods, shifted_annotations, exclusions)


def main():
    parser = argparse.ArgumentParser(description='Write a synthetic ActivityWatch bucket export.')
    parser.add_argument('history', type=Path, help='Completed-history JSON fixture')
    parser.add_argument('output', type=Path, help='Destination JSON file')
    args = parser.parse_args()
    raw = export_history(json.loads(args.history.read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
