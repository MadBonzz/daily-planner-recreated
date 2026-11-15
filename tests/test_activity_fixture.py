import json
from copy import deepcopy
from datetime import datetime

import pytest

import activity
from examples.estimation.activity_fixture import export_history, replay_history


def study_history():
    def video(minutes, title, label, **extra):
        return {'minutes': minutes, 'app': 'Google Chrome', 'domain': 'youtube.com',
                'title': title, 'label': label, 'audible': True, **extra}

    signals = {'id': 'signals', 'title': 'Review Fourier series', 'description': 'Use video lecture 3 and notes.',
               'day': '2026-09-01', 'sessions_minutes': [10, 3], 'activity': [
                   [video(4, 'Fourier series — lecture 3', 'on_task'),
                    video(2, 'Funny cat compilation', 'off_task', pause_after_minutes=6),
                    video(1, 'Test billing details', 'uncertain', excluded=True),
                    video(1, 'Test private preferences', 'uncertain', incognito=True),
                    video(2, 'Fourier series — lecture 3', 'uncertain', idle=True)],
                   [{'minutes': 3, 'app': 'Obsidian', 'title': 'Fourier notes', 'label': 'on_task'}]]}
    clinic = {'id': 'call', 'title': 'Call with Casey', 'day': signals['day'], 'sessions_minutes': [4],
              'completion_note': 'Casey joined late.', 'activity': [[
                  {'minutes': 4, 'app': 'Zoom', 'title': 'Meeting', 'label': 'uncertain'}]]}
    return {'id': 'test', 'tasks': [signals, clinic], 'holdouts': [{'id': 'future', 'activity': 'SPOILER'}]}


def test_export_has_native_sensor_payloads_and_no_reviewer_data():
    history = study_history()
    exported = export_history(history)
    assert exported == export_history(deepcopy(history))
    text = json.dumps(exported)
    assert all(s not in text for s in ('label', 'SPOILER', 'Casey', 'on_task', 'off_task'))
    by_kind = {bucket['type']: bucket for bucket in exported['buckets'].values()}
    assert set(by_kind) == {'web.tab.current', 'currentwindow', 'afkstatus'}
    payloads = {'currentwindow': {'app', 'title'}, 'web.tab.current': {'url', 'title', 'audible', 'incognito'},
                'afkstatus': {'status'}}
    for kind, bucket in by_kind.items():
        assert exported['buckets'][bucket['id']] == bucket
        assert bucket['hostname'] == 'demo-test' and bucket['client'] in bucket['id']
        for row in bucket['events']:
            assert datetime.fromisoformat(row['timestamp']).utcoffset().total_seconds() == 0
            assert row['duration'] > 0 and set(row['data']) == payloads[kind]
    assert len(by_kind['afkstatus']['events']) < len(by_kind['currentwindow']['events'])


def test_replay_obeys_timer_boundaries_privacy_and_historical_annotations(monkeypatch):
    history = study_history()
    monkeypatch.setattr(activity, 'chat_json', lambda *_a, **_kw: pytest.fail('Unexpected network request'))
    replays = replay_history(history, export=json.loads(json.dumps(export_history(history))))
    study = replays['signals']
    assert [len(s) for s in study['sessions']] == [2, 1]
    assert study['sessions'][0][1][0] - study['sessions'][0][0][1] == 6 * 60
    assert study['sessions'][0][-1][1] < replays['call']['periods'][0][0] < study['sessions'][1][0][0]
    report = study['report']
    assert report['classification_source'] == 'synthetic_annotations'
    assert report['totals'] == {'on_task': 7 * 60, 'off_task': 2 * 60, 'uncertain': 4 * 60}
    assert report['elapsed_seconds'] == 13 * 60 and report['switches'] == 1
    assert sum(r['seconds'] for r in report['intervals'] if r['excluded']) == 120
    assert all(s not in json.dumps(report) for s in ('Test billing', 'Test private', 'private-fixture', 'Calendar', 'Zoom'))
    assert 'Casey' not in json.dumps(replays['call']['report'])


def test_model_replay_uses_production_classifier_with_only_visible_intervals(monkeypatch):
    requests = []

    def classify(messages, schema, **kwargs):
        query = json.loads(messages[-1]['content'])
        requests.append(query)
        assert {str(i['id']) for i in query['intervals']} == set(schema['required'])
        return {str(i['id']): {'label': 'uncertain', 'reason': 'Model example.'} for i in query['intervals']}

    monkeypatch.setattr(activity, 'chat_json', classify)
    report = replay_history(study_history(), use_model=True)['signals']['report']
    assert report['totals']['uncertain'] == 13 * 60 and report['classification_source'] == 'model'
    assert len(requests) == 2
    assert len(requests[0]['intervals']) == 3
    text = json.dumps(requests)
    assert 'Funny cat' in text and 'Fourier series' in text
    assert all(s not in text for s in ('joined late', 'fixture annotation', 'Test billing', 'Test private', 'Calendar'))


def test_legacy_history_and_invalid_session_duration():
    h = study_history()
    h['tasks'] = [dict(id='old', day='2026-09-01', sessions_minutes=[5])]
    assert replay_history(h)['old']['report']['totals']['uncertain'] == 300
    h = study_history()
    h['tasks'][0]['activity'][0][0]['minutes'] += 1
    with pytest.raises(ValueError, match='activity duration differs'):
        export_history(h)


def test_seeded_activity_can_be_clipped_and_does_not_extend_to_new_time():
    from examples.estimation.activity_fixture import replay_task_at

    task = study_history()['tasks'][0]
    destinations = [[(0, 360), (720, 960)], [(1800, 1980)]]
    report = replay_task_at('test', task, destinations, [(0, 300), (720, 960), (1800, 2120)],
                            exclusions=['Funny cat'])
    assert report['elapsed_seconds'] == 860
    assert report['totals'] == {'on_task': 420, 'off_task': 0, 'uncertain': 440}
    assert report['intervals'][-1]['app'] == '' and report['intervals'][-1]['seconds'] == 140
    assert 'Funny cat' not in json.dumps(report)


def test_packaged_database_integrity_and_private_annotations(tmp_path):
    import hashlib
    import sqlite3
    from examples.estimation.activity_fixture import write_database

    h = study_history()
    path = tmp_path / 'activitywatch' / 'test.db'
    path.parent.mkdir()
    write_database(export_history(h), path)
    h['activitywatch_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    for row in h['tasks']:
        for session in row['activity']:
            for scene in session:
                for key in set(scene) - {'minutes', 'label', 'pause_after_minutes'}:
                    del scene[key]
    assert replay_history(h, directory=tmp_path)['signals']['report']['totals']['on_task'] == 420
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone() == (6,)
        raw = [r[0] for r in db.execute('SELECT data FROM events')]
        assert all('Casey' not in s and 'on_task' not in s and 'SPOILER' not in s for s in raw)
        db.execute('DELETE FROM events')
    with pytest.raises(ValueError, match='does not match'):
        replay_history(h, directory=tmp_path)


def test_db_uses_model_for_open_phone_activity_with_seed_clock(tmp_path, monkeypatch):
    import dev_workspace as dev
    from examples.estimation.run import DATASET, histories

    h = histories('07')[0]
    assert (DATASET / 'activitywatch' / (h['id'] + '.db')).is_file()
    store = dev.open_dev_workspace(tmp_path, h['id'])
    laundry = next(t for t in store.list_tasks() if t['status'] == 'paused')
    store.transition(laundry['id'], 'stop')
    calls = []

    def classify(messages, schema, **kwargs):
        payload = json.loads(messages[-1]['content'])
        calls.extend(payload['intervals'])
        return {key: {'label': 'uncertain', 'reason': 'Phone app alone is ambiguous.'} for key in schema['required']}

    monkeypatch.setattr(activity, 'chat_json', classify)
    report = dev.dev_activity_report(store, store.get_task(laundry['id']))
    assert report['classification_source'] == 'model'
    assert {r.get('phone_app') for r in calls} == {'Clock', 'Messages'}
    assert all(not r['app'] and r['idle'] for r in calls)
    assert report['elapsed_seconds'] == pytest.approx(420)
    assert sum(r['seconds'] for r in report['intervals']) == pytest.approx(420)
    assert all('label' not in row and 'package' not in row and 'classname' not in row for row in calls)


def test_all_packaged_histories_show_laptop_and_phone_distractions_from_completed_tasks():
    import sqlite3
    from examples.estimation.run import DATASET, histories

    for history in histories(None):
        path = DATASET / 'activitywatch' / (history['id'] + '.db')
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            assert db.execute("SELECT count(*) FROM buckets WHERE client='aw-watcher-android'").fetchone()[0]
        sets = replay_history(history, directory=DATASET)
        all_events = [r for task in sets.values() for r in task['report']['intervals']]
        distractions = [r for r in all_events if r['label'] == 'off_task']
        assert {activity.device_label(r) for r in distractions} >= {'Laptop', 'Phone'}, history['id']
        assert all(r['label_source'] == 'sample annotation' for r in distractions)
        for task in history['tasks']:
            report = sets[task['id']]['report']
            assert report['elapsed_seconds'] == pytest.approx(task['actual_minutes'] * 60)
        if history['id'] == '04-kernel-workbench':
            assert sum(r['seconds'] for r in distractions if r['domain'] == 'youtube.com') == 660
            assert sum(r['seconds'] for r in distractions if r.get('phone_app') == 'FotMob') == 480


def test_device_labels_identify_legacy_foreground_and_redacted_rows():
    assert activity.device_label({'phone_app': 'Clock', 'idle': True}) == 'Phone'
    assert activity.device_label({'title': 'Spreadsheet', 'phone_app': 'Clock'}) == 'Laptop + phone'
    assert activity.device_label({'domain': 'arxiv.org'}) == 'Laptop'
    assert activity.device_label({'idle': True}) == 'Laptop'
    assert activity.device_label({'idle': None}) == 'Unknown'
    assert activity.device_label({'excluded': True, 'app': 'Firefox'}) == 'Unknown'


def test_existing_gpu_profile_refreshes_only_unmodified_sample_reports(tmp_path, monkeypatch):
    import dev_workspace as dev

    monkeypatch.setattr(activity, 'chat_json', lambda *_a, **_kw: pytest.fail('Unexpected model request'))
    profile = '04-kernel-workbench'
    store = dev.open_dev_workspace(tmp_path, profile)
    tasks = {r['title']: r for r in store.completed_tasks()}
    fixture_task = tasks['Fix corruption on partial warp tails']
    reviewed = next(t for t in tasks.values() if t['id'] != fixture_task['id'])
    before = store.periods(fixture_task['id'])
    old = {'classification_source': 'synthetic_annotations', 'summary': 'Old fixture', 'intervals': []}
    store.save_report(fixture_task['id'], old, refresh_estimates=False)
    user_report = {**reviewed['report'], 'classification_source': 'model', 'summary': 'Personally reviewed'}
    store.save_report(reviewed['id'], user_report, refresh_estimates=False)
    store.set_setting('dev_sample_activity_sha256', 'prior-fixture')
    reopened = dev.open_dev_workspace(tmp_path, profile)
    refreshed = reopened.get_task(fixture_task['id'])
    assert refreshed['report']['totals']['off_task'] == 360
    assert refreshed['day'] == fixture_task['day'] and reopened.periods(fixture_task['id']) == before
    assert reopened.get_task(reviewed['id'])['report'] == user_report


def test_logged_session_starts_keep_their_dates_offsets_and_pause_boundaries():
    history = study_history()
    task, clinic = history['tasks']
    task['session_starts'] = ['2026-03-07T19:42-05:00', '2026-03-09T09:17-04:00']
    clinic['session_starts'] = ['2026-03-07T20:14-05:00']
    observed = replay_history(history)
    sessions = observed['signals']['sessions']
    first, second = map(activity._stamp, task['session_starts'])
    assert sessions == [[(first, first + 360), (first + 720, first + 960)],
                        [(second, second + 180)]]
    assert observed['signals']['report']['intervals'][-1]['title'] == 'Fourier notes'
    starts = [r['start'] for r in observed['call']['report']['intervals']]
    assert starts == [activity._stamp(clinic['session_starts'][0])]
