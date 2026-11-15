"""Controlled stress tests for the active-task matrix and forecast queue."""
from collections import Counter
from contextlib import ExitStack
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import estimate_worker
from examples.estimation.audit_matrix import write
from planner_store import PlannerStore


def stress_seed(seed, boards):
    """Model-independent state machine test with a separate co-presence and source-snapshot oracle."""
    import random
    from examples.estimation.scale_matrix import create
    rng = random.Random(seed)
    chosen = rng.sample(boards, 2)
    tasks = [dict(t, key=str(i) + t['key']) for i, b in enumerate(chosen) for t in b['tasks']]
    mapping = {t['title']: t['key'] for t in tasks}
    related = {frozenset((str(i) + p[0], str(i) + p[1]))
               for i, b in enumerate(chosen) for p in b['positive_pairs']}
    def match(a, c):
        return frozenset((a, c)) in related
    rng.shuffle(tasks)
    actual, snapshots, ids, key_by_id, active, expected_pairs = {}, {}, {}, {}, set(), set()
    counts = Counter()
    now = [1_800_000_000]
    def estimate(t, history, _):
        key = mapping[t['title']]
        values = {mapping[h['title']]: int(h['actual_seconds']) // 60 for h in history
                  if h.get('matrix_related', match(key, mapping[h['title']]))}
        return {'point_min': 30, 'explanation': 'Controlled timing snapshot.', 'oracle_inputs': values,
                'related_tasks': [{'task_id': h['id']} for h in history if mapping[h['title']] in values]}
    ctx = ExitStack()
    with TemporaryDirectory() as tmp, ctx:
        s = PlannerStore(tmp, clock=lambda: now[0])
        s.set_setting('dev_replay_enabled', True)
        ctx.enter_context(patch.object(estimate_worker.estimator, 'predict', estimate))
        ctx.enter_context(patch.object(estimate_worker.estimator, 'index_current_task', lambda *_: None))
        ctx.enter_context(patch.object(estimate_worker.retrieval, 'decide_related',
            lambda a, b, **kw: match(mapping[a['title']], mapping[b['title']])))
        def step():
            # A pending initial must precede refreshes and pair decisions in the physical transaction queue.
            with s._db() as db:
                starts = {i for i, in db.execute(
                    'SELECT j.task_id FROM prediction_jobs j JOIN tasks t ON j.task_id=t.id WHERE t.creation_prediction IS NULL')}
            history = s.claim_prediction()
            if starts and (not history or history['task']['id'] not in starts):
                raise AssertionError('Initial forecast lost priority.')
            if history:
                counts['initial_estimates' if history['task']['id'] in starts else 'refresh_estimates'] += 1
                estimate_worker._process(s, Path(tmp)/'index', history)
                return True
            edge = s.claim_relation()
            if edge:
                estimate_worker._process_relation(s, edge)
                counts['pair_jobs'] += 1
                return True
            return False
        def checkpoint():
            while step():
                pass
            target_pairs = {p for p in expected_pairs if p & active}
            with s._db() as db:
                have = {frozenset((key_by_id[a], key_by_id[b])): v for a, b, v in db.execute(
                    'SELECT left_id,right_id,related FROM task_relations')}
            if not set(have) <= expected_pairs or any(have.get(p) != int(p in related) for p in target_pairs):
                raise AssertionError('Incorrect stored matrix or an edge for tasks that did not overlap.')
            counts['matrix_verdicts_checked'] += len(target_pairs)
            for k, identifier in ids.items():
                t = s.get_task(identifier)
                if t['creation_prediction']['oracle_inputs'] != snapshots[k]:
                    raise AssertionError(f'Initial snapshot {k} expected {snapshots[k]}, received {t["creation_prediction"]}')
                counts['initial_snapshots_checked'] += 1
                if k in active:
                    current = {src:v for src, v in actual.items() if match(k, src)}
                    if t['prediction']['oracle_inputs'] != current:
                        raise AssertionError(f'Latest {k} should use {current}, has {t["prediction"]}')
                    counts['current_snapshots_checked'] += 1
            counts['drain_checkpoints'] += 1
        def finish(k):
            if s.get_task(ids[k])['status'] == 'running':
                s.transition(ids[k], 'stop')
            minutes = rng.choice((12, 27, 64, 93))
            s.transition(ids[k], 'finish', spent_seconds=minutes*60)
            active.remove(k)
            actual[k] = minutes
            counts['completions'] += 1
        for idx, t in enumerate(tasks):
            now[0] += 20
            k = t['key']
            ids[k] = create(s, t, initial=True)
            key_by_id[ids[k]] = k
            snapshots[k] = {src:v for src, v in actual.items() if match(k, src)}
            expected_pairs.update(frozenset((k, other)) for other in active)
            active.add(k)
            counts['created'] += 1
            if idx == 3 or rng.random() < .4:
                s.transition(ids[k], 'start')
                counts['starts'] += 1
                if rng.random() < .4:
                    s.transition(ids[k], 'pause')
                    counts['pauses'] += 1
            if idx in (3, 6, 8):
                if idx == 3:
                    counts['four_task_bursts'] += 1
                for source in rng.sample(sorted(active), rng.choice((1, 2))):
                    finish(source)
            if idx >= 3:
                for _ in range(rng.randrange(4)):
                    step()
            if idx == 5 or idx == 9:
                checkpoint()
        # A retrospective timing correction should refresh only currently open eligible neighbors.
        if actual:
            src = rng.choice(sorted(actual))
            baseline = {k: len(s.forecast_timeline(ids[k])) for k in active}
            actual[src] += 23
            s.transition(ids[src], 'adjust_time', spent_seconds=actual[src]*60)
            counts['time_corrections'] += 1
            checkpoint()
            for k in active:
                changed = len(s.forecast_timeline(ids[k])) > baseline[k]
                if changed != match(k, src):
                    raise AssertionError(f'Incorrect correction queue for {src}, {k}.')
                counts['correction_targets_checked'] += 1
    return dict(counts)


def crowded_stress():
    """Create 36 cards in one burst, then complete six before the worker can react."""
    from examples.estimation.scale_matrix import DAY

    with TemporaryDirectory() as directory, ExitStack() as context:
        store = PlannerStore(directory)
        store.set_setting('dev_replay_enabled', True)
        index = Path(directory)/'vectors'
        # Five six-task workstreams, plus six genuinely independent tasks.
        families = [slot//6 if slot < 30 else slot for slot in range(36)]
        key = {}
        history = {}
        calls = []
        def judge(a, b, **kwargs):
            left, right = key[a['id']], key[b['id']]
            calls.append(tuple(sorted((left, right))))
            return families[left] == families[right]
        def estimate(t, sources, index):
            rows = [s for s in sources if s.get('matrix_related')]
            history.setdefault(key[t['id']], []).append([key[s['id']] for s in rows])
            return {'point_min': 30, 'explanation': 'Controlled state-machine response.',
                    'related_tasks': [{'task_id': s['id']} for s in rows]}
        context.enter_context(patch.object(estimate_worker.retrieval, 'decide_related', judge))
        context.enter_context(patch.object(estimate_worker.estimator, 'predict', estimate))
        context.enter_context(patch.object(estimate_worker.estimator, 'index_current_task', lambda *_:None))
        ids = []
        for i in range(36):
            t = store.create_task(day=DAY, title=f'Card {i}', description='A controlled bulk submission.',
                                category='Load test', estimate_minutes=30)
            ids.append(t['id'])
            key[t['id']] = i
            if i == 3:
                store.transition(t['id'], 'start')
        pending = sum(t['creation_prediction'] is None for t in [store.get_task(k) for k in ids])
        while estimate_worker.run_one_job(store, index):
            pass
        with store._db() as db:
            physical = {(min(key[a], key[b]), max(key[a], key[b])):r for a, b, r in db.execute(
                      'SELECT left_id,right_id,related FROM task_relations')}
        truth = {(a, b):int(families[a] == families[b]) for a, b in combinations(range(36), 2)}
        if pending != 36 or len(calls) != 630 or len(set(calls)) != 630 or physical != truth:
            raise AssertionError('Incomplete or duplicate matrix judgments after the submission burst.')
        finished = {0, 1, 6, 7, 12, 30}
        for k in sorted(finished):
            store.transition(ids[k], 'finish', spent_seconds=(70+k)*60)
        expected = {k for k in range(36) if k not in finished and any(families[k] == families[src] for src in finished)}
        queued = {k for k in range(36) if store.get_task(ids[k])['estimate_updating']}
        jobs = 0
        while estimate_worker.run_one_job(store, index):
            jobs += 1
        observed = {k for k in range(36) if len(store.forecast_timeline(ids[k])) > 1}
        if expected != queued or expected != observed or jobs != len(expected):
            raise AssertionError('Completion burst failed to coalesce its affected neighbors.')
        for k in expected:
            sources = {src for src in finished if families[k] == families[src]}
            if len(history[k]) != 2 or set(history[k][-1]) != sources:
                raise AssertionError('Refresh dropped a source or ran twice.')
        return {'simultaneous_initial_estimates': pending, 'distinct_pair_jobs': len(calls),
                'positive_cells': sum(truth.values()), 'negative_cells': len(truth)-sum(truth.values()),
                'completions_in_one_wave': len(finished), 'affected_survivors': len(expected),
                'untouched_survivors': 36-len(finished)-len(expected), 'saved_refreshes': jobs}


def stress():
    from examples.estimation.scale_matrix import DIR, fixture, new_run
    path = DIR/'stress.json'
    run = new_run(path, seed_count=96, method='Controlled semantic model, ten cards from two boards, independent timing and co-presence oracle.', seeds={})
    boards = fixture()['boards']
    for seed in range(run['seed_count']):
        if str(seed) not in run['seeds']:
            try:
                run['seeds'][str(seed)] = {'counts': stress_seed(seed, boards)}
            except AssertionError as exc:
                run['seeds'][str(seed)] = {'error': str(exc)}
            write(path, run)
    counts = Counter()
    for r in run['seeds'].values():
        counts.update(r.get('counts', {}))
    run['totals'] = dict(counts)
    run['crowded_burst'] = crowded_stress()
    run['failed_seeds'] = [k for k, v in run['seeds'].items() if 'error' in v]
    write(path, run)
    print(run['totals'], 'failed seeds', run['failed_seeds'], flush=True)
