"""Score independently labeled active task boards and their real SQLite completion waves.

Run from the repo with python -m examples.estimation.scale_matrix vectors|kev|queue|forecast|stress|report.
The stages checkpoint by request/board. Supply PLANNER_* endpoints as for the application.
"""
import argparse
import base64
import hashlib
import json
import math
from collections import Counter
from contextlib import ExitStack
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import zlib

import numpy as np

import estimate_worker
import retrieval
from examples.estimation.audit_matrix import payload_key, write
from model_client import planner_settings
from planner_store import PlannerStore

DIR = Path(__file__).resolve().parents[2] / 'data/results/component-scale/matrix'
REFERENCE = DIR / 'reference.json'
LABEL_DIGEST = '89d6b6fe78cf28d7205fd670c01655fd35ea0e3a83624c226bdf17be42f0b00c'
# A prior task on every board exercises ordinary Qwen retrieval in the integrated run.
PRIOR = dict(key='prior', title='Digitally renew my parking permit', minutes=15,
             description='Enter the registration and permit number in the municipal portal, pay the annual fee and download the receipt.')
DAY = '2026-11-15'


def fixture():
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != LABEL_DIGEST:
        raise ValueError('Frozen task boards or independent reference labels differ.')
    return json.loads(REFERENCE.read_text())


def new_run(path, **kwargs):
    if path.exists():
        run = json.loads(path.read_text())
        if run.get('reference_sha256') != LABEL_DIGEST:
            raise ValueError(f'{path} was recorded against a different reference.')
        return run
    return {'reference_sha256': LABEL_DIGEST, **kwargs}


def typed(t):
    return {'title': t['title'], 'description': t['description']}


def vectors():
    """Fetch both Qwen roles once; serialize the model's float32 output losslessly."""
    cfg = planner_settings()
    model = cfg.get('PLANNER_EMBEDDING_MODEL') or retrieval.EMBEDDING_MODEL
    path = DIR / 'qwen.json'
    run = new_run(path, model=model, encoding='zlib-base64 little-endian float32', responses={})
    for board in fixture()['boards']:
        for t in [*board['tasks'], PRIOR]:
            for role in ([False] if t is PRIOR else [False, True]):
                body = {'model': model, 'input': retrieval.embedding_input(typed(t), role)}
                key = payload_key(body)
                if key in run['responses']:
                    continue
                vector = retrieval.embed_task(typed(t), cfg, query=role)
                run['responses'][key] = {'board': board['id'] if t is not PRIOR else 'shared',
                                         'task': t['key'], 'query': role,
                                         'vector': base64.b64encode(zlib.compress(
                                             np.asarray(vector, dtype='<f4').tobytes())).decode()}
            # Saving one task at a time avoids losing an expensive partial run.
            write(path, run)
        print(board['id'], 'Qwen requests', len(run['responses']), flush=True)


def vector(run, body):
    item = run['responses'][payload_key(body)]
    return np.frombuffer(zlib.decompress(base64.b64decode(item['vector'])), dtype='<f4')


def prior_candidates(inputs, board):
    def lookup(t, query):
        return vector(inputs, dict(model=inputs['model'], input=retrieval.embedding_input(typed(t), query)))
    prior = lookup(PRIOR, False)
    result = {}
    for t in board['tasks']:
        scores = {}
        for role, text in [('instructed', True), ('bare', False)]:
            q = lookup(t, text)
            scores[role] = float(np.dot(q, prior) / np.linalg.norm(q) / np.linalg.norm(prior))
        result[t['key']] = scores
    return result


def capture_kev(run, path, cfg, body):
    key = payload_key(body)
    if key not in run['responses']:
        endpoint = cfg.get('PLANNER_DECISION_ENDPOINT') or 'http://127.0.0.1:1234/v1/systemone'
        result = retrieval._local_post(endpoint, body, cfg)
        score = result['answers']['relevant']['noul']
        if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f'Invalid Kev response for {key}: {result}')
        run['responses'][key] = score
        write(path, run)
    return key, run['responses'][key]


def kev(prior_only=False):
    """Collect actual, orientation-canonical active-pair judgments plus above-threshold prior cases."""
    cfg = planner_settings()
    model = cfg.get('PLANNER_DECISION_MODEL') or retrieval.DECISION_MODEL
    path = DIR / 'kev.json'
    run = new_run(path, model=model, responses={}, boards={})
    for b in fixture()['boards']:
        report = run['boards'].setdefault(b['id'], {'pairs': {}, 'prior': {}})
        if not prior_only:
            for a, c in combinations(b['tasks'], 2):
                first, second = retrieval.active_decision_order(typed(a), typed(c))
                body = retrieval.decision_payload(first, second, (), model)
                key, score = capture_kev(run, path, cfg, body)
                report['pairs'][a['key'] + c['key']] = {'request_sha256': key, 'score': score,
                                                         'related': score >= .5}
                write(path, run)
        if (DIR / 'qwen.json').exists():
            qwen = new_run(DIR / 'qwen.json')
            try:
                sims = prior_candidates(qwen, b)
            except KeyError:
                sims = {}
            for t in b['tasks']:
                if (s := sims.get(t['key'])) is None:
                    continue
                item = {**s, 'cosine': max(s.values()), 'related': False}
                if item['cosine'] >= retrieval.MINIMUM_THRESHOLD:
                    body = retrieval.decision_payload(typed(t), typed(PRIOR), (), model)
                    key, score = capture_kev(run, path, cfg, body)
                    item.update(request_sha256=key, score=score, related=score >= .5)
                report['prior'][t['key']] = item
            write(path, run)
        print(b['id'], 'Kev requests', len(run['responses']), flush=True)


def pair(a, b):
    return ''.join(sorted((a, b)))


def neighbors(cells, sources, active):
    return sorted(t for t in active if any(cells.get(pair(t, s)) for s in sources))


def create(store, t, initial=False):
    pred = None if initial else {'point_min': t['minutes'], 'low_min': t['minutes'],
                                  'high_min': t['minutes'], 'explanation': 'Scenario starting budget.'}
    return store.create_task(day=DAY, **typed(t), category='Daily work',
                             estimate_minutes=t['minutes'], prediction=pred)['id']


def cells(store, keys):
    with store._db() as conn:
        return {pair(keys[a], keys[b]): r for a, b, r in conn.execute(
            'SELECT left_id,right_id,related FROM task_relations')}


def state(store, keys, ids):
    result = {}
    for k in ids:
        snapshot = store.get_task(k)['prediction']
        result[keys[k]] = {
            'versions': len(store.forecast_timeline(k)),
            'prediction_sha256': payload_key(snapshot),
            'point': snapshot.get('point_min'),
            'basis': (snapshot.get('provenance') or {}).get('decision', 'seeded'),
            'selected': sorted(keys[x['task_id']] for x in snapshot.get('related_tasks', [])),
            'cited': sorted(keys[x] for x in snapshot.get('used_task_ids', [])),
        }
        for field in ['model_error', 'retrieval_error']:
            if snapshot.get(field):
                result[keys[k]][field] = snapshot[field]
    return result


def comparable_direction(board, target, just_finished, completed, actuals, gold):
    # Score only the first comparable timing evidence. Subsequent same-bias evidence
    # need not cause a second move, and a mixed operation need not scale as a whole.
    targets = [k for k in just_finished if pair(target, k) in gold]
    if not targets or any(pair(target, k) in gold for k in completed):
        return None
    mixed = {'textile-mending': 'D', 'video-edit': 'E'}.get(board['id'])
    if target == mixed or mixed in targets:
        return None
    budgets = {t['key']: t['minutes'] for t in board['tasks']}
    ratios = [actuals[k] / budgets[k] for k in targets]
    if all(r >= 1.15 for r in ratios):
        return 1
    if all(r <= .85 for r in ratios):
        return -1
    return None


def check_inputs(b, qwen, kev_run):
    judgments = kev_run['boards'].get(b['id'], {})
    if qwen.get('model') != retrieval.EMBEDDING_MODEL or kev_run.get('model') != retrieval.DECISION_MODEL:
        raise ValueError('Expected recordings from Qwen3-Embedding-0.6B and Kev-4B.')
    if retrieval.relevance_threshold(planner_settings()) != .5:
        raise ValueError('These reference screens require a .50 similarity threshold.')
    if len(judgments.get('pairs', {})) != 10:
        raise ValueError(f"Incomplete Kev matrix for {b['id']}")
    for a, c in combinations(b['tasks'], 2):
        first, second = retrieval.active_decision_order(typed(a), typed(c))
        key = payload_key(retrieval.decision_payload(first, second, (), kev_run['model']))
        vote = judgments['pairs'][pair(a['key'], c['key'])]
        if vote['request_sha256'] != key or kev_run['responses'][key] != vote['score']:
            raise ValueError(f"Exact production Kev input no longer matches {b['id']} {a['key']}{c['key']}.")
    similarity_by_task = prior_candidates(qwen, b)
    for task in b['tasks']:
        key = task['key']
        similarity = judgments.get('prior', {}).get(key)
        if (not similarity or not math.isclose(similarity['cosine'], max(similarity_by_task[key].values()), abs_tol=1e-6)
                or (similarity['cosine'] >= .5 and (similarity.get('request_sha256') not in kev_run['responses']
                    or similarity['request_sha256'] != payload_key(retrieval.decision_payload(typed(task), typed(PRIOR), (), kev_run['model']))))):
            raise ValueError(f"Missing historical Kev receipt or Qwen cutoff for {b['id']} {key}")
    if len(qwen['responses']) != 161:
        raise ValueError('Expected 161 exact Qwen inputs before live forecasting.')


def queue_probe(task, history, _):
    selected = [h for h in history if h.get('matrix_related')]
    return {'point_min': task['estimate_minutes'], 'basis': 'Controlled queue probe',
            'explanation': 'Probe saved this forecast to count actual updates.',
            'related_tasks': [{'task_id': s['id']} for s in selected],
            'provenance': {'decision': 'probe'}}


def do_board(b, kev_run, qwen=None, model_call=None):
    prior = qwen is not None
    if prior:
        check_inputs(b, qwen, kev_run)
    ticks = []
    names = {t['title']: t['key'] for t in [*b['tasks'], PRIOR]}
    def observe_forecast(messages, schema, **kwargs):
        body = json.loads(messages[-1]['content'])
        ticks.append({'target': names[body['new_task']['title']],
                      'history': {names[row['title']]: {'actual': row['actual_minutes'],
                                                        'budget': row['user_estimate_minutes']}
                                  for row in body['history']}})
        return model_call(messages, schema, **kwargs)
    model_pairs = kev_run['boards'][b['id']]['pairs']
    gold = set(b['positive_pairs'])
    keys, ids, additions, waves, calls, missing = {}, {}, [], [], [], []
    # Sinks still active after the final wave provide the independently added cold forecast.
    completed_later = {k for wave in b['completion_waves'] for k in wave}
    sinks = sorted(set('ABCDE') - completed_later)
    cold = max(sinks, key=lambda k: sum(k in p for p in gold))
    def replay(_, body, config):
        k = payload_key(body)
        try:
            if 'input' in body:
                assert qwen is not None
                result = {'data': [{'embedding': vector(qwen, body).tolist()}]}
                role = 'qwen'
            else:
                result = {'answers': {'relevant': {'noul': kev_run['responses'][k]}}}
                role = 'kev'
        except (AssertionError, KeyError):
            missing.append(k)
            raise AssertionError(f'No exact measured input {k}.')
        calls.append([role, k])
        return result
    with TemporaryDirectory() as temp, ExitStack() as ctx:
        now = [1_800_000_000]
        store = PlannerStore(temp, clock=lambda: now[0])
        store.set_setting('dev_replay_enabled', True)
        index = Path(temp) / 'vectors'
        ctx.enter_context(patch.object(retrieval, '_local_post', replay))
        if not prior:
            ctx.enter_context(patch.object(estimate_worker.estimator, 'predict', queue_probe))
            ctx.enter_context(patch.object(estimate_worker.estimator, 'index_current_task', lambda *_: None))
        elif model_call:
            ctx.enter_context(patch.object(estimate_worker.estimator, 'chat_json', observe_forecast))
        if prior:
            old = create(store, PRIOR)
            keys[old] = 'prior'
            store.transition(old, 'finish', spent_seconds=14 * 60)
        for t in b['tasks']:
            now[0] += 60
            key = t['key']
            ids[key] = create(store, t, initial=(prior and key == cold))
            keys[ids[key]] = key
            initial = cells(store, keys)
            while estimate_worker.run_one_job(store, index):
                pass
            final = cells(store, keys)
            if missing or any(v is None for v in final.values()):
                raise RuntimeError(f'Unresolved matrix or missing receipts in {b["id"]}: {missing}, {final}')
            additions.append({'added': key, 'before_worker': initial, 'after_worker': final})
        all_states = state(store, keys, ids.values())
        if prior and all_states[cold].get('retrieval_error'):
            raise ValueError(f'Incomplete cold forecast retrieval for {b["id"]}')
        initial_requests = list(ticks)
        active = set(ids)
        actuals = {}
        true_cells = {k: True for k in gold}
        model_cells = {k: item['related'] for k, item in model_pairs.items()}
        if cells(store, keys) != {k: int(v) for k, v in model_cells.items()}:
            raise AssertionError('Persisted cells differ from the exact model votes.')
        for wave in b['completion_waves']:
            call_offset = len(ticks)
            now[0] += 3600
            active -= wave.keys()
            before = state(store, keys, [ids[k] for k in sorted(active)])
            for k, minutes in wave.items():
                store.transition(ids[k], 'finish', spent_seconds=minutes * 60,
                                 completion_note=b.get('notes', {}).get(k, ''))
            actuals.update(wave)
            booked = sorted(k for k in active if store.get_task(ids[k])['estimate_updating'])
            jobs = 0
            while estimate_worker.run_one_job(store, index):
                jobs += 1
            after = state(store, keys, [ids[k] for k in sorted(active)])
            if missing or any(s.get('retrieval_error') for s in after.values()):
                raise RuntimeError(f'Retrieval was incomplete: {b["id"]} {missing}')
            observed = sorted(k for k in active if after[k]['versions'] > before[k]['versions'])
            expected_gold = neighbors(true_cells, wave, active)
            expected_matrix = neighbors(model_cells, wave, active)
            directions = {k: d for k in sorted(active) if (d := comparable_direction(
                b, k, wave, set(actuals) - wave.keys(), actuals, gold))}
            expected_inputs = {k: sorted(src for src in actuals if model_cells.get(pair(k, src)))
                               for k in observed}
            # The historical permit source is independently screened by dual-query Qwen and Kev.
            if prior:
                for k in expected_inputs:
                    if kev_run['boards'][b['id']]['prior'][k]['related']:
                        expected_inputs[k].append('prior')
            waves.append({'finished': wave, 'before': before, 'after': after, 'new_jobs': jobs,
                          'queue': booked, 'observed': observed, 'reference_targets': expected_gold,
                          'matrix_targets': expected_matrix, 'reference_direction': directions,
                          'expected_sources': expected_inputs, 'forecast_requests': ticks[call_offset:]})
        return {'cold_target': cold if prior else None, 'initial': all_states, 'additions': additions,
                'waves': waves, 'initial_requests': initial_requests, 'inference_inputs': calls}


def make_model_recorder(path, model, original, attempts):
    """Checkpoint each completed raw forecast. No fallback or transport exception is cached."""
    receipts = new_run(path, model=model, responses={})
    if receipts['model'] != model:
        raise ValueError('Forecast receipts were written by a different model.')
    def record(messages, schema, **kwargs):
        body = {'model': model, 'messages': messages, 'schema': schema, 'kwargs': kwargs}
        k = payload_key(body)
        if k not in receipts['responses']:
            try:
                response = original(messages, schema, **kwargs)
            except estimate_worker.ModelUnavailable as exc:
                from model_client import ModelResponseInvalid, ModelContextExceeded
                kind = ('invalid_response' if isinstance(exc, ModelResponseInvalid) else
                        'context_exceeded' if isinstance(exc, ModelContextExceeded) else 'transport')
                issue = {'request_sha256': k, 'kind': kind, 'message': str(exc)}
                attempts.append(issue)
                receipts.setdefault('errors', []).append(issue)
                write(path, receipts)
                raise
            receipts['responses'][k] = {'input': json.loads(messages[-1]['content']),
                                        'response': response}
            write(path, receipts)
        return receipts['responses'][k]['response']
    return record


def forecast_board(job):
    from model_client import model_settings
    from urllib.parse import urlsplit

    tag, b, retry_transport = job
    endpoint, model = model_settings()
    wanted = {'4b': 'ternary-bonsai-4b', '9b': 'ternary-bonsai-2-9b'}[tag]
    if model != wanted or urlsplit(endpoint).hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError(f'Expected local {wanted}; set the forecast model and endpoint.')
    parts = DIR / f'forecast-{tag}-boards'
    path = parts / (b['id'] + '.json')
    if path.exists():
        run = new_run(path)
        if run['model'] != model:
            raise ValueError('Wrong measured model.')
        if not retry_transport or not run['transport_failures']:
            return b['id'], run
        for attempt in range(1, 1000):
            previous = parts / (b['id'] + f'.attempt-{attempt}.json')
            if not previous.exists():
                write(previous, run)
                break
    errors = []
    record = make_model_recorder(parts / (b['id'] + '-receipts.json'), model,
                                 estimate_worker.estimator.chat_json, errors)
    board = do_board(b, new_run(DIR / 'kev.json'), new_run(DIR / 'qwen.json'), record)
    result = {'reference_sha256': LABEL_DIGEST, 'model': model,
              'transport_failures': [e for e in errors if e['kind'] == 'transport'],
              'response_failures': [e for e in errors if e['kind'] != 'transport'], **board}
    write(path, result)
    return b['id'], result


def scenarios(tag=None, workers=1, retry_transport=False):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    path = DIR / (f'forecast-{tag}.json' if tag else 'queue.json')
    if not tag:
        inputs = new_run(DIR / 'kev.json')
        run = new_run(path, model='queue probe (fixed point, no LLM)', boards={})
        for b in fixture()['boards']:
            if b['id'] in run['boards']:
                continue
            out = do_board(b, inputs)
            run['boards'][b['id']] = out
            write(path, run)
        return
    run = new_run(path, model={'4b': 'ternary-bonsai-4b', '9b': 'ternary-bonsai-2-9b'}[tag], boards={})
    if workers < 1:
        raise ValueError('--workers must be positive.')
    jobs = [(tag, b, retry_transport) for b in fixture()['boards']]
    def store(item):
        name, data = item
        # Per-board files contain full snapshots and request receipts. This manifest is the single writer.
        run['boards'][name] = {'path': f'forecast-{tag}-boards/{name}.json',
                               'refresh_jobs': sum(w['new_jobs'] for w in data['waves']),
                               'transport_failures': len(data['transport_failures'])}
        write(path, run)
        print(name, run['boards'][name], flush=True)
    if workers == 1:
        for job in jobs:
            store(forecast_board(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(forecast_board, job) for job in jobs]
            for f in as_completed(futures):
                store(f.result())


def stress_seed(seed, boards):
    """Model-independent state machine test with a separate co-presence and source-snapshot oracle."""
    import random
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


def stress():
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
    run['failed_seeds'] = [k for k, v in run['seeds'].items() if 'error' in v]
    write(path, run)
    print(run['totals'], 'failed seeds', run['failed_seeds'], flush=True)


def metrics(rows):
    counts = Counter({k: 0 for k in ('tp', 'fp', 'fn', 'tn')})
    for ref, actual in rows:
        counts[('tp' if ref else 'fp') if actual else ('fn' if ref else 'tn')] += 1
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    return {**counts, 'precision': round(tp/(tp+fp), 4) if tp+fp else None,
            'recall': round(tp/(tp+fn), 4) if tp+fn else None,
            'f1': round(2*tp/(2*tp+fp+fn), 4) if tp+fp+fn else None}


def summarize_boards(boards, counted, live=False):
    counts = Counter()
    vs_gold, vs_matrix, queue_gold, matrix_actual = [], [], [], []
    directions, source_errors, state_errors = [], [], []
    citations = Counter()
    for b in boards:
        if b['id'] not in counted:
            continue
        out = counted[b['id']]
        counts['boards'] += 1
        ctrl = False
        for i, a in enumerate(out['additions']):
            k = a['added']
            pre, post = a['before_worker'], a['after_worker']
            prev = out['additions'][i-1]['after_worker'] if i else {}
            new = {cell for cell in post if k in cell}
            ok = (len(new) == i and all(pre[cell] is None and post[cell] in (0, 1) for cell in new)
                  and {key: v for key, v in pre.items() if key not in new} == prev
                  and all(post[cell] == value for cell, value in prev.items()))
            counts['addition_snapshots_checked'] += 1
            if not ok:
                state_errors.append({'board': b['id'], 'added': k, 'error': 'matrix expansion'})
        if live:
            counts['cold_direct'] += out['initial'][out['cold_target']]['basis'] == 'direct'
            counts['transport_failures'] += len(out['transport_failures'])
            counts['response_failures_or_repairs'] += len(out['response_failures'])
        budgets = {t['key']: t['minutes'] for t in [*b['tasks'], PRIOR]}
        times = {'prior': 14}
        for i, w in enumerate(out['waves'], 1):
            times.update(w['finished'])
            if live:
                for key in w['observed']:
                    got = [call['history'] for call in w['forecast_requests'] if call['target'] == key]
                    want = {s:{'budget': budgets[s], 'actual': times[s]} for s in w['expected_sources'][key]}
                    counts['refresh_payloads_checked'] += 1
                    if not got or any(batch != want for batch in got):
                        source_errors.append({'board': b['id'], 'wave': i, 'target': key,
                                              'expected_timing_payload': want, 'observed_payloads': got})
                for call in w['forecast_requests']:
                    if call['target'] not in w['observed']:
                        state_errors.append({'board': b['id'], 'wave': i, 'error': 'chat request for an unchanged target'})
            counts['waves'] += 1
            counts['individual_completions'] += len(w['finished'])
            counts['multi_completion_waves'] += len(w['finished']) > 1
            if not w['observed']:
                counts['waves_without_refresh'] += 1
            counts['jobs'] += w['new_jobs']
            if w['new_jobs'] != len(w['observed']):
                state_errors.append({'board': b['id'], 'wave': i, 'error': 'unexpected worker job count'})
            for k in w['before']:
                positive = k in w['reference_targets']
                routed = k in w['matrix_targets']
                queued = k in w['queue']
                ran = k in w['observed']
                vs_gold.append((positive, ran))
                vs_matrix.append((routed, ran))
                queue_gold.append((positive, queued))
                matrix_actual.append((routed, queued))
                counts['target_transitions'] += 1
                bef, aft = w['before'][k], w['after'][k]
                delta = aft['versions'] - bef['versions']
                if (ran and delta != 1) or (not ran and aft != bef):
                    state_errors.append({'board': b['id'], 'wave': i, 'target': k, 'error': 'forecast persistence'})
                if not positive and not ran:
                    ctrl = True
                    counts['reference_no_change_controls'] += 1
                if not ran:
                    continue
                if aft['selected'] != sorted(w['expected_sources'][k]):
                    source_errors.append({'board': b['id'], 'wave': i, 'target': k, 'expected': w['expected_sources'][k], 'sent': aft['selected']})
                if live:
                    direct = aft['basis'] == 'direct'
                    counts['refresh_direct' if direct else 'refresh_fallback'] += 1
                    if direct:
                        actual_cited = set(aft['cited'])
                        expected_source = set(w['finished']) & set(w['expected_sources'][k])
                        citations['jobs_with_new_selected_source'] += bool(expected_source)
                        citations['jobs_citing_any_new_source'] += bool(actual_cited & expected_source)
                        citations['new_slots'] += len(expected_source)
                        citations['new_slots_cited'] += len(actual_cited & expected_source)
                        citations['outside_selected'] += len(actual_cited - set(aft['selected']))
            for k, direction in w['reference_direction'].items():
                before, after = w['before'][k], w['after'][k]
                if not live:
                    continue
                available = bool(set(w['finished']) & set(after['selected']))
                ran = k in w['observed']
                direct = ran and after['basis'] == 'direct'
                change = after['point']-before['point']
                budget = next(t['minutes'] for t in b['tasks'] if t['key'] == k)
                directions.append({'board': b['id'], 'wave': i, 'target': k, 'expected': 'up' if direction == 1 else 'down',
                    'before': before['point'], 'before_source': before['basis'], 'after': after['point'],
                    'related_evidence_available': available, 'direct_refresh': direct,
                    'moved_expected_way': direct and direction*change > 0,
                    'on_expected_side_of_budget': direct and direction*(after['point']-budget) > 0})
        counts['boards_with_reference_no_change_control'] += int(ctrl)
    return {'counts': dict(counts), 'persisted_vs_reference': metrics(vs_gold),
            'queued_vs_reference': metrics(queue_gold), 'persisted_vs_kev': metrics(vs_matrix),
            'queued_vs_kev': metrics(matrix_actual), 'state_errors': state_errors, 'source_errors': source_errors,
            'citations': dict(citations), 'directions': directions}


def report():
    inputs = new_run(DIR/'kev.json')
    boards = fixture()['boards']
    independent = json.loads((DIR/'second-labels.json').read_text())
    if hashlib.sha256((DIR/'second-labels.json').read_bytes()).hexdigest() != '8dd3fa47738c763e139f5d908eea741f9ed2a52f80c931775eff7f9362865b90':
        raise ValueError('Blind labels differ from the pre-inference second review.')
    confusion, consensus, misses, qwen_rows = [], [], [], []
    qwen = new_run(DIR/'qwen.json')
    topology, by_board = {}, {}
    for b in boards:
        truth = set(b['positive_pairs'])
        topology[b['id']] = len(truth)
        check_inputs(b, qwen, inputs)
        by_key = {t['key']:t for t in b['tasks']}
        def lookup(t, role):
            return vector(qwen, {'model': qwen['model'], 'input': retrieval.embedding_input(t, role)})
        votes = []
        for key, r in inputs['boards'][b['id']]['pairs'].items():
            positive = key in truth
            accepted = r['related']
            confusion.append((positive, accepted))
            votes.append((positive, accepted))
            first, second = retrieval.active_decision_order(typed(by_key[key[0]]), typed(by_key[key[1]]))
            doc = lookup(second, False)
            scores = {}
            for role, is_query in [('instructed', True), ('bare', False)]:
                q = lookup(first, is_query)
                scores[role] = float(np.dot(q, doc)/np.linalg.norm(q)/np.linalg.norm(doc))
            qwen_rows.append({'board': b['id'], 'pair': key, 'reference': positive, **scores})
            consensus.append((positive, key in independent[b['id']]))
            if positive != accepted:
                misses.append({'board': b['id'], 'pair': key, 'reference': positive, 'score': r['score']})
        by_board[b['id']] = metrics(votes)
    out = {'reference_sha256': LABEL_DIGEST, 'independent_review': metrics(consensus), 'matrix_by_board': by_board,
         'matrix': metrics(confusion), 'matrix_disagreements': misses, 'positives_per_board': topology,
         'qwen_only_active_pairs': {'description': 'Diagnostic using the production dual-query Qwen screen on the same canonical active pairs. The actual active matrix uses Kev for every pair.',
            'threshold': .5, 'metrics': metrics((r['reference'], max(r['instructed'], r['bare']) >= .5) for r in qwen_rows),
            'pairs': qwen_rows},
         'qwen_calls': len(new_run(DIR/'qwen.json')['responses']),
         'prior_screen_passes': sum(r['cosine'] >= .5 for b in inputs['boards'].values() for r in b['prior'].values()),
         'prior_kev_acceptances': sum(r['related'] for b in inputs['boards'].values() for r in b['prior'].values())}
    if (DIR/'queue.json').exists():
        out['queue'] = summarize_boards(boards, new_run(DIR/'queue.json')['boards'])
    if (DIR/'stress.json').exists():
        stress_result = new_run(DIR/'stress.json')
        out['stress'] = {k:stress_result[k] for k in ('seed_count', 'totals', 'failed_seeds') if k in stress_result}
    for tag in ('4b', '9b'):
        manifest = DIR/f'forecast-{tag}.json'
        if not manifest.exists():
            continue
        saved = {k:json.loads((DIR/v['path']).read_text()) for k, v in new_run(manifest)['boards'].items()}
        result = summarize_boards(boards, saved, True)
        result['unique_successful_requests'] = sum(len(new_run(DIR/f'forecast-{tag}-boards/{k}-receipts.json')['responses']) for k in saved)
        direction_summary = {}
        for label, rows in [('all', result['directions']),
                            ('up', [r for r in result['directions'] if r['expected'] == 'up']),
                            ('down', [r for r in result['directions'] if r['expected'] == 'down']),
                            ('model_initial', [r for r in result['directions'] if r['before_source'] == 'direct']),
                            ('seeded_initial', [r for r in result['directions'] if r['before_source'] == 'seeded'])]:
            conditional = [r for r in rows if r['direct_refresh'] and r['related_evidence_available']]
            direction_summary[label] = {
                'reference_cases': len(rows), 'direct_with_evidence': len(conditional),
                'moved_expected_way': sum(r['moved_expected_way'] for r in conditional),
                'opposite_move': sum(r['after'] != r['before'] and not r['moved_expected_way'] for r in conditional),
                'unchanged_point': sum(r['after'] == r['before'] for r in conditional),
                'on_expected_side_of_budget': sum(r['on_expected_side_of_budget'] for r in conditional)}
        result['direction_summary'] = direction_summary
        out[tag] = result
    write(DIR/'metrics.json', out)
    render_results(out)
    print({k:out[k] for k in ('independent_review', 'matrix', 'matrix_disagreements', 'prior_screen_passes')})
    if 'queue' in out: print('Queue', out['queue']['counts'], out['queue']['persisted_vs_reference'])
    for tag in ('4b', '9b'):
        if tag in out: print(tag, out[tag]['counts'], out[tag]['citations'])


def render_results(out):
    def perf(label, scores):
        def pct(v):
            return '—' if v is None else f'{100*v:.1f}%'
        return (f"| {label} | {scores['tp']} | {scores['fp']} | {scores['fn']} | {scores['tn']} "
                f"| {pct(scores['precision'])} | {pct(scores['recall'])} |")
    text = [
        '# Selective re-estimation results',
        '',
        'The same 160 pair labels were reviewed independently by two assistants. '
        'All 41 positive and 119 negative labels agreed. Counts below use the pair or surviving-target '
        'opportunities named in each row. [Reference](reference.json) · [Full metrics and per-case scores](metrics.json)',
        '',
        '| Component and denominator | TP | FP | FN | TN | Precision | Recall |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        perf('Qwen dual-query screen alone, 160 active pairs', out['qwen_only_active_pairs']['metrics']),
        perf('Kev, 160 active pairs', out['matrix'])]
    if 'queue' in out:
        queue = out['queue']
        c = queue['counts']
        n = c['target_transitions']
        text.extend([perf(f'Saved refresh vs semantic reference, {n} survivors', queue['persisted_vs_reference']),
                     perf(f'Saved refresh vs stored Kev decision, {n} survivors', queue['persisted_vs_kev']), '',
                     f"The worker saved {c['jobs']} refreshes over {c['waves']} waves on {c['boards']} boards. "
                     f"{c['multi_completion_waves']} waves included multiple completions before the worker drained. "
                     f"{c['reference_no_change_controls']} unrelated survivor opportunities kept their prior forecast. "
                     f"There were {len(queue['state_errors'])} state or queue-order errors over "
                     f"{c['addition_snapshots_checked']} task additions and {n} survivor transitions. "
                     "[Queue traces](queue.json)"])
    if out['matrix_disagreements']:
        text.extend(['', '### Kev disagreements', '',
                     '| Board | Pair | Reference | Kev score |', '| --- | --- | --- | ---: |'])
        for r in out['matrix_disagreements']:
            text.append(f"| {r['board']} | {r['pair']} | {'Relevant' if r['reference'] else 'Unrelated'} | {r['score']:.3f} |")
    text.extend(['', f"Actual Qwen embeddings were recorded for 80 task documents, their 80 instructed queries, "
                 f"and the shared permit-renewal document. {out['prior_screen_passes']} of the 80 tasks passed the "
                 f"dual-role .50 threshold for that unrelated historical source; Kev accepted {out['prior_kev_acceptances']}. "
                 "[Embedding receipts](qwen.json) · [Relevance receipts](kev.json)"])
    live = [t for t in ('4b', '9b') if t in out]
    if live:
        text.extend(['', '## Live forecasts', '',
            '| Model | Boards | Cold model outputs | Direct refreshes | Jobs citing a newly finished source | New timing payload checks | Payload mismatches | Transport failures |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for t in live:
            r = out[t]
            c = r['counts']
            a = r['citations']
            text.append(f"| Bonsai {t.upper()} | {c['boards']}/16 | {c['cold_direct']}/{c['boards']} | "
                        f"{c.get('refresh_direct', 0)}/{c['jobs']} | {a.get('jobs_citing_any_new_source', 0)}/"
                        f"{a.get('jobs_with_new_selected_source', 0)} | {c.get('refresh_payloads_checked', 0)} | "
                        f"{len(r['source_errors'])} | {c.get('transport_failures', 0)} |")
        text.extend(['', 'The movement test uses a task’s first comparable measured overrun or underrun. '
                     'The columns count only direct model refreshes that received the relevant evidence. '
                     'The remaining task has no known actual duration.', '',
                     '| Model and expected direction | Reference cases | Received evidence and a model refresh | Moved as expected | Unchanged | Opposite move | On expected side of user budget |',
                     '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for t in live:
            for k in ('up', 'down', 'model_initial', 'seeded_initial'):
                g = out[t]['direction_summary'][k]
                text.append(f"| Bonsai {t.upper()} · {k.replace('_', ' ')} | {g['reference_cases']} | {g['direct_with_evidence']} | "
                            f"{g['moved_expected_way']} | {g['unchanged_point']} | {g['opposite_move']} | {g['on_expected_side_of_budget']} |")
        text.extend(['', 'The initial-model and seeded-initial rows split the same directional cases by the source '
                     'of their previous point estimate. Full persisted before/after states, input/source IDs and model responses are linked from the manifests. '
                     + ' · '.join(f'[{t.upper()} manifest](forecast-{t}.json)' for t in live)])
    if 'stress' in out and 'totals' in out['stress']:
        g = out['stress']
        c = g['totals']
        text.extend(['', '## Interleaving stress', '',
            f"{g['seed_count']} ten-card schedules included {c['created']} submissions, {c['completions']} completions, "
            f"{c['starts']} timer starts, {c['pauses']} pauses, and {c['time_corrections']} retroactive time corrections. "
            f"The runner checked {c['matrix_verdicts_checked']:,} persisted cell values, "
            f"{c['initial_snapshots_checked']:,} creation snapshots, {c['current_snapshots_checked']:,} current forecasts, "
            f"and {c['correction_targets_checked']} correction-target decisions. {len(g['failed_seeds'])} seeds failed. "
            "The test substitutes controlled model outputs and runs the real store and inference workers. [Per-seed counts](stress.json)"])
    (DIR/'findings.md').write_text('\n'.join(text)+'\n')


def later_stage(args):
    if args.stage == 'queue':
        scenarios()
    elif args.stage == 'forecast':
        if not args.model_tag:
            raise ValueError('--model-tag 4b or 9b is required')
        scenarios(args.model_tag, args.workers, args.retry_transport)
    elif args.stage == 'stress':
        stress()
    else:
        report()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['vectors', 'kev', 'prior', 'queue', 'forecast', 'stress', 'report'])
    parser.add_argument('--model-tag', choices=['4b', '9b'])
    parser.add_argument('--workers', type=int, default=1, help='Independent forecast processes, one SQLite store per board.')
    parser.add_argument('--retry-transport', action='store_true', help='Preserve and rerun boards with a transport failure; reuse successful exact requests.')
    args = parser.parse_args()
    if args.stage == 'vectors':
        vectors()
    elif args.stage in ('kev', 'prior'):
        kev(prior_only=args.stage == 'prior')
    else:
        later_stage(args)


if __name__ == '__main__':
    main()
