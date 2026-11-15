"""Score independently labeled active task boards and their real SQLite completion waves.

Run from the repo with python -m examples.estimation.scale_matrix vectors|kev|queue|forecast|stress|report.
The stages checkpoint by request/board. Supply PLANNER_* endpoints as for the application.
"""
import argparse
import base64
import hashlib
import json
import math
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
from examples.estimation.scale_matrix_reporting import report, verify_forecasts
from examples.estimation.scale_matrix_stress import stress
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
            'range': [snapshot.get('low_min'), snapshot.get('high_min')],
            'confidence': snapshot.get('confidence'),
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
            # Each measured duration fits between this and the preceding user action.
            now[0] += max(60, sum(wave.values())) * 60
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


def later_stage(args):
    if args.stage == 'queue':
        scenarios()
    elif args.stage == 'forecast':
        if not args.model_tag:
            raise ValueError('--model-tag 4b or 9b is required')
        scenarios(args.model_tag, args.workers, args.retry_transport)
    elif args.stage == 'verify':
        if not args.model_tag:
            raise ValueError('--model-tag identifies the measured forecaster.')
        outcome = verify_forecasts(args.model_tag)
        print('Replayed exact forecast receipts for', len(outcome['boards']), 'boards.', flush=True)
    elif args.stage == 'stress':
        stress()
    else:
        report()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['vectors', 'kev', 'prior', 'queue', 'forecast', 'stress', 'report', 'verify'])
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
