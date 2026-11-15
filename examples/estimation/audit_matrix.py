"""Evaluate active-task judgments and selective re-estimation using real worker transactions.

Run with python -m examples.estimation.audit_matrix <output.json> [--lifecycle].
Set the local Qwen, Kev and forecast endpoints via the normal planner settings.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import estimate_worker
import retrieval
from model_client import planner_settings
from planner_store import PlannerStore


BOARDS = Path(__file__).resolve().parents[2] / 'data/results/component-audit/matrix/boards.json'
DAY = '2026-11-15'


def write(output, result):
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix('.tmp')
    staged.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    staged.replace(output)


def add(store, task, *, initial=True):
    key, title, scope = task
    result = store.create_task(title=title, description=scope, category='Project', estimate_minutes=30,
                               day=DAY, prediction={'point_min': 30, 'explanation': 'Seeded budget.'}
                               if initial else None)
    return result['id']


def matrix(store, keys):
    with store._db() as db:
        return {'/'.join(sorted((keys[left], keys[right]))): state for left, right, state in
                db.execute('SELECT left_id,right_id,related FROM task_relations')}


def confusion(boards):
    counts = {k: 0 for k in ('tp', 'fp', 'tn', 'fn', 'unresolved')}
    for board in boards:
        for pair in board['pairs']:
            actual, expect = pair['decision'], pair['expected']
            key = 'unresolved' if actual is None else (
                ('tp' if expect else 'fp') if actual else ('fn' if expect else 'tn'))
            counts[key] += 1
    def rate(n, d):
        return round(n / d, 4) if d else None
    return dict(counts, precision=rate(counts['tp'], counts['tp'] + counts['fp']),
                recall=rate(counts['tp'], counts['tp'] + counts['fn']))


def run_boards(out, split, method, canonical, reference):
    fixtures = json.loads(BOARDS.read_text())
    known = {}
    if reference:
        for board in json.loads(reference.read_text())['boards']:
            for pair in board['pairs']:
                known[pair['request_sha256']] = pair['score']
    run = {'orientation': 'canonical' if canonical else 'insertion', 'labels_sha256': hashlib.sha256(BOARDS.read_bytes()).hexdigest(),
           'model': retrieval.DECISION_MODEL, 'variant': 'method' if method else 'production at run time', 'boards': []}
    for board in fixtures:
        if split and board['split'] != split:
            continue
        with TemporaryDirectory() as tmp:
            store = PlannerStore(tmp)
            keys, steps, pairs = {}, [], []
            gold = {frozenset(related) for related in board['related']}
            for task in board['tasks']:
                keys[add(store, task)] = task[0]
                before = matrix(store, keys)
                while job := store.claim_relation():
                    pair = sorted(keys[k] for k in job['pair'])
                    calls = []
                    send = retrieval._local_post
                    def observe(endpoint, body, config):
                        if method:
                            from examples.estimation.audit_retrieval import METHOD_QUESTION
                            body['questions']['relevant']['instructions'] = METHOD_QUESTION
                        digest = payload_key(body)
                        if digest in known:
                            reply = {'answers': {'relevant': {'type': 'noul', 'noul': known[digest]}}}
                        else:
                            reply = send(endpoint, body, config)
                        calls.append((digest, reply['answers']['relevant']['noul'], digest in known))
                        return reply
                    order = retrieval.active_decision_order if canonical else lambda a, b, examples: (a, b)
                    first, _ = order(job["target"], job["candidate"], job["examples"])
                    with patch.object(retrieval, '_local_post', observe), patch.object(retrieval, 'active_decision_order', order):
                        estimate_worker._process_relation(store, job)
                    after = matrix(store, keys)['/'.join(pair)]
                    if len(calls) != 1 or after is None:
                        raise RuntimeError(f'Expected a live decision for {pair}, observed {calls} and {after}.')
                    pairs.append({'pair': pair, 'target': keys[job['target']['id']],
                                  'expected': frozenset(pair) in gold, 'decision': bool(after), 'model_target': keys[first['id']],
                                  'score': calls[0][1], 'request_sha256': calls[0][0], 'reused_exact_input': calls[0][2]})
                steps.append({'added': task[0], 'before_judging': before, 'after_judging': matrix(store, keys)})
            run['boards'].append({'name': board['name'], 'split': board['split'], 'pairs': pairs, 'steps': steps})
            run['metrics'] = {part: confusion(b for b in run['boards'] if b['split'] == part)
                              for part in ('development', 'validation')}
            write(out, run)
            print(board['name'], run['metrics'], flush=True)


def payload_key(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def prepare(out):
    """Record the real Qwen vectors and Kev probabilities for the exact lifecycle inputs."""
    board = next(b for b in json.loads(BOARDS.read_text()) if b['name'] == 'release desk')
    records = {k: dict(title=t, description=d) for k, t, d in board['tasks']}
    config = planner_settings()
    em = config.get('PLANNER_EMBEDDING_MODEL') or retrieval.EMBEDDING_MODEL
    dm = config.get('PLANNER_DECISION_MODEL') or retrieval.DECISION_MODEL
    cache = json.loads(out.read_text()) if out.exists() else {'responses': {}}
    def capture(body, endpoint, name):
        if payload_key(body) in cache['responses']:
            return
        reply = retrieval._local_post(endpoint, body, config)
        cache['responses'][payload_key(body)] = {'input': name, 'model': body['model'], 'response': reply}
        write(out, cache)
    for key, task in records.items():
        for query in ([False, True] if key == 'verify' else [False]):
            capture({'model': em, 'input': retrieval.embedding_input(task, query)},
                    config['PLANNER_EMBEDDING_ENDPOINT'], key + (' (instructed)' if query else ' (document)'))
    for target, candidate in [('memo', 'sign'), ('verify', 'sign'), ('verify', 'memo'), ('memo', 'verify'), ('verify', 'bike')]:
        capture(retrieval.decision_payload(records[target], records[candidate], (), dm),
                config['PLANNER_DECISION_ENDPOINT'], target + '/' + candidate)


def lifecycle(out, inputs):
    board = next(b for b in json.loads(BOARDS.read_text()) if b['name'] == 'release desk')
    tasks = {t[0]: t for t in board['tasks']}
    calls = []
    responses = json.loads(inputs.read_text())['responses'] if inputs else None
    model = planner_settings()['PLANNER_MODEL']
    run = {'model': model, 'budget_minutes': 30, 'control_cards': 'Independent memo and related signing task start with a saved 30-minute estimate.',
           'steps': [], 'http': calls}
    send = retrieval._local_post
    def observe(endpoint, body, config):
        if responses is None:
            reply = send(endpoint, body, config)
        else:
            reply = responses[payload_key(body)]['response']
        calls.append({'stage': 'qwen' if 'input' in body else 'kev', 'model': body['model'],
                      'from_recorded_run': responses is not None, 'request_sha256': payload_key(body)})
        return reply
    predict = estimate_worker.estimator.chat_json
    def observe_forecast(*args, **kwargs):
        calls.append({'stage': 'forecast', 'model': model})
        return predict(*args, **kwargs)
    with TemporaryDirectory() as tmp, patch.object(retrieval, '_local_post', observe), \
            patch.object(estimate_worker.estimator, 'chat_json', observe_forecast):
        store = PlannerStore(tmp)
        store.set_setting('dev_replay_enabled', True)
        index = Path(tmp) / 'vectors'
        # Supply unrelated historical work so the first forecast exercises Qwen and its threshold.
        old = add(store, tasks['bike'])
        store.transition(old, 'finish', spent_seconds=20 * 60)
        a = add(store, tasks['sign'])
        b = add(store, tasks['memo'])
        c = add(store, tasks['verify'], initial=False)
        keys = {a: 'sign', b: 'memo', c: 'verify', old: 'bike'}
        for stage in ('after_add', 'after_independent_completion', 'after_related_completion'):
            if stage == 'after_independent_completion':
                store.transition(b, 'finish', spent_seconds=75 * 60)
            elif stage == 'after_related_completion':
                store.transition(a, 'finish', spent_seconds=90 * 60,
                                 completion_note='The token needed reauthorization before verification.')
            queued = store.get_task(c)['estimate_updating']
            jobs = 0
            while estimate_worker.run_one_job(store, index):
                jobs += 1
            pred = store.get_task(c)['prediction']
            cells = matrix(store, keys)
            if any(v is None for v in cells.values()) or pred.get('retrieval_error'):
                raise RuntimeError(f'Incomplete retrieval or matrix: {cells}, {pred.get("retrieval_error")}')
            run['steps'].append({'stage': stage, 'queued': bool(queued), 'jobs': jobs, 'matrix': matrix(store, keys),
                                 'timeline_length': len(store.forecast_timeline(c)), 'minutes': pred['point_min'],
                                 'range': [pred['low_min'], pred['high_min']], 'provenance': pred['provenance'],
                                 'rationale': pred['explanation'], 'model_error': pred.get('model_error'),
                                 'retrieval_error': pred.get('retrieval_error'),
                                 'selected': [{**r, 'task_id': keys[r['task_id']]} for r in pred['related_tasks']],
                                 'cited': [keys[k] for k in pred['used_task_ids']]})
            if stage == 'after_related_completion':
                initial, independent, final = run['steps']
                run['checks'] = {
                    'expected_matrix': initial['matrix'] == {'memo/sign': 0, 'memo/verify': 0, 'sign/verify': 1},
                    'independent_queued_nothing': independent['jobs'] == 0 and not independent['queued'],
                    'independent_preserved_forecast': independent['timeline_length'] == initial['timeline_length']
                        and independent['minutes'] == initial['minutes'],
                    'related_triggered_one_refresh': final['queued'] and final['jobs'] == 1
                        and final['timeline_length'] == independent['timeline_length'] + 1,
                    'only_related_evidence_reached_forecaster': [r['task_id'] for r in final['selected']] == ['sign'],
                    'model_cited_related_source': 'sign' in final['cited'],
                    'model_revised_up_after_overrun': final['minutes'] > initial['minutes'],
                    'model_wrote_both_forecasts': initial['provenance']['decision'] == final['provenance']['decision'] == 'direct',
                }
            write(out, run)
            print(stage, pred['point_min'], pred['provenance'], flush=True)


def archived_peers(out):
    """Replay the historical .50 screen for accepted co-active peers in each published schedule."""
    from examples.estimation.calibrate_retrieval import cosine
    from examples.estimation.run import histories, DATASET
    import sample_retrieval

    records = []
    for history in histories(None):
        name = history['id']
        index = sample_retrieval.DIRECTORY / (name + '.json')
        pairs = json.loads(index.read_text())['decisions'].values()
        decisions = {(r['target'], r['candidate']): r['related'] for r in pairs}
        tasks = {r['id']: r for group in ('tasks', 'open_tasks') for r in history[group]}
        active, edges, end = set(), [], {}
        events = json.loads((DATASET / 'schedules' / (name + '.json')).read_text())
        for at, event in enumerate(events):
            task = event['task']
            if event['action'] == 'create':
                edges.extend((task, peer) for peer in sorted(active))
                active.add(task)
            elif event['action'] == 'finish':
                end[task] = at
                active.remove(task)
        for newer, older in edges:
            if not decisions[newer, older] or any(tasks[k].get('fixed_duration') for k in (newer, older)):
                continue
            if newer not in end and older not in end:
                continue
            source, target = sorted((newer, older), key=lambda k: end.get(k, float('inf')))
            v = sample_retrieval.read_vectors(index, target)
            prior = sample_retrieval.read_vectors(index, source)['document']
            score = cosine(v['query'], prior)
            reversed_decision = decisions.get((target, source))
            records.append({'profile': name, 'active_target': newer, 'active_candidate': older, 'remaining': target,
                            'completed': source, 'active_decision': True, 'rejudged_decision': reversed_decision,
                            'query_cosine': round(score, 5), 'bare_cosine': round(cosine(v['document'], prior), 5),
                            'old_refresh_would_include': score >= .5 and reversed_decision is True})
    write(out, records)


def archived_orientation(out):
    """Check order invariance on every physically overlapping pair in the ten public schedules."""
    from examples.estimation.package_retrieval import inputs, matrix_pairs
    from examples.estimation.run import histories
    import sample_retrieval

    result = {'actual_overlaps': 0, 'both_orders_archived': 0, 'old_disagreements': [],
              'legacy_reverse_reused': 0, 'new_order_disagreements': []}
    for history in histories(None):
        tasks = inputs(history)
        dataset = json.loads((sample_retrieval.DIRECTORY / (history['id'] + '.json')).read_text())
        votes = {(r['target'], r['candidate']): r['related'] for r in dataset['decisions'].values()}
        holdouts = {r['id'] for r in history['holdouts']}
        for target, candidate in sorted(matrix_pairs(history)):
            if target in holdouts or candidate in holdouts:
                continue
            result['actual_overlaps'] += 1
            first, second = tasks[target], tasks[candidate]
            order = retrieval.active_decision_order(first, second)
            reverse_order = retrieval.active_decision_order(second, first)
            metadata = {'history': history['id'], 'target': target, 'candidate': candidate}
            if order != reverse_order:
                result['new_order_disagreements'].append(metadata)
            model_keys = (target, candidate) if order[0] is first else (candidate, target)
            if (candidate, target) in votes:
                result['both_orders_archived'] += 1
                if votes[target, candidate] != votes[candidate, target]:
                    result['old_disagreements'].append({**metadata, 'newer_first': votes[target, candidate],
                                                        'older_first': votes[candidate, target],
                                                        'canonical_vote': votes[model_keys]})
            canonical = sorted((first, second), key=lambda t: tuple(t.get(k) or '' for k in
                               ('title', 'description', 'details', 'pdf_text')))
            result['legacy_reverse_reused'] += order[0] != canonical[0]
    write(out, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--split', choices=['development', 'validation'])
    parser.add_argument('--lifecycle', action='store_true')
    parser.add_argument('--method', action='store_true', help='Evaluate the timing-method prompt candidate')
    parser.add_argument('--prepare', action='store_true', help='Record exact live Qwen/Kev inputs for the lifecycle')
    parser.add_argument('--inputs', type=Path, help='Replay recorded Qwen/Kev HTTP responses; Bonsai stays live')
    parser.add_argument('--archived-peers', action='store_true', help='Audit accepted peers in packaged schedules')
    parser.add_argument('--canonical', action='store_true', help='Use stable active-pair ordering from the patched app')
    parser.add_argument('--reference', type=Path, help='Reuse previously measured probabilities for identical payloads')
    parser.add_argument('--archived-orientation', action='store_true', help='Audit direction invariance of shipped matrix decisions')
    args = parser.parse_args()
    if args.archived_orientation:
        archived_orientation(args.output)
    elif args.archived_peers:
        archived_peers(args.output)
    elif args.prepare:
        prepare(args.output)
    elif args.lifecycle:
        lifecycle(args.output, args.inputs)
    else:
        run_boards(args.output, args.split, args.method, args.canonical, args.reference)
