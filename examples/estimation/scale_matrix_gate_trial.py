"""Paired, separately scored Kev prompt ablation on shared substantive operations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from itertools import combinations
import json

import retrieval
from model_client import planner_settings
from examples.estimation.audit_matrix import payload_key, write
from examples.estimation.scale_matrix import DIR, fixture, metrics, typed

FIXTURE_SHA = '1577f0727175d760b23fda7859cff12ca3f6cd23e625e69dc35680abcc524946'
# Frozen before the holdout or alternative prompt was run. Baseline payload stays byte-for-byte identical.
SHARED_WORK = (
    " Identify the substantial hands-on or intellectual steps each person must actually perform. "
    "A multi-part task can share a timing comparison with a task that performs one substantial part, "
    "even when the totals differ. The duration can inform that part. "
    "A task that merely names the operation, presents its finished results, builds a tool for someone "
    "else to do it, or uses an artifact already prepared by someone else does not perform that operation. "
    "A shared minor administrative step by itself is insufficient."
)


def cases():
    frozen = DIR/'composite-holdout.json'
    if sha256(frozen.read_bytes()).hexdigest() != FIXTURE_SHA:
        raise ValueError('Mixed-work reference changed.')
    extra = json.loads(frozen.read_text())
    second = json.loads((DIR/'composite-second-labels.json').read_text())
    for b in extra['boards']:
        gold = set(extra['positive_pairs'][b['id']])
        if gold != set(second[b['id']]):
            raise ValueError(f'Independent reference disagreement on {b["id"]}.')
        for a, c in combinations(b['tasks'], 2):
            pk = a['key'] + c['key']
            yield 'fresh', b['id'], pk, typed(a), typed(c), pk in gold
    for b in fixture()['boards']:
        if b['id'] not in ('textile-mending', 'video-edit'):
            continue
        split = 'development' if b['id'] == 'textile-mending' else 'film'
        for a, c in combinations(b['tasks'], 2):
            pk = a['key'] + c['key']
            yield split, b['id'], pk, typed(a), typed(c), pk in b['positive_pairs']
    friday = next(b for b in json.loads((DIR/'mixed-boards.json').read_text()) if b['name'] == 'Friday close')
    gold = {frozenset(p) for p in friday['related']}
    for a, c in combinations(friday['tasks'], 2):
        pair = frozenset((a[0], c[0]))
        yield 'friday', friday['name'], '/'.join(sorted(pair)), dict(title=a[1], description=a[2]), dict(title=c[1], description=c[2]), pair in gold


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--workers', type=int, default=1, help='Parallel HTTP requests for a multi-slot Kev runtime')
    cli.add_argument('--fresh-only', action='store_true', help='Score only the six new holdouts')
    args = cli.parse_args()
    cfg = planner_settings()
    model = cfg.get('PLANNER_DECISION_MODEL') or retrieval.DECISION_MODEL
    if model != retrieval.DECISION_MODEL:
        raise ValueError('Use Kev-4B for both sides of the ablation.')
    output = DIR/'composite-ablation.json'
    run = json.loads(output.read_text()) if output.exists() else {
        'reference_sha256': FIXTURE_SHA, 'prompt_addition': SHARED_WORK, 'responses': {}, 'scores': {}}
    if (run['reference_sha256'], run['prompt_addition']) != (FIXTURE_SHA, SHARED_WORK):
        raise ValueError('Checkpoint belongs to a different prompt or labels.')

    known = {}
    def import_known(key, value, origin):
        if key not in known:
            known[key] = {'score': value, 'origin': origin}
    if (DIR/'kev.json').exists():
        for k, v in json.loads((DIR/'kev.json').read_text())['responses'].items():
            import_known(k, v, 'baseline matrix')
    if (DIR/'mixed-decisions.json').exists():
        for b in json.loads((DIR/'mixed-decisions.json').read_text())['boards']:
            for row in b['pairs']:
                import_known(row['request_sha256'], row['score'], 'baseline mixed board')
    known.update(run['responses'])
    pending = {}
    rows = {}
    for split, name, pair, left, right, expected in cases():
        if args.fresh_only and split != 'fresh':
            continue
        a, b = retrieval.active_decision_order(left, right)
        row = {'split': split, 'board': name, 'pair': pair, 'reference': expected}
        for strategy in ('original', 'shared_operations'):
            body = retrieval.decision_payload(a, b, (), model)
            if strategy == 'shared_operations':
                body['questions']['relevant']['instructions'] += SHARED_WORK
            key = payload_key(body)
            row[strategy] = key
            if key in known:
                run['responses'][key] = known[key]
            else:
                pending[key] = body
        rows[(name, pair)] = row
    def save():
        scored = []
        for row in rows.values():
            if all(k in run['responses'] for k in (row['original'], row['shared_operations'])):
                scored.append(row)
        run['pairs'] = scored
        groups = ['fresh'] if args.fresh_only else ['development', 'film', 'fresh', 'friday']
        run['scores'] = {group: {strategy: metrics((r['reference'], run['responses'][r[strategy]]['score'] >= .5)
                                for r in scored if r['split'] == group) for strategy in ('original', 'shared_operations')}
                         for group in groups}
        write(output, run)
    def infer(item):
        key, body = item
        response = retrieval._local_post(cfg.get('PLANNER_DECISION_ENDPOINT') or 'http://127.0.0.1:1234/v1/systemone', body, cfg)
        score = response['answers']['relevant']['noul']
        if type(score) not in (float, int) or not 0 <= score <= 1:
            raise ValueError(f'Invalid response {key}.')
        return key, {'score': score, 'origin': 'live ablation'}
    save()
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        futures = {workers.submit(infer, item):item[0] for item in pending.items()}
        failures = []
        for i, f in enumerate(as_completed(futures), 1):
            try:
                key, reply = f.result()
            except retrieval.ModelUnavailable as exc:
                failures.append(futures[f])
                run.setdefault('transport_errors', []).append({'request_sha256': futures[f], 'message': str(exc)})
            else:
                run['responses'][key] = reply
            save()
            if i%10 == 0 or i == len(futures):
                print('Processed', i, 'of', len(futures), 'new requests', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} calls remain. Rerun with the same model to resume.')
    print(run['scores'], flush=True)


if __name__ == '__main__':
    main()
