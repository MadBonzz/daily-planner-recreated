"""Paired, separately scored Kev prompt ablation on shared substantive operations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from itertools import combinations
import json

import retrieval
from model_client import planner_settings
from examples.estimation.audit_matrix import payload_key, write
from examples.estimation.scale_matrix import DIR, fixture, typed
from examples.estimation.scale_matrix_reporting import metrics

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


def publish(run):
    if len(run['pairs']) != 66:
        return
    lines = ['# Shared-operation prompt experiment', '',
             'The baseline missed two cases where hemming a pair of trousers could inform the hem portion of a combined hem-and-zipper job. The candidate explicitly asks Kev to distinguish performing a substantial operation from referring to it, showing its results, or building a tool for someone else. Both prompts use identical task text, example records, orientation and .50 Kev cutoff. [Frozen prompt, scores and request hashes](composite-ablation.json)', '',
             '| Cohort | Prompt | TP | FP | FN | TN | Precision | Recall |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    names = {'development': 'Textile repair · development', 'film': 'Film editing · validation',
           'fresh': 'Six new mixed-work boards · validation', 'friday': 'Friday close · false-link guard'}
    for split in names:
        for name in ('original', 'shared_operations'):
            s = run['scores'][split][name]
            def percent(v):
                return '—' if v is None else f'{100*v:.1f}%'
            lines.append(f"| {names[split]} | {name.replace('_', ' ')} | {s['tp']} | {s['fp']} | {s['fn']} | "
                         f"{s['tn']} | {percent(s['precision'])} | {percent(s['recall'])} |")
    # Predict only whether a completion wave would request a survivor refresh under each
    # measured decision set. This is a scheduling counterfactual; Bonsai was not rerun.
    scheduled = []
    for board in fixture()['boards']:
        if board['id'] not in ('textile-mending', 'video-edit'):
            continue
        pairs = {row['pair']: row for row in run['pairs'] if row['board'] == board['id']}
        active = set('ABCDE')
        for wave in board['completion_waves']:
            active -= wave.keys()
            for target in sorted(active):
                keys = [''.join(sorted((target, source))) for source in wave]
                reference = any(key in board['positive_pairs'] for key in keys)
                record = {'board': board['id'], 'finished': wave, 'target': target, 'reference': reference}
                for strategy in ('original', 'shared_operations'):
                    record[strategy] = any(run['responses'][pairs[key][strategy]]['score'] >= .5 for key in keys)
                scheduled.append(record)
    run['projected_completion_waves'] = {'description': 'Two measured mixed-task boards under the frozen schedules. '
        'This projects the binary refresh trigger and excludes final forecasts.', 'targets': scheduled,
        'scores': {strategy:metrics((row['reference'], row[strategy]) for row in scheduled)
                  for strategy in ('original', 'shared_operations')}}
    write(DIR/'composite-ablation.json', run)
    lines.extend(['', '## Completion effects on the two mixed-task schedules', '',
                  'Applying the measured gates to the original textile and film completion waves gives the following '
                  'projected refresh targets. These rows count surviving tasks after each wave.', '',
                  '| Gate | Correct refresh | Unnecessary refresh | Missed refresh | Correct no change |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for strategy, scores in run['projected_completion_waves']['scores'].items():
        lines.append(f"| {strategy.replace('_', ' ')} | {scores['tp']} | {scores['fp']} | {scores['fn']} | {scores['tn']} |")
    changes = []
    for r in run['pairs']:
        baseline = run['responses'][r['original']]['score']
        candidate = run['responses'][r['shared_operations']]['score']
        if (baseline >= .5) != (candidate >= .5):
            changes.append((r, baseline, candidate))
    if changes:
        lines.extend(['', '## Changed decisions', '',
                      '| Board | Pair | Reference | Original Kev score | Candidate Kev score | Outcome |',
                      '| --- | --- | --- | ---: | ---: | --- |'])
        for r, a, b in changes:
            outcome = 'Corrected' if (b >= .5) == r['reference'] else 'Regressed'
            lines.append(f"| {r['board']} | {r['pair']} | {'Related' if r['reference'] else 'Unrelated'} | {a:.3f} | {b:.3f} | {outcome} |")
    lines.extend(['', 'The six new boards contain 18 pairs. Two assistant reviewers labeled all pairs independently before inference and agreed on six positives and twelve negatives. Their domains differ, but each board is built around one mixed deliverable, a shared operation and a related artifact. The Friday guard covers same-title and cross-domain confounders. The separate equipment-handover challenge was assessed only with the original prompt. [New holdouts](composite-holdout.json) · [Second review](composite-second-labels.json) · [Friday challenge](mixed-boards.json)'])
    (DIR/'composite-trial.md').write_text('\n'.join(lines)+'\n')


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
    publish(run)
    print(run['scores'], flush=True)


if __name__ == '__main__':
    main()
