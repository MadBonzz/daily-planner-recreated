"""Audit Qwen screening and Kev's timing-evidence decision on the labeled fixture pairs."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import retrieval
import sample_retrieval
from examples.estimation.package_retrieval import inputs
from examples.estimation.run import REPO, histories, write_json
from examples.estimation.scoring import cosine, pair_metrics


FROZEN = REPO / 'data/results/component-audit/retrieval/judge-inputs.json'

METHOD_QUESTION = (
    "Would the time to do the candidate's requested work help estimate the target's requested work? "
    "Answer yes when the main work uses the same process with comparable effort drivers even if the "
    "topic, input, specific bug, destination or output differs. Someone measuring and plotting a second "
    "lab trial can use the duration of another measured and plotted trial. "
    "Answer no when only the topic, place, broad category or handout matches and the actual operations "
    "differ (for example, building a fixture versus writing about results that already exist). "
    "Use the title, description and details to identify assigned work; PDFs can also describe work "
    "outside the assignment. The tasks do not have to be interchangeable."
)


def frozen_request(fixture, history, first, second, method):
    scopes = fixture['tasks'][history]
    state = {'examples': fixture['examples'], 'reviewed_examples': [],
             'target': scopes[first], 'candidate': scopes[second]}
    request = {'model': fixture['model'], 'state': json.dumps(state, ensure_ascii=False),
               'questions': {'relevant': fixture['method_question' if method else 'baseline_question']}}
    if method:
        digest = fixture['receipts'][f'{history}/{first}/{second}']['method_request_sha256']
    else:
        digest = fixture['packaged_baseline'][history][f'{first}/{second}']['sha256']
    if sample_retrieval.decision_hash(request) != digest:
        raise ValueError(f'Frozen judge input mismatch for {history}/{first}/{second}')
    return request


def pair_rows():
    fixture = json.loads(FROZEN.read_text())
    rows = []
    tasks = {}
    for h in histories(None):
        h_id = h['id']
        tasks[h_id] = inputs(h)
        path = sample_retrieval.DIRECTORY / (h_id + '.json')
        vec = {i: sample_retrieval.read_vectors(path, i) for i in tasks[h_id]}
        for q in h['holdouts']:
            if q.get('fixed_duration'):
                continue
            for c in h['tasks']:
                if c.get('fixed_duration'):
                    continue
                a, b = q['id'], c['id']
                saved = fixture['packaged_baseline'][h_id].get(f'{a}/{b}')
                if saved is not None:
                    frozen_request(fixture, h_id, a, b, False)
                rows.append({'history': h_id, 'target': a, 'candidate': b,
                             'split': 'development' if int(h_id[:2]) % 2 else 'validation',
                             'expected': b in q['evaluation']['evidence_ids'],
                             'instructed': cosine(vec[a]['query'], vec[b]['document']),
                             'bare': cosine(vec[a]['document'], vec[b]['document']),
                             'reverse': cosine(vec[a]['document'], vec[b]['query']),
                             'baseline_kev': saved['related'] if saved is not None else None})
    return rows, tasks


def dual_score(row):
    return max(row['instructed'], row['bare'])


def verify_ranking(rows, tasks):
    """Replay the actual search, matching cutoff, IDs, deduplication and scores on every target."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row['history'], row['target']), []).append(row)
    with TemporaryDirectory() as temp:
        for (profile, target), group in grouped.items():
            sources = [dict(tasks[profile][r['candidate']], id=r['candidate']) for r in group]
            with_threshold = {r['candidate']: dual_score(r) for r in group if dual_score(r) >= .50}
            hits = retrieval.search(tasks[profile][target], sources, Path(temp) / profile)
            found = {row['id']: score for row, score in hits}
            if (len(hits) != len(found) or found.keys() != with_threshold.keys() or
                    any(abs(score - with_threshold[key]) > 1e-5 for key, score in found.items())):
                raise RuntimeError(f'Qdrant disagrees with offline scores for {profile}/{target}')


def orientations(fixture):
    pairs = []
    for history, records in sorted(fixture['packaged_baseline'].items()):
        for key, record in sorted(records.items()):
            a, b = key.split('/')
            reverse = records.get(f'{b}/{a}')
            if a < b and reverse is not None:
                frozen_request(fixture, history, a, b, False)
                frozen_request(fixture, history, b, a, False)
                pairs.append({'history': history, 'a': a, 'b': b, 'a_to_b': record['related'],
                              'b_to_a': reverse['related']})
    return {'matched_orders': len(pairs), 'disagreement': sum(r['a_to_b'] != r['b_to_a'] for r in pairs),
            'pairs': pairs}


def rank_metrics(rows, score):
    positive_ranks = []
    by_target = {}
    for row in rows:
        by_target.setdefault((row['history'], row['target']), []).append(row)
    for candidates in by_target.values():
        if not any(r['expected'] for r in candidates):
            continue
        ranking = sorted((r for r in candidates if score(r) >= .50), key=score, reverse=True)
        rank = next((i for i, row in enumerate(ranking, 1) if row['expected']), None)
        positive_ranks.append(rank)
    return {'targets_with_authored_match': len(positive_ranks), 'top_result_relevant': positive_ranks.count(1),
            'targets_with_a_retrieved_match': sum(r is not None for r in positive_ranks),
            'mean_reciprocal_rank': sum(1/r for r in positive_ranks if r) / len(positive_ranks)}


def score(directory, rows, tasks):
    verify_ranking(rows, tasks)
    fixture = json.loads(FROZEN.read_text())
    for row in rows:
        if dual_score(row) < .50:
            continue
        group = tasks[row['history']]
        payload = retrieval.decision_payload(group[row['target']], group[row['candidate']], (), retrieval.DECISION_MODEL)
        value = sample_retrieval.decision_for(payload)
        if value is None:
            raise ValueError(f"Missing production Kev result for {row['target']}/{row['candidate']}")
        row['production_kev_sha256'] = sample_retrieval.decision_hash(payload)
        row['production_kev'] = value
    outputs = {}
    for part in ('development', 'validation'):
        path = directory / (part + '.json')
        if path.exists():
            outputs.update(json.loads(path.read_text()))
    report = {}
    for split in ('development', 'validation', 'all'):
        group = [r for r in rows if split == 'all' or r['split'] == split]
        candidates = [r for r in group if dual_score(r) >= .50]
        if any(r['baseline_kev'] is None for r in candidates):
            raise ValueError('No baseline response for a screened candidate')
        methods = {'instructed_qwen': lambda r: r['instructed'] >= .50,
                   'bare_qwen': lambda r: r['bare'] >= .50,
                   'bidirectional_qwen': lambda r: max(r['reverse'], r['instructed']) >= .50,
                   'dual_query_qwen': lambda r: dual_score(r) >= .50,
                   'instructed_then_kev': lambda r: r['instructed'] >= .50 and r['baseline_kev'],
                   'dual_query_then_kev': lambda r: dual_score(r) >= .50 and r['baseline_kev'],
                   'dual_query_then_scoped_kev': lambda r: dual_score(r) >= .50 and r['production_kev']}
        report[split] = {name: pair_metrics(group, pred) for name, pred in methods.items()}
        report[split]['ranking'] = {'instructed': rank_metrics(group, lambda r: r['instructed']),
                                   'dual_query': rank_metrics(group, dual_score)}
        report[split]['kev_on_dual_candidates'] = pair_metrics(candidates, lambda r: r['baseline_kev'])
        report[split]['scoped_kev_on_dual_candidates'] = pair_metrics(candidates, lambda r: r['production_kev'])
        received = []
        for row in candidates:
            request = frozen_request(fixture, row['history'], row['target'], row['candidate'], True)
            hit = outputs.get(sample_retrieval.decision_hash(request))
            if hit is not None:
                if any(row[k] != hit[k] for k in ('history', 'candidate', 'target', 'expected')):
                    raise ValueError('Probe input or reference changed')
                received.append({**row, 'p': hit['probability']})
        report[split]['method_probe'] = {
            'attempts_with_response': len(received), 'eligible': len(candidates),
            'production_same_pairs': pair_metrics(received, lambda r: r['baseline_kev']) if received else None,
            'method_same_pairs': pair_metrics(received, lambda r: r['p'] >= .50) if received else None,
            'decision_threshold_sensitivity': {
                str(cutoff): pair_metrics(received, lambda r: r['p'] >= cutoff) for cutoff in (.25, .35, .5, .65)
            } if received else None,
        }
    report['development_selection'] = choose_challenger(report['development'])
    write_json(directory / 'summary.json', report)
    write_json(directory / 'pairs.json', rows)
    write_json(directory / 'order-sensitivity.json', orientations(fixture))


def choose_challenger(development):
    """Require more correct matches without extra false positives on development."""
    probe = development['method_probe']
    if probe['attempts_with_response'] != probe['eligible']:
        return {'decision': 'awaiting development responses'}
    baseline = probe['production_same_pairs']
    eligible = [(float(cutoff), row) for cutoff, row in probe['decision_threshold_sensitivity'].items()
                if row['tp'] > baseline['tp'] and row['fp'] <= baseline['fp']]
    if not eligible:
        return {'decision': 'retain production question'}
    cutoff, best = min(eligible, key=lambda item: (item[1]['fn'], item[1]['fp'], abs(item[0]-.5)))
    return {'decision': 'candidate for validation and portability checks', 'cutoff': cutoff,
            'development': best}


def verify_live(directory, tasks):
    first = tasks['02-migration-lab']
    profile_path = sample_retrieval.DIRECTORY / '02-migration-lab.json'
    config = retrieval.planner_settings()
    results = []
    for task_id in ('patch-tool', 'suite-report'):
        source = first[task_id]
        stored = sample_retrieval.read_vectors(profile_path, task_id)
        for role in ('document', 'query'):
            content = retrieval.embedding_input(source, role == 'query')
            live = retrieval._live_vector(content, retrieval.EMBEDDING_MODEL, config)
            match = cosine(live, stored[role])
            if match < .9999:
                raise RuntimeError(f'Live vector differs for {task_id}/{role}: {match}')
            results.append({'task': task_id, 'role': role, 'input_sha256': sample_retrieval.text_hash(content),
                            'dimensions': len(live), 'cosine_against_published_vector': match})
    write_json(directory / 'live-qwen.json', results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--split', choices=['development', 'validation'])
    action.add_argument('--score', action='store_true')
    action.add_argument('--verify-live-qwen', action='store_true')
    args = p.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    rows, tasks = pair_rows()
    if args.score:
        return score(args.directory, rows, tasks)
    if args.verify_live_qwen:
        return verify_live(args.directory, tasks)
    config = retrieval.planner_settings()
    endpoint = config.get('PLANNER_DECISION_ENDPOINT') or 'http://127.0.0.1:1234/v1/systemone'
    path = args.directory / (args.split + '.json')
    recorded = json.loads(path.read_text()) if path.exists() else {}
    fixture = json.loads(FROZEN.read_text())
    for row in rows:
        if row['split'] != args.split or max(row['instructed'], row['bare']) < .50:
            continue
        data = frozen_request(fixture, row['history'], row['target'], row['candidate'], True)
        key = sample_retrieval.decision_hash(data)
        if key not in recorded:
            response = retrieval._local_post(endpoint, data, config)
            prob = response['answers']['relevant']['noul']
            assert 0 <= prob <= 1
            recorded[key] = {**row, 'probability': prob}
            write_json(path, recorded)
        print(args.split, row['target'], row['candidate'], recorded[key]['probability'], flush=True)


if __name__ == '__main__':
    main()
