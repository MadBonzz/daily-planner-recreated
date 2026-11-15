"""Compare bare and instructed Qwen queries on the same forecast requests and histories."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
from tempfile import TemporaryDirectory
from unittest.mock import patch

import estimator
import retrieval
import sample_retrieval
from examples.estimation import run
from examples.estimation.calibrate_retrieval import cosine, metrics as pair_metrics, write_json
from model_client import ModelUnavailable, model_settings, planner_settings


ARMS = ('bare', 'instructed')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def fixture_signature():
    files = sorted(p for name in ('histories', 'activitywatch', 'pdfs') for p in (run.DATASET / name).rglob('*')
                   if p.is_file() and p.suffix in {'.json', '.db', '.pdf'})
    return fingerprint({path.relative_to(run.DATASET).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in files})


def verified_pairs(profiles):
    """Inspect exact bundled inputs and require a Kev response for each cross-arm candidate."""
    pairs = []
    for history in profiles:
        index_path = sample_retrieval.DIRECTORY / (history['id'] + '.json')
        bundle = json.loads(index_path.read_text())
        rows = {r['id']: r for r in history['tasks'] + history['holdouts'] if not r.get('fixed_duration')}
        tasks = {key: run.task_inputs(row, history['id'], run.DATASET)[0] for key, row in rows.items()}
        vectors = {}
        for key, task in tasks.items():
            vectors[key] = {}
            for role in ('query', 'document'):
                record = bundle['vectors'][key][role]
                text = retrieval.embedding_input(task, role == 'query')
                if sample_retrieval.text_hash(text) != record['input_sha256']:
                    raise ValueError(f"Stale {role} for {key}")
                vectors[key][role] = sample_retrieval.read_vectors(index_path, key)[role]
        for target in history['holdouts']:
            if target.get('fixed_duration'):
                continue
            target_id = target['id']
            for past in history['tasks']:
                if past.get('fixed_duration'):
                    continue
                prior_id = past['id']
                scores = {arm: cosine(vectors[target_id][role], vectors[prior_id]['document'])
                          for arm, role in zip(ARMS, ('document', 'query'), strict=True)}
                decision = None
                if max(scores.values()) >= 0.50:
                    key = sample_retrieval.decision_hash(retrieval.decision_payload(
                        tasks[target_id], tasks[prior_id], (), retrieval.DECISION_MODEL))
                    decision = bundle['decisions'][key]['related']
                pairs.append({'history': history['id'], 'task': target_id, 'candidate': prior_id,
                              'expected': prior_id in target['evaluation']['evidence_ids'],
                              'cosine': scores, 'kev': decision})
    return pairs


class PairedRequests:
    """Use one actual provider result for each distinct, complete forecasting request."""

    def __init__(self, directory):
        self.path = directory / 'requests.json'
        self.receipts = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.case_calls = []
        self.endpoint, self.model = model_settings()
        self.send = estimator.chat_json

    def __call__(self, messages, schema, **options):
        request_id = fingerprint({'endpoint': self.endpoint, 'model': self.model, 'temperature': 0,
                                  'messages': messages, 'schema': schema, 'options': options})
        reuse = request_id in self.receipts
        if not reuse:
            receipt = {'model': self.model, 'attempts': []}
            for attempt in range(2):
                start = time.monotonic()
                try:
                    receipt['response'] = self.send(messages, schema, **options)
                except ModelUnavailable as error:
                    message = str(error)
                    receipt['attempts'].append({'seconds': round(time.monotonic()-start, 3), 'error': message})
                    if attempt == 0 and ('HTTP 429' in message or 'HTTP 5' in message):
                        time.sleep(60 if 'HTTP 429' in message else 4)
                        continue
                    receipt['error'] = message
                else:
                    receipt['attempts'].append({'seconds': round(time.monotonic()-start, 3)})
                break
            self.receipts[request_id] = receipt
            write_json(self.path, self.receipts)
        self.case_calls.append({'request_id': request_id, 'source': 'identical request' if reuse else 'live endpoint'})
        receipt = self.receipts[request_id]
        if 'error' in receipt:
            raise ModelUnavailable(receipt['error'])
        return receipt['response']


def score_case(history, task, arm, directory, requests):
    ranked = []
    production_search = estimator._search

    def record_search(*args, **kwargs):
        result = production_search(*args, **kwargs)
        ranked.extend({'candidate': row['id'], 'cosine': score} for row, score in result)
        return result

    requests.case_calls.clear()
    query = patch.object(retrieval, 'embed_query', retrieval.embed_task) if arm == 'bare' else nullcontext()
    with query, patch.object(estimator, '_search', record_search), patch.object(estimator, 'chat_json', requests):
        row = run.evaluate(history, run.DATASET, directory / history['id'], case=task['id'])[0]
    if row['retrieval_error']:
        raise RuntimeError(row['retrieval_error'])
    accepted = {r['task_id'] for r in row['related_tasks']}
    row['qwen_candidates'] = [dict(r, kev_accepted=r['candidate'] in accepted) for r in ranked]
    row['forecast_calls'] = requests.case_calls.copy()
    return row


def compare(rows, pairs):
    timing = {}
    for name, group in [('all', lambda r: True), ('flexible', lambda r: not r['fixed_duration']),
                        ('validation', lambda r: int(r['history'][:2]) % 2 == 0)]:
        by_arm = {arm: [r for r in rows[arm] if group(r)] for arm in ARMS}
        diffs = [abs(b['planner']-b['actual']) - abs(a['planner']-a['actual'])
                 for a, b in zip(by_arm['bare'], by_arm['instructed'], strict=True)]
        timing[name] = {arm: run.metrics(by_arm[arm]) for arm in ARMS}
        timing[name]['changes'] = {'closer': sum(d < 0 for d in diffs), 'tied': diffs.count(0),
                                   'further': sum(d > 0 for d in diffs)}
    predicted = {arm: {(r['history'], r['task'], c['candidate']): c for r in rows[arm]
                        for c in r['qwen_candidates']} for arm in ARMS}
    for p in pairs:
        for arm in ARMS:
            candidate = predicted[arm].get((p['history'], p['task'], p['candidate']))
            if (candidate is not None) != (p['cosine'][arm] >= 0.50):
                raise ValueError(f"Qdrant cutoff disagrees with recorded Qwen vectors: {p}")
            if candidate and (candidate['kev_accepted'] != p['kev'] or abs(candidate['cosine']-p['cosine'][arm]) > 1e-5):
                raise ValueError(f"Different score or Kev judgment for {p}")
    retrieval_scores = {}
    for name, group in [('all', pairs), ('validation', [r for r in pairs if int(r['history'][:2]) % 2 == 0])]:
        retrieval_scores[name] = {arm: {
            'qwen': pair_metrics(group, lambda r: r['cosine'][arm] >= 0.50),
            'qwen_and_kev': pair_metrics(group, lambda r: r['cosine'][arm] >= 0.50 and r['kev'])}
            for arm in ARMS}
    return {'timing': timing, 'retrieval': retrieval_scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Fresh per-run output directory')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    settings = planner_settings()
    if retrieval.relevance_threshold(settings) != 0.50:
        parser.error('Set PLANNER_RELEVANCE_THRESHOLD=0.50 for this ablation.')
    requests = PairedRequests(args.directory)
    profiles = run.histories(None)
    pairs = verified_pairs(profiles)
    source_files = ['sample_retrieval.py', 'activity.py', 'examples/estimation/activity_fixture.py', 'estimator.py', 'model_client.py', 'retrieval.py', 'examples/estimation/run.py',
                    'examples/estimation/compare_instruction.py']
    source_hashes = {f: hashlib.sha256((run.REPO / f).read_bytes()).hexdigest() for f in source_files}
    manifest = {'model': requests.model, 'endpoint': requests.endpoint, 'temperature': 0,
                'cosine_cutoff': 0.50, 'query_instruction': retrieval.QUERY_INSTRUCTION,
                'retrieval_sha256': fingerprint(pairs), 'fixture_sha256': fixture_signature(), 'implementation_sha256': source_hashes}
    manifest_path = args.directory / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error('Inputs or implementations have changed. Choose a new output directory.')
    write_json(manifest_path, manifest)
    rows = {arm: [] for arm in ARMS}
    with TemporaryDirectory(prefix='planner-query-ablation-') as tmp, patch.object(
            retrieval, '_local_post', side_effect=AssertionError('Incomplete sample retrieval bundle')):
        for index, history in enumerate(profiles):
            for task in history['holdouts']:
                # Alternate the first arm for each history. An identical provider request always shares its response.
                for arm in (ARMS if index % 2 == 0 else reversed(ARMS)):
                    folder = args.directory / arm
                    folder.mkdir(exist_ok=True)
                    result_path = folder / (task['id'] + '.json')
                    if result_path.exists():
                        row = json.loads(result_path.read_text())
                    else:
                        row = score_case(history, task, arm, Path(tmp) / arm, requests)
                        write_json(result_path, row)
                    rows[arm].append(row)
                    print(arm, history['id'], task['id'], row['model_status'], row['planner'], flush=True)
    summary = compare(rows, pairs)
    summary['distinct_provider_requests'] = len(requests.receipts)
    summary['provider_attempts'] = sum(len(r['attempts']) for r in requests.receipts.values())
    summary['same_first_request'] = sum(bool(a['forecast_calls'] and b['forecast_calls']) and
                                        a['forecast_calls'][0]['request_id'] == b['forecast_calls'][0]['request_id']
                                        for a, b in zip(*rows.values(), strict=True))
    write_json(args.directory / 'summary.json', summary)
    write_json(args.directory / 'pairs.json', {'cosine_cutoff': 0.50, 'pairs': pairs})


if __name__ == '__main__':
    main()
