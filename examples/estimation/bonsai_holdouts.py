"""Run a local Bonsai against the twenty instructed-.50 production forecast cases.

Start llama-server with the desired GGUF, then pass its model alias, the weights,
and the run directory. The first request for each flexible task must match the
published Cohere request in every aspect except provider and model.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

import estimator
import model_client
import retrieval
from examples.estimation import compare_instruction as paired
from examples.estimation import run

BASELINE = run.REPO / 'data/results/holdouts/2026-11-15-qwen-query-instruction'
WEIGHTS = {
    'ternary-bonsai-4b': '9d968b04a3c9a794897bcc744c8072fb6a061c0e42efd03c989401ddf8baef0c',
    'ternary-bonsai-2-9b': '1f884aa56aacd2bbbb2e4b7abaead2b1102b74a9c75a23fc33dab916813a384d',
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


class ObservedClient:
    def __init__(self, client, receipt, target):
        self.client, self.receipt, self.target = client, receipt, target

    def open(self, request, **kwargs):
        if request.full_url != self.target:
            raise RuntimeError('Unexpected inference endpoint')
        sent = json.loads(request.data)
        self.receipt['payload_sha256'] = paired.fingerprint(sent)
        with self.client.open(request, **kwargs) as result:
            self.receipt['http_status'] = result.status
            raw = result.read()
        wire = json.loads(raw)
        choice = wire['choices'][0]
        self.receipt.update(response_model=wire['model'], finish_reason=choice['finish_reason'],
                            usage=wire.get('usage'), timings=wire.get('timings'),
                            content_sha256=hashlib.sha256(choice['message']['content'].encode()).hexdigest())
        # Preserve unparseable/generated JSON for diagnosis. No hidden reasoning channel is exported.
        self.receipt['content'] = re.sub(r'<think>.*?(?:</think>|$)', '', choice['message']['content'], flags=re.S)
        return io.BytesIO(raw)


class ForecastCalls:
    def __init__(self, reference, checkpoint, preflight):
        self.endpoint, self.model = model_client.model_settings()
        self.case_calls = []
        self.reference = reference
        self.checkpoint = checkpoint
        self.records = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
        self.preflight = preflight

    def __call__(self, messages, schema, **options):
        inputs = {'temperature': 0, 'messages': messages, 'schema': schema, 'options': options}
        baseline = json.loads((BASELINE / 'manifest.json').read_text())
        reference_id = paired.fingerprint({**inputs, 'endpoint': baseline['endpoint'], 'model': baseline['model']})
        if not self.case_calls and reference_id != self.reference['forecast_calls'][0]['request_id']:
            raise RuntimeError('The primary prompt, schema or generation settings differ from the paired reference.')
        rid = paired.fingerprint({**inputs, 'endpoint': self.endpoint, 'model': self.model})
        self.case_calls.append({'request_id': rid, 'same_input_cohere_id': reference_id})
        if self.preflight:
            raise model_client.ModelUnavailable('Benchmark preparation')
        if rid not in self.records:
            receipt = {'request_id': rid, 'started_utc': utc(), 'max_tokens': options['max_tokens']}
            started = time.monotonic()
            with patch.object(model_client, 'build_opener', lambda *h: ObservedClient(build_opener(*h), receipt, self.endpoint)):
                try:
                    reply = model_client.chat_json(messages, schema, **options)
                except model_client.ModelUnavailable as exc:
                    if not receipt.get('http_status'):
                        raise RuntimeError(f'No model completion was received: {exc}') from exc
                    receipt['error'] = str(exc)
                    receipt['truncated'] = isinstance(exc, model_client.ModelResponseTruncated)
                else:
                    receipt.pop('content', None)
                    receipt['response'] = reply
            receipt.update(ended_utc=utc(), seconds=round(time.monotonic()-started, 3))
            if receipt['response_model'] != self.model:
                raise RuntimeError('The server returned a different model alias.')
            self.records[rid] = receipt
            save(self.checkpoint, self.records)
        receipt = self.records[rid]
        if 'error' in receipt:
            error = model_client.ModelResponseTruncated if receipt['truncated'] else model_client.ModelUnavailable
            raise error(receipt['error'])
        return receipt['response']


def evaluate_case(history, task, directory, preflight=False):
    name = task['id']
    baseline = json.loads((BASELINE / 'instructed' / (name + '.json')).read_text())
    checkpoint = directory / ('.' + name + '.resume.json')
    calls = ForecastCalls(baseline, checkpoint, preflight)
    started = utc()
    with TemporaryDirectory() as tmp, patch.object(retrieval, '_local_post', side_effect=AssertionError('Incomplete retrieval package')):
        row = paired.score_case(history, task, 'instructed', Path(tmp), calls)
    def comparable(value):
        if isinstance(value, list):
            return [comparable(item) for item in value]
        if isinstance(value, dict):
            return {k: comparable(v) for k, v in value.items()}
        return round(value, 5) if isinstance(value, float) else value

    if any(comparable(row[field]) != comparable(baseline[field]) for field in ('related_tasks', 'qwen_candidates')):
        raise RuntimeError(f'Retrieved evidence changed for {name}.')
    if not preflight:
        row['http_receipts'] = [calls.records[call['request_id']] for call in calls.case_calls]
        for receipt in row['http_receipts']:
            if 'response' in receipt:
                receipt.pop('response')  # run.evaluate already records the validated response in model_calls.
        row['started_utc'], row['ended_utc'] = started, utc()
        row['wall_model_seconds'] = round(sum(r['seconds'] for r in row['http_receipts']), 3)
        save(directory / 'cases' / (name + '.json'), row)
        checkpoint.unlink(missing_ok=True)
    return name, row['model_status'], row['planner']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--model', choices=WEIGHTS, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:1234/v1/chat/completions')
    parser.add_argument('--slots', type=int, default=4)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if urlsplit(args.endpoint).hostname not in {'127.0.0.1', 'localhost', '::1'}:
        parser.error('This benchmark expects a loopback model server.')
    os.environ.update(PLANNER_MODEL=args.model, PLANNER_MODEL_ENDPOINT=args.endpoint,
                      PLANNER_RELEVANCE_THRESHOLD='0.50', PLANNER_MODEL_TIMEOUT_SECONDS='900')
    with args.weights.open('rb') as file:
        digest = hashlib.file_digest(file, 'sha256').hexdigest()
    if digest != WEIGHTS[args.model]:
        parser.error('GGUF checksum does not match the public Bonsai weights.')
    profiles = run.histories(None)
    cohort = [(h, t) for h in profiles for t in h['holdouts']]
    baseline = json.loads((BASELINE / 'manifest.json').read_text())
    fixture_hash = paired.fixture_signature()
    if fixture_hash != baseline['fixture_sha256'] or len(cohort) != 20:
        parser.error('Published evaluation fixtures changed.')
    directory = args.directory
    if args.preflight:
        for h, t in cohort:
            evaluate_case(h, t, directory, True)
        print('Verified 20 reference cases, first forecast prompts and exact Qwen/Kev selections.', flush=True)
        return
    (directory / 'cases').mkdir(parents=True, exist_ok=True)
    props_url = args.endpoint.removesuffix('/v1/chat/completions') + '/props'
    with build_opener(ProxyHandler({})).open(props_url) as response:
        props = json.load(response)
    manifest = {'model': args.model, 'file': args.weights.name, 'file_sha256': digest, 'bytes': args.weights.stat().st_size,
                'reference': BASELINE.name, 'cases': len(cohort), 'fixture_sha256': fixture_hash,
                'cosine_cutoff': 0.5, 'query_instruction': retrieval.QUERY_INSTRUCTION, 'temperature': 0,
                'prompt_parity': 'SHA256 of messages, schema, temperature and token budget against the instructed Cohere primary request',
                'client_processes': args.slots, 'server_slots': props['total_slots'],
                'context_per_slot': props['default_generation_settings']['n_ctx'],
                'endpoint': args.endpoint, 'http_timeout_seconds': 900,
                'implementation_sha256': {name: hashlib.sha256((run.REPO / name).read_bytes()).hexdigest() for name in (
                    'estimator.py','model_client.py','retrieval.py','examples/estimation/run.py','examples/estimation/bonsai_holdouts.py')}}
    marker = directory / 'manifest.json'
    if marker.exists() and json.loads(marker.read_text()) != manifest:
        parser.error('Run inputs or settings changed since the checkpoint.')
    save(marker, manifest)
    work = [(h, t) for h, t in cohort if not (directory / 'cases' / (t['id']+'.json')).exists()]
    with ProcessPoolExecutor(args.slots, mp_context=multiprocessing.get_context('fork')) as pool:
        futures = [pool.submit(evaluate_case, h, t, directory) for h, t in work]
        for future in as_completed(futures):
            print(*future.result(), flush=True)


if __name__ == '__main__':
    main()
