"""Score stored model receipts and SQLite transitions against frozen reference labels."""
from collections import Counter
import hashlib
import json

import numpy as np

import retrieval
from examples.estimation.audit_matrix import write


def metrics(rows):
    counts = Counter({k: 0 for k in ('tp', 'fp', 'fn', 'tn')})
    for ref, actual in rows:
        counts[('tp' if ref else 'fp') if actual else ('fn' if ref else 'tn')] += 1
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    return {**counts, 'precision': round(tp/(tp+fp), 4) if tp+fp else None,
            'recall': round(tp/(tp+fn), 4) if tp+fn else None,
            'f1': round(2*tp/(2*tp+fp+fn), 4) if tp+fp+fn else None}


def summarize_boards(boards, counted, live=False):
    from examples.estimation.scale_matrix import PRIOR, pair
    counts = Counter()
    vs_gold, vs_matrix, queue_gold, matrix_actual = [], [], [], []
    directions, source_errors, state_errors, routing_errors = [], [], [], []
    citations = Counter()
    new_evidence, all_evidence, evidence_disagreements = [], [], []
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
                if positive != ran:
                    routing_errors.append({'board': b['id'], 'wave': i, 'finished': w['finished'],
                                           'surviving_task': k, 'expected_refresh': positive})
                vs_gold.append((positive, ran))
                vs_matrix.append((routed, ran))
                queue_gold.append((positive, queued))
                matrix_actual.append((routed, queued))
                counts['target_transitions'] += 1
                bef, aft = w['before'][k], w['after'][k]
                for source in w['finished']:
                    expected_source = pair(k, source) in b['positive_pairs']
                    actually_selected = ran and source in aft['selected']
                    new_evidence.append((expected_source, actually_selected))
                    if expected_source != actually_selected:
                        evidence_disagreements.append({'board': b['id'], 'wave': i, 'target': k,
                            'source': source, 'expected': expected_source, 'selected': actually_selected,
                            'target_refreshed': ran})
                if ran:
                    for source in times.keys() - {'prior'}:
                        all_evidence.append((pair(k, source) in b['positive_pairs'], source in aft['selected']))
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
                        citations['all_source_citations'] += len(actual_cited)
                        citations['unrelated_source_citations'] += sum(
                            source == 'prior' or pair(k, source) not in b['positive_pairs'] for source in actual_cited)
            for k, direction in w['reference_direction'].items():
                before, after = w['before'][k], w['after'][k]
                if not live:
                    continue
                reference_sources = {src for src in w['finished'] if pair(k, src) in b['positive_pairs']}
                available = bool(reference_sources & set(after['selected']))
                ran = k in w['observed']
                direct = ran and after['basis'] == 'direct'
                change = after['point']-before['point']
                budget = budgets[k]
                scaled = [budget * w['finished'][src] / budgets[src] for src in reference_sources]
                directions.append({'board': b['id'], 'wave': i, 'target': k, 'expected': 'up' if direction == 1 else 'down',
                    'before': before['point'], 'before_source': before['basis'], 'after': after['point'],
                    'related_evidence_available': available, 'direct_refresh': direct,
                    'user_budget': budget, 'new_analogue_range': [round(min(scaled), 1), round(max(scaled), 1)],
                    'beyond_all_new_analogues': direct and (
                        (direction == 1 and after['point'] > max(scaled)) or (direction == -1 and after['point'] < min(scaled))),
                    'moved_expected_way': direct and direction*change > 0,
                    'on_expected_side_of_budget': direct and direction*(after['point']-budget) > 0})
        counts['boards_with_reference_no_change_control'] += int(ctrl)
    return {'counts': dict(counts), 'persisted_vs_reference': metrics(vs_gold),
            'queued_vs_reference': metrics(queue_gold), 'persisted_vs_kev': metrics(vs_matrix),
            'queued_vs_kev': metrics(matrix_actual), 'state_errors': state_errors, 'source_errors': source_errors,
            'citations': dict(citations), 'directions': directions, 'routing_errors': routing_errors,
            'new_source_selection_vs_reference': metrics(new_evidence),
            'all_source_slots_on_refreshed_tasks_vs_reference': metrics(all_evidence),
            'new_source_selection_disagreements': evidence_disagreements}


def report():
    from examples.estimation.scale_matrix import DIR, LABEL_DIGEST, check_inputs, fixture, new_run, typed, vector
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
        out['stress'] = {k:stress_result[k] for k in ('seed_count', 'totals', 'failed_seeds', 'crowded_burst') if k in stress_result}
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
                'on_expected_side_of_budget': sum(r['on_expected_side_of_budget'] for r in conditional),
                'beyond_all_new_analogues': sum(r['beyond_all_new_analogues'] for r in conditional)}
        result['direction_summary'] = direction_summary
        out[tag] = result
    write(DIR/'metrics.json', out)
    render_results(out)
    print({k:out[k] for k in ('independent_review', 'matrix', 'matrix_disagreements', 'prior_screen_passes')})
    if 'queue' in out: print('Queue', out['queue']['counts'], out['queue']['persisted_vs_reference'])
    for tag in ('4b', '9b'):
        if tag in out: print(tag, out[tag]['counts'], out[tag]['citations'])


def render_results(out):
    from examples.estimation.scale_matrix import DIR, fixture
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
                     perf(f'Saved refresh vs stored Kev decision, {n} survivors', queue['persisted_vs_kev']),
                     perf('Newly finished source reached the forecast vs reference', queue['new_source_selection_vs_reference']), '',
                     f"The worker saved {c['jobs']} refreshes over {c['waves']} waves on {c['boards']} boards. "
                     f"{c['multi_completion_waves']} waves included multiple completions before the worker drained. "
                     f"{c['reference_no_change_controls']} unrelated survivor opportunities kept their prior forecast. "
                     f"There were {len(queue['state_errors'])} state or queue-order errors over "
                     f"{c['addition_snapshots_checked']} task additions and {n} survivor transitions. "
                     "[Queue traces](queue.json)"])
    titles = {b['id']: {t['key']: t['title'] for t in b['tasks']} for b in fixture()['boards']}
    if 'queue' in out and out['queue'].get('routing_errors'):
        text.extend(['', '### Completion waves that differed from the reference', '',
                     '| Completed task and actual | Open target | Observed effect |',
                     '| --- | --- | --- |'])
        for err in out['queue']['routing_errors']:
            names = titles[err['board']]
            source = ', '.join(f"{names[k]} ({n} min)" for k, n in err['finished'].items())
            effect = 'Missed refresh' if err['expected_refresh'] else 'Unnecessary refresh'
            text.append(f"| {source} | {names[err['surviving_task']]} | {effect} |")
    if 'queue' in out:
        hidden = [e for e in out['queue']['new_source_selection_disagreements'] if e['target_refreshed']
                  and not any(r['board']==e['board'] and r['wave']==e['wave'] and r['surviving_task']==e['target']
                          for r in out['queue']['routing_errors'])]
        if hidden:
            text.extend(['','Two coalesced waves illustrate why a correctly queued target also needs a source-level check.',''])
            for err in hidden:
                names=titles[err['board']]
                verb='omitted' if err['expected'] else 'included an unrelated source,'
                text.append(f"- **{names[err['target']]}** refreshed as expected. Its forecast input {verb} "
                            f"“{names[err['source']]}.”")
    if out['matrix_disagreements']:
        text.extend(['', '### Kev disagreements', '',
                     '| Board | Task pair | Reference | Kev score |', '| --- | --- | --- | ---: |'])
        for r in out['matrix_disagreements']:
            names = titles[r['board']]
            first, second = r['pair']
            text.append(f"| {r['board']} | {names[first]} / {names[second]} | "
                        f"{'Relevant' if r['reference'] else 'Unrelated'} | {r['score']:.3f} |")
    text.extend(['', f"Actual Qwen embeddings were recorded for 80 task documents, their 80 instructed queries, "
                 f"and the shared permit-renewal document. {out['prior_screen_passes']} of the 80 tasks passed the "
                 f"dual-role .50 threshold for that unrelated historical source. Kev accepted {out['prior_kev_acceptances']}. "
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
        if 'crowded_burst' in g:
            crowded = g['crowded_burst']
            text.extend(['', f"A separate single-board burst created {crowded['simultaneous_initial_estimates']} pending forecasts "
                f"and {crowded['distinct_pair_jobs']} distinct matrix judgments. Six simultaneous completions produced "
                f"{crowded['saved_refreshes']} coalesced refreshes and left {crowded['untouched_survivors']} "
                "other open forecasts untouched."])
    (DIR/'findings.md').write_text('\n'.join(text)+'\n')


def verify_forecasts(tag, artifacts=None, save=True):
    """Replay raw Qwen/Kev/Bonsai responses by exact request hash, in a fresh app store."""
    import os
    from unittest.mock import patch
    from model_client import ModelContextExceeded, ModelResponseInvalid, ModelResponseTruncated, ModelUnavailable
    from examples.estimation.scale_matrix import DIR, LABEL_DIGEST, do_board, fixture, new_run, payload_key

    artifacts = artifacts or DIR
    manifest = new_run(artifacts / f'forecast-{tag}.json')
    expected_model = {'4b':'ternary-bonsai-4b', '9b':'ternary-bonsai-2-9b'}[tag]
    if manifest.get('model') != expected_model:
        raise ValueError('The raw runs identify a different forecast model.')
    boards = fixture()['boards']
    if set(manifest['boards']) != {b['id'] for b in boards}:
        raise ValueError('Cannot verify an incomplete board manifest.')
    kev = new_run(DIR / 'kev.json')
    qwen = new_run(DIR / 'qwen.json')
    summary = {'model':expected_model, 'reference_sha256':LABEL_DIGEST, 'boards':{}}
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    config = {'PLANNER_MODEL':expected_model, 'PLANNER_MODEL_ENDPOINT':'http://127.0.0.1:1234/v1/chat/completions',
              'PLANNER_EMBEDDING_MODEL':retrieval.EMBEDDING_MODEL, 'PLANNER_DECISION_MODEL':retrieval.DECISION_MODEL,
              'PLANNER_RELEVANCE_THRESHOLD':'.50'}
    with patch.dict(os.environ, config):
        for board in boards:
            name = board['id']
            parts = artifacts / f'forecast-{tag}-boards'
            raw = parts / (name + '-receipts.json')
            result = parts / (name + '.json')
            saved = new_run(result)
            logged = new_run(raw)
            if logged['model'] != expected_model or saved['model'] != expected_model:
                raise ValueError(f'Unexpected model for {name}.')
            seen, missing = [], []
            recorded_errors = {e['request_sha256']:e for e in logged.get('errors', [])}
            def read(messages, schema, **kwargs):
                body = {'model':expected_model, 'messages':messages, 'schema':schema, 'kwargs':kwargs}
                key = payload_key(body)
                seen.append(key)
                if key in logged['responses']:
                    response = logged['responses'][key]
                    if response['input'] != json.loads(messages[-1]['content']):
                        missing.append(key)
                        raise ValueError('Receipt content differs from the actual worker prompt.')
                    return response['response']
                if key in recorded_errors:
                    err = recorded_errors[key]
                    typ = {'invalid_response':ModelResponseInvalid, 'context_exceeded':ModelContextExceeded,
                           'transport':ModelUnavailable}[err['kind']]
                    if typ is ModelResponseInvalid and 'reached its response length limit' in err['message']:
                        typ = ModelResponseTruncated
                    raise typ(err['message'])
                missing.append(key)
                raise ValueError('Fresh request has no exact inference receipt.')
            replay = do_board(board, kev, qwen, read)
            if missing:
                raise ValueError(f"{name} missing exact forecast receipts: {missing}")
            for field in ('cold_target', 'additions', 'initial_requests', 'inference_inputs'):
                if saved[field] != replay[field]:
                    raise ValueError(f'{name}: replay differs for {field}')
            def check_states(a, b):
                if a.keys() != b.keys():
                    raise ValueError(f'{name}: replay differs in card IDs')
                for task, state in a.items():
                    # Original UUIDs differ in the stored raw prediction hash. All stable fields must match.
                    for k,v in state.items():
                        if k != 'prediction_sha256' and (k not in b[task] or v != b[task][k]):
                            raise ValueError(f'{name}/{task}: saved {k} differs from receipt replay')
            check_states(saved['initial'], replay['initial'])
            for before, after in zip(saved['waves'], replay['waves'], strict=True):
                for field, value in before.items():
                    if field in ('before','after'):
                        check_states(value,after[field])
                    elif value != after.get(field):
                        raise ValueError(f'{name}: replayed completion wave differs in {field}')
            summary['boards'][name] = {
                'saved_state_sha256':sha(result), 'raw_receipt_file_sha256':sha(raw),
                'forecast_calls_replayed':len(seen), 'distinct_exact_requests':len(set(seen)),
                'unused_success_receipts':len(set(logged['responses'])-set(seen)),
                'surviving_target_states_replayed':sum(len(w['after']) for w in saved['waves'])}
    if save:
        write(artifacts / f'verified-{tag}.json', summary)
    return summary
