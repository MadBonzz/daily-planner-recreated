import json

import pytest

from examples.estimation import run


@pytest.mark.parametrize('provider_available', [True, False])
def test_classify_history_uses_only_prior_evidence_and_records_failures(tmp_path, monkeypatch, provider_available):
    history, = run.histories('01')
    holdout = history['holdouts'][0]
    monkeypatch.setattr("retrieval.embed_task", lambda *args: [1.0, 0.0])
    monkeypatch.setattr("retrieval.decide_related", lambda *args: True)
    requests = []

    def classify(messages, schema, **kwargs):
        payload = json.loads(messages[-1]['content'])
        requests.append(payload)
        if not provider_available:
            raise run.ModelUnavailable('Classifier unavailable')
        return {str(row['id']): {'label': 'on_task', 'reason': 'Editing the named task.'}
                for row in payload['intervals']}

    monkeypatch.setattr(run.activity, 'chat_json', classify)
    monkeypatch.setattr(run.estimator, 'chat_json', lambda *a, **kw: {
        'point_min': 90, 'low_min': 68, 'high_min': 113, 'confidence': 'medium',
        'rationale': 'Proof needs a separate trace.', 'confidence_reason': 'Scope is bounded.',
        'comparisons': [], 'warnings': []})
    dest = tmp_path / 'results.json'
    status = run.main(['evaluate', '01', '--case', holdout['id'], '--classify-history',
                       '--strict-model', '--json', str(dest)])
    result = json.loads(dest.read_text())
    assert status == (0 if provider_available else 1)
    assert requests and any('pdf_text' in request['task'] for request in requests)
    previous_titles = {task['title'] for task in history['tasks']}
    assert all(request['task']['title'] in previous_titles for request in requests)
    forbidden = {'actual_minutes', 'completion_note', 'check', 'label', 'sessions_minutes', 'evaluation', 'activity'}
    for request in requests:
        assert not forbidden.intersection(request['task'])
        assert all(not forbidden.intersection(interval) for interval in request['intervals'])
        assert 'private-fixture.example' not in json.dumps(request)
    assert result['configuration']['history_activity_labels'] == 'configured_model'
    assert result['summary']['history_classification_calls'] == len(requests)
    assert result['summary']['history_classification_failed_calls'] == (0 if provider_available else len(requests))
    trace, = result['history_classification']
    assert trace['history'] == history['id']
    assert all(('response' if provider_available else 'error') in call for call in trace['calls'])
    if not provider_available:
        assert all(task['label_seconds']['on_task'] == task['label_seconds']['off_task'] == 0 for task in trace['tasks'])
    assert result['cases'][0]['model_status'] == 'ok'


def test_classify_history_requires_live_model(capsys):
    with pytest.raises(SystemExit) as exc:
        run.main(['evaluate', '--offline', '--classify-history'])
    assert exc.value.code == 2
    assert '--classify-history requires an online run' in capsys.readouterr().err
