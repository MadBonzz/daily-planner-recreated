import io
import json
import multiprocessing
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

import estimator
import local_model_runtime
import model_client
import planner_analytics
import retrieval
import sample_retrieval
from chat_tasks import apply_action
from examples.estimation.package_retrieval import inputs
from examples.estimation.run import histories
from fallback_provenance import capture_fallbacks
from examples.estimation import check_local_models
from planner_chat import planning_context
from planner_store import PlannerStore


QWEN = "text-embedding-qwen3-embedding-0.6b"
KEV = "kev-4b"
BONSAI = "ternary-bonsai-2-4b-mlx"
MODELS = {QWEN: "embedding", KEV: "decision", BONSAI: "llm"}
FORECAST = {
    "point_min": 20, "low_min": 15, "high_min": 25, "confidence": "low",
    "rationale": "Compare each printed copy with the checklist.",
    "confidence_reason": "The number of corrections is unknown.",
}
CHAT_DRAFT = {
    "action": "create", "title": "Review Paxos proofs", "description": "Review consensus proofs.",
    "details": "", "estimate_minutes": 240, "estimate_phrase": "4 hours", "category": "",
    "day": "", "when": "", "fixed_duration": False, "file_ids": [], "reply": "",
}


class BionicHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/api/v1/models":
            return self.reply({}, 404)
        inventory = []
        for key, kind in MODELS.items():
            instances = [{"id": instance_id, "config": {"context_length": context,
                          **self.server.instance_options.get(instance_id, {})}}
                         for instance_id, (installed, context) in self.server.loaded.items()
                         if key == installed]
            entry = {"key": key, "type": kind, "loaded_instances": instances}
            if key == QWEN:
                # The native embedding inventory omits architecture and separates trained and loaded context.
                entry.update(publisher="Qwen", display_name="Qwen3 Embedding 0.6B", format="gguf",
                             max_context_length=32768, quantization={"name": "Q8_0", "bits_per_weight": 8})
            inventory.append(entry)
        self.reply({"models": inventory})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server = self.server
        if self.path == "/api/v1/models/unload":
            target = body["instance_id"]
            server.events.append(("unload", target))
            if server.loaded.pop(target, None) is None:
                return self.reply({"error": "unknown instance"}, 404)
            return self.reply({"instance_id": target})
        if self.path == "/api/v1/models/load":
            key, context = body["model"], body["context_length"]
            server.events.append(("load", key, context))
            server.load_requests.append(body)
            if server.reject_parallel and "parallel" in body:
                return self.reply({"error": {"message": "Unrecognized key: parallel"}}, 400)
            if server.fail_load:
                server.fail_load = False
                return self.reply({"error": {"message": "load failed"}}, 500)
            if key not in MODELS or server.loaded:
                return self.reply({"error": "unknown model or insufficient free memory"}, 400)
            # Exercise the native API's choice of an instance ID distinct from the installed key.
            instance_id = f"{key}:1"
            server.loaded[instance_id] = key, context
            if "parallel" in body:
                server.instance_options[instance_id] = {"parallel": body["parallel"]}
            return self.reply({"instance_id": instance_id, "status": "loaded", "type": MODELS[key],
                               "load_config": {"context_length": context}, "load_time_seconds": 0.1})

        instance_id = body["model"]
        server.events.append((self.path, instance_id))
        if self.path == "/v1/chat/completions":
            server.chat_started.set()
            if not server.release_chat.wait(10):
                return self.reply({"error": "timed out waiting for the test"}, 503)
        if instance_id not in server.loaded:
            return self.reply({"error": "instance unavailable"}, 404)
        if self.path == "/v1/embeddings":
            text = body["input"]
            server.embedding_inputs.append(text)
            if server.reject_embeddings:
                return self.reply({"error": {"type": "context_length_exceeded", "message": "Input exceeds context."}}, 400)
            # Simulate a backend that embeds only the prefix fitting its context.
            words = text.split()
            context = server.loaded[instance_id][1]
            if len(words) > context:
                server.truncated_inputs.append(text)
            return self.reply({"data": [{"embedding": [0.03125] * 1024}],
                               "usage": {"prompt_tokens": min(len(words), context)}})
        if self.path == "/v1/systemone":
            return self.reply({"answers": {"relevant": {"noul": 0.9}}})
        if self.path == "/v1/chat/completions":
            if server.chat_replies:
                server.forecast_requests.append(body)
                return self.reply({"choices": [{"message": server.chat_replies.pop(0), "finish_reason": "stop"}]})
            answer = dict(FORECAST)
            if "response_format" in body:
                schema = body["response_format"]["json_schema"]["schema"]
            else:
                schema = json.loads(body["messages"][0]["content"].split("matching this schema:\n")[-1])
            properties = schema["properties"]
            if "warnings" in properties:
                answer["warnings"] = []
            if "comparisons" in properties:
                answer["comparisons"] = [{"slot": 1, "reason": "Both tasks compare printed checklists with a master copy."}]
            return self.reply({"choices": [{"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}]})
        self.reply({}, 404)


@pytest.fixture
def bionic(monkeypatch, tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), BionicHandler)
    server.loaded = {}
    server.events = []
    server.embedding_inputs = []
    server.truncated_inputs = []
    server.reject_embeddings = False
    server.instance_options = {}
    server.load_requests = []
    server.reject_parallel = False
    server.fail_load = False
    server.chat_replies = []
    server.forecast_requests = []
    server.chat_started = threading.Event()
    server.release_chat = threading.Event()
    server.release_chat.set()
    runner = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    runner.start()
    base = f"http://127.0.0.1:{server.server_port}/v1"
    config = {
        "PLANNER_SINGLE_MODEL": "1", "PLANNER_EMBEDDING_MODEL": "qwen3-embedding-0.6b-live",
        "PLANNER_EMBEDDING_ENDPOINT": base + "/embeddings", "PLANNER_DECISION_MODEL": "kev-4b-live",
        "PLANNER_DECISION_ENDPOINT": base + "/systemone", "PLANNER_MODEL": BONSAI,
        "PLANNER_MODEL_ENDPOINT": base + "/chat/completions", "PLANNER_MODEL_TIMEOUT_SECONDS": "15",
        "PLANNER_DATA_DIR": str(tmp_path),
    }
    monkeypatch.setattr(model_client, "ENV_FILE", tmp_path / ".env")
    for name in ("PLANNER_EMBEDDING_MODEL_KEY", "PLANNER_DECISION_MODEL_KEY", "PLANNER_MODEL_KEY",
                 "PLANNER_EMBEDDING_CONTEXT_LENGTH", "PLANNER_DECISION_CONTEXT_LENGTH", "PLANNER_CHAT_CONTEXT_LENGTH"):
        monkeypatch.delenv(name, raising=False)
    for name, value in config.items():
        monkeypatch.setenv(name, value)
    try:
        yield server
    finally:
        server.release_chat.set()
        server.shutdown()
        server.server_close()
        runner.join()


def check(role):
    return check_local_models.check(role, model_client.planner_settings())


@pytest.fixture
def sdk_unavailable(monkeypatch):
    def failed_count(args, **kwargs):
        raise subprocess.CalledProcessError(1, args, stderr=b"SDK protocol mismatch")

    monkeypatch.setattr(subprocess, "run", failed_count)


def test_full_local_check_switches_once_per_role_and_reuses_bonsai(bionic, sdk_unavailable):
    assert "1024 dimensions" in check("embedding")
    assert "as related" in check("decision")
    assert "validated forecast of 20 min" in check("forecast")
    assert "validated forecast of 20 min" in check("forecast")
    assert bionic.events == [
        ("load", QWEN, 2048), ("/v1/embeddings", QWEN + ":1"), ("unload", QWEN + ":1"),
        ("load", KEV, 4096), ("/v1/systemone", KEV + ":1"), ("unload", KEV + ":1"),
        ("load", BONSAI, 8192), ("/v1/chat/completions", BONSAI + ":1"), ("/v1/chat/completions", BONSAI + ":1"),
    ]
    assert [body.get("parallel") for body in bionic.load_requests] == [None, None, 1]


def ask_about_paxos(question):
    now = datetime(2026, 11, 15, 9)
    return planner_analytics.ask_analytics(question, [], [], plan=planning_context([], now.date(), now),
                                          create_categories=["Study"])


def test_chat_router_and_creation_share_one_managed_bonsai_instance(bionic, tmp_path, monkeypatch):
    bionic.loaded.update({"qwen-resident": (QWEN, 2048), "kev-resident": (KEV, 4096)})
    monkeypatch.setenv("PLANNER_CHAT_CONTEXT_LENGTH", "6144")
    bionic.chat_replies = [{"content": json.dumps(answer)} for answer in ({"intent": "create_task"}, CHAT_DRAFT)]
    result = ask_about_paxos("I need to review Paxos proofs, which should take 4 hours.")
    assert result["tool"] == "create_task" and result["query_source"] == "model"
    store = PlannerStore(tmp_path / "plan")
    saved = apply_action(store, result, {}, "paxos-chat-turn")
    task = store.get_task(saved["affected_task_id"])
    assert (task["title"], task["estimate_minutes"]) == (CHAT_DRAFT["title"], 240)
    assert set(bionic.events[:2]) == {("unload", "qwen-resident"), ("unload", "kev-resident")}
    assert bionic.events[2:] == [("load", BONSAI, 6144), ("/v1/chat/completions", BONSAI + ":1"),
                                 ("/v1/chat/completions", BONSAI + ":1")]
    assert bionic.load_requests == [{"model": BONSAI, "context_length": 6144, "parallel": 1}]
    assert bionic.loaded == {BONSAI + ":1": (BONSAI, 6144)}


def test_background_forecast_waits_for_chat_before_switching_to_qwen(bionic, tmp_path, monkeypatch):
    bionic.chat_replies = [{"content": json.dumps(CHAT_DRAFT)}]
    bionic.release_chat.clear()
    forecast_waiting = threading.Event()
    lock = local_model_runtime.portalocker.lock

    def observed_lock(handle, flags):
        if bionic.chat_started.is_set():
            forecast_waiting.set()
        return lock(handle, flags)

    monkeypatch.setattr(local_model_runtime.portalocker, "lock", observed_lock)
    task = {**check_local_models.TASK, "id": "new"}
    prior = {**check_local_models.PRIOR, "id": "prior", "estimate_minutes": 20, "actual_seconds": 1500}
    with ThreadPoolExecutor(max_workers=2) as pool:
        chat = pool.submit(ask_about_paxos, "Add a task to review Paxos proofs. I expect it to take 4 hours.")
        try:
            assert bionic.chat_started.wait(5)
            forecast = pool.submit(estimator.predict, task, [prior], tmp_path / "vectors")
            assert forecast_waiting.wait(5)
            assert bionic.events == [("load", BONSAI, 8192), ("/v1/chat/completions", BONSAI + ":1")]
        finally:
            bionic.release_chat.set()
        assert chat.result()["new_task"]["estimate_minutes"] == 240
        assert forecast.result()["provenance"]["decision"] == "direct"
    assert [event[1] for event in bionic.events if event[0] == "load"] == [BONSAI, QWEN, KEV, BONSAI]


def test_regular_forecast_uses_both_qwen_queries_and_past_timing_without_sdk(bionic, sdk_unavailable, tmp_path):
    task = {**check_local_models.TASK, "id": "new", "pdf_text": "Compare each checklist annotation. " * 800}
    prior = {**check_local_models.PRIOR, "id": "prior", "estimate_minutes": 20,
             "actual_seconds": 1500, "status": "completed"}
    result = estimator.predict(task, [prior], tmp_path / "index")
    assert result["provenance"]["decision"] == "direct"
    assert "retrieval_error" not in result
    assert not result.get("retrieval_fallbacks")
    assert result["used_task_ids"] == ["prior"]
    assert result["evidence"][0]["actual_min"] == 25
    assert bionic.embedding_inputs == [retrieval.embedding_input(task, query=True),
                                       retrieval.embedding_input(task), retrieval.embedding_input(prior)]
    assert bionic.truncated_inputs == bionic.embedding_inputs[:2]
    assert [event[1] for event in bionic.events if event[0] == "load"] == [QWEN, KEV, BONSAI]


def test_nora_marker_word_recovers_a_real_forecast_after_reasoning_only_reply(bionic, tmp_path):
    history, = histories("02-migration-lab")
    rows = inputs(history)
    sample = next(task for task in history["tasks"] if task["id"] == "ring-port")
    prior = {**rows["ring-port"], "id": "ring-port", "actual_seconds": sample["actual_minutes"] * 60,
             "completion_note": sample["completion_note"]}
    task = {**rows["held-seed"], "id": "held-seed", "relevance_overrides": {"ring-port": True}}
    answer = {**FORECAST, "point_min": 74, "low_min": 60, "high_min": 85,
              "rationale": "Minimizing the failing trace takes multiple replay attempts.",
              "comparisons": [{"slot": 1, "reason": "Both tasks involve reproducing a failing fixture."}],
              "warnings": [{"observation_id": "S1.1", "advice": "Leave time to reproduce the minimized fixture."}]}
    draft = json.dumps({**answer, "point_min": 65})
    bionic.chat_replies = [
        {"content": "", "reasoning_content": draft},
        {"content": [{"type": "reasoning", "text": draft},
                     {"type": "text", "text": "```json\n" + json.dumps(answer) + "\n```"}]},
    ]
    forecast = estimator.predict(task, [prior], tmp_path / "index")
    assert forecast["provenance"]["decision"] == "direct"
    assert forecast["point_min"] == 74
    assert forecast["evidence"][0]["title"] == "Port the wraparound header reader"
    assert forecast["cautions"][0]["sources"][0]["task_id"] == "ring-port"
    first, retry = bionic.forecast_requests
    for payload in (first, retry):
        assert "response_format" not in payload
        assert payload["model"] == BONSAI + ":1"
        assert payload["reasoning_effort"] == "none"
        assert {"max_tokens", "max_completion_tokens", "max_output_tokens"}.isdisjoint(payload)
    assert first["messages"][-1] == retry["messages"][-1]
    assert "point_min" in retry["messages"][0]["content"]


def test_exhausted_bonsai_retry_preserves_fallback_and_keeps_reasoning_private(bionic, tmp_path):
    draft = json.dumps({**FORECAST, "low_min": 700, "point_min": 777, "high_min": 800})
    bionic.chat_replies = [{"content": "", "reasoning_content": draft},
                           {"content": f"<think>{draft}</think>\nPRIVATE notes suggest 777 minutes."}]
    result = estimator.predict(check_local_models.TASK, [], tmp_path / "index")
    assert result["provenance"]["decision"] == "fallback"
    assert result["point_min"] == check_local_models.TASK["estimate_minutes"]
    assert result["model_error"] and "PRIVATE" not in result["model_error"]
    assert len(bionic.forecast_requests) == 2


def test_http_rejection_reuses_only_an_exact_packaged_vector(bionic, sdk_unavailable):
    task = inputs(histories(None)[0])["bridge-proof"]
    bionic.reject_embeddings = True
    for query in (False, True):
        text = retrieval.embedding_input(task, query)
        archived = sample_retrieval.vector_for(retrieval.EMBEDDING_MODEL, text)
        assert archived is not None
        with capture_fallbacks() as used:
            assert retrieval.embed_task(task, query=query) == archived
            assert used == {"embedding"}
        assert bionic.embedding_inputs[-1] == text

    changed = {**task, "description": "Check a newly recorded hardware trace."}
    with capture_fallbacks() as used, pytest.raises(retrieval.DocumentTooLong):
        retrieval.embed_task(changed)
    assert not used
    assert bionic.embedding_inputs[-1] == retrieval.embedding_input(changed)


def test_retains_existing_target_and_releases_other_loaded_role(bionic):
    bionic.loaded.update({"kev-4b-live": (KEV, 4096), "existing-qwen": (QWEN, 2048)})
    assert "as related" in check("decision")
    assert bionic.events == [("unload", "existing-qwen"), ("/v1/systemone", "kev-4b-live")]


def test_explicit_installed_key_and_context_use_returned_instance(bionic, monkeypatch):
    monkeypatch.setenv("PLANNER_DECISION_MODEL", "kev-custom-profile")
    monkeypatch.setenv("PLANNER_DECISION_MODEL_KEY", KEV)
    monkeypatch.setenv("PLANNER_DECISION_CONTEXT_LENGTH", "1536")
    assert "as related" in check("decision")
    assert bionic.events == [("load", KEV, 1536), ("/v1/systemone", KEV + ":1")]


def test_missing_model_keeps_resident_instance(bionic, monkeypatch):
    bionic.loaded["already-running"] = QWEN, 2048
    monkeypatch.setenv("PLANNER_DECISION_MODEL_KEY", "uninstalled-model")
    with pytest.raises(model_client.ModelUnavailable):
        check("decision")
    assert not bionic.events
    assert bionic.loaded == {"already-running": (QWEN, 2048)}


@pytest.mark.parametrize("parallel", [4, None])
def test_bonsai_single_slot_load_and_older_inventories(bionic, parallel):
    bionic.loaded["bonsai-running"] = BONSAI, 8192
    if parallel is not None:
        bionic.instance_options["bonsai-running"] = {"parallel": parallel}
    assert "validated forecast" in check("forecast")
    if parallel == 4:
        assert bionic.events == [("unload", "bonsai-running"), ("load", BONSAI, 8192),
                                 ("/v1/chat/completions", BONSAI + ":1")]
        assert bionic.load_requests[0]["parallel"] == 1
    else:
        assert bionic.events == [("/v1/chat/completions", "bonsai-running")]


def test_older_server_rejecting_parallel_loads_bonsai_once_without_it(bionic):
    bionic.reject_parallel = True
    assert "validated forecast" in check("forecast")
    assert [body.get("parallel") for body in bionic.load_requests] == [1, None]
    assert "validated forecast" in check("forecast")
    assert len(bionic.load_requests) == 2


def test_failed_load_retries_against_fresh_inventory(bionic):
    bionic.loaded["other"] = QWEN, 2048
    bionic.fail_load = True
    with pytest.raises(model_client.ModelUnavailable):
        check("decision")
    assert bionic.events == [("unload", "other"), ("load", KEV, 4096)]
    bionic.loaded["reloaded-outside-planner"] = BONSAI, 8192
    assert "as related" in check("decision")
    assert bionic.events[2:] == [("unload", "reloaded-outside-planner"), ("load", KEV, 4096),
                                 ("/v1/systemone", KEV + ":1")]


def start_decision(ready):
    ready.set()
    assert "as related" in check("decision")


def test_separate_worker_cannot_unload_an_inflight_forecast(bionic):
    bionic.release_chat.clear()
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    worker = context.Process(target=start_decision, args=(ready,))
    with ThreadPoolExecutor(max_workers=1) as pool:
        forecast = pool.submit(check, "forecast")
        try:
            assert bionic.chat_started.wait(5)
            worker.start()
            assert ready.wait(8)
            worker.join(timeout=0.3)
            assert worker.is_alive()
            assert bionic.events == [("load", BONSAI, 8192), ("/v1/chat/completions", BONSAI + ":1")]
        finally:
            bionic.release_chat.set()
            if worker.pid is not None:
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
        assert "validated forecast" in forecast.result()
    assert worker.exitcode == 0
    assert bionic.events[-3:] == [("unload", BONSAI + ":1"), ("load", KEV, 4096), ("/v1/systemone", KEV + ":1")]


def test_disabled_mode_leaves_other_models_loaded(bionic, monkeypatch):
    monkeypatch.setenv("PLANNER_SINGLE_MODEL", "0")
    bionic.loaded.update({BONSAI: (BONSAI, 8192), "kev-4b-live": (KEV, 4096)})
    assert "as related" in check("decision")
    assert bionic.events == [("/v1/systemone", "kev-4b-live")]
    assert set(bionic.loaded) == {BONSAI, "kev-4b-live"}


def test_hosted_cohere_sends_only_its_ordinary_completion(bionic, monkeypatch):
    endpoint = "https://api.cohere.ai/compatibility/v1/chat/completions"
    monkeypatch.setenv("PLANNER_MODEL_ENDPOINT", endpoint)
    monkeypatch.setenv("PLANNER_MODEL", "command-r7b-12-2024")
    monkeypatch.setenv("COHERE_API", "test-key")
    response = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}
    with patch("model_client.build_opener") as opener:
        opener.return_value.open.return_value = io.BytesIO(json.dumps(response).encode())
        assert model_client.chat_json([], {"type": "object"}) == {}
        opener.return_value.open.assert_called_once()
        assert opener.return_value.open.call_args.args[0].full_url == endpoint
    assert not bionic.events
