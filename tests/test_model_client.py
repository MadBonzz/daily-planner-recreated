import io
import json
import os
import subprocess
import sys
from http.client import IncompleteRead
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from model_client import API_URL, MODEL_ID, ModelContextExceeded, ModelResponseInvalid, ModelResponseTruncated, ModelUnavailable, chat_json


class ModelClientTests(unittest.TestCase):
    schema = {
        "type": "object",
        "required": ["label"],
        "properties": {"label": {"enum": ["on_task", "off_task", "uncertain"]}},
        "additionalProperties": False,
    }

    def setUp(self):
        self.env_file = Path(self.enterContext(TemporaryDirectory())) / ".env"
        self.enterContext(patch("model_client.ENV_FILE", self.env_file))
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def reply(self, content=None, finish_reason="stop", *, message=None):
        return io.BytesIO(json.dumps({"choices": [
            {"finish_reason": finish_reason, "message": message if message is not None else {"content": content}}
        ]}).encode())

    @patch("model_client.build_opener")
    def test_local_schema_request(self, opener):
        opener.return_value.open.return_value = self.reply('{"label":"on_task"}')
        self.assertEqual(chat_json([], self.schema), {"label": "on_task"})
        self.assertEqual(opener.return_value.open.call_args.kwargs['timeout'], 90)
        request = opener.return_value.open.call_args.args[0]
        self.assertEqual(request.full_url, API_URL)
        body = json.loads(request.data)
        self.assertEqual(body["model"], MODEL_ID)
        self.assertTrue({"max_tokens", "max_completion_tokens", "max_output_tokens"}.isdisjoint(body))
        self.assertEqual(body["response_format"]["json_schema"]["schema"], self.schema)

    @patch("model_client.build_opener")
    def test_command_r_turn_delimiter_only_applies_to_local_inference(self, opener):
        model = "command-r7b-12-2024"
        for endpoint, key, local in [(API_URL, model, True), (API_URL, "other-model", False),
                                     ("https://api.cohere.ai/compatibility/v1/chat/completions", model, False)]:
            with self.subTest(endpoint=endpoint, key=key), patch.dict(os.environ,
                    PLANNER_MODEL=key, PLANNER_MODEL_ENDPOINT=endpoint, COHERE_API="test"):
                opener.return_value.open.return_value = self.reply('{"label":"on_task"}')
                self.assertEqual(chat_json([], self.schema), {"label": "on_task"})
                payload = json.loads(opener.return_value.open.call_args.args[0].data)
                self.assertEqual(payload.get("stop"), ["<|END_OF_TURN_TOKEN|>"] if local else None)
                self.assertEqual(payload["response_format"]["json_schema"]["schema"], self.schema)
                self.assertNotIn("max_tokens", payload)
                if local:
                    self.assertIn(json.dumps(self.schema), payload["messages"][0]["content"])

    @patch("model_client.build_opener")
    def test_bonsai_2_mlx_uses_prompt_schema_from_first_request(self, opener):
        for model, key, mlx in [
            ("ternary-bonsai-2-4b-mlx", "", True),
            ("demo-alias", "prism-ml/Ternary-Bonsai-2-4B-mlx-2bit", True),
            ("ternary-bonsai-2-4b-gguf", "", False),
        ]:
            with self.subTest(model=model), patch.dict(os.environ, PLANNER_MODEL=model, PLANNER_MODEL_KEY=key):
                opener.return_value.open.reset_mock()
                opener.return_value.open.return_value = self.reply('{"label":"on_task"}')
                self.assertEqual(chat_json([], self.schema), {"label": "on_task"})
                opener.return_value.open.assert_called_once()
                payload = json.loads(opener.return_value.open.call_args.args[0].data)
                self.assertEqual("response_format" not in payload, mlx)
                self.assertEqual(payload["reasoning_effort"], "none")
                self.assertIn(json.dumps(self.schema), payload["messages"][0]["content"])

    @patch("model_client.build_opener")
    def test_only_local_bonsai_retries_unsupported_reasoning_effort(self, opener):
        for local, endpoint in [(True, API_URL), (False, "https://api.cohere.ai/compatibility/v1/chat/completions")]:
            with self.subTest(endpoint=endpoint), patch.dict(
                os.environ, PLANNER_MODEL="ternary-bonsai-2-4b-mlx", PLANNER_MODEL_ENDPOINT=endpoint, COHERE_API="test"
            ):
                opener.reset_mock()
                issue = {"error": {"param": "reasoning_effort", "message": "Unsupported parameter"}}
                failure = HTTPError(endpoint, 400, "Bad Request", {}, io.BytesIO(json.dumps(issue).encode()))
                transport = opener.return_value.open
                transport.side_effect = [failure, self.reply('{"label":"on_task"}')]
                if local:
                    self.assertEqual(chat_json([], self.schema), {"label": "on_task"})
                    first, retry = [json.loads(call.args[0].data) for call in transport.call_args_list]
                    self.assertEqual(first.pop("reasoning_effort"), "none")
                    self.assertEqual(retry, first)
                else:
                    with self.assertRaisesRegex(ModelUnavailable, "HTTP 400"):
                        chat_json([], self.schema)
                    transport.assert_called_once()

    @patch("model_client.build_opener")
    def test_bad_schema_or_truncated_completion_is_recoverable(self, opener):
        for content, reason in [('{"label":"unknown"}', "stop"), ('{"label":', "stop"), ('{"label":', "length")]:
            with self.subTest(reason=reason):
                opener.return_value.open.return_value = self.reply(content, reason)
                with self.assertRaises(ModelResponseInvalid) as exc:
                    chat_json([], self.schema)
                self.assertEqual(isinstance(exc.exception, ModelResponseTruncated), reason == "length")

    @patch("model_client.build_opener")
    def test_local_completion_errors_identify_the_failed_stage_without_echoing_server_content(self, opener):
        private = "PRIVATE TASK CONTENT"
        cases = [
            (b"Not JSON: " + private.encode(), "decoded as JSON"),
            (b"\xff\x90", "decoded as JSON"),
            (json.dumps([private]).encode(), "without a chat completion object"),
            (json.dumps({"error": {"message": private}}).encode(), "an error object"),
            (b'{}', "choices array"),
            (b'{"choices":[]}', "choices array"),
            (b'{"choices":[null]}', "invalid first choice"),
            (b'{"choices":[{}]}', "without an assistant message"),
            (b'{"choices":[{"finish_reason":"content_filter"}]}', "filtered the response"),
            (self.reply(message={"refusal": private, "content": None}).getvalue(), "a refusal"),
            (self.reply(message={"content": [{"type": "refusal", "refusal": private}]}).getvalue(), "a refusal"),
        ]
        for raw, category in cases:
            with self.subTest(category=category):
                opener.return_value.open.reset_mock()
                opener.return_value.open.return_value = io.BytesIO(raw)
                with self.assertRaisesRegex(ModelResponseInvalid, category) as failure:
                    chat_json([], self.schema)
                self.assertNotIn(private, str(failure.exception))
                opener.return_value.open.assert_called_once()
        opener.return_value.open.side_effect = IncompleteRead(private.encode(), 80)
        with self.assertRaisesRegex(ModelResponseInvalid, "closed the HTTP response") as failure:
            chat_json([], self.schema)
        self.assertNotIn(private, str(failure.exception))

    @patch("model_client.build_opener")
    def test_local_bonsai_reads_only_the_final_schema_object(self, opener):
        os.environ["PLANNER_MODEL"] = "ternary-bonsai-2-4b-mlx"
        final = '{"label":"on_task"}'
        draft = '{"label":"off_task"}'
        messages = [
            {"content": f"<think>{draft}</think>\n```json\n{final}\n```"},
            {"content": f"Here is the final result.\n{final}"},
            {"content": f"{draft}</think>\n{final}"},
            {"content": f"```json\n{final}"},
            {"content": [{"type": "reasoning", "text": draft}, {"type": "text", "text": final[:7]},
                         {"type": "text", "text": final[7:]}]},
            {"reasoning_content": draft, "content": [{"type": "output_text", "text": final}]},
        ]
        for message in messages:
            with self.subTest(message=message):
                opener.return_value.open.return_value = self.reply(message=message)
                self.assertEqual(chat_json([], self.schema), {"label": "on_task"})

    @patch("model_client.build_opener")
    def test_reasoning_or_ambiguous_or_invalid_bonsai_output_cannot_be_a_result(self, opener):
        os.environ["PLANNER_MODEL"] = "ternary-bonsai-2-4b-mlx"
        draft = '{"label":"off_task"}'
        messages = [
            {"content": "", "reasoning_content": draft},
            {"content": f"<think>{draft}"},
            {"content": [{"type": "reasoning", "text": draft}]},
            {"content": f"{draft} or " + '{"label":"on_task"}'},
            {"content": 'It took around 65 minutes.'},
            {"content": '<think>' + draft + '</think>\n{"label":"invalid"}'},
            {"content": '{"label":"on_task"'},
            {"content": '{"broken":' + draft},
        ]
        for message in messages:
            with self.subTest(message=message):
                opener.return_value.open.return_value = self.reply(message=message)
                with self.assertRaises(ModelResponseInvalid):
                    chat_json([], self.schema)

    @patch("model_client.build_opener")
    def test_hosted_answer_with_a_preamble_needs_only_one_request(self, opener):
        os.environ.update(PLANNER_MODEL="command-r7b-12-2024",
                          PLANNER_MODEL_ENDPOINT="https://api.cohere.ai/compatibility/v1/chat/completions",
                          COHERE_API="test-token")
        opener.return_value.open.return_value = self.reply('Here is my guess. {"label":"on_task"}')
        self.assertEqual(chat_json([], self.schema), {"label": "on_task"})
        opener.return_value.open.assert_called_once()

    @patch("model_client.build_opener")
    def test_schema_integer_slots_are_safe_indices_and_only_typed_integers_are_converted(self, opener):
        schema = {"type": "object", "properties": {
            "minutes": {"type": "integer"}, "slots": {"type": "array", "items": {"type": "integer"}},
            "weight": {"type": "number"}, "flag": {"type": "boolean"}}}
        opener.return_value.open.return_value = self.reply('{"minutes":20.0,"slots":[1.0],"weight":2.0,"flag":true}')
        obj = chat_json([], schema)
        self.assertIs(type(obj["minutes"]), int)
        self.assertEqual(["a", "b"][obj["slots"][0]], "b")
        self.assertIs(type(obj["weight"]), float)
        self.assertIs(obj["flag"], True)
        for invalid in ["1.2", "true"]:
            opener.return_value.open.return_value = self.reply('{"slots":[' + invalid + ']}')
            with self.assertRaises(ModelResponseInvalid):
                chat_json([], schema)

    @patch("model_client.build_opener")
    def test_context_rejection_is_identified_without_echoing_provider_content(self, opener):
        for status, code, typed in [(400, "exceed_context_size_error", True),
                                    (413, "context_length_exceeded", True), (400, "other", False)]:
            payload = {"error": {"type": code, "n_prompt_tokens": 25803, "n_ctx": 6144,
                                 "message": "PRIVATE PROMPT ECHO"}}
            opener.return_value.open.side_effect = HTTPError(
                API_URL, status, "bad request", {}, io.BytesIO(json.dumps(payload).encode()))
            with self.assertRaises(ModelUnavailable) as exc:
                chat_json([], self.schema)
            self.assertEqual(isinstance(exc.exception, ModelContextExceeded), typed)
            self.assertNotIn("PRIVATE PROMPT", str(exc.exception))
            if typed:
                self.assertIn("25,803 input tokens, 6,144 available", str(exc.exception))
            else:
                self.assertIn("HTTP 400", str(exc.exception))

    @patch("model_client.build_opener")
    def test_rejected_schema_retries_as_prompt_at_the_same_endpoint(self, opener):
        for url in ("https://api.cohere.ai/compatibility/v1/chat/completions", API_URL):
            with self.subTest(url=url), patch.dict(os.environ, PLANNER_MODEL_ENDPOINT=url, COHERE_API="test"):
                issue = {"error": {"param": "response_format", "message": "unsupported schema - PRIVATE ECHO"}}
                transport = opener.return_value.open
                transport.reset_mock()
                transport.side_effect = [HTTPError(url, 400, "Bad Request", {}, io.BytesIO(json.dumps(issue).encode())),
                                         self.reply('{"label":"on_task"}')]
                self.assertEqual(chat_json([{"role": "user", "content": "Classify"}], self.schema), {"label": "on_task"})
                first, retry = [json.loads(call.args[0].data) for call in transport.call_args_list]
                self.assertIn("response_format", first)
                self.assertNotIn("response_format", retry)
                self.assertIn(json.dumps(self.schema), retry["messages"][0]["content"])
                self.assertNotIn("PRIVATE ECHO", json.dumps(retry))
                self.assertEqual(first["messages"][-1], retry["messages"][-1])

    @patch("model_client.build_opener")
    def test_complete_schema_valid_answer_reused_across_wire_formats(self, opener):
        answer = {"label": "on_task"}
        messages = [{"content": answer}, {"content": None, "parsed": answer},
                    {"content": [{"type": "reasoning", "text": "draft"},
                                 {"type": "text", "text": "```json\n"},
                                 {"type": "output_text", "text": json.dumps(answer) + "\n```"}]},
                    {"content": "<think>draft</think>\n" + json.dumps(answer)}]
        for endpoint in [API_URL, "https://api.cohere.ai/compatibility/v1/chat/completions"]:
            with self.subTest(endpoint=endpoint), patch.dict(os.environ, PLANNER_MODEL_ENDPOINT=endpoint, COHERE_API="test"):
                for message in messages:
                    opener.return_value.open.reset_mock()
                    opener.return_value.open.return_value = self.reply(message=message, finish_reason="length")
                    self.assertEqual(chat_json([], self.schema), answer)
                    opener.return_value.open.assert_called_once()
                opener.return_value.open.return_value = self.reply(message={"content": None, "reasoning_content": json.dumps(answer)})
                with self.assertRaisesRegex(ModelResponseInvalid, "without a final"):
                    chat_json([], self.schema)

    @patch("model_client.build_opener")
    def test_configured_cohere_request_loads_dotenv_and_keeps_prompt(self, opener):
        self.env_file.write_text(
            'COHERE_API="test-token"\nPLANNER_MODEL=c4ai-aya-expanse-32b\n'
            'PLANNER_MODEL_ENDPOINT=https://api.cohere.ai/compatibility/v1/chat/completions\n'
        )
        opener.return_value.open.return_value = self.reply('```json\n{"label":"on_task"}\n```')
        messages = [{"role": "system", "content": "Classify the activity."},
                    {"role": "user", "content": "CUDA docs"}]
        self.assertEqual(chat_json(messages, self.schema), {"label": "on_task"})
        req = opener.return_value.open.call_args.args[0]
        self.assertEqual(req.get_header("Authorization"), "Bearer test-token")
        payload = json.loads(req.data)
        self.assertEqual(payload["model"], "c4ai-aya-expanse-32b")
        self.assertTrue({"max_tokens", "max_completion_tokens", "max_output_tokens"}.isdisjoint(payload))
        self.assertEqual(payload["messages"][1], messages[1])
        self.assertIn(messages[0]["content"], payload["messages"][0]["content"])
        self.assertIn(json.dumps(self.schema), payload["messages"][0]["content"])
        self.assertNotIn("response_format", payload)
        self.assertEqual(messages[0]["content"], "Classify the activity.")

    @patch("model_client.build_opener")
    def test_command_r7b_compatibility_request_with_nested_strict_schema(self, opener):
        self.env_file.write_text(
            "COHERE_API=test-token\nPLANNER_MODEL=command-r7b-12-2024\n"
            "PLANNER_MODEL_ENDPOINT=https://api.cohere.ai/compatibility/v1/chat/completions\n"
        )
        opener.return_value.open.return_value = self.reply('{"label":"on_task"}')
        messages = [{"role": "user", "content": "Classify the task."}]
        self.assertEqual(chat_json(messages, self.schema), {"label": "on_task"})
        request = opener.return_value.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "command-r7b-12-2024")
        self.assertTrue({"max_tokens", "max_completion_tokens", "max_output_tokens"}.isdisjoint(payload))
        self.assertEqual(payload["response_format"], {
            "type": "json_schema",
            "json_schema": {"name": "planner_response", "strict": True, "schema": self.schema},
        })
        self.assertEqual(payload["messages"], messages)

        # Validate the response even if the endpoint ignores a requested constraint.
        opener.return_value.open.return_value = self.reply('{"label":"on_task","extra":1}')
        with self.assertRaises(ModelUnavailable):
            chat_json(messages, self.schema)

    @patch("model_client.build_opener")
    def test_cohere_key_only_sent_to_official_host(self, opener):
        self.env_file.write_text("COHERE_API=unit-test\nPLANNER_MODEL=stale\n")
        for endpoint in [API_URL, "https://api.cohere.ai.example.org/v1/chat/completions"]:
            with self.subTest(endpoint=endpoint), patch.dict(
                os.environ, PLANNER_MODEL_ENDPOINT=endpoint, PLANNER_MODEL="chosen-model"
            ):
                opener.return_value.open.return_value = self.reply('{"label":"uncertain"}')
                chat_json([], self.schema)
                req = opener.return_value.open.call_args.args[0]
                self.assertFalse(req.has_header("Authorization"))
                self.assertEqual(json.loads(req.data)["model"], "chosen-model")

    @patch("model_client.build_opener")
    def test_missing_key_retired_model_and_insecure_cohere_are_actionable(self, opener):
        host = "https://api.cohere.ai/compatibility/v1/chat/completions"
        for endpoint, model, message in [
            (host, "c4ai-aya-expanse-32b", "COHERE_API"),
            (host, "c4ai-aya-expanse-8b", "retired"),
            (host.replace("https:", "http:"), "test", "Invalid"),
        ]:
            with self.subTest(model=model), patch.dict(
                os.environ, PLANNER_MODEL_ENDPOINT=endpoint, PLANNER_MODEL=model
            ), self.assertRaisesRegex(ModelUnavailable, message):
                chat_json([], self.schema)
        opener.assert_not_called()


    def test_import_and_dotenv_settings_without_python_dotenv(self):
        self.env_file.write_text(
            'COHERE_API="fake-test-credential"\n'
            r'PLANNER_ACTIVITYWATCH_DB="C:\Users\A B\activitywatch\sqlite.db" # local store' "\n"
            "PLANNER_MODEL='command-r7b-12-2024' # hosted model\n"
            "PLANNER_MODEL_ENDPOINT=https://api.cohere.ai/compatibility/v1/chat/completions\n",
            encoding="utf-8-sig",
        )
        script = r"""
import os, sys
from pathlib import Path
sys.modules['dotenv'] = None
import model_client
model_client.ENV_FILE = Path(sys.argv[1])
endpoint, model = model_client.model_settings()
settings = model_client.planner_settings()
assert settings['COHERE_API'] == 'fake-test-credential'
assert settings['PLANNER_ACTIVITYWATCH_DB'] == r'C:\Users\A B\activitywatch\sqlite.db'
assert model == 'command-r7b-12-2024'
assert endpoint == 'https://api.cohere.ai/compatibility/v1/chat/completions'
os.environ['PLANNER_MODEL'] = 'env-override'
assert model_client.model_settings() == (endpoint, 'env-override')
"""
        subprocess.run([sys.executable, "-c", script, str(self.env_file)], check=True,
                       cwd=Path(__file__).resolve().parents[1])


    @patch("model_client.build_opener")
    def test_forecast_timeout_distinguishes_waiting_for_a_model_from_unreachable_endpoint(self, opener):
        with patch.dict(os.environ, PLANNER_MODEL_TIMEOUT_SECONDS="900"):
            for error in (TimeoutError(), URLError(TimeoutError())):
                opener.return_value.open.side_effect = error
                with self.assertRaisesRegex(ModelUnavailable, "timed out after 900 seconds"):
                    chat_json([], self.schema)
            opener.return_value.open.side_effect = URLError(ConnectionRefusedError())
            with self.assertRaisesRegex(ModelUnavailable, "is unavailable"):
                chat_json([], self.schema)

    @patch("model_client.build_opener")
    def test_configurable_timeout_is_positive_and_finite(self, opener):
        self.env_file.write_text("PLANNER_MODEL_TIMEOUT_SECONDS=240.5")
        opener.return_value.open.return_value = self.reply('{"label":"on_task"}')
        chat_json([], self.schema)
        self.assertEqual(opener.return_value.open.call_args.kwargs["timeout"], 240.5)
        for invalid in ["-1", "0", "inf", "nan", "nonsense"]:
            with self.subTest(invalid=invalid), patch.dict(os.environ, PLANNER_MODEL_TIMEOUT_SECONDS=invalid):
                with self.assertRaisesRegex(ModelUnavailable, "positive finite"):
                    chat_json([], self.schema)


def test_windows_powershell_utf16_env_and_explicit_hosted_endpoint_default(tmp_path, monkeypatch):
    import model_client
    env_file = tmp_path / ".env"
    env_file.write_text("PLANNER_MODEL_ENDPOINT=https://api.cohere.ai/compatibility/v1/chat/completions\n"
                        "COHERE_API=test-token\n", encoding="utf-16")
    monkeypatch.setattr(model_client, "ENV_FILE", env_file)
    monkeypatch.delenv("PLANNER_MODEL", raising=False)
    monkeypatch.delenv("PLANNER_MODEL_ENDPOINT", raising=False)
    for parser in (model_client.dotenv_values, None):
        monkeypatch.setattr(model_client, "dotenv_values", parser)
        config = model_client.planner_settings()
        assert config["COHERE_API"] == "test-token"
        assert model_client.model_settings(config) == ("https://api.cohere.ai/compatibility/v1/chat/completions",
                                                      "command-r7b-12-2024")
