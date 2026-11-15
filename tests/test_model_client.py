import io
import json
import os
import subprocess
import sys
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

    def reply(self, content, finish_reason="stop"):
        return io.BytesIO(json.dumps({"choices": [
            {"finish_reason": finish_reason, "message": {"content": content}}
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
    def test_bad_schema_or_truncated_completion_is_recoverable(self, opener):
        for content, reason in [('{"label":"unknown"}', "stop"), ('{"label":', "stop"), ('{"label":"on_task"}', "length")]:
            with self.subTest(reason=reason):
                opener.return_value.open.return_value = self.reply(content, reason)
                with self.assertRaises(ModelResponseInvalid) as exc:
                    chat_json([], self.schema)
                self.assertEqual(isinstance(exc.exception, ModelResponseTruncated), reason == "length")

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
