import io
import json
import unittest
from unittest.mock import patch

from model_client import API_URL, MODEL_ID, ModelUnavailable, chat_json


class ModelClientTests(unittest.TestCase):
    schema = {
        "type": "object",
        "required": ["label"],
        "properties": {"label": {"enum": ["on_task", "off_task", "uncertain"]}},
        "additionalProperties": False,
    }

    def reply(self, content, finish_reason="stop"):
        return io.BytesIO(json.dumps({"choices": [
            {"finish_reason": finish_reason, "message": {"content": content}}
        ]}).encode())

    @patch("model_client.build_opener")
    def test_local_schema_request(self, opener):
        opener.return_value.open.return_value = self.reply('{"label":"on_task"}')
        self.assertEqual(chat_json([], self.schema), {"label": "on_task"})
        request = opener.return_value.open.call_args.args[0]
        self.assertEqual(request.full_url, API_URL)
        body = json.loads(request.data)
        self.assertEqual(body["model"], MODEL_ID)
        self.assertEqual(body["response_format"]["json_schema"]["schema"], self.schema)

    @patch("model_client.build_opener")
    def test_bad_schema_or_truncated_completion_is_recoverable(self, opener):
        for content, reason in [('{"label":"unknown"}', "stop"), ('{"label":"on_task"}', "length")]:
            with self.subTest(reason=reason):
                opener.return_value.open.return_value = self.reply(content, reason)
                with self.assertRaises(ModelUnavailable):
                    chat_json([], self.schema)
