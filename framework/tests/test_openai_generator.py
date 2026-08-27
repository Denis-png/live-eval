import unittest
from types import SimpleNamespace
from unittest import mock

from framework.generators.openai_generator import OpenAIGenerator


def _response(content="ok", **overrides):
    message_fields = {"content": content}
    message_fields.update(overrides.pop("message", {}))
    choice_fields = {
        "message": SimpleNamespace(**message_fields),
        "finish_reason": overrides.pop("finish_reason", "stop"),
    }
    choice_fields.update(overrides.pop("choice", {}))
    response_fields = {
        "id": overrides.pop("response_id", "resp_1"),
        "model": overrides.pop("response_model", "model-name"),
        "choices": [SimpleNamespace(**choice_fields)],
    }
    if "usage" in overrides:
        response_fields["usage"] = SimpleNamespace(**overrides.pop("usage"))
    response_fields.update(overrides)
    return SimpleNamespace(**response_fields)


class OpenAIGeneratorTests(unittest.TestCase):
    def test_passes_max_tokens_when_configured(self):
        with mock.patch("framework.generators.openai_generator.OpenAI") as openai:
            client = openai.return_value
            client.chat.completions.create.return_value = _response()
            gen = OpenAIGenerator({
                "provider": "openrouter",
                "model": "m",
                "api_key": "k",
                "temperature": 0,
                "max_tokens": 123,
            })
            self.assertEqual(gen.call_api("hello"), "ok")

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["max_tokens"], 123)

    def test_omits_max_tokens_when_not_configured(self):
        with mock.patch("framework.generators.openai_generator.OpenAI") as openai:
            client = openai.return_value
            client.chat.completions.create.return_value = _response()
            gen = OpenAIGenerator({
                "provider": "openrouter",
                "model": "m",
                "api_key": "k",
                "temperature": 0,
            })
            gen.call_api("hello")

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertNotIn("max_tokens", kwargs)

    def test_normal_string_content_leaves_response_diagnostic_empty(self):
        with mock.patch("framework.generators.openai_generator.OpenAI") as openai:
            client = openai.return_value
            client.chat.completions.create.return_value = _response("hello")
            gen = OpenAIGenerator({
                "provider": "openrouter",
                "model": "m",
                "api_key": "k",
                "temperature": 0,
            })
            self.assertEqual(gen.call_api("prompt"), "hello")

        self.assertIsNone(gen.last_response_diagnostic)

    def test_none_content_records_safe_response_shape(self):
        with mock.patch("framework.generators.openai_generator.OpenAI") as openai:
            client = openai.return_value
            client.chat.completions.create.return_value = _response(
                None,
                finish_reason="length",
                response_id="resp_123",
                response_model="xiaomi/mimo-v2.5",
                usage={"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33},
            )
            gen = OpenAIGenerator({
                "provider": "openrouter",
                "model": "m",
                "api_key": "k",
                "temperature": 0,
            })
            self.assertIsNone(gen.call_api("prompt"))

        self.assertEqual(gen.last_response_diagnostic["finish_reason"], "length")
        self.assertEqual(gen.last_response_diagnostic["response_id"], "resp_123")
        self.assertEqual(gen.last_response_diagnostic["response_model"], "xiaomi/mimo-v2.5")
        self.assertEqual(gen.last_response_diagnostic["usage"]["prompt_tokens"], 11)
        self.assertEqual(gen.last_response_diagnostic["usage"]["completion_tokens"], 22)
        self.assertEqual(gen.last_response_diagnostic["usage"]["total_tokens"], 33)

    def test_reasoning_shape_is_recorded_without_reasoning_text(self):
        secret = "hidden chain of thought"
        with mock.patch("framework.generators.openai_generator.OpenAI") as openai:
            openai.return_value.chat.completions.create.return_value = _response(
                None,
                message={
                    "reasoning": secret,
                    "reasoning_details": [
                        {"type": "reasoning_text", "text": secret},
                        SimpleNamespace(text=secret),
                    ],
                },
            )
            gen = OpenAIGenerator({
                "provider": "openrouter",
                "model": "m",
                "api_key": "k",
                "temperature": 0,
            })
            gen.call_api("prompt")

        diagnostic = gen.last_response_diagnostic
        self.assertTrue(diagnostic["reasoning_present"])
        self.assertEqual(diagnostic["reasoning_length"], len(secret))
        self.assertTrue(diagnostic["reasoning_details_present"])
        self.assertEqual(diagnostic["reasoning_details_count"], 2)
        self.assertEqual(diagnostic["reasoning_details_types"], ["reasoning_text", "SimpleNamespace"])
        self.assertNotIn(secret, str(diagnostic))

    def test_tool_calls_refusal_and_function_call_are_recorded_safely(self):
        with mock.patch("framework.generators.openai_generator.OpenAI") as openai:
            openai.return_value.chat.completions.create.return_value = _response(
                None,
                message={
                    "tool_calls": [object(), object()],
                    "function_call": object(),
                    "refusal": "No thanks",
                },
            )
            gen = OpenAIGenerator({
                "provider": "openrouter",
                "model": "m",
                "api_key": "k",
                "temperature": 0,
            })
            gen.call_api("prompt")

        diagnostic = gen.last_response_diagnostic
        self.assertEqual(diagnostic["tool_calls_count"], 2)
        self.assertTrue(diagnostic["function_call_present"])
        self.assertTrue(diagnostic["refusal_present"])
        self.assertEqual(diagnostic["refusal_length"], len("No thanks"))
        self.assertEqual(diagnostic["refusal_preview"], "No thanks")

    def test_absent_optional_fields_do_not_crash(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )
        with mock.patch("framework.generators.openai_generator.OpenAI") as openai:
            openai.return_value.chat.completions.create.return_value = response
            gen = OpenAIGenerator({
                "provider": "openrouter",
                "model": "m",
                "api_key": "k",
                "temperature": 0,
            })
            self.assertIsNone(gen.call_api("prompt"))

        self.assertEqual(gen.last_response_diagnostic, {})


if __name__ == "__main__":
    unittest.main()
