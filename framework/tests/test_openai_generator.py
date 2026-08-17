import unittest
from unittest import mock

from framework.generators.openai_generator import OpenAIGenerator


class _Message:
    content = "ok"


class _Choice:
    message = _Message()


class _Response:
    choices = [_Choice()]


class OpenAIGeneratorTests(unittest.TestCase):
    def test_passes_max_tokens_when_configured(self):
        with mock.patch("framework.generators.openai_generator.OpenAI") as openai:
            client = openai.return_value
            client.chat.completions.create.return_value = _Response()
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
            client.chat.completions.create.return_value = _Response()
            gen = OpenAIGenerator({
                "provider": "openrouter",
                "model": "m",
                "api_key": "k",
                "temperature": 0,
            })
            gen.call_api("hello")

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertNotIn("max_tokens", kwargs)


if __name__ == "__main__":
    unittest.main()
