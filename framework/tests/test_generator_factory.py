import unittest
import types
from unittest import mock

from framework.generators.factory import load_generator


class GeneratorFactoryTests(unittest.TestCase):
    def test_openrouter_uses_openai_compatible_generator(self):
        calls = []

        class FakeOpenAI:
            def __init__(self, config):
                calls.append(config)

        module = types.SimpleNamespace(OpenAIGenerator=FakeOpenAI)
        cfg = {"provider": "openrouter", "model": "m", "api_key": "k", "temperature": 0}
        with mock.patch.dict("sys.modules", {"framework.generators.openai_generator": module}):
            load_generator(cfg)
        self.assertEqual(calls, [cfg])

    def test_minimax_gets_anthropic_compatible_base_url(self):
        calls = []

        class FakeAnthropic:
            def __init__(self, config):
                calls.append(config)

        module = types.SimpleNamespace(AnthropicGenerator=FakeAnthropic)
        cfg = {"provider": "minimax", "model": "m", "api_key": "k",
               "temperature": 0, "max_tokens": 10}
        with mock.patch.dict("sys.modules", {"framework.generators.anthropic_generator": module}):
            load_generator(cfg)
        self.assertEqual(calls[0]["base_url"], "https://api.minimax.io/anthropic")

    def test_google_uses_google_generator(self):
        calls = []

        class FakeGoogle:
            def __init__(self, config):
                calls.append(config)

        module = types.SimpleNamespace(GoogleGenerator=FakeGoogle)
        cfg = {"provider": "google", "model": "m", "api_key": "k", "temperature": 0}
        with mock.patch.dict("sys.modules", {"framework.generators.google_generator": module}):
            load_generator(cfg)
        self.assertEqual(calls, [cfg])

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            load_generator({"provider": "unknown"})


if __name__ == "__main__":
    unittest.main()
