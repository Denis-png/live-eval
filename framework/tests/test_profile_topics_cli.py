import unittest
from unittest import mock

from framework.profile_dataset import _build_topic_call


class FakeGenerator:
    def __init__(self, config):
        self.config = config

    def call_api(self, prompt):
        return "ok"


class BuildTopicCallTests(unittest.TestCase):
    def _config(self, **overrides):
        config = {
            "api_keys": {"openrouter": "key-or", "groq": "key-groq"},
            "generation": {"provider": "groq", "model": "gen-model"},
            "profiling": {"provider": "openrouter", "model": "prof-model"},
        }
        config.update(overrides)
        return config

    @mock.patch("framework.pipeline.load_generator")
    def test_uses_profiling_block_with_resolved_key(self, load_gen):
        load_gen.side_effect = FakeGenerator
        call = _build_topic_call(self._config())
        cfg = load_gen.call_args.args[0]
        self.assertEqual(cfg["provider"], "openrouter")
        self.assertEqual(cfg["model"], "prof-model")
        self.assertEqual(cfg["api_key"], "key-or")
        self.assertEqual(call("hi"), "ok")

    @mock.patch("framework.pipeline.load_generator")
    def test_falls_back_to_generation_block_with_warning(self, load_gen):
        load_gen.side_effect = FakeGenerator
        _build_topic_call(self._config(profiling={}))
        cfg = load_gen.call_args.args[0]
        self.assertEqual(cfg["provider"], "groq")
        self.assertEqual(cfg["model"], "gen-model")
        self.assertEqual(cfg["api_key"], "key-groq")

    def test_neither_block_configured_exits(self):
        with self.assertRaises(SystemExit):
            _build_topic_call({"api_keys": {}})

    @mock.patch("framework.pipeline.load_generator")
    def test_missing_api_key_exits(self, load_gen):
        config = self._config(api_keys={})
        with self.assertRaises(SystemExit):
            _build_topic_call(config)

    @mock.patch("framework.pipeline.load_generator")
    def test_explicit_api_key_preserved(self, load_gen):
        load_gen.side_effect = FakeGenerator
        config = self._config()
        config["profiling"]["api_key"] = "explicit"
        _build_topic_call(config)
        self.assertEqual(load_gen.call_args.args[0]["api_key"], "explicit")


if __name__ == "__main__":
    unittest.main()
