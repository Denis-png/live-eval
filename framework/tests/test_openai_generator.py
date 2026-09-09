import types
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
        # finish_reason is deliberately NOT "length" here: this test covers the
        # no-usable-content diagnostic path, and a length stop is now a
        # TruncatedResponse (see TruncationGuardTests). Using "length" would
        # conflate the two.
        with mock.patch("framework.generators.openai_generator.OpenAI") as openai:
            client = openai.return_value
            client.chat.completions.create.return_value = _response(
                None,
                finish_reason="content_filter",
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

        self.assertEqual(gen.last_response_diagnostic["finish_reason"], "content_filter")
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


class TruncationGuardTests(unittest.TestCase):
    """A max_tokens stop must never be parsed.

    A truncated reasoning-model response often still carries the answer field
    names with a half-written value after the last one, so parsing it yields a
    sample whose gold reference is a fragment. That silently corrupts the
    benchmark, where a skip merely costs a sample.
    """

    def _generator(self, finish_reason, content="Generated: x\nGround truth: y"):
        from framework.generators.openai_generator import OpenAIGenerator

        gen = OpenAIGenerator.__new__(OpenAIGenerator)
        gen.model, gen.temperature, gen.max_tokens = "m", 1.0, 1024
        gen.last_response_diagnostic = None

        choice = types.SimpleNamespace(
            finish_reason=finish_reason,
            message=types.SimpleNamespace(content=content),
        )
        response = types.SimpleNamespace(choices=[choice])
        gen.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kw: response)
            )
        )
        return gen

    def test_length_finish_raises_rather_than_returning_partial_text(self):
        from framework.generators.base_generator import TruncatedResponse

        with self.assertRaises(TruncatedResponse) as ctx:
            self._generator("length").call_api("p")
        # The message must name the knob to raise, since that is the fix.
        self.assertIn("max_tokens=1024", str(ctx.exception))

    def test_truncated_response_is_not_parsed_even_when_fields_are_present(self):
        # The silent-corruption case: all three fields present, but the last
        # value is a fragment. Before the guard this was ACCEPTED.
        from framework.generators.base_generator import TruncatedResponse

        raw = "Error type: article\nGenerated: I am surprise to hear\nGround truth: I am"
        with self.assertRaises(TruncatedResponse):
            self._generator("length", raw).call_api("p")

    def test_normal_stop_is_unaffected(self):
        self.assertEqual(
            self._generator("stop", "Corrupted: hello there").call_api("p"),
            "Corrupted: hello there",
        )

    def test_missing_finish_reason_is_not_treated_as_truncation(self):
        self.assertEqual(
            self._generator(None, "Corrupted: hello there").call_api("p"),
            "Corrupted: hello there",
        )


class TruncationSkipsLoudlyTests(unittest.TestCase):
    """The guard must degrade to a counted skip, not kill the run."""

    def test_generation_loop_survives_a_truncated_response(self):
        from framework.generators.base_generator import BaseGenerator, TruncatedResponse

        class _Truncating(BaseGenerator):
            def __init__(self):
                self.calls = 0

            def call_api(self, prompt):
                self.calls += 1
                if self.calls == 1:
                    raise TruncatedResponse("response truncated at max_tokens=1024")
                return ("Error type: article\n"
                        "Generated: I go to the school yesterday\n"
                        "Ground truth: I went to school yesterday.")

        gen = _Truncating()
        out = gen.generate_forward(
            real_samples=[{"incorrect": "a b c"}, {"incorrect": "d e f"}],
            error_types=["article"],
            prompt_instruction="Corrupt: {sentence} ({error_type})",
            sample_size=2,
        )
        # First sample skipped, second kept — the run continues.
        self.assertEqual(len(out), 1)
        self.assertEqual(gen.calls, 2)


class TruncationDiagnosticTests(unittest.TestCase):
    def test_truncation_records_the_provider_diagnostic_before_raising(self):
        """Raising must not cost the observability the None-content path gives."""
        from framework.generators.base_generator import TruncatedResponse

        with mock.patch("framework.generators.openai_generator.OpenAI") as openai:
            openai.return_value.chat.completions.create.return_value = _response(
                None,
                finish_reason="length",
                response_id="resp_9",
                response_model="minimax-m3",
                usage={"prompt_tokens": 11, "completion_tokens": 1024,
                       "total_tokens": 1035},
            )
            gen = OpenAIGenerator({"provider": "openrouter", "model": "m",
                                   "api_key": "k", "temperature": 0,
                                   "max_tokens": 1024})
            with self.assertRaises(TruncatedResponse):
                gen.call_api("prompt")
        self.assertEqual(gen.last_response_diagnostic["finish_reason"], "length")
