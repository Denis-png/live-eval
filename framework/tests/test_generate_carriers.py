import unittest

from framework.generators.base_generator import BaseGenerator


class FakeGenerator(BaseGenerator):
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def call_api(self, prompt):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


PROMPT = "Write a sentence.\n\n{spec}\n\nSentence: <text>"


class GenerateCarriersTests(unittest.TestCase):
    def test_one_call_per_spec_and_tagged_parse(self):
        gen = FakeGenerator(["Sentence: The train arrives at six.",
                             "Sentence: We walked to the old harbour."])
        carriers = gen.generate_carriers(
            ["topic: travel; roughly 6 words", "topic: travel; roughly 7 words"],
            PROMPT, "Sentence",
        )
        self.assertEqual(carriers, ["The train arrives at six.",
                                    "We walked to the old harbour."])
        self.assertEqual(len(gen.prompts), 2)
        self.assertIn("roughly 6 words", gen.prompts[0])

    def test_refusal_and_parse_failure_are_skipped(self):
        gen = FakeGenerator(["I'm sorry, I can't help with that.",
                             "line one\nline two",
                             "Sentence: A quiet morning by the river."])
        carriers = gen.generate_carriers(["a", "b", "c"], PROMPT, "Sentence")
        self.assertEqual(carriers, ["A quiet morning by the river."])

    def test_api_exception_skips_only_that_spec(self):
        gen = FakeGenerator([RuntimeError("boom"), "Sentence: A short walk home."])
        carriers = gen.generate_carriers(["a", "b"], PROMPT, "Sentence")
        self.assertEqual(carriers, ["A short walk home."])

    def test_too_short_carrier_rejected(self):
        gen = FakeGenerator(["Sentence: Hi there", "Sentence: The bus was very late today."])
        carriers = gen.generate_carriers(["a", "b"], PROMPT, "Sentence")
        self.assertEqual(carriers, ["The bus was very late today."])

    def test_empty_specs_returns_empty(self):
        gen = FakeGenerator([])
        self.assertEqual(gen.generate_carriers([], PROMPT, "Sentence"), [])


if __name__ == "__main__":
    unittest.main()
