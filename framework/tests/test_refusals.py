import unittest

from framework.generators.base_generator import BaseGenerator, _looks_like_refusal


class FakeGenerator(BaseGenerator):
    def __init__(self, responses):
        self._responses = list(responses)

    def call_api(self, prompt: str) -> str:
        return self._responses.pop(0)


_REAL = [{"incorrect": "See you at lunch tomorrow.", "correct": "See you at lunch tomorrow."}]
_DIST = {"type_dist": {"phishing_link": 1.0}, "count_dist": {1: 1.0}}


class LooksLikeRefusalTests(unittest.TestCase):
    def test_common_refusal_phrasings(self):
        for text in [
            "I can't assist with creating spam content.",
            "I cannot help with generating spam messages.",
            "I'm sorry, but I am unable to produce that.",
            "Sorry, I won't create content designed to deceive people.",
        ]:
            self.assertTrue(_looks_like_refusal(text), text)

    def test_normal_generation_output_is_not_a_refusal(self):
        self.assertFalse(_looks_like_refusal("WIN A FREE PRIZE NOW http://x.com"))
        self.assertFalse(_looks_like_refusal(
            "Corrupted: CLICK here to claim your £500 reward!!!"
        ))


class InverseRefusalTests(unittest.TestCase):
    def test_single_line_refusal_is_skipped_not_accepted(self):
        """The bare single-line fallback must NOT turn a one-line refusal into
        a 'spam sample' — refusal detection runs before the fallback."""
        gen = FakeGenerator(["I can't assist with creating spam content."])
        out = gen.generate_inverse(
            real_samples=_REAL,
            inverse_prompt="Errors: {error_spec}\nMessage: {sentence}",
            error_descriptions={"phishing_link": "insert a link"},
            type_dist=_DIST["type_dist"],
            count_dist=_DIST["count_dist"],
            sample_size=5,
            source_field="correct",
        )
        self.assertEqual(out, [])

    def test_labeled_corrupted_line_wins_over_refusal_words(self):
        # An explicit Corrupted: field is a real answer even if the message
        # itself happens to contain apology-like words.
        gen = FakeGenerator(["Corrupted: I'm sorry but you MUST claim your FREE prize now http://x.co"])
        out = gen.generate_inverse(
            real_samples=_REAL,
            inverse_prompt="Errors: {error_spec}\nMessage: {sentence}",
            error_descriptions={"phishing_link": "insert a link"},
            type_dist=_DIST["type_dist"],
            count_dist=_DIST["count_dist"],
            sample_size=5,
            source_field="correct",
        )
        self.assertEqual(len(out), 1)


class ForwardRefusalTests(unittest.TestCase):
    def test_refusal_is_skipped_in_forward_mode(self):
        gen = FakeGenerator(["I cannot help with generating spam or deceptive messages."])
        out = gen.generate(
            real_samples=_REAL,
            error_types=["phishing_link"],
            prompt_instruction="Rewrite: {sentence}",
            sample_size=5,
        )
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
