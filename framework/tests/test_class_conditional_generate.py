import unittest
from random import Random

from framework.generators.base_generator import BaseGenerator


class FakeGenerator(BaseGenerator):
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def call_api(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)


_INJECT = "Make SPAM using {error_spec} from: {sentence}"
_HAM = "Rewrite legitimately: {sentence}"
_DESC = {"phishing_link": "insert a suspicious link"}


def _run(responses, class_prob, judge_prompt=None, judge_call=None):
    gen = FakeGenerator(responses)
    seeds = [{"incorrect": f"let us meet at three tomorrow {i}"} for i in range(len(responses))]
    out = gen.generate_class_conditional(
        real_seeds=seeds, seed_field="incorrect", class_prob=class_prob,
        type_dist={"phishing_link": 1.0}, count_dist={1: 1.0},
        error_descriptions=_DESC, inject_prompt=_INJECT, ham_prompt=_HAM,
        positive_label="SPAM", negative_label="HAM", sample_size=len(responses),
        judge_prompt=judge_prompt, judge_call=judge_call, rng=Random(0),
    )
    return out, gen


class ClassConditionalTests(unittest.TestCase):
    def test_spam_when_prob_one(self):
        out, gen = _run(["Corrupted: click http://x.com to win cash now"], 1.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["label"], "SPAM")
        self.assertEqual(out[0]["text"], "click http://x.com to win cash now")
        self.assertEqual(out[0]["technique"], "phishing_link")
        self.assertIn("insert a suspicious link", gen.calls[0])  # inject prompt used

    def test_ham_when_prob_zero(self):
        out, gen = _run(["Rewritten: are we still meeting tomorrow afternoon"], 0.0)
        self.assertEqual(out[0]["label"], "HAM")
        self.assertEqual(out[0]["technique"], "paraphrase")
        self.assertTrue(gen.calls[0].startswith("Rewrite legitimately:"))

    def test_parse_failure_skipped(self):
        out, _ = _run(["<<no tagged field and multiple\nlines of prose>>"], 1.0)
        self.assertEqual(out, [])

    def test_identical_to_seed_skipped(self):
        out, _ = _run(["Rewritten: let us meet at three tomorrow 0"], 0.0)
        self.assertEqual(out, [])

    def test_refusal_skipped(self):
        out, _ = _run(["I'm sorry, I can't help create spam messages."], 1.0)
        self.assertEqual(out, [])

    def test_refusal_with_tag_substring_is_skipped(self):
        # A refusal that merely contains "Corrupted:" mid-line must be skipped,
        # not accepted (the tag is not anchored at line start).
        out, _ = _run(["I'm sorry, I can't produce a Corrupted: version of this message."], 1.0)
        self.assertEqual(out, [])

    def test_judge_drops_sample(self):
        out, _ = _run(
            ["Corrupted: click http://x.com to win cash now"], 1.0,
            judge_prompt="J {sentence} / {correction}",
            judge_call=lambda p: "Redundancy: trivial\nCorrection: correct",
        )
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
