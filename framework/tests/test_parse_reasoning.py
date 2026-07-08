"""Parsing must survive reasoning-model output (e.g. minimax-m3), which wraps a
<think> chain-of-thought and glues the answer onto it without a line break or a
closing </think>. Regression for the "every sample parse failed / refused" bug."""
import unittest
from random import Random

from framework.generators.base_generator import (
    BaseGenerator,
    _looks_like_refusal,
    _parse_generation,
    _parse_tagged,
)

# Real captured minimax-m3 responses (answer glued to the reasoning, no </think>):
_SPAM_RAW = (
    "<think>\nThe user wants me to transform a legitimate message by inserting a "
    "suspicious link/URL and promising money. Let me pick something that flows "
    "naturally.Corrupted: Neva mind it's ok.. claim your $500 reward now at "
    "http://claim-prize-fast.xyz/win"
)
_HAM_RAW = (
    "<think>\nThe user wants me to rewrite the message as a natural, legitimate "
    "message. Just a simple, casual rephrase.Rewritten: No worries, it's all good."
)


class ParseReasoningTests(unittest.TestCase):
    def test_spam_answer_extracted_from_reasoning(self):
        self.assertEqual(
            _parse_tagged(_SPAM_RAW, "Corrupted"),
            "Neva mind it's ok.. claim your $500 reward now at http://claim-prize-fast.xyz/win",
        )

    def test_ham_answer_extracted_from_reasoning(self):
        self.assertEqual(_parse_tagged(_HAM_RAW, "Rewritten"), "No worries, it's all good.")

    def test_closed_think_block_stripped(self):
        raw = "<think>reasoning here</think>\nCorrupted: buy now at http://x.com cheap"
        self.assertEqual(_parse_tagged(raw, "Corrupted"), "buy now at http://x.com cheap")

    def test_plain_anchored_still_works(self):
        self.assertEqual(_parse_tagged("Corrupted: hello there friend", "Corrupted"), "hello there friend")

    def test_reasoning_not_treated_as_refusal(self):
        raw = "<think>\nI can't be too obvious here.Corrupted: WIN cash http://x.com now"
        self.assertFalse(_looks_like_refusal(raw))

    def test_real_refusal_still_detected(self):
        self.assertTrue(_looks_like_refusal("I'm sorry, I can't help with creating spam."))


class _ReasoningGen(BaseGenerator):
    def __init__(self, responses):
        self._responses = list(responses)

    def call_api(self, prompt: str) -> str:
        return self._responses.pop(0)


class GenerateClassConditionalReasoningTests(unittest.TestCase):
    def test_reasoning_spam_kept(self):
        gen = _ReasoningGen([_SPAM_RAW])
        out = gen.generate_class_conditional(
            real_seeds=[{"incorrect": "neva mind it's ok"}], seed_field="incorrect",
            class_prob=1.0, type_dist={"phishing_link": 1.0}, count_dist={1: 1.0},
            error_descriptions={"phishing_link": "add a link"},
            inject_prompt="{sentence} {error_spec}", ham_prompt="{sentence}",
            positive_label="SPAM", negative_label="HAM", sample_size=1, rng=Random(0),
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["label"], "SPAM")
        self.assertIn("claim your $500", out[0]["text"])


# GEC forward path (3-field CoT response) from a reasoning model: the first field
# is glued to the end of the reasoning, the rest are on their own lines.
_GEC_FORWARD_RAW = (
    "<think>\nThe user wants me to corrupt the sentence by changing the verb "
    "tense. Let me write it out.Error type: verb tense\n"
    "Generated: She go to school yesterday.\n"
    "Ground truth: She went to school yesterday."
)


class ParseGenerationReasoningTests(unittest.TestCase):
    def test_forward_fields_extracted_from_reasoning(self):
        self.assertEqual(
            _parse_generation(_GEC_FORWARD_RAW),
            ("verb tense", "She go to school yesterday.", "She went to school yesterday."),
        )

    def test_forward_plain_still_works(self):
        raw = "Error type: tense\nGenerated: he go home\nGround truth: he goes home"
        self.assertEqual(_parse_generation(raw), ("tense", "he go home", "he goes home"))

    def test_forward_closed_think_stripped(self):
        raw = "<think>plan</think>\nError type: tense\nGenerated: he go home\nGround truth: he goes home"
        self.assertEqual(_parse_generation(raw), ("tense", "he go home", "he goes home"))


class GenerateForwardReasoningTests(unittest.TestCase):
    def test_reasoning_forward_kept(self):
        gen = _ReasoningGen([_GEC_FORWARD_RAW])
        out = gen.generate(
            real_samples=[{"incorrect": "she go to school yesterday"}],
            error_types=["verb tense"],
            prompt_instruction="fix {sentence} ({error_type})",
            sample_size=1,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["corrupted"], "She go to school yesterday.")
        self.assertEqual(out[0]["original"], "She went to school yesterday.")


if __name__ == "__main__":
    unittest.main()
