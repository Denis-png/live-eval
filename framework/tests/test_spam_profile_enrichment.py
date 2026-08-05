import json
import unittest
from unittest import mock

from framework.profiling.spam_profiler import profile_spam_dataset

ROWS = [
    {"text": "WIN a free prize now!", "label": "SPAM"},
    {"text": "are we still on for lunch?", "label": "HAM"},
    {"text": "i'll be there at 5", "label": "HAM"},
]

LABELS_RESPONSE = "1: chat\n2: chat\n3: chat"
CONSOLIDATION = json.dumps(
    {"chat": {"description": "Everyday chat.", "members": ["chat"]}}
)


class FakeCallApi:
    def __init__(self, responses):
        self.responses = list(responses)

    def __call__(self, prompt):
        return self.responses.pop(0)


@mock.patch(
    "framework.profiling.spam_profiler.load_spam_rows", return_value=ROWS
)
class SpamProfileEnrichmentTests(unittest.TestCase):
    def test_per_label_characteristic_blocks(self, _mock_rows):
        profile = profile_spam_dataset()
        self.assertEqual(profile["profile_version"], 2)
        for label in ("HAM", "SPAM"):
            words = profile["length_distributions_per_label"][label]["words"]
            self.assertAlmostEqual(sum(words["bins"].values()), 1.0, places=3)
            self.assertIn("question_rate", profile["style_per_label"][label])
            self.assertIn("type_token_ratio", profile["vocabulary_per_label"][label])
        self.assertAlmostEqual(
            profile["style_per_label"]["SPAM"]["uppercase_word_rate"], 1.0
        )
        self.assertNotIn("topics_per_label", profile)  # no LLM injected

    def test_v1_keys_unchanged(self, _mock_rows):
        profile = profile_spam_dataset()
        for key in (
            "num_samples", "label_distribution", "label_percentages",
            "text_length_stats_per_label", "word_count_stats_per_label",
            "top_frequent_words_per_label", "examples_per_label", "spam_signals",
        ):
            self.assertIn(key, profile)

    def test_topics_per_label_when_call_api_given(self, _mock_rows):
        # Labels are requested per label, HAM first (sorted): HAM has 2 texts,
        # SPAM has 1. Each pass is one label call + one consolidation call.
        call = FakeCallApi(
            ["1: chat\n2: chat", CONSOLIDATION,
             "1: promo", json.dumps({"promo": {"description": "", "members": ["promo"]}})]
        )
        profile = profile_spam_dataset(topic_call_api=call, topic_sample_size=10)
        self.assertIn("HAM", profile["topics_per_label"])
        self.assertIn("SPAM", profile["topics_per_label"])
        self.assertEqual(profile["topics_per_label"]["SPAM"]["n_sampled"], 1)


if __name__ == "__main__":
    unittest.main()
