import json
import os
import random
import tempfile
import unittest

from framework.profiling.spec_sampler import (
    load_profile,
    render_spec,
    sample_content_spec,
)

GEC_PROFILE = {
    "profile_version": 2,
    "topics": {
        "travel": {"fraction": 0.75, "description": "Trips and holidays.", "examples": []},
        "school": {"fraction": 0.25, "description": "Classes and study.", "examples": []},
    },
    "length_distributions": {
        "correct": {"words": {"bins": {"1-5": 0.0, "6-10": 1.0, "11-15": 0.0,
                                       "16-20": 0.0, "21-30": 0.0, "31-50": 0.0,
                                       "51+": 0.0},
                              "quantiles": {"p10": 6, "p25": 7, "p50": 8,
                                            "p75": 9, "p90": 10},
                              "count": 10}},
        "incorrect": {"words": {"bins": {"1-5": 1.0, "6-10": 0.0, "11-15": 0.0,
                                         "16-20": 0.0, "21-30": 0.0, "31-50": 0.0,
                                         "51+": 0.0},
                                "quantiles": {"p10": 2, "p25": 3, "p50": 4,
                                              "p75": 5, "p90": 5},
                                "count": 10}},
    },
    "style": {
        "correct": {"question_rate": 1.0, "exclaim_rate": 0.0,
                    "first_person_rate": 0.0, "second_person_rate": 0.0,
                    "contraction_rate": 0.0, "digit_rate": 0.0,
                    "uppercase_word_rate": 0.0, "texting_slang_rate": 0.0,
                    "punctuation_density": 0.1},
        "incorrect": {"question_rate": 0.0, "exclaim_rate": 0.0,
                      "first_person_rate": 0.0, "second_person_rate": 0.0,
                      "contraction_rate": 0.0, "digit_rate": 0.0,
                      "uppercase_word_rate": 0.0, "texting_slang_rate": 0.0,
                      "punctuation_density": 0.1},
    },
}

SPAM_PROFILE = {
    "profile_version": 2,
    "topics_per_label": {
        "HAM": {"topics": {"chat": {"fraction": 1.0, "description": "Everyday chat.",
                                    "examples": []}}},
        "SPAM": {"topics": {"prizes": {"fraction": 1.0, "description": "Prize scams.",
                                       "examples": []}}},
    },
    "length_distributions_per_label": {
        "HAM": {"words": {"bins": {"6-10": 1.0}, "quantiles": {"p90": 9}, "count": 5}},
        "SPAM": {"words": {"bins": {"51+": 1.0}, "quantiles": {"p90": 60}, "count": 5}},
    },
    "style_per_label": {
        "HAM": {"question_rate": 0.0, "exclaim_rate": 0.0},
        "SPAM": {"question_rate": 0.0, "exclaim_rate": 1.0},
    },
}


def _write(profile):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(profile, f)
    return path


class LoadProfileTests(unittest.TestCase):
    def test_missing_file_names_the_profiling_command(self):
        with self.assertRaises(RuntimeError) as ctx:
            load_profile("/nonexistent/gec_profile.json")
        self.assertIn("profile_dataset", str(ctx.exception))
        self.assertIn("--topics", str(ctx.exception))

    def test_old_profile_version_rejected(self):
        path = _write({"topics": {"a": {"fraction": 1.0}}})
        try:
            with self.assertRaises(RuntimeError) as ctx:
                load_profile(path)
            self.assertIn("profile_version", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_missing_topics_block_rejected(self):
        path = _write({"profile_version": 2})
        try:
            with self.assertRaises(RuntimeError) as ctx:
                load_profile(path)
            self.assertIn("--topics", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_valid_profile_loads(self):
        path = _write(GEC_PROFILE)
        try:
            self.assertEqual(load_profile(path)["profile_version"], 2)
        finally:
            os.unlink(path)

    def test_spam_profile_validated_against_its_topics_key(self):
        path = _write(SPAM_PROFILE)
        try:
            profile = load_profile(path, topics_key="topics_per_label")
            self.assertIn("SPAM", profile["topics_per_label"])
        finally:
            os.unlink(path)


class SampleContentSpecTests(unittest.TestCase):
    def test_length_falls_inside_the_drawn_bin(self):
        rng = random.Random(0)
        for _ in range(20):
            spec = sample_content_spec(GEC_PROFILE, rng, side="correct")
            self.assertGreaterEqual(spec["target_words"], 6)
            self.assertLessEqual(spec["target_words"], 10)

    def test_side_selects_the_right_length_block(self):
        rng = random.Random(0)
        spec = sample_content_spec(GEC_PROFILE, rng, side="incorrect")
        self.assertLessEqual(spec["target_words"], 5)

    def test_open_ended_bin_uses_p90_as_upper_bound(self):
        rng = random.Random(0)
        spec = sample_content_spec(SPAM_PROFILE, rng, label="SPAM")
        self.assertGreaterEqual(spec["target_words"], 51)
        self.assertLessEqual(spec["target_words"], 60)

    def test_topic_drawn_with_its_description(self):
        rng = random.Random(1)
        topics = {sample_content_spec(GEC_PROFILE, rng)["topic"] for _ in range(50)}
        self.assertEqual(topics, {"travel", "school"})
        spec = sample_content_spec(SPAM_PROFILE, rng, label="HAM")
        self.assertEqual(spec["topic"], "chat")
        self.assertEqual(spec["topic_description"], "Everyday chat.")

    def test_style_features_fire_at_their_rates(self):
        rng = random.Random(2)
        spec = sample_content_spec(GEC_PROFILE, rng, side="correct")
        self.assertIn("question_rate", spec["style_features"])
        self.assertNotIn("exclaim_rate", spec["style_features"])

    def test_punctuation_density_is_never_a_style_feature(self):
        rng = random.Random(3)
        for _ in range(10):
            spec = sample_content_spec(GEC_PROFILE, rng, side="correct")
            self.assertNotIn("punctuation_density", spec["style_features"])

    def test_label_selects_per_label_blocks(self):
        rng = random.Random(4)
        spec = sample_content_spec(SPAM_PROFILE, rng, label="SPAM")
        self.assertEqual(spec["topic"], "prizes")
        self.assertIn("exclaim_rate", spec["style_features"])

    def test_all_zero_length_bins_raise(self):
        profile = json.loads(json.dumps(GEC_PROFILE))
        profile["length_distributions"]["correct"]["words"]["bins"] = {"1-5": 0.0}
        with self.assertRaises(RuntimeError):
            sample_content_spec(profile, random.Random(0), side="correct")


class RenderSpecTests(unittest.TestCase):
    def test_includes_topic_length_and_fired_features_only(self):
        text = render_spec({
            "topic": "travel", "topic_description": "Trips and holidays.",
            "target_words": 12, "style_features": ["question_rate"],
        })
        self.assertIn("travel", text)
        self.assertIn("Trips and holidays.", text)
        self.assertIn("12", text)
        self.assertIn("question", text)
        self.assertNotIn("exclamation", text)

    def test_no_style_features_still_renders_topic_and_length(self):
        text = render_spec({"topic": "school", "topic_description": "",
                            "target_words": 8, "style_features": []})
        self.assertIn("school", text)
        self.assertIn("8", text)


if __name__ == "__main__":
    unittest.main()
