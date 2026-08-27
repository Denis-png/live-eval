import unittest

from framework.tasks.sentiment.task import SentimentTask


class SentimentConfigTests(unittest.TestCase):
    """The task config lives at configs/sentiment/sentiment.json, matching the
    per-task layout every other task uses."""

    def setUp(self):
        self.task = SentimentTask()

    def test_config_loads_with_required_keys(self):
        for key in ("error_types", "evaluators", "models", "prompt"):
            self.assertIn(key, self.task._config)

    def test_prompt_has_the_sentence_placeholder(self):
        self.assertIn("{sentence}", self.task.get_prompt_instruction())

    def test_every_declared_evaluator_has_a_function(self):
        fns = self.task.get_evaluator_fns()
        missing = [name for name in self.task.get_evaluators() if name not in fns]
        self.assertEqual(missing, [])

    def test_task_name(self):
        self.assertEqual(self.task.get_task_name(), "sentiment")

    def test_unsupported_model_type_names_the_config_file(self):
        with self.assertRaises(ValueError) as ctx:
            self.task.get_model({"type": "nope"})
        message = str(ctx.exception)
        self.assertIn("nope", message)
        self.assertIn("configs/sentiment/sentiment.json", message)


class SentimentLabelTests(unittest.TestCase):
    """get_label derives ground truth from the corruption technique, so a
    technique whose resulting sentiment is not determined must yield None
    rather than a guess."""

    def setUp(self):
        self.task = SentimentTask()

    def test_negative_techniques(self):
        for error_type in ("sentiment_flip_negative", "sarcasm_injection",
                           "negation_insertion"):
            self.assertEqual(self.task.get_label({"error_type": error_type}),
                             "NEGATIVE", error_type)

    def test_positive_technique(self):
        self.assertEqual(
            self.task.get_label({"error_type": "sentiment_flip_positive"}), "POSITIVE")

    def test_intensity_reduction_is_neutral(self):
        self.assertEqual(
            self.task.get_label({"error_type": "intensity_reduction"}), "NEUTRAL")

    def test_paraphrase_and_unknown_have_no_deterministic_label(self):
        self.assertIsNone(self.task.get_label({"error_type": "paraphrase"}))
        self.assertIsNone(self.task.get_label({"error_type": "something_else"}))
        self.assertIsNone(self.task.get_label({}))

    def test_every_configured_error_type_is_classified_or_deliberately_skipped(self):
        # Guards against a new error_type being added to the config without a
        # matching rule here — it would silently vanish from every benchmark.
        labelled = {e: self.task.get_label({"error_type": e})
                    for e in self.task.get_error_types()}
        unlabelled = [e for e, label in labelled.items() if label is None]
        self.assertEqual(unlabelled, ["paraphrase"], labelled)


class SentimentEvalSamplesTests(unittest.TestCase):
    def setUp(self):
        self.task = SentimentTask()

    def test_uses_corrupted_text_and_derived_label(self):
        samples = self.task.get_eval_samples([
            {"corrupted": "awful film", "original": "great film",
             "error_type": "sentiment_flip_negative"},
        ])
        self.assertEqual(samples, [{
            "corrupted": "awful film", "original": "great film",
            "error_type": "sentiment_flip_negative",
            "text": "awful film", "label": "NEGATIVE",
        }])

    def test_items_without_a_deterministic_label_are_dropped(self):
        samples = self.task.get_eval_samples([
            {"corrupted": "a", "error_type": "paraphrase"},
            {"corrupted": "b b b", "error_type": "sentiment_flip_positive"},
        ])
        self.assertEqual([s["label"] for s in samples], ["POSITIVE"])


class SentimentParseRowTests(unittest.TestCase):
    def setUp(self):
        self.task = SentimentTask()

    def test_numeric_labels_map_to_class_names(self):
        for raw, expected in ((0, "NEGATIVE"), (1, "NEUTRAL"), (2, "POSITIVE")):
            row = self.task.parse_row({"text": "x", "label": raw})
            self.assertEqual(row["sentiment_label"], expected)

    def test_accepts_alternative_text_fields(self):
        for field in ("text", "sentence", "review"):
            row = self.task.parse_row({field: "hello", "label": 1})
            self.assertEqual(row["incorrect"], "hello")

    def test_row_without_text_is_skipped(self):
        self.assertIsNone(self.task.parse_row({"label": 0}))
        self.assertIsNone(self.task.parse_row({"text": "", "label": 0}))

    def test_unmapped_label_passes_through_as_string(self):
        row = self.task.parse_row({"text": "x", "label": "POSITIVE"})
        self.assertEqual(row["sentiment_label"], "POSITIVE")


class SentimentRealEvalSamplesTests(unittest.TestCase):
    def test_incomplete_rows_are_filtered(self):
        task = SentimentTask()
        samples = task.get_real_eval_samples({}, [
            {"incorrect": "good movie", "sentiment_label": "POSITIVE"},
            {"incorrect": "", "sentiment_label": "NEGATIVE"},
            {"incorrect": "no label here"},
        ])
        self.assertEqual(samples, [{"text": "good movie", "label": "POSITIVE"}])



class SentimentMissingLabelTests(unittest.TestCase):
    """A row whose label cannot be resolved must be skipped, not stringified.
    Previously `str(None)` produced the class "None", which is truthy, so it
    survived get_real_eval_samples' filter and entered the real baseline as a
    class no model can predict — every model scored wrong on it and the
    baseline was silently depressed."""

    def setUp(self):
        self.task = SentimentTask()

    def test_absent_label_skips_the_row(self):
        self.assertIsNone(self.task.parse_row({"text": "a review"}))

    def test_explicit_none_label_skips_the_row(self):
        self.assertIsNone(self.task.parse_row({"text": "a review", "label": None}))

    def test_blank_label_skips_the_row(self):
        self.assertIsNone(self.task.parse_row({"text": "a review", "label": ""}))
        self.assertIsNone(self.task.parse_row({"text": "a review", "label": "   "}))

    def test_zero_is_a_valid_label_not_a_missing_one(self):
        # 0 is falsy but means NEGATIVE — a truthiness check here would silently
        # discard every negative row in the dataset.
        row = self.task.parse_row({"text": "a review", "label": 0})
        self.assertEqual(row["sentiment_label"], "NEGATIVE")

    def test_digit_strings_map_like_their_integers(self):
        # A local CSV delivers every field as text, so "0" must not become a
        # phantom "0" class.
        for raw, expected in (("0", "NEGATIVE"), ("1", "NEUTRAL"), ("2", "POSITIVE")):
            row = self.task.parse_row({"text": "a review", "label": raw})
            self.assertEqual(row["sentiment_label"], expected, raw)

    def test_class_name_labels_still_pass_through(self):
        row = self.task.parse_row({"text": "a review", "label": "POSITIVE"})
        self.assertEqual(row["sentiment_label"], "POSITIVE")

    def test_no_phantom_class_reaches_the_real_baseline(self):
        rows = [r for r in (self.task.parse_row({"text": "labelled", "label": 2}),
                            self.task.parse_row({"text": "unlabelled"})) if r]
        samples = self.task.get_real_eval_samples({}, rows)
        self.assertEqual(samples, [{"text": "labelled", "label": "POSITIVE"}])
        self.assertNotIn("None", [s["label"] for s in samples])

if __name__ == "__main__":
    unittest.main()
