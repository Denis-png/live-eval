import argparse
import math
import os
import tempfile
import unittest

from framework.pipeline import _get_field, aggregate, load_real_data, _mean_std


class _SpamLikeTask:
    def parse_row(self, row):
        if str(row.get("label", "")).lower() in ("spam", "1"):
            return None
        text = row.get("text")
        return {"incorrect": text} if text else None


class _GecLikeTask:
    def parse_row(self, row):
        if row.get("incorrect") and row.get("correct"):
            return {"incorrect": row["incorrect"], "correct": row["correct"]}
        return None


class LoadRealDataLocalTests(unittest.TestCase):
    def test_local_csv_through_parse_row_respects_sample_size(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "spam.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("label,text\nham,one\nSPAM,ignore me\nham,two\nham,three\n")
            config = {"dataset": {"source": "local", "local": {"path": path}},
                      "generation": {"sample_size": 2}}
            samples = load_real_data(config, _SpamLikeTask())
        # SPAM row filtered by parse_row; first-N sampling stops at 2.
        self.assertEqual(samples, [{"incorrect": "one"}, {"incorrect": "two"}])

    def test_local_m2_produces_incorrect_correct_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fce.m2")
            with open(path, "w", encoding="utf-8") as f:
                f.write("S He go home .\n"
                        "A 1 2|||R:VERB:SVA|||goes|||REQUIRED|||-NONE-|||0\n")
            config = {"dataset": {"source": "local",
                                  "local": {"path": path, "format": "m2"}},
                      "generation": {"sample_size": 5}}
            samples = load_real_data(config, _GecLikeTask())
        self.assertEqual(samples, [{"incorrect": "He go home .",
                                    "correct": "He goes home ."}])

    def test_warns_when_pool_smaller_than_requested(self):
        """sample_size counts USABLE (post-filter) samples; if the source can't
        fill the pool, say so instead of silently running on fewer."""
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "spam.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("label,text\nham,one\nham,two\n")
            config = {"dataset": {"source": "local", "local": {"path": path}},
                      "generation": {"sample_size": 5}}
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                samples = load_real_data(config, _SpamLikeTask())
        self.assertEqual(len(samples), 2)
        self.assertIn("2", stderr.getvalue())
        self.assertIn("5", stderr.getvalue())

    def test_local_missing_file_raises_with_path(self):
        config = {"dataset": {"source": "local", "local": {"path": "no/such.csv"}},
                  "generation": {"sample_size": 5}}
        with self.assertRaises(ValueError) as ctx:
            load_real_data(config, _SpamLikeTask())
        self.assertIn("no/such.csv", str(ctx.exception))


class GetFieldTests(unittest.TestCase):
    def test_returns_first_matching_candidate(self):
        row = {"input": "bad sentence", "output": "good sentence"}
        self.assertEqual(_get_field(row, ["input", "text"]), "bad sentence")

    def test_skips_empty_candidate(self):
        row = {"input": "", "text": "fallback text"}
        self.assertEqual(_get_field(row, ["input", "text"]), "fallback text")

    def test_returns_none_when_no_candidate_matches(self):
        # Must NOT silently grab an arbitrary unrelated column.
        row = {"some_unrelated_column": "noise", "another": "more noise"}
        self.assertIsNone(_get_field(row, ["input", "output"]))


class AggregateTests(unittest.TestCase):
    def test_scalar_mean_and_sample_std(self):
        runs = [
            {"m": {"gleu": 0.2}},
            {"m": {"gleu": 0.4}},
            {"m": {"gleu": 0.6}},
        ]
        out = aggregate(runs)
        self.assertAlmostEqual(out["m"]["gleu"]["mean"], 0.4)
        # sample std (ddof=1) of [.2,.4,.6] = 0.2, not population std 0.1633
        self.assertAlmostEqual(out["m"]["gleu"]["std"], 0.2)

    def test_single_run_has_zero_std(self):
        out = aggregate([{"m": {"gleu": 0.5}}])
        self.assertEqual(out["m"]["gleu"]["mean"], 0.5)
        self.assertEqual(out["m"]["gleu"]["std"], 0.0)
        self.assertFalse(math.isnan(out["m"]["gleu"]["std"]))

    def test_nested_dict_metric(self):
        runs = [
            {"m": {"errant": {"precision": 0.5}}},
            {"m": {"errant": {"precision": 0.7}}},
        ]
        out = aggregate(runs)
        self.assertAlmostEqual(out["m"]["errant"]["precision"]["mean"], 0.6)

    def test_run_missing_a_model_does_not_crash(self):
        # A later run that failed to score a model should be skipped for that
        # model, not raise KeyError mid-aggregation.
        runs = [
            {"a": {"gleu": 0.2}, "b": {"gleu": 0.8}},
            {"a": {"gleu": 0.4}},  # model "b" missing this run
        ]
        out = aggregate(runs)
        self.assertAlmostEqual(out["a"]["gleu"]["mean"], 0.3)
        self.assertAlmostEqual(out["b"]["gleu"]["mean"], 0.8)


if __name__ == "__main__":
    unittest.main()
