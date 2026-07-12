import unittest

from framework.tasks.gec.task import GECTask
from framework.tasks.spam.task import SpamTask
import framework.profiling.spam_profiler as sp


class RealEvalSamplesTests(unittest.TestCase):
    def test_gec_reshapes_pairs(self):
        real = [{"incorrect": "he go home", "correct": "he goes home"},
                {"incorrect": "", "correct": "x"}]  # dropped: empty incorrect
        out = GECTask().get_real_eval_samples({}, real)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["text"], "he go home")
        self.assertEqual(out[0]["corrupted"], "he go home")
        self.assertEqual(out[0]["original"], "he goes home")

    def test_spam_maps_labeled_reference(self):
        original = sp.load_spam_rows
        sp.load_spam_rows = lambda **kw: [
            {"text": "win money now", "label": "SPAM"},
            {"text": "lunch tomorrow", "label": "HAM"},
        ]
        try:
            config = {"dataset": {"huggingface": {"name": "d", "split": "train"}},
                      "generation": {"inverse": {"profile_size": 10}}}
            out = SpamTask().get_real_eval_samples(config, [])
        finally:
            sp.load_spam_rows = original
        self.assertEqual({r["label"] for r in out}, {"SPAM", "HAM"})
        self.assertEqual(out[0]["text"], "win money now")


if __name__ == "__main__":
    unittest.main()
