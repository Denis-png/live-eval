import unittest
from unittest import mock

from framework.tasks.spam.task import SpamTask


class SpamProfileDatasetConfigTests(unittest.TestCase):
    def test_profiler_reads_nested_huggingface_block(self):
        """The inverse-mode spam profiler must see the same dataset settings as
        the main loader — including the nested dataset.huggingface block."""
        config = {
            "dataset": {
                "source": "huggingface",
                "sample_size": 50,
                "huggingface": {"name": "nested/ds", "split": "validation"},
            },
            "generation": {"inverse": {"profile_size": 7}},
        }
        with mock.patch(
            "framework.profiling.spam_profiler.load_spam_rows", return_value=[]
        ) as load_rows:
            with mock.patch(
                "framework.profiling.spam_distribution.profile_spam_distribution",
                return_value={},
            ):
                SpamTask().profile_error_distribution([], config=config)
        kwargs = load_rows.call_args.kwargs
        self.assertEqual(kwargs["dataset_name"], "nested/ds")
        self.assertEqual(kwargs["split"], "validation")
        self.assertEqual(kwargs["sample_size"], 7)


    def test_profiler_uses_local_file_when_source_is_local(self):
        """With source: local, SPAM signal profiling must come from the local
        file's SPAM rows — not silently from a different (HF) dataset."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sms.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("label,text\n"
                        "spam,WIN A FREE PRIZE NOW http://x.com\n"
                        "ham,see you at lunch\n"
                        "spam,URGENT! claim your £500 reward\n")
            config = {
                "dataset": {"source": "local", "sample_size": 50,
                            "local": {"path": path, "format": "csv"}},
                "generation": {"inverse": {"profile_size": 10}},
            }
            with mock.patch(
                "framework.profiling.spam_profiler.load_spam_rows"
            ) as load_rows:
                with mock.patch(
                    "framework.profiling.spam_distribution.profile_spam_distribution",
                    return_value={},
                ) as profile:
                    SpamTask().profile_error_distribution([], config=config)
            load_rows.assert_not_called()
            spam_rows = profile.call_args.args[0]
            self.assertEqual(len(spam_rows), 2)  # the two SPAM rows from the file
            self.assertTrue(all(r["label"] == "SPAM" for r in spam_rows))


if __name__ == "__main__":
    unittest.main()
