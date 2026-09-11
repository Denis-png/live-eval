"""load_real_data warns when the source yields fewer usable rows than
generation.sample_size -- except for structured tasks, where it was noise.

A structured sample is a whole generated artifact, and the real rows are whole
ontologies its seed pool is cut from: Pizza is ONE row, so every taxonomy run
warned that "the source only yielded 1", which reads like a broken benchmark.
"""
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from framework import pipeline


class _Task:
    def __init__(self, strategy):
        self.strategy = strategy

    def get_generation_strategy(self):
        return self.strategy

    def parse_row(self, row):
        return row


def _load(strategy):
    config = {"dataset": {"source": "local", "local": {"path": "x.jsonl"}},
              "generation": {"sample_size": 10}}
    err = io.StringIO()
    with mock.patch.object(pipeline, "iter_local_rows", return_value=[{"a": 1}]), \
            redirect_stdout(io.StringIO()), redirect_stderr(err):
        rows = pipeline.load_real_data(config, _Task(strategy))
    return rows, err.getvalue()


class ShortPoolWarningTests(unittest.TestCase):
    def test_a_sentence_task_is_told_its_pool_ran_short(self):
        rows, err = _load("corruption")
        self.assertEqual(len(rows), 1)
        self.assertIn("only yielded 1", err)

    def test_a_structured_task_is_not(self):
        rows, err = _load("structured")
        self.assertEqual(len(rows), 1)
        self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
