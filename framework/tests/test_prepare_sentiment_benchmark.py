import os
import tempfile
import unittest

from framework.data_loading import iter_local_rows
from framework.tasks.sentiment.task import SentimentTask
from scripts.prepare_sentiment_benchmark import write_benchmark


class WriteBenchmarkTests(unittest.TestCase):
    """The CSV is only useful if the sentiment task reads back exactly the rows
    written, so each test round-trips through the local loader and parse_row."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "benchmarks", "sentiment", "bench.csv")

    def _load(self):
        task = SentimentTask()
        return [task.parse_row(row) for row in iter_local_rows(self.path)]

    def test_first_n_rows_round_trip_through_the_task_parser(self):
        rows = [{"text": 'loved it, "really"', "label": 2},
                {"text": "meh\nwhatever", "label": 1},
                {"text": "awful", "label": 0},
                {"text": "past the cut", "label": 2}]
        counts = write_benchmark(rows, self.path, n=3)
        parsed = self._load()
        self.assertEqual([p["sentiment_label"] for p in parsed],
                         ["POSITIVE", "NEUTRAL", "NEGATIVE"])
        # Commas, quotes and newlines survive the CSV quoting.
        self.assertEqual([p["incorrect"] for p in parsed],
                         ['loved it, "really"', "meh\nwhatever", "awful"])
        self.assertEqual(counts, {2: 1, 1: 1, 0: 1})

    def test_rows_the_task_would_drop_are_skipped_not_counted(self):
        rows = [{"text": "  ", "label": 1}, {"text": "no label", "label": None},
                {"text": "kept", "label": 0}, {"text": "also kept", "label": 1}]
        write_benchmark(rows, self.path, n=2)
        self.assertEqual([p["incorrect"] for p in self._load()], ["kept", "also kept"])

    def test_a_short_source_refuses_instead_of_writing_fewer_rows(self):
        with self.assertRaisesRegex(ValueError, "Only 1 usable rows"):
            write_benchmark([{"text": "one", "label": 0}], self.path, n=2)
        self.assertFalse(os.path.exists(self.path))


if __name__ == "__main__":
    unittest.main()
