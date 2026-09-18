"""The spam, gec and taxonomy preparation scripts, offline.

Each script rebuilds a gitignored benchmark file the task config points at, so
what it writes must load back through the task exactly as the sweep's file did.
The network steps (HuggingFace, the FCE archive, the Pizza ontology) are the
only parts not exercised here; each script checks its live result against the
sweep file's md5 when it runs.
"""
import io
import os
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

import pandas as pd

from framework.data_loading import iter_local_rows
from framework.tasks.spam.task import SpamTask
from scripts.benchmarks import prepare_gec_benchmark as gec
from scripts.benchmarks import prepare_spam_benchmark as spam
from scripts.benchmarks import prepare_taxonomy_benchmark as taxonomy


class SpamBenchmarkTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "spam", "bench.csv")
        self.source = pd.DataFrame({
            "label": ["ham", "spam", "ham"],
            "text": ["see you, \"later\"", "WIN a prize now", "glued\nham\tsecond message"],
        })

    def test_fixed_rows_are_written_in_their_order_and_load_back(self):
        counts = spam.write_benchmark(self.source, self.path, rows=(1, 2, 0))
        self.assertEqual(counts, {"SPAM": 1, "HAM": 2})
        rows = list(iter_local_rows(self.path))
        self.assertEqual([r["id"] for r in rows], ["1", "2", "3"])
        self.assertEqual([r["label"] for r in rows], ["SPAM", "HAM", "HAM"])
        # Quotes, commas and a glued multi-line message survive the round trip.
        self.assertEqual(rows[1]["text"], "glued\nham\tsecond message")
        self.assertEqual(rows[2]["text"], 'see you, "later"')
        self.assertEqual(SpamTask().parse_row(rows[2]), {"incorrect": 'see you, "later"'})

    def test_the_file_uses_the_sweep_file_s_crlf_line_endings(self):
        spam.write_benchmark(self.source, self.path, rows=(1,))
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), b"id,label,text\r\n1,SPAM,WIN a prize now\r\n")

    def test_the_fixed_rows_are_300_distinct_positions(self):
        self.assertEqual(len(spam.SOURCE_ROWS), 300)
        self.assertEqual(len(set(spam.SOURCE_ROWS)), 300)


class GecBenchmarkTests(unittest.TestCase):
    M2 = b"S A cat sat .\nA 1 2|||R:NOUN|||dog|||REQUIRED|||-NONE-|||0\n\nS Fine .\nA 0 1|||R:ADJ|||Good|||REQUIRED|||-NONE-|||0\n\n"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.archive = os.path.join(tmp.name, "fce.tar.gz")
        with tarfile.open(self.archive, "w:gz") as tar:
            info = tarfile.TarInfo(gec.member_for("test"))
            info.size = len(self.M2)
            tar.addfile(info, io.BytesIO(self.M2))

    def test_the_split_is_copied_unmodified_and_loads_as_m2(self):
        out = os.path.join(self.dir, "gec", "fce.m2")
        self.assertEqual(gec.extract_split(self.archive, "test", out), 2)
        with open(out, "rb") as f:
            self.assertEqual(f.read(), self.M2)
        rows = list(iter_local_rows(out))
        self.assertEqual(rows[0], {"incorrect": "A cat sat .", "correct": "A dog sat ."})

    def test_a_split_missing_from_the_archive_is_named(self):
        with self.assertRaisesRegex(ValueError, "fce.dev.gold.bea19.m2"):
            gec.extract_split(self.archive, "dev", os.path.join(self.dir, "x.m2"))


class TaxonomyBenchmarkArgsTests(unittest.TestCase):
    def _args(self, *argv):
        with mock.patch.object(sys, "argv", ["prepare", *argv]):
            return taxonomy.parse_args()

    def test_no_arguments_builds_the_pinned_pizza_benchmark(self):
        args = self._args()
        self.assertIsNone(args.input)
        self.assertEqual(args.output, taxonomy.DEFAULT_OUTPUT)
        self.assertIn("/4922ecbdf5535a00a6515276324c3aa7f5e4407a/", taxonomy.PIZZA_URL)

    def test_an_input_ontology_must_be_named(self):
        with mock.patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            self._args("my.owl", "out.jsonl")
        self.assertEqual(self._args("my.owl", "out.jsonl", "--ontology-id", "x",
                                    "--domain", "y").input, "my.owl")


if __name__ == "__main__":
    unittest.main()
