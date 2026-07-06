import os
import tempfile
import textwrap
import unittest

from framework.data_loading import (
    infer_format,
    iter_local_rows,
    resolve_dataset_config,
)

# Two sentences with annotator-0 edits, one already-correct sentence (no edits),
# and one annotator-1-only edit that annotator 0 must ignore.
SAMPLE_M2 = textwrap.dedent("""\
    S He go to school .
    A 1 2|||R:VERB:SVA|||goes|||REQUIRED|||-NONE-|||0

    S This sentence are fine .
    A 2 3|||R:VERB:SVA|||is|||REQUIRED|||-NONE-|||0
    A 4 4|||M:ADV|||really|||REQUIRED|||-NONE-|||1

    S Nothing wrong here .

    S Only other annotator touched this .
    A 0 1|||R:DET|||The|||REQUIRED|||-NONE-|||1
""")


def _write(dirname, name, content):
    path = os.path.join(dirname, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


class InferFormatTests(unittest.TestCase):
    def test_infers_from_extension(self):
        self.assertEqual(infer_format("data/fce.m2"), "m2")
        self.assertEqual(infer_format("data/spam.csv"), "csv")
        self.assertEqual(infer_format("data/spam.tsv"), "tsv")

    def test_unknown_extension_returns_none(self):
        self.assertIsNone(infer_format("data/spam.jsonl"))


class M2LoadingTests(unittest.TestCase):
    def _rows(self, content=SAMPLE_M2):
        with tempfile.TemporaryDirectory() as d:
            path = _write(d, "sample.m2", content)
            return list(iter_local_rows(path, "m2"))

    def test_applies_annotator_zero_edits(self):
        rows = self._rows()
        self.assertEqual(rows[0], {"incorrect": "He go to school .",
                                   "correct": "He goes to school ."})

    def test_ignores_other_annotators_edits(self):
        rows = self._rows()
        # Second block: only the annotator-0 edit is applied, not annotator 1's.
        self.assertEqual(rows[1]["correct"], "This sentence is fine .")

    def test_skips_sentences_without_annotator_zero_edits(self):
        # "Nothing wrong here" (no edits) and the annotator-1-only sentence
        # carry no error signal for GEC — both are skipped.
        self.assertEqual(len(self._rows()), 2)

    def test_multiple_edits_applied_right_to_left(self):
        content = textwrap.dedent("""\
            S a b c d
            A 0 1|||R:X|||A|||REQUIRED|||-NONE-|||0
            A 2 3|||R:X|||C|||REQUIRED|||-NONE-|||0
        """)
        rows = self._rows(content)
        self.assertEqual(rows[0]["correct"], "A b C d")

    def test_deletion_edit_with_empty_correction(self):
        content = textwrap.dedent("""\
            S he is is here
            A 1 2|||U:VERB|||-NONE-|||REQUIRED|||-NONE-|||0
            A 2 3|||U:VERB||||||REQUIRED|||-NONE-|||0
        """)
        rows = self._rows(content)
        # -NONE- correction ignored; empty correction deletes the span.
        self.assertEqual(rows[0]["correct"], "he is here")


class DelimitedLoadingTests(unittest.TestCase):
    def test_csv_rows_as_dicts(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write(d, "spam.csv", "id,label,text\n1,ham,hello there\n2,SPAM,BUY NOW\n")
            rows = list(iter_local_rows(path, "csv"))
        self.assertEqual(rows[0], {"id": "1", "label": "ham", "text": "hello there"})
        self.assertEqual(len(rows), 2)

    def test_tsv_rows_as_dicts(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write(d, "spam.tsv", "label\ttext\nham\thi\n")
            rows = list(iter_local_rows(path, "tsv"))
        self.assertEqual(rows[0], {"label": "ham", "text": "hi"})

    def test_format_inferred_when_omitted(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write(d, "spam.csv", "label,text\nham,hi\n")
            rows = list(iter_local_rows(path, None))
        self.assertEqual(rows[0]["text"], "hi")


class LocalRowErrorTests(unittest.TestCase):
    def test_missing_file_raises_value_error_with_path(self):
        with self.assertRaises(ValueError) as ctx:
            list(iter_local_rows("no/such/file.csv", "csv"))
        self.assertIn("no/such/file.csv", str(ctx.exception))

    def test_unknown_format_raises_value_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write(d, "data.jsonl", "{}\n")
            with self.assertRaises(ValueError) as ctx:
                list(iter_local_rows(path, None))
            self.assertIn("format", str(ctx.exception).lower())


class ResolveDatasetConfigTests(unittest.TestCase):
    def test_nested_blocks(self):
        ds = {
            "source": "local",
            "sample_size": 50,
            "huggingface": {"name": "d/ds", "split": "train", "streaming": True},
            "local": {"path": "framework/data/spam/x.csv", "format": "csv"},
        }
        res = resolve_dataset_config(ds)
        self.assertEqual(res["source"], "local")
        self.assertEqual(res["path"], "framework/data/spam/x.csv")
        self.assertEqual(res["format"], "csv")
        self.assertEqual(res["name"], "d/ds")   # hf settings still resolvable
        self.assertTrue(res["streaming"])
        self.assertEqual(res["sample_size"], 50)

    def test_legacy_flat_keys_still_work(self):
        ds = {"source": "local", "sample_size": 10,
              "name": "d/ds", "split": "test",
              "local_path": "data/fce.m2", "format": "m2"}
        res = resolve_dataset_config(ds)
        self.assertEqual(res["path"], "data/fce.m2")
        self.assertEqual(res["format"], "m2")
        self.assertEqual(res["split"], "test")

    def test_defaults_to_huggingface_source(self):
        res = resolve_dataset_config({"name": "d/ds", "split": "train", "sample_size": 5})
        self.assertEqual(res["source"], "huggingface")
        self.assertFalse(res["streaming"])

    def test_nested_block_wins_over_flat_key(self):
        ds = {"sample_size": 5, "name": "flat/ds",
              "huggingface": {"name": "nested/ds", "split": "train"}}
        self.assertEqual(resolve_dataset_config(ds)["name"], "nested/ds")


if __name__ == "__main__":
    unittest.main()
