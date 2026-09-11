"""Per-run paired real scoring.

A seeded taxonomy session's real side was the WHOLE seed pool, scored once, while
each run's synthetic side is what that run drew and what passed verification --
which drops the largest subtrees most often. Micro scores are dominated by those
items, so the generated-vs-real gap mixed generation fidelity with attrition.
Each run is now also scored against the real items it actually delivered.
"""
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from framework import pipeline
from framework.tasks.base_task import BaseTask
from framework.tasks.taxonomy.task import TaxonomyTask

_REAL = [{"pool_index": 0, "classes": ["A"]}, {"pool_index": 1, "classes": ["B"]},
         {"pool_index": 2, "classes": ["C"]}]


class PairingHookTests(unittest.TestCase):
    def test_the_base_task_does_not_pair(self):
        self.assertIsNone(BaseTask.paired_real_indices(object(), _REAL, [{}]))

    def test_each_record_maps_to_the_real_item_with_its_index(self):
        synthetic = [{"source_pool_index": 2}, {"source_pool_index": 0}]
        self.assertEqual(TaxonomyTask().paired_real_indices(_REAL, synthetic), [2, 0])

    def test_a_repeated_draw_is_paired_once_per_repeat(self):
        synthetic = [{"source_pool_index": 1}, {"source_pool_index": 1}]
        self.assertEqual(TaxonomyTask().paired_real_indices(_REAL, synthetic), [1, 1])

    def test_positions_not_indices_are_returned(self):
        # real_sample.json need not be in pool order.
        real = [_REAL[2], _REAL[0], _REAL[1]]
        synthetic = [{"source_pool_index": 2}]
        self.assertEqual(TaxonomyTask().paired_real_indices(real, synthetic), [0])

    def test_seedless_records_are_not_paired(self):
        # A seedless artifact comes from no particular real item.
        self.assertIsNone(TaxonomyTask().paired_real_indices(_REAL, [{"classes": ["X"]}]))

    def test_an_index_matching_no_real_item_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "7"):
            TaxonomyTask().paired_real_indices(_REAL, [{"source_pool_index": 7}])


if __name__ == "__main__":
    unittest.main()
