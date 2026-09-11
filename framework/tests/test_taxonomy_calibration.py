"""Taxonomy calibration: measurements, keys, actuators, and the calibrate loop.

Taxonomy used to opt out of calibration, so the ablation's phase B skipped it.
Seeded cells lose their largest subtrees to verification; inverse+seedless
imposes the real ontology's structure and the model honours some depth and
branching bins more readily than others. Both are measurable, so both can be
steered: seed weights over max-depth buckets, and the imposed distributions.
"""
import io
import json
import os
import random
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from framework.profiling.taxonomy_fidelity import structure_measurements
from framework.tasks.base_task import BaseTask
from framework.tasks.taxonomy.task import TaxonomyTask

_CHAIN = {"n_classes": 4, "max_depth": 2, "depth_distribution": {"0": 1, "1": 2, "2": 1},
          "child_count_distribution": {"0": 2, "1": 1, "2": 1}}
_PAIR = {"n_classes": 2, "max_depth": 1, "depth_distribution": {"0": 1, "1": 1},
         "child_count_distribution": {"0": 1, "1": 1}}
_CYCLE = {"n_classes": 2, "max_depth": None, "depth_distribution": {},
          "child_count_distribution": {"1": 2}}


class MeasurementTests(unittest.TestCase):
    def test_depth_and_branching_are_pooled_over_classes(self):
        m = structure_measurements([_CHAIN, _PAIR])
        self.assertEqual(m["depth_dist"], {"0": 2 / 6, "1": 3 / 6, "2": 1 / 6})
        self.assertEqual(m["child_count_dist"], {"0": 3 / 6, "1": 2 / 6, "2": 1 / 6})

    def test_the_max_depth_mix_weights_each_taxonomy_by_its_classes(self):
        # 4 classes at max depth 2, 2 at max depth 1 -- not one taxonomy each.
        m = structure_measurements([_CHAIN, _PAIR])
        self.assertEqual(m["max_depth_mix"], {"1": 2 / 6, "2": 4 / 6})

    def test_a_cyclic_taxonomy_counts_everywhere_but_depth(self):
        m = structure_measurements([_CHAIN, _CYCLE])
        self.assertEqual(m["n_classes_total"], 6)
        self.assertEqual(m["depth_dist"], {"0": 1 / 4, "1": 2 / 4, "2": 1 / 4})
        self.assertEqual(m["max_depth_mix"], {"2": 1.0})
        self.assertEqual(sum(m["child_count_dist"].values()), 1.0)

    def test_nothing_measured_is_empty_not_an_error(self):
        self.assertEqual(structure_measurements([]),
                         {"depth_dist": {}, "child_count_dist": {},
                          "max_depth_mix": {}, "n_classes_total": 0})

    def test_the_fidelity_profile_carries_them(self):
        profile = TaxonomyTask().build_fidelity_profile([
            {"domain": "d", "classes": ["A", "B", "C"],
             "subclass_axioms": [["B", "A"], ["C", "B"]]}])
        self.assertEqual(profile["depth_dist"], {"0": 1 / 3, "1": 1 / 3, "2": 1 / 3})
        self.assertEqual(profile["max_depth_mix"], {"2": 1.0})
        self.assertEqual(profile["n_classes_total"], 3)


class KeyTests(unittest.TestCase):
    def test_taxonomy_reuses_the_two_slots(self):
        self.assertEqual(TaxonomyTask().get_calibration_keys(),
                         {"type_dist": "depth_dist", "count_dist": "child_count_dist"})

    def test_seed_mode_measures_the_max_depth_mix(self):
        self.assertEqual(TaxonomyTask().get_seed_calibration_key(), "max_depth_mix")

    def test_other_tasks_keep_their_seed_key(self):
        self.assertIsNone(BaseTask.get_seed_calibration_key(object()))


if __name__ == "__main__":
    unittest.main()
