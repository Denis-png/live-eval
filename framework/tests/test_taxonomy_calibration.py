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

from framework import pipeline
from framework.calibration.artifact import write_calibration
from framework.profiling.taxonomy_fidelity import structure_measurements
from framework.tasks.base_task import BaseTask
from framework.tasks.taxonomy.task import TaxonomyTask
from framework.tests.test_taxonomy_real_reference import _E2E_REAL, _POOL_OPTS
from framework.tests.test_taxonomy_seeded_dispatch import _Gen

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


_Random = random.Random


class _Workspace(unittest.TestCase):
    """_E2E_REAL on disk: a pool of 3 subtrees -- 6 classes at max depth 2,
    10 at max depth 3, 5 at max depth 2."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.data = os.path.join(self.tmp, "onto.jsonl")
        with open(self.data, "w", encoding="utf-8") as f:
            f.write(json.dumps(_E2E_REAL) + "\n")
        from framework.profiling.taxonomy_profiler import profile_taxonomy_rows
        self.profile_path = os.path.join(self.tmp, "onto_taxonomy_profile.json")
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_taxonomy_rows([_E2E_REAL]), f)

    def config(self, mode, seedless, **generation):
        return {
            "task": {"name": "taxonomy"},
            "dataset": {"source": "local", "local": {"path": self.data, "format": "jsonl"}},
            "generation": {"provider": "stub", "model": "stub", "mode": mode,
                           "seedless": seedless, "num_runs": 1, "sample_size": 3,
                           "max_parse_attempts": 1, "profile_path": self.profile_path,
                           "feedback": {"max_rounds": 0},
                           "seed_pool": {**_POOL_OPTS, "domains": ["marine biology"]},
                           **generation},
            "task_models": [{"name": "lexical", "type": "lexical"}],
            "output": {"base_dir": self.tmp, "plots": False},
        }

    def artifact(self, name, target, calibrated):
        path = os.path.join(self.tmp, name)
        write_calibration(path, {"meta": {}, "target": target, "calibrated": calibrated,
                                 "selected_round": 0, "rounds": []})
        return path

    def context(self, cfg):
        with mock.patch.object(pipeline, "load_generator", return_value=_Gen()), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
            ctx = pipeline.build_generation_context(cfg)
        return ctx, err.getvalue()


class SeedPoolWeightingTests(_Workspace):
    def test_weights_draw_sample_size_seeds_with_repeats(self):
        cfg = self.config("forward", False, sample_size=6)
        pool = TaxonomyTask().get_seed_pool(cfg, [_E2E_REAL], "forward",
                                            seed_weights={"3": 1.0}, rng=_Random(0))
        self.assertEqual(len(pool), 6)
        self.assertEqual({seed["pool_index"] for seed in pool}, {1})  # the depth-3 subtree

    def test_no_weights_is_the_whole_pool_as_before(self):
        pool = TaxonomyTask().get_seed_pool(self.config("forward", False), [_E2E_REAL],
                                            "forward")
        self.assertEqual([seed["pool_index"] for seed in pool], [0, 1, 2])


class SeedWeightConsumptionTests(_Workspace):
    def _target(self):
        real = TaxonomyTask().get_real_eval_samples(self.config("forward", False),
                                                    [_E2E_REAL])
        return {"type_dist": TaxonomyTask().build_fidelity_profile(real)["max_depth_mix"]}

    def test_a_seeded_run_draws_by_the_artifacts_weights(self):
        path = self.artifact("a_calibration.json", self._target(),
                             {"type_dist": {"3": 1.0}, "seed_weights": {"3": 1.0}})
        cfg = self.config("forward", False, calibration_path=path)
        ctx, _ = self.context(cfg)
        self.assertEqual(ctx["seed_weights"], {"3": 1.0})
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            synthetic = pipeline._run_generation(_Gen(), TaxonomyTask(), cfg,
                                                 ctx["real_data"], None, None, None,
                                                 profile=ctx["profile"])
        self.assertEqual([r["source_pool_index"] for r in synthetic], [1, 1, 1])
        self.assertEqual(pipeline._LAST_CALIBRATION["path"], path)

    def test_a_stale_artifact_is_ignored_with_a_warning(self):
        path = self.artifact("b_calibration.json", {"type_dist": {"9": 1.0}},
                             {"type_dist": {"3": 1.0}, "seed_weights": {"3": 1.0}})
        ctx, err = self.context(self.config("forward", False, calibration_path=path))
        self.assertIsNone(ctx["seed_weights"])
        self.assertIn("different real reference", err)

    def test_a_malformed_target_is_ignored_with_a_warning_not_a_crash(self):
        path = self.artifact("c_calibration.json", "not-a-dict",
                             {"type_dist": {"3": 1.0}, "seed_weights": {"3": 1.0}})
        ctx, err = self.context(self.config("forward", False, calibration_path=path))
        self.assertIsNone(ctx["seed_weights"])
        self.assertIn("malformed", err)


if __name__ == "__main__":
    unittest.main()
