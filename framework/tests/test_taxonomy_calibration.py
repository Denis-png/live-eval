"""Taxonomy calibration: measurements, keys, actuators, and the calibrate loop.

Taxonomy used to opt out of calibration, so the ablation's phase B skipped it.
inverse+seedless imposes the real ontology's structure and the model honours
some depth and branching bins more readily than others, so the imposed
distributions are steered. Seeded cells refuse: their only control input would
be seed weights over max-depth buckets, and on a pool of ~10 subtrees weighted
draws with replacement add more structural noise than the verification
attrition they would correct. A seeded run still consumes an existing artifact.
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

from framework import calibrate, pipeline
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


class SeedWeightMessageTests(_Workspace):
    """Without weights a structured seeded run draws the WHOLE pool; only GEC's
    corruption cell falls back to a first-N order. And seeded taxonomy cannot
    be calibrated, so its note must not point at a command that refuses."""

    def _load(self, task, strategy, cfg, real=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            weights = pipeline._load_seed_weights(cfg, task, strategy, "forward", False,
                                                  real)
        self.assertIsNone(weights)
        return out.getvalue() + err.getvalue()

    def test_a_structured_cell_without_an_artifact_draws_the_whole_pool(self):
        text = self._load(TaxonomyTask(), "structured",
                          self.config("forward", False, calibration_path=None))
        self.assertIn("the whole pool", text)
        self.assertNotIn("first-N", text)
        self.assertNotIn("framework.calibrate", text)

    def test_a_structured_cell_with_a_malformed_artifact_draws_the_whole_pool(self):
        path = os.path.join(self.tmp, "broken_calibration.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json")
        text = self._load(TaxonomyTask(), "structured",
                          self.config("forward", False, calibration_path=path))
        self.assertIn("malformed", text)
        self.assertIn("the whole pool", text)
        self.assertNotIn("first-N", text)

    def test_a_structured_cell_with_a_stale_artifact_draws_the_whole_pool(self):
        path = self.artifact("stale_calibration.json", {"type_dist": {"9": 1.0}},
                             {"type_dist": {"3": 1.0}, "seed_weights": {"3": 1.0}})
        cfg = self.config("forward", False, calibration_path=path)
        text = self._load(TaxonomyTask(), "structured", cfg,
                          TaxonomyTask().get_real_eval_samples(cfg, [_E2E_REAL]))
        self.assertIn("different real reference", text)
        self.assertIn("the whole pool", text)
        self.assertNotIn("Recalibrate", text)

    def test_gec_keeps_its_first_n_wording_and_its_build_hint(self):
        from framework.tasks.gec.task import GECTask
        cfg = {"generation": {"mode": "forward", "seedless": False,
                              "calibration_path": None}}
        text = self._load(GECTask(), "corruption", cfg)
        self.assertIn("unweighted first-N order", text)
        self.assertIn("python -m framework.calibrate", text)


_PROFILE = {"taxonomies": [{"domain": "d", "n_classes": 10, "n_leaves": 6,
                            "max_depth": 3, "mean_depth": 1.5,
                            "depth_distribution": {"0": 1, "1": 4, "2": 4, "3": 1},
                            "child_count_distribution": {"0": 6, "1": 2, "2": 2}}]}


class ApplyStructureTests(unittest.TestCase):
    def setUp(self):
        self.request = {"type_dist": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4},
                        "count_dist": {0: 0.5, 1: 0.25, 2: 0.25}}   # int keys, as loaded
        self.out = TaxonomyTask().apply_calibrated_structure(_PROFILE, self.request)
        self.target = self.out["taxonomies"][0]

    def test_the_request_becomes_class_counts_summing_to_n_classes(self):
        self.assertEqual(self.target["depth_distribution"], {"0": 1, "1": 2, "2": 3, "3": 4})
        self.assertEqual(sum(self.target["child_count_distribution"].values()), 10)

    def test_dependent_fields_are_recomputed(self):
        self.assertEqual(self.target["mean_depth"], 2.0)       # (0+2+6+12)/10
        self.assertEqual(self.target["n_leaves"],
                         self.target["child_count_distribution"]["0"])

    def test_other_fields_and_the_input_are_untouched(self):
        self.assertEqual((self.target["domain"], self.target["n_classes"]), ("d", 10))
        self.assertEqual(_PROFILE["taxonomies"][0]["depth_distribution"],
                         {"0": 1, "1": 4, "2": 4, "3": 1})
        self.assertNotIn("n_roots", _PROFILE["taxonomies"][0])

    def test_roots_and_max_depth_follow_the_requested_depths(self):
        # A request that moves mass off the deepest level and onto the roots
        # must not leave the spec claiming the real ontology's max depth and
        # root count beside counts that contradict them.
        out = TaxonomyTask().apply_calibrated_structure(
            _PROFILE, {"type_dist": {"0": 0.2, "1": 0.5, "2": 0.3}})
        target = out["taxonomies"][0]
        self.assertEqual(target["depth_distribution"], {"0": 2, "1": 5, "2": 3})
        self.assertEqual((target["n_roots"], target["max_depth"]), (2, 2))

    def test_max_depth_is_the_deepest_non_empty_bin(self):
        # A zero-mass bin is dropped by the rounding and sets no depth.
        out = TaxonomyTask().apply_calibrated_structure(
            _PROFILE, {"type_dist": {"0": 0.1, "1": 0.4, "4": 0.5, "5": 0.0}})
        self.assertEqual(out["taxonomies"][0]["max_depth"], 4)

    def test_a_request_with_no_roots_still_imposes_one(self):
        out = TaxonomyTask().apply_calibrated_structure(
            _PROFILE, {"type_dist": {"1": 0.5, "2": 0.5}})
        self.assertEqual(out["taxonomies"][0]["n_roots"], 1)

    def test_a_count_only_request_leaves_roots_and_depth(self):
        out = TaxonomyTask().apply_calibrated_structure(
            {"taxonomies": [{**_PROFILE["taxonomies"][0], "n_roots": 3}]},
            {"count_dist": {"0": 1.0}})
        self.assertEqual((out["taxonomies"][0]["n_roots"],
                          out["taxonomies"][0]["max_depth"]), (3, 3))

    def test_rounding_keeps_the_total_exact(self):
        out = TaxonomyTask().apply_calibrated_structure(
            _PROFILE, {"type_dist": {"0": 1 / 3, "1": 1 / 3, "2": 1 / 3},
                       "count_dist": {"0": 1.0}})
        self.assertEqual(sum(out["taxonomies"][0]["depth_distribution"].values()), 10)


class StructureConsumptionTests(_Workspace):
    def _target(self):
        real = TaxonomyTask().get_real_eval_samples(self.config("inverse", True), [_E2E_REAL])
        measured = TaxonomyTask().build_fidelity_profile(real)
        return {"type_dist": measured["depth_dist"], "count_dist": measured["child_count_dist"]}

    def test_the_prompt_and_the_feedback_carry_the_calibrated_counts(self):
        request = {"type_dist": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4},
                   "count_dist": {"0": 0.6, "1": 0.2, "2": 0.2}}
        path = self.artifact("c_calibration.json", self._target(), request)
        ctx, _ = self.context(self.config("inverse", True, calibration_path=path))
        self.assertEqual(ctx["profile"]["taxonomies"][0]["depth_distribution"],
                         {"0": 1, "1": 2, "2": 3, "3": 4})
        prompt = TaxonomyTask().build_structured_generation_prompt(ctx["profile"],
                                                                   mode="inverse")
        self.assertIn('"3": 4', prompt)
        artifact = {"domain": "d", "classes": ["A", "B"], "subclass_axioms": [["B", "A"]]}
        feedback = TaxonomyTask().build_structural_feedback(ctx["profile"], artifact)
        # The loop compares each artifact against the substituted reference.
        self.assertEqual(
            feedback["comparison"]["distribution_characteristics"]
            ["depth_distribution"]["real"],
            {"0": 1, "1": 2, "2": 3, "3": 4})

    def test_a_stale_artifact_leaves_the_real_structure(self):
        path = self.artifact("d_calibration.json",
                             {"type_dist": {"9": 1.0}, "count_dist": {"0": 1.0}},
                             {"type_dist": {"0": 1.0}, "count_dist": {"0": 1.0}})
        ctx, err = self.context(self.config("inverse", True, calibration_path=path))
        self.assertEqual(ctx["profile"]["taxonomies"][0]["depth_distribution"],
                         json.load(open(self.profile_path))["taxonomies"][0]["depth_distribution"])
        self.assertIn("different real reference", err)

    def test_a_malformed_target_is_ignored_with_a_warning_not_a_crash(self):
        # A structurally corrupt target ("not-a-dict") must not crash
        # _structured_target_matches -- it takes the "malformed" path instead
        # of the "different real reference" one (task-3 decision 1).
        path = self.artifact("e_calibration.json", "not-a-dict",
                             {"type_dist": {"0": 1.0}, "count_dist": {"0": 1.0}})
        ctx, err = self.context(self.config("inverse", True, calibration_path=path))
        self.assertEqual(ctx["profile"]["taxonomies"][0]["depth_distribution"],
                         json.load(open(self.profile_path))["taxonomies"][0]["depth_distribution"])
        self.assertIn("malformed", err)

    def test_a_massless_type_dist_is_ignored_with_a_warning_not_a_crash(self):
        # A CURRENT artifact (target matches the real reference) can still carry
        # a degenerate calibrated request: an all-zero type_dist would otherwise
        # pass validation, print the normal success line, and leave
        # apply_calibrated_structure imposing {} while n_classes stays.
        request = {"type_dist": {"0": 0.0, "1": 0.0}, "count_dist": {"0": 1.0}}
        path = self.artifact("f_calibration.json", self._target(), request)
        ctx, err = self.context(self.config("inverse", True, calibration_path=path))
        self.assertEqual(ctx["profile"]["taxonomies"][0]["depth_distribution"],
                         json.load(open(self.profile_path))["taxonomies"][0]["depth_distribution"])
        self.assertIn("malformed", err)


class _Shallow(_Gen):
    """Always delivers a star: one root, every other class directly beneath it."""

    def call_api(self, prompt):
        self.calls += 1
        leaves = [f"L{i}" for i in range(5)]
        return json.dumps({"domain": "d", "classes": ["Root", *leaves],
                           "subclass_axioms": [[leaf, "Root"] for leaf in leaves]})


class CalibrateLoopTests(_Workspace):
    def _calibrate(self, cfg, generator, **kwargs):
        with mock.patch.object(pipeline, "load_generator", return_value=generator), \
                mock.patch("random.Random", lambda *a, **k: _Random(1234)), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return calibrate.run_calibration(
                cfg, output_path=os.path.join(self.tmp, "cal.json"), **kwargs)

    def test_both_seeded_cells_refuse_before_generating(self):
        # Seed weights are bucket DRAW probabilities while the target mix is
        # class-weighted, and on a small pool weighted draws with replacement add
        # more structural noise than the attrition they would correct: a seeded
        # calibration shipped a worse mix and called it converged. Both seeded
        # cells refuse, and before a single generation is paid for.
        for mode in ("forward", "inverse"):
            with self.subTest(mode=mode):
                generator = _Gen()
                with self.assertRaisesRegex(RuntimeError, "cannot be calibrated") as err:
                    self._calibrate(self.config(mode, False, sample_size=12), generator,
                                    rounds=1, sample_size=12)
                self.assertIn(f"{mode}_seeded", str(err.exception))
                self.assertIn("inverse+seedless", str(err.exception))
                self.assertEqual(generator.calls, 0)
                self.assertFalse(os.path.exists(os.path.join(self.tmp, "cal.json")))

    def test_inverse_seedless_calibration_asks_for_more_depth(self):
        payload = self._calibrate(self.config("inverse", True), _Shallow(),
                                  rounds=1, sample_size=3)
        first, second = (r["request"]["type_dist"] for r in payload["rounds"][:2])
        deep = lambda d: sum(v for k, v in d.items() if int(k) >= 2)
        self.assertGreater(deep(second), deep(first))

    def test_forward_seedless_refuses(self):
        with self.assertRaisesRegex(RuntimeError, "nothing to calibrate"):
            self._calibrate(self.config("forward", True), _Shallow(), rounds=0, sample_size=3)

    def test_the_informative_count_is_classes(self):
        profile = TaxonomyTask().build_fidelity_profile([_E2E_REAL])
        self.assertEqual(calibrate.informative_count(TaxonomyTask(), [], profile), 10)

    def test_structured_calibration_defaults_to_three_times_the_sample(self):
        settings = calibrate.calibration_settings({"task": {"name": "taxonomy"},
                                                   "generation": {"sample_size": 10}})
        self.assertEqual(settings["sample_size"], 30)
        gec = calibrate.calibration_settings({"task": {"name": "gec"},
                                              "generation": {"sample_size": 150}})
        self.assertEqual(gec["sample_size"], 150)


if __name__ == "__main__":
    unittest.main()
