"""Seed-weight calibration for the one cell with no injectable distribution.

GEC forward+seeded lets the generator identify the error type itself, so the
control input is WHICH seeds it is fed. These tests cover the index, the draw,
the widened get_seed_pool signature, and — the part that has no other guard —
that a calibration artifact's weights actually reach the draw at run time.
"""

import os
import tempfile
import unittest
from collections import Counter
from random import Random
from unittest import mock

from framework import calibrate, pipeline
from framework.calibration.artifact import write_calibration
from framework.calibration.seeds import draw_weighted_seeds
from framework.generators.base_generator import BaseGenerator
from framework.profiling.gec_profiler import index_seed_edit_types


class _Edit:
    def __init__(self, type_):
        self.type = type_


class _Annotator:
    """Returns edits keyed off a marker in the incorrect text, so the test needs
    neither spaCy nor ERRANT."""

    def parse(self, text):
        return text

    def annotate(self, src, ref):
        return [_Edit(t) for t in src.split("|")[1].split(",")]


_ROWS = [
    {"incorrect": "a|R:DET", "correct": "a"},
    {"incorrect": "b|R:PREP", "correct": "b"},
    {"incorrect": "c|R:DET,R:PREP", "correct": "c"},
    {"incorrect": "d|R:DET", "correct": "d"},
]


class IndexTests(unittest.TestCase):
    def test_row_indexed_under_every_type_it_carries(self):
        index = index_seed_edit_types(_ROWS, annotator=_Annotator())
        self.assertEqual(index["R:DET"], [0, 2, 3])
        self.assertEqual(index["R:PREP"], [1, 2])

    def test_rows_without_usable_fields_are_skipped(self):
        rows = _ROWS + [{"incorrect": "", "correct": "x"}]
        index = index_seed_edit_types(rows, annotator=_Annotator())
        self.assertNotIn(4, index.get("R:DET", []))

    def test_empty_input_gives_empty_index(self):
        self.assertEqual(index_seed_edit_types([], annotator=_Annotator()), {})

    def test_generated_field_names_are_accepted_too(self):
        rows = [{"corrupted": "a|R:DET", "original": "a"}]
        self.assertEqual(index_seed_edit_types(rows, annotator=_Annotator()),
                         {"R:DET": [0]})


class DrawTests(unittest.TestCase):
    def setUp(self):
        self.index = index_seed_edit_types(_ROWS, annotator=_Annotator())

    def test_draw_length_matches_request(self):
        out = draw_weighted_seeds(_ROWS, self.index,
                                  {"R:DET": 0.5, "R:PREP": 0.5}, 10, Random(0))
        self.assertEqual(len(out), 10)

    def test_weights_shift_the_type_mix(self):
        heavy = draw_weighted_seeds(_ROWS, self.index,
                                    {"R:DET": 0.95, "R:PREP": 0.05}, 400, Random(1))
        # Row 1 is the only PREP-exclusive seed, so it should be rare.
        counts = Counter(row["incorrect"] for row in heavy)
        self.assertLess(counts["b|R:PREP"], 60)

    def test_deterministic_under_a_fixed_rng(self):
        a = draw_weighted_seeds(_ROWS, self.index, {"R:DET": 0.5, "R:PREP": 0.5},
                                20, Random(7))
        b = draw_weighted_seeds(_ROWS, self.index, {"R:DET": 0.5, "R:PREP": 0.5},
                                20, Random(7))
        self.assertEqual([r["incorrect"] for r in a], [r["incorrect"] for r in b])

    def test_type_absent_from_the_pool_is_skipped_with_a_warning(self):
        out = draw_weighted_seeds(_ROWS, self.index,
                                  {"R:DET": 0.5, "R:MORPH": 0.5}, 20, Random(0))
        self.assertEqual(len(out), 20)
        self.assertTrue(all("R:DET" in r["incorrect"] for r in out))

    def test_no_weights_falls_back_to_first_n_order(self):
        out = draw_weighted_seeds(_ROWS, self.index, {}, 3, Random(0))
        self.assertEqual([r["incorrect"] for r in out],
                         [r["incorrect"] for r in _ROWS[:3]])

    def test_empty_index_falls_back_to_first_n_order(self):
        out = draw_weighted_seeds(_ROWS, {}, {"R:DET": 1.0}, 2, Random(0))
        self.assertEqual(len(out), 2)

    def test_all_zero_weights_fall_back_to_first_n_order(self):
        out = draw_weighted_seeds(_ROWS, self.index, {"R:DET": 0.0}, 2, Random(0))
        self.assertEqual([r["incorrect"] for r in out],
                         [r["incorrect"] for r in _ROWS[:2]])


class SeedPoolSignatureTests(unittest.TestCase):
    def test_base_get_seed_pool_accepts_weights_and_ignores_them(self):
        from framework.tasks.gec.task import GECTask
        rows = [{"incorrect": "x", "correct": "y"}]
        out = GECTask().get_seed_pool({}, rows, "forward", seed_weights=None,
                                      rng=Random(0))
        self.assertEqual(out, rows)
        # Identity, not just equality: without weights _run_generation must hand
        # generate() the very list it handed it before this feature existed.
        self.assertIs(out, rows)

    def test_spam_get_seed_pool_still_returns_labeled_rows(self):
        from framework.tasks.spam.task import SpamTask
        cfg = {"dataset": {"source": "local",
                           "local": {"path": "framework/data/benchmarks/spam/x.csv",
                                     "format": "csv"}}}
        try:
            SpamTask().get_seed_pool(cfg, [], "forward", seed_weights=None,
                                     rng=Random(0))
        except TypeError:
            self.fail("spam get_seed_pool must accept seed_weights/rng")
        except Exception:
            pass  # missing fixture file is fine; the signature is what matters

    def test_gec_inverse_mode_ignores_weights_entirely(self):
        # Inverse mode injects a distribution directly; its seed pool is the
        # unchanged real data even when an artifact carries weights.
        from framework.tasks.gec.task import GECTask
        out = GECTask().get_seed_pool({"generation": {"sample_size": 2}}, _ROWS,
                                      "inverse", seed_weights={"R:DET": 1.0},
                                      rng=Random(0))
        self.assertIs(out, _ROWS)

    def test_gec_forward_with_weights_draws_through_the_index(self):
        from framework.tasks.gec.task import GECTask
        with mock.patch("framework.evaluators.gec._errant_shared.get_annotator",
                        return_value=_Annotator()):
            out = GECTask().get_seed_pool({"generation": {"sample_size": 8}}, _ROWS,
                                          "forward",
                                          seed_weights={"R:DET": 1.0, "R:PREP": 0.0},
                                          rng=Random(3))
        self.assertEqual(len(out), 8)
        self.assertTrue(all("R:DET" in r["incorrect"] for r in out))


# ── Run-time wiring (controller ruling R3) ────────────────────
#
# GEC forward+seeded never calls load_error_distribution, so _apply_calibration
# never runs for it: without the build_generation_context hook, an artifact's
# seed weights would silently never apply. These tests pin that path end to end.

# Seed texts are single whitespace tokens so a prompt can be read back as the
# exact seed it was built from.
_SEEDS = [("det_one", "R:DET"), ("det_two", "R:DET"), ("det_three", "R:DET"),
          ("prep_one", "R:PREP"), ("prep_two", "R:PREP"), ("prep_three", "R:PREP")]
_CSV = "text,correct\n" + "".join(
    f"{text}|{etype},{text}\n" for text, etype in _SEEDS
)


class _RecordingGenerator(BaseGenerator):
    """Echoes the seed's error type into a new sentence, the way the forward
    prompt asks the real generator to."""

    def __init__(self):
        self.prompts = []

    def call_api(self, prompt):
        self.prompts.append(prompt)
        etype = "R:PREP" if "R:PREP" in prompt else "R:DET"
        return (f"Error type: {etype}\n"
                f"Generated: a fresh new sentence|{etype}\n"
                "Ground truth: a fresh new sentence")

    def seeds_seen(self):
        """The seed sentence embedded in each generation prompt."""
        return [line for prompt in self.prompts
                for line in prompt.split() if "|" in line]


def _gec_config(path, sample_size=6, calibration_path=...):
    cfg = {
        "dataset": {"source": "local", "local": {"path": path, "format": "csv"}},
        "generation": {"provider": "openai", "model": "gpt-x", "num_runs": 1,
                       "sample_size": sample_size, "mode": "forward",
                       "seedless": False},
        "task": {"name": "gec"},
        "task_models": [],
    }
    if calibration_path is not ...:
        cfg["generation"]["calibration_path"] = calibration_path
    return cfg


class _GecBench(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.bench = os.path.join(self.dir.name, "bench.csv")
        with open(self.bench, "w", encoding="utf-8") as f:
            f.write(_CSV)
        self.artifact = os.path.join(self.dir.name, "cal.json")
        self.annotator = _Annotator()

    def _write_artifact(self, weights):
        write_calibration(self.artifact, {
            "meta": {}, "target": {"type_dist": {"R:DET": 0.5, "R:PREP": 0.5}},
            "calibrated": {"type_dist": dict(weights), "seed_weights": dict(weights)},
            "selected_round": 1, "rounds": [],
        })

    def _generate(self, cfg):
        gen = _RecordingGenerator()
        with mock.patch.object(pipeline, "load_generator", return_value=gen), \
             mock.patch("framework.evaluators.gec._errant_shared.get_annotator",
                        return_value=self.annotator):
            ctx = pipeline.build_generation_context(cfg)
            pipeline._run_generation(ctx["generator"], ctx["task"], cfg,
                                     ctx["real_data"], ctx["error_dist"],
                                     ctx["judge_call"], ctx["class_prob"],
                                     profile=ctx["profile"])
        return ctx, gen


class SeedWeightWiringTests(_GecBench):
    def test_artifact_weights_reach_the_draw(self):
        self._write_artifact({"R:DET": 1.0, "R:PREP": 0.0})
        ctx, gen = self._generate(_gec_config(self.bench,
                                              calibration_path=self.artifact))
        self.assertEqual(ctx["seed_weights"], {"R:DET": 1.0, "R:PREP": 0.0})
        seeds = gen.seeds_seen()
        self.assertEqual(len(seeds), 6)
        self.assertTrue(all(s.endswith("|R:DET") for s in seeds), seeds)

    def test_weights_are_published_onto_the_generation_config(self):
        self._write_artifact({"R:DET": 0.75, "R:PREP": 0.25})
        cfg = _gec_config(self.bench, calibration_path=self.artifact)
        with mock.patch.object(pipeline, "load_generator",
                               return_value=_RecordingGenerator()), \
             mock.patch("framework.evaluators.gec._errant_shared.get_annotator",
                        return_value=self.annotator):
            pipeline.build_generation_context(cfg)
        self.assertEqual(cfg["generation"]["seed_weights"],
                         {"R:DET": 0.75, "R:PREP": 0.25})

    def test_without_an_artifact_the_pool_is_the_unchanged_first_n_order(self):
        # The uncalibrated cell must generate exactly what it does today.
        ctx, gen = self._generate(_gec_config(self.bench, calibration_path=None))
        self.assertIsNone(ctx["seed_weights"])
        self.assertEqual(gen.seeds_seen(),
                         [f"{text}|{etype}" for text, etype in _SEEDS])

    def test_artifact_without_seed_weights_is_ignored(self):
        write_calibration(self.artifact, {
            "meta": {}, "target": {}, "calibrated": {"type_dist": {"R:DET": 1.0}},
            "selected_round": 0, "rounds": [],
        })
        ctx, gen = self._generate(_gec_config(self.bench,
                                              calibration_path=self.artifact))
        self.assertIsNone(ctx["seed_weights"])
        self.assertEqual(gen.seeds_seen(),
                         [f"{text}|{etype}" for text, etype in _SEEDS])

    def test_unreadable_artifact_falls_back_instead_of_crashing(self):
        with open(self.artifact, "w", encoding="utf-8") as f:
            f.write("{not json")
        ctx, gen = self._generate(_gec_config(self.bench,
                                              calibration_path=self.artifact))
        self.assertIsNone(ctx["seed_weights"])
        self.assertEqual(gen.seeds_seen(),
                         [f"{text}|{etype}" for text, etype in _SEEDS])

    def test_structurally_corrupt_artifact_falls_back_instead_of_crashing(self):
        # JSON-valid but not an object: .get() on it would raise AttributeError
        # deep inside the loader, which must not take a run down.
        with open(self.artifact, "w", encoding="utf-8") as f:
            f.write('["not", "a", "payload"]')
        ctx, _ = self._generate(_gec_config(self.bench,
                                            calibration_path=self.artifact))
        self.assertIsNone(ctx["seed_weights"])

    def test_non_numeric_weights_fall_back_instead_of_crashing(self):
        # A bad value would otherwise raise inside draw_weighted_seeds, mid-run.
        self._write_artifact({"R:DET": "lots"})
        ctx, gen = self._generate(_gec_config(self.bench,
                                              calibration_path=self.artifact))
        self.assertIsNone(ctx["seed_weights"])
        self.assertEqual(gen.seeds_seen(),
                         [f"{text}|{etype}" for text, etype in _SEEDS])

    def test_mixed_weights_with_one_bad_value_fall_back(self):
        # The good value comes FIRST on purpose: an `any(float(w) > 0 ...)` guard
        # short-circuits on it and never coerces the bad one, publishes the dict,
        # and then raises ValueError inside draw_weighted_seeds mid-run. Every
        # value must be validated, not just up to the first positive one.
        self._write_artifact({"R:DET": 1.0, "R:PREP": "bad"})
        ctx, gen = self._generate(_gec_config(self.bench,
                                              calibration_path=self.artifact))
        self.assertIsNone(ctx["seed_weights"])
        self.assertEqual(gen.seeds_seen(),
                         [f"{text}|{etype}" for text, etype in _SEEDS])

    def test_all_zero_weights_fall_back(self):
        self._write_artifact({"R:DET": 0.0, "R:PREP": 0.0})
        ctx, _ = self._generate(_gec_config(self.bench,
                                            calibration_path=self.artifact))
        self.assertIsNone(ctx["seed_weights"])

    def test_inverse_cell_never_looks_for_seed_weights(self):
        self._write_artifact({"R:DET": 1.0, "R:PREP": 0.0})
        cfg = _gec_config(self.bench, calibration_path=self.artifact)
        cfg["generation"]["mode"] = "inverse"
        with mock.patch.object(pipeline, "load_generator",
                               return_value=_RecordingGenerator()), \
             mock.patch("framework.evaluators.gec._errant_shared.get_annotator",
                        return_value=self.annotator):
            ctx = pipeline.build_generation_context(cfg)
        self.assertIsNone(ctx["seed_weights"])
        self.assertNotIn("seed_weights", cfg["generation"])


class SeedWeightProvenanceTests(_GecBench):
    """results.json is the only surviving record of what produced a benchmark
    (artifacts are gitignored), so a run steered by seed weights must say so."""

    def _meta(self, cfg):
        from framework.tasks.gec.task import GECTask
        return pipeline._build_meta(cfg, GECTask(), runs_completed=1,
                                    effective_samples_per_run=[6],
                                    real_baseline=False)

    def test_meta_records_the_seed_weight_artifact(self):
        self._write_artifact({"R:DET": 1.0, "R:PREP": 0.0})
        cfg = _gec_config(self.bench, calibration_path=self.artifact)
        self._generate(cfg)
        meta = self._meta(cfg)
        self.assertEqual(meta["calibration"]["path"], self.artifact)
        self.assertEqual(meta["calibration"]["selected_round"], 1)

    def test_meta_calibration_is_none_without_seed_weights(self):
        # The Task 6 isolation guarantee: no artifact must still end at None.
        cfg = _gec_config(self.bench, calibration_path=None)
        self._generate(cfg)
        self.assertIsNone(self._meta(cfg)["calibration"])

    def test_seed_weights_do_not_leak_into_the_next_config(self):
        # This change adds a SECOND writer of _LAST_CALIBRATION, so the Task 6
        # isolation guarantee now has a new direction to protect: a
        # seed-weighted run followed by an uncalibrated one in the same process
        # (scripts/compare_models.py loops configs) must not inherit the
        # artifact. build_generation_context's unconditional reset is what
        # holds this, and the write must stay after it.
        self._write_artifact({"R:DET": 1.0, "R:PREP": 0.0})
        calibrated = _gec_config(self.bench, calibration_path=self.artifact)
        self._generate(calibrated)
        self.assertIsNotNone(pipeline._LAST_CALIBRATION)  # contamination source is real

        plain = _gec_config(self.bench, calibration_path=None)
        self._generate(plain)
        self.assertIsNone(pipeline._LAST_CALIBRATION)
        self.assertIsNone(self._meta(plain)["calibration"])

    def test_a_failed_artifact_read_leaves_no_provenance(self):
        with open(self.artifact, "w", encoding="utf-8") as f:
            f.write("{not json")
        cfg = _gec_config(self.bench, calibration_path=self.artifact)
        self._generate(cfg)
        self.assertIsNone(self._meta(cfg)["calibration"])


class _BiasedGenerator(_RecordingGenerator):
    """Deterministically leaks every other R:DET seed out as R:PREP — the lossy
    channel seed weights exist to invert."""

    def __init__(self):
        super().__init__()
        self.det_calls = 0

    def call_api(self, prompt):
        self.prompts.append(prompt)
        etype = "R:PREP" if "R:PREP" in prompt else "R:DET"
        if etype == "R:DET":
            self.det_calls += 1
            if self.det_calls % 2 == 0:
                etype = "R:PREP"
        return (f"Error type: {etype}\n"
                f"Generated: a fresh new sentence|{etype}\n"
                "Ground truth: a fresh new sentence")


class SeedModeCalibrationTests(_GecBench):
    def _run(self, generator, rounds=1, sample_size=60):
        cfg = _gec_config(self.bench, sample_size=sample_size,
                          calibration_path=None)
        out = os.path.join(self.dir.name, "out.json")
        with mock.patch.object(pipeline, "load_generator", return_value=generator), \
             mock.patch("framework.evaluators.gec._errant_shared.get_annotator",
                        return_value=self.annotator):
            return calibrate.run_calibration(
                cfg, rounds=rounds, alpha=0.5, tolerance=0.001,
                sample_size=sample_size, output_path=out,
            )

    def test_target_comes_from_the_seed_pools_own_profile(self):
        # R1: ctx["error_dist"] is None for this cell, so the setpoint is the
        # ERRANT profile of the real seed pool.
        payload = self._run(_RecordingGenerator())
        self.assertEqual(payload["target"]["type_dist"],
                         {"R:DET": 0.5, "R:PREP": 0.5})
        self.assertEqual(payload["rounds"][0]["request"]["type_dist"],
                         payload["target"]["type_dist"])

    def test_count_dist_is_a_diagnostic_not_a_calibrated_dimension(self):
        # R2: forward mode gives no control input for edit count, so steering on
        # it would make convergence unreachable.
        payload = self._run(_RecordingGenerator())
        self.assertEqual(set(payload["target"]), {"type_dist"})
        self.assertEqual(set(payload["rounds"][0]["jsd"]), {"type_dist"})
        self.assertIn("count_dist", payload["rounds"][0]["diagnostic"])

    def test_artifact_records_seed_weights_under_the_name_runs_read(self):
        payload = self._run(_RecordingGenerator())
        self.assertEqual(payload["calibrated"]["seed_weights"],
                         payload["calibrated"]["type_dist"])

    def test_written_artifact_is_consumable_by_a_run(self):
        # The driver and the run must agree on the artifact's shape.
        payload = self._run(_RecordingGenerator())
        write_calibration(self.artifact, payload)
        cfg = _gec_config(self.bench, calibration_path=self.artifact)
        from framework.tasks.gec.task import GECTask
        weights = pipeline._load_seed_weights(cfg, GECTask(), "corruption",
                                              "forward", False)
        self.assertEqual(weights, payload["calibrated"]["seed_weights"])

    def test_biased_channel_pushes_the_under_delivered_type_up(self):
        payload = self._run(_BiasedGenerator())
        self.assertGreater(payload["rounds"][0]["jsd"]["type_dist"], 0.001)
        self.assertGreater(payload["rounds"][1]["request"]["type_dist"]["R:DET"],
                           payload["target"]["type_dist"]["R:DET"])


if __name__ == "__main__":
    unittest.main()
