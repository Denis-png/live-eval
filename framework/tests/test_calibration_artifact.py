import json
import os
import tempfile
import unittest
from unittest import mock

from framework.calibration.artifact import (
    calibration_filename,
    default_calibration_path,
    load_calibration,
    resolve_calibration_path,
    targets_match,
    write_calibration,
)


def _config(path="framework/data/benchmarks/spam/sms_spam_ham_300.csv"):
    return {"dataset": {"source": "local", "local": {"path": path, "format": "csv"}},
            "generation": {"mode": "inverse", "seedless": False}}


class _Task:
    def get_task_name(self):
        return "spam"


class NamingTests(unittest.TestCase):
    def test_filename_carries_benchmark_size_and_cell(self):
        self.assertEqual(
            calibration_filename(_config(), "spam", "class_conditional", 120),
            "sms_spam_ham_300_120_inverse_seeded_calibration.json",
        )

    def test_filename_cannot_collide_with_a_profile_glob(self):
        # _resolve_profile_path globs "*_<task>_profile.json".
        name = calibration_filename(_config(), "spam", "class_conditional", 120)
        self.assertFalse(name.endswith("_spam_profile.json"))

    def test_default_path_lands_in_the_task_profile_dir(self):
        path = default_calibration_path(_config(), "spam", "class_conditional", 120)
        self.assertEqual(os.path.dirname(path), "framework/data/profiles/spam")


class RoundTripTests(unittest.TestCase):
    def test_count_dist_int_keys_survive_json(self):
        # JSON has no int keys. Without coercion _sample_categories draws a str
        # as n and raises TypeError on `n > len(keys)` deep in the run loop.
        payload = {
            "meta": {"task": "spam"},
            "target": {"type_dist": {"a": 1.0}, "count_dist": {1: 0.4, 2: 0.6}},
            "calibrated": {"type_dist": {"a": 1.0}, "count_dist": {1: 0.3, 2: 0.7}},
            "rounds": [],
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            write_calibration(path, payload)
            raw = json.load(open(path, encoding="utf-8"))
            self.assertEqual(sorted(raw["calibrated"]["count_dist"]), ["1", "2"])

            loaded = load_calibration(path)
            self.assertEqual(sorted(loaded["calibrated"]["count_dist"]), [1, 2])
            self.assertEqual(sorted(loaded["target"]["count_dist"]), [1, 2])

    def test_type_dist_keys_are_left_as_strings(self):
        payload = {"meta": {}, "target": {"type_dist": {"R:DET": 1.0}, "count_dist": {}},
                   "calibrated": {"type_dist": {"R:DET": 1.0}, "count_dist": {}},
                   "rounds": []}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            write_calibration(path, payload)
            self.assertEqual(list(load_calibration(path)["calibrated"]["type_dist"]),
                             ["R:DET"])

    def test_write_creates_missing_directories(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nested", "deeper", "c.json")
            write_calibration(path, {"meta": {}, "target": {}, "calibrated": {},
                                     "rounds": []})
            self.assertTrue(os.path.exists(path))


class ResolveTests(unittest.TestCase):
    def test_explicit_path_wins(self):
        cfg = _config()
        cfg["generation"]["calibration_path"] = "/tmp/pinned.json"
        self.assertEqual(resolve_calibration_path(cfg, _Task(), "class_conditional"),
                         "/tmp/pinned.json")

    def test_explicit_null_disables_lookup(self):
        cfg = _config()
        cfg["generation"]["calibration_path"] = None
        self.assertIsNone(resolve_calibration_path(cfg, _Task(), "class_conditional"))

    def test_missing_artifact_resolves_to_none(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _config(os.path.join(d, "bench.csv"))
            self.assertIsNone(
                resolve_calibration_path(cfg, _Task(), "class_conditional")
            )

    def test_single_artifact_discovered_by_glob(self):
        # Glob-discovery path: no calibration_path key, artifact exists on disk
        cfg = _config()
        # Ensure no calibration_path key (it should not be present)
        cfg["generation"].pop("calibration_path", None)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temporary profile directory
            filename = calibration_filename(cfg, "spam", "class_conditional", 120)
            expected_path = os.path.join(tmpdir, filename)

            # Write a real artifact file
            payload = {
                "meta": {"task": "spam"},
                "target": {"type_dist": {"a": 1.0}, "count_dist": {1: 1.0}},
                "calibrated": {"type_dist": {"a": 1.0}, "count_dist": {1: 1.0}},
                "rounds": [],
            }
            write_calibration(expected_path, payload)

            # Patch profile_dir to return tmpdir
            with mock.patch("framework.pipeline.profile_dir", return_value=tmpdir):
                result = resolve_calibration_path(cfg, _Task(), "class_conditional")
                self.assertEqual(result, expected_path)

    def test_newest_artifact_wins_among_several(self):
        # Glob-discovery with tie-breaking: multiple artifacts, newest by mtime wins
        cfg = _config()
        cfg["generation"].pop("calibration_path", None)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two artifacts for the same benchmark and cell, different sample sizes
            path_1 = os.path.join(tmpdir, calibration_filename(cfg, "spam", "class_conditional", 100))
            path_2 = os.path.join(tmpdir, calibration_filename(cfg, "spam", "class_conditional", 120))

            payload = {
                "meta": {"task": "spam"},
                "target": {"type_dist": {"a": 1.0}, "count_dist": {1: 1.0}},
                "calibrated": {"type_dist": {"a": 1.0}, "count_dist": {1: 1.0}},
                "rounds": [],
            }
            write_calibration(path_1, payload)
            write_calibration(path_2, payload)

            # Set explicit mtimes: path_1 older, path_2 newer
            os.utime(path_1, (1000000, 1000000))
            os.utime(path_2, (2000000, 2000000))

            # Patch profile_dir to return tmpdir
            with mock.patch("framework.pipeline.profile_dir", return_value=tmpdir):
                result = resolve_calibration_path(cfg, _Task(), "class_conditional")
                self.assertEqual(result, path_2)


class TargetMatchTests(unittest.TestCase):
    def test_identical_targets_match(self):
        t = {"type_dist": {"a": 0.5, "b": 0.5}, "count_dist": {1: 1.0}}
        self.assertTrue(targets_match(t, {"type_dist": {"a": 0.5, "b": 0.5},
                                          "count_dist": {1: 1.0}}))

    def test_drifted_target_does_not_match(self):
        a = {"type_dist": {"a": 0.5, "b": 0.5}, "count_dist": {1: 1.0}}
        b = {"type_dist": {"a": 0.9, "b": 0.1}, "count_dist": {1: 1.0}}
        self.assertFalse(targets_match(a, b))

    def test_new_category_does_not_match(self):
        a = {"type_dist": {"a": 1.0}, "count_dist": {1: 1.0}}
        b = {"type_dist": {"a": 1.0, "c": 0.0}, "count_dist": {1: 1.0}}
        self.assertFalse(targets_match(a, b))

    def test_float_noise_within_atol_matches(self):
        a = {"type_dist": {"a": 0.5000000001}, "count_dist": {}}
        b = {"type_dist": {"a": 0.5}, "count_dist": {}}
        self.assertTrue(targets_match(a, b))
