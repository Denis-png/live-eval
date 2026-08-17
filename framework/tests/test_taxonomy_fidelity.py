import json
import os
import tempfile
import unittest
from unittest import mock

from framework.pipeline import run_pipeline
from framework.profiling.taxonomy_fidelity import (
    compare_distribution,
    compare_scalar,
    compare_taxonomy_profiles,
    sanitize_taxonomy_profile,
)
from framework.profiling.taxonomy_profiler import profile_taxonomy_rows
from framework.tasks.taxonomy import TaxonomyTask


class FakeGenerator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def call_api(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _taxonomy(domain="demo", classes=None, axioms=None):
    class_list = classes or ["Root", "Child"]
    return {
        "domain": domain,
        "classes": class_list,
        "subclass_axioms": axioms if axioms is not None else [[class_list[1], class_list[0]]],
    }


def _profile(rows):
    return sanitize_taxonomy_profile(profile_taxonomy_rows(rows))


class ScalarComparisonTests(unittest.TestCase):
    def test_identical_scalar_profile_zero_difference(self):
        out = compare_scalar({"n_classes": 2}, {"n_classes": 2}, "n_classes")
        self.assertEqual(out["absolute_difference"], 0.0)
        self.assertEqual(out["relative_difference"], 0.0)

    def test_scalar_difference_calculations(self):
        out = compare_scalar({"n_classes": 10}, {"n_classes": 12}, "n_classes")
        self.assertEqual(out["real"], 10)
        self.assertEqual(out["synthetic"], 12)
        self.assertEqual(out["absolute_difference"], 2.0)
        self.assertEqual(out["relative_difference"], 0.2)

    def test_zero_denominator_safe(self):
        same = compare_scalar({"n_roots": 0}, {"n_roots": 0}, "n_roots")
        different = compare_scalar({"n_roots": 0}, {"n_roots": 2}, "n_roots")
        self.assertEqual(same["relative_difference"], 0.0)
        self.assertIsNone(different["relative_difference"])


class DistributionComparisonTests(unittest.TestCase):
    def test_identical_distribution_jsd_zero(self):
        out = compare_distribution(
            {"depth_distribution": {"0": 1, "1": 2}},
            {"depth_distribution": {"0": 1, "1": 2}},
            "depth_distribution",
        )
        self.assertEqual(out["jensen_shannon_divergence"], 0.0)

    def test_different_distribution_jsd_positive(self):
        out = compare_distribution(
            {"depth_distribution": {"0": 3}},
            {"depth_distribution": {"1": 3}},
            "depth_distribution",
        )
        self.assertGreater(out["jensen_shannon_divergence"], 0.0)

    def test_missing_bins_are_aligned(self):
        out = compare_distribution(
            {"depth_distribution": {"0": 1, "1": 1}},
            {"depth_distribution": {"1": 1, "2": 1}},
            "depth_distribution",
        )
        self.assertGreater(out["jensen_shannon_divergence"], 0.0)
        self.assertLessEqual(out["jensen_shannon_divergence"], 1.0)


class TaxonomyFidelityTests(unittest.TestCase):
    def test_one_real_vs_one_synthetic_profile(self):
        real = _profile([_taxonomy(classes=["A", "B"], axioms=[["B", "A"]])])
        synthetic = _profile([_taxonomy(classes=["X", "Y"], axioms=[["Y", "X"]])])
        out = compare_taxonomy_profiles(real, synthetic)
        self.assertEqual(len(out["comparisons"]), 1)
        self.assertEqual(
            out["comparisons"][0]["scalar_characteristics"]["n_classes"]["absolute_difference"],
            0.0,
        )
        self.assertEqual(
            out["comparisons"][0]["distribution_characteristics"]["depth_distribution"][
                "jensen_shannon_divergence"
            ],
            0.0,
        )

    def test_one_real_vs_multiple_synthetic_profiles_and_aggregate(self):
        real = _profile([_taxonomy(classes=["A", "B"], axioms=[["B", "A"]])])
        synthetic = _profile([
            _taxonomy(classes=["X", "Y"], axioms=[["Y", "X"]]),
            _taxonomy(classes=["R", "S", "T"], axioms=[["S", "R"], ["T", "R"]]),
        ])
        out = compare_taxonomy_profiles(real, synthetic)
        self.assertEqual(out["aggregate"]["n_synthetic_taxonomies"], 2)
        n_classes = out["aggregate"]["scalar_characteristics"]["n_classes"]
        self.assertEqual(n_classes["synthetic"]["mean"], 2.5)
        self.assertEqual(n_classes["synthetic"]["min"], 2.0)
        self.assertEqual(n_classes["synthetic"]["max"], 3.0)

    def test_generated_taxonomy_is_profiled_with_existing_profiler(self):
        task = TaxonomyTask()
        profile = task.profile_dataset([
            _taxonomy(
                classes=["A", "B", "C", "D"],
                axioms=[["B", "A"], ["C", "A"], ["D", "B"], ["D", "C"]],
            )
        ])
        taxonomy = profile["taxonomies"][0]
        self.assertEqual(taxonomy["n_roots"], 1)
        self.assertEqual(taxonomy["max_depth"], 2)
        self.assertEqual(taxonomy["parent_count_distribution"], {"0": 1, "1": 2, "2": 1})

    def test_fidelity_output_excludes_class_uri_maps_and_real_class_lists(self):
        raw = profile_taxonomy_rows([
            {
                "ontology_id": "real",
                "domain": "demo",
                "classes": ["SecretRoot", "SecretChild"],
                "subclass_axioms": [["SecretChild", "SecretRoot"]],
                "metadata": {"class_uri_map": {"SecretRoot": "http://example.org/SecretRoot"}},
            }
        ])
        sanitized = sanitize_taxonomy_profile(raw)
        text = json.dumps(sanitized)
        self.assertNotIn("SecretRoot", text)
        self.assertNotIn("SecretChild", text)
        self.assertNotIn("class_uri_map", text)

    def test_multiple_real_profiles_require_explicit_reference(self):
        real = _profile([
            _taxonomy(classes=["A", "B"], axioms=[["B", "A"]]),
            _taxonomy(classes=["C", "D"], axioms=[["D", "C"]]),
        ])
        synthetic = _profile([_taxonomy(classes=["X", "Y"], axioms=[["Y", "X"]])])
        with self.assertRaises(ValueError):
            compare_taxonomy_profiles(real, synthetic)

    def test_fidelity_data_does_not_enter_generation_prompts(self):
        real_profile = {
            "profile_type": "taxonomy_structure",
            "taxonomies": [
                {
                    "domain": "demo",
                    "n_classes": 2,
                    "n_subclass_axioms": 1,
                    "n_roots": 1,
                    "n_leaves": 1,
                    "max_depth": 1,
                    "mean_depth": 0.5,
                    "depth_distribution": {"0": 1, "1": 1},
                    "parent_count_distribution": {"0": 1, "1": 1},
                    "child_count_distribution": {"0": 1, "1": 1},
                    "multiple_parent_fraction": 0.0,
                    "comparisons": [{"absolute_difference": 999}],
                    "fidelity": {"secret_feedback": "make it closer"},
                }
            ],
        }
        prompt = TaxonomyTask().build_structured_generation_prompt(real_profile)
        self.assertNotIn("secret_feedback", prompt)
        self.assertNotIn("absolute_difference", prompt)

    def test_pipeline_writes_taxonomy_fidelity_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = os.path.join(tmp, "taxonomy.jsonl")
            profile_path = os.path.join(tmp, "profile.json")
            output_dir = os.path.join(tmp, "runs")
            real_row = {
                "ontology_id": "real",
                "domain": "demo",
                "classes": ["RealRoot", "RealChild"],
                "subclass_axioms": [["RealChild", "RealRoot"]],
            }
            with open(dataset_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(real_row) + "\n")
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_taxonomy_rows([real_row]), f)

            generation_generator = FakeGenerator([
                json.dumps(_taxonomy(classes=["GeneratedRoot", "GeneratedChild"]))
            ])
            model_generator = FakeGenerator([
                '{"subclass_axioms": [["GeneratedChild", "GeneratedRoot"]]}'
            ])
            config = {
                "dataset": {
                    "source": "local",
                    "local": {"path": dataset_path, "format": "jsonl"},
                },
                "generation": {
                    "provider": "openrouter",
                    "model": "generator",
                    "api_key": "key",
                    "temperature": 0,
                    "sample_size": 1,
                    "num_runs": 1,
                    "profile_path": profile_path,
                },
                "task": {"name": "taxonomy"},
                "task_models": [
                    {"type": "llm", "name": "evaluator", "provider": "openrouter", "api_key": "key"}
                ],
                "evaluation": {"real_baseline": False},
                "output": {"base_dir": output_dir, "plots": False, "session_id": "test"},
            }
            with mock.patch("framework.pipeline.load_generator", return_value=generation_generator), \
                 mock.patch("framework.generators.factory.load_generator", return_value=model_generator):
                run_pipeline(config)

            profile_json = os.path.join(output_dir, "taxonomy", "test", "profile.json")
            with open(profile_json, encoding="utf-8") as f:
                artifact = json.load(f)

        self.assertEqual(artifact["fidelity"]["profile_type"], "taxonomy_structural_fidelity")
        self.assertEqual(len(artifact["fidelity"]["synthetic_profiles"]), 1)
        self.assertNotIn("RealRoot", json.dumps(artifact["fidelity"]))
        self.assertNotIn("RealChild", json.dumps(artifact["fidelity"]))
        self.assertNotIn("GeneratedChild\", \"GeneratedRoot", model_generator.prompts[0])


if __name__ == "__main__":
    unittest.main()
