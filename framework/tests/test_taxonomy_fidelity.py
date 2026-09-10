import json
import os
import tempfile
import unittest
from unittest import mock

from framework.pipeline import run_pipeline
from framework.profiling.taxonomy_fidelity import (
    DISTRIBUTION_KEYS,
    SCALAR_KEYS,
    compare_distribution,
    compare_scalar,
    compare_taxonomy_profiles,
    pool_taxonomy_profiles,
    sanitize_taxonomy_profile,
    select_reference_taxonomy_profile,
)
from framework.profiling.taxonomy_profiler import profile_taxonomy_rows
from framework.tasks.taxonomy import TaxonomyTask
from framework.generators.base_generator import BaseGenerator


class FakeGenerator(BaseGenerator):
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
        profile = task.build_fidelity_profile([
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

    def test_sanitising_keeps_the_ontology_id(self):
        # The benchmark's own id, not a class identifier. Without it nothing
        # downstream can tell k subtrees of ONE ontology from k ontologies.
        raw = profile_taxonomy_rows([{
            "ontology_id": "onto", "domain": "demo",
            "classes": ["SecretRoot", "SecretChild"],
            "subclass_axioms": [["SecretChild", "SecretRoot"]],
        }])
        self.assertEqual(sanitize_taxonomy_profile(raw)["taxonomies"][0]["ontology_id"], "onto")

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


# One ontology, a DAG (DessertPizza has two parents), whose seed pool under
# max_depth 3 / min_classes 3 is three subtrees of different sizes and shapes:
# Dessert (6 classes), Food (10), Pizza (5).
_ONTOLOGY = [{
    "ontology_id": "onto",
    "domain": "food",
    "classes": ["Food", "Pizza", "Dessert", "Margherita", "Napoletana", "Gelato",
                "Sorbet", "Tiramisu", "DessertPizza", "Nutella"],
    "subclass_axioms": [["Pizza", "Food"], ["Dessert", "Food"],
                        ["Margherita", "Pizza"], ["Napoletana", "Pizza"],
                        ["Gelato", "Dessert"], ["Sorbet", "Dessert"],
                        ["Tiramisu", "Dessert"], ["DessertPizza", "Pizza"],
                        ["DessertPizza", "Dessert"], ["Nutella", "DessertPizza"]],
}]
_SEEDED = {"task": {"name": "taxonomy"},
           "generation": {"mode": "forward", "seedless": False,
                          "seed_pool": {"max_depth": 3, "min_classes": 3}}}


def _renamed(subtree):
    """The subtree's exact structure under new names -- what forward+seeded
    produces from it when verification accepts it."""
    names = {c: f"X{i}" for i, c in enumerate(subtree["classes"])}
    return {"domain": "marine biology",
            "classes": [names[c] for c in subtree["classes"]],
            "subclass_axioms": [[names[c], names[p]] for c, p in subtree["subclass_axioms"]]}


def _item(ontology_id, **stats):
    base = {"ontology_id": ontology_id, "domain": "d", "n_classes": 2,
            "n_subclass_axioms": 1, "n_roots": 1, "n_leaves": 1, "max_depth": 1,
            "mean_depth": 0.5, "multiple_parent_fraction": 0.0,
            "depth_distribution": {"0": 1, "1": 1},
            "parent_count_distribution": {"0": 1, "1": 1},
            "child_count_distribution": {"0": 1, "1": 1},
            "has_cycle": False,
            "validation": {"unknown_class_edges": 0, "self_loops": 0,
                           "duplicate_subclass_axioms": 0}}
    return {**base, **stats}


class PooledReferenceTests(unittest.TestCase):
    """A seeded session's real side is several subtrees of ONE ontology.

    select_reference_taxonomy_profile raised on more than one real taxonomy, so
    every seeded taxonomy session died in the fidelity step -- after all the
    generation and evaluation spend, with no profile.json and no plots. Subtrees
    of one ontology are now pooled into one reference; the strict raise stays
    for genuinely different ontologies."""

    def test_subtrees_of_one_ontology_pool_into_one_reference(self):
        task = TaxonomyTask()
        ref = task.get_real_eval_samples(_SEEDED, _ONTOLOGY)
        self.assertEqual(len(ref), 3)
        real = task.build_fidelity_profile(ref)
        synthetic = task.build_fidelity_profile([_renamed(r) for r in ref])
        out = task.compare_fidelity_profiles(real, synthetic)
        self.assertIsInstance(out["real_profile"], dict)
        self.assertEqual(out["real_profile"]["pooled_taxonomies"], 3)
        self.assertEqual(out["real_profile"]["ontology_id"], "onto")
        self.assertEqual(out["real_profile"]["domain"], "food")

    def test_the_pool_is_summarised_with_the_synthetic_sides_statistics(self):
        # THE invariant that justifies mean-of-items. A synthetic side that is
        # structurally the pool itself must show no gap at all: if the real side
        # were summarised with a different statistic than the synthetic side,
        # the gap would measure the summary rather than the generation.
        from framework.plotting.plots import _taxonomy_distribution_series

        task = TaxonomyTask()
        ref = task.get_real_eval_samples(_SEEDED, _ONTOLOGY)
        real = task.build_fidelity_profile(ref)
        synthetic = task.build_fidelity_profile([_renamed(r) for r in ref])
        out = task.compare_fidelity_profiles(real, synthetic)

        scalars = out["aggregate"]["scalar_characteristics"]
        for key in SCALAR_KEYS:
            self.assertEqual(out["real_profile"][key], scalars[key]["synthetic"]["mean"], key)
        for key in DISTRIBUTION_KEYS:
            series = _taxonomy_distribution_series(out, key)
            self.assertTrue(series["labels"], key)
            for label, real_p, synth_p in zip(series["labels"], series["real"],
                                              series["synthetic_mean"]):
                self.assertAlmostEqual(real_p, synth_p, places=9, msg=f"{key}[{label}]")

    def test_distributions_are_pooled_as_probabilities_not_summed_counts(self):
        # Summed counts would weight a large item over a small one, which the
        # synthetic side never does: 4 classes vs 2 here, 91 vs 5 on Pizza.
        pooled = pool_taxonomy_profiles([
            _item("onto", n_classes=2, depth_distribution={"0": 1, "1": 1}),
            _item("onto", n_classes=4, depth_distribution={"0": 1, "1": 3}),
        ])
        self.assertEqual(pooled["n_classes"], 3.0)
        dist = pooled["depth_distribution"]
        self.assertEqual(list(dist), ["0", "1"])
        self.assertAlmostEqual(dist["0"], 0.375)     # (1/2 + 1/4) / 2; summed: 2/6
        self.assertAlmostEqual(dist["1"], 0.625)     # (1/2 + 3/4) / 2; summed: 4/6

    def test_the_pool_reports_its_size_flags_and_shared_identity(self):
        pooled = pool_taxonomy_profiles([
            _item("onto", has_cycle=False, domain="food",
                  validation={"unknown_class_edges": 0, "self_loops": 1,
                              "duplicate_subclass_axioms": 0}),
            _item("onto", has_cycle=True, domain="food",
                  validation={"unknown_class_edges": 2, "self_loops": 0,
                              "duplicate_subclass_axioms": 0}),
            _item("onto", has_cycle=False, domain="food"),
        ])
        self.assertEqual(pooled["pooled_taxonomies"], 3)
        self.assertEqual(pooled["ontology_id"], "onto")
        self.assertEqual(pooled["domain"], "food")
        self.assertIs(pooled["has_cycle"], True)
        self.assertEqual(pooled["validation"], {"unknown_class_edges": 2, "self_loops": 1,
                                                "duplicate_subclass_axioms": 0})

    def test_a_pool_across_domains_has_no_single_domain(self):
        pooled = pool_taxonomy_profiles([_item("onto", domain="a"), _item("onto", domain="b")])
        self.assertIsNone(pooled["domain"])

    def test_a_scalar_undefined_for_one_item_is_left_out_of_its_mean(self):
        # A cyclic taxonomy has no depth; aggregate_comparisons skips such None
        # values on the synthetic side, so the real side must skip them too.
        pooled = pool_taxonomy_profiles([
            _item("onto", max_depth=None, mean_depth=None, depth_distribution={}),
            _item("onto", max_depth=3, mean_depth=1.5, depth_distribution={"0": 1, "1": 3}),
        ])
        self.assertEqual(pooled["max_depth"], 3.0)
        self.assertEqual(pooled["mean_depth"], 1.5)
        self.assertEqual(pooled["depth_distribution"], {"0": 0.25, "1": 0.75})

    def test_one_taxonomy_is_still_returned_unpooled(self):
        only = _item("onto")
        self.assertIs(select_reference_taxonomy_profile({"taxonomies": [only]}), only)

    def test_a_named_reference_matching_the_shared_id_pools(self):
        profile = {"taxonomies": [_item("onto"), _item("onto")]}
        ref = select_reference_taxonomy_profile(profile, ontology_id="onto")
        self.assertEqual(ref["pooled_taxonomies"], 2)

    def test_a_named_reference_other_than_the_shared_id_raises(self):
        profile = {"taxonomies": [_item("onto"), _item("onto")]}
        with self.assertRaises(ValueError):
            select_reference_taxonomy_profile(profile, ontology_id="other")

    def test_taxonomies_of_different_ontologies_still_raise(self):
        profile = {"taxonomies": [_item("pizza"), _item("wine")]}
        with self.assertRaises(ValueError):
            select_reference_taxonomy_profile(profile)

    def test_a_taxonomy_without_an_id_still_raises(self):
        # "Same ontology" cannot be assumed of an item that does not say which.
        for ids in ((None, None), ("onto", None)):
            with self.subTest(ids=ids):
                profile = {"taxonomies": [_item(i) for i in ids]}
                with self.assertRaises(ValueError):
                    select_reference_taxonomy_profile(profile)


if __name__ == "__main__":
    unittest.main()
