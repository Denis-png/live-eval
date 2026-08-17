import json
import unittest
from unittest import mock

from framework.pipeline import _run_generation
from framework.profiling.taxonomy_fidelity import (
    build_generation_feedback,
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


PROFILE = {
    "profile_type": "taxonomy_structure",
    "taxonomies": [
        {
            "ontology_id": "secret_ontology",
            "domain": "demo",
            "n_classes": 4,
            "n_subclass_axioms": 3,
            "n_roots": 1,
            "n_leaves": 2,
            "roots": ["SecretRoot"],
            "leaves": ["SecretLeaf"],
            "max_depth": 2,
            "mean_depth": 1.0,
            "depth_distribution": {"0": 1, "1": 2, "2": 1},
            "parent_count_distribution": {"0": 1, "1": 3},
            "child_count_distribution": {"0": 2, "1": 1, "2": 1},
            "multiple_parent_fraction": 0.0,
            "subclass_axioms": [["SecretLeaf", "SecretRoot"]],
            "metadata": {"class_uri_map": {"SecretRoot": "http://example.org/SecretRoot"}},
            "class_depths": {"SecretRoot": 0, "SecretLeaf": 1},
        }
    ],
}


def _taxonomy(classes, axioms):
    return {"domain": "demo", "classes": classes, "subclass_axioms": axioms}


def _response(classes, axioms):
    return json.dumps(_taxonomy(classes, axioms))


def _profile(row):
    return sanitize_taxonomy_profile(profile_taxonomy_rows([row]))


def _feedback_for(synthetic_row, tolerances=None):
    real = sanitize_taxonomy_profile(PROFILE)
    synthetic = _profile(synthetic_row)
    comparison = compare_taxonomy_profiles(real, synthetic)
    return build_generation_feedback(
        comparison["real_profile"],
        comparison["synthetic_profiles"][0],
        comparison["comparisons"][0],
        tolerances,
    )


class TaxonomyFeedbackBuilderTests(unittest.TestCase):
    def test_metrics_within_tolerance_produce_no_feedback(self):
        row = _taxonomy(
            ["R", "A", "B", "C"],
            [["A", "R"], ["B", "R"], ["C", "A"]],
        )
        feedback = _feedback_for(row)
        self.assertTrue(feedback["within_tolerance"])
        self.assertEqual(feedback["messages"], [])

    def test_too_many_roots_produces_reduction_guidance(self):
        row = _taxonomy(["A", "B", "C", "D"], [])
        feedback = _feedback_for(row, {"distribution_jsd": 1.0})
        text = " ".join(feedback["messages"])
        self.assertIn("too many root classes", text)
        self.assertIn("fewer roots", text)

    def test_too_few_subclass_axioms_produces_connectivity_guidance(self):
        row = _taxonomy(["R", "A", "B", "C"], [["A", "R"]])
        feedback = _feedback_for(row, {"distribution_jsd": 1.0})
        text = " ".join(feedback["messages"])
        self.assertIn("too few direct subclass relations", text)
        self.assertIn("Increase connectivity", text)

    def test_shallow_and_deep_hierarchy_feedback(self):
        shallow = _taxonomy(["R", "A", "B", "C"], [["A", "R"], ["B", "R"], ["C", "R"]])
        deep = _taxonomy(["R", "A", "B", "C"], [["A", "R"], ["B", "A"], ["C", "B"]])
        shallow_text = " ".join(_feedback_for(shallow, {"distribution_jsd": 1.0})["messages"])
        deep_text = " ".join(_feedback_for(deep, {"distribution_jsd": 1.0})["messages"])
        self.assertIn("shallower than the target", shallow_text)
        self.assertIn("deeper than the target", deep_text)

    def test_multiple_parent_fraction_feedback(self):
        real = _profile(_taxonomy(
            ["R1", "R2", "A", "B"],
            [["A", "R1"], ["A", "R2"], ["B", "A"]],
        ))
        synthetic = _profile(_taxonomy(
            ["R1", "R2", "A", "B"],
            [["A", "R1"], ["B", "A"]],
        ))
        comparison = compare_taxonomy_profiles(real, synthetic)
        feedback = build_generation_feedback(
            comparison["real_profile"],
            comparison["synthetic_profiles"][0],
            comparison["comparisons"][0],
            {"distribution_jsd": 1.0},
        )
        self.assertIn("Multiple inheritance was below the target", " ".join(feedback["messages"]))

    def test_distribution_jsd_inside_tolerance(self):
        row = _taxonomy(["R", "A", "B", "C"], [["A", "R"], ["B", "R"], ["C", "A"]])
        feedback = _feedback_for(row, {"distribution_jsd": 0.0})
        self.assertTrue(feedback["within_tolerance"])

    def test_distribution_jsd_outside_tolerance(self):
        row = _taxonomy(["A", "B", "C", "D"], [])
        feedback = _feedback_for(row, {"count_relative": 1.0, "depth_absolute": 9.0,
                                       "rate_absolute": 1.0, "distribution_jsd": 0.01})
        text = " ".join(feedback["messages"])
        self.assertIn("depth distribution", text)
        self.assertIn("parent-count distribution", text)

    def test_zero_denominator_count_feedback(self):
        comparison = {
            "scalar_characteristics": {
                "n_classes": {"real": 0, "synthetic": 0},
                "n_subclass_axioms": {"real": 0, "synthetic": 2},
            },
            "distribution_characteristics": {},
        }
        feedback = build_generation_feedback({}, {}, comparison)
        text = " ".join(feedback["messages"])
        self.assertNotIn("classes", text)
        self.assertIn("too many direct subclass relations", text)

    def test_missing_depth_values_skip_depth_feedback(self):
        comparison = {
            "scalar_characteristics": {
                "max_depth": {"real": None, "synthetic": 3},
                "mean_depth": {"real": 1.0, "synthetic": None},
            },
            "distribution_characteristics": {},
        }
        feedback = build_generation_feedback({}, {}, comparison)
        self.assertTrue(feedback["within_tolerance"])
        self.assertEqual(feedback["messages"], [])

    def test_feedback_prompt_contains_only_structural_guidance(self):
        row = _taxonomy(["A", "B", "C", "D"], [])
        feedback = _feedback_for(row)
        prompt = TaxonomyTask().build_structured_generation_prompt(PROFILE, feedback=feedback)
        self.assertIn("Feedback from previous generation", prompt)
        self.assertIn("too many root classes", prompt)
        self.assertNotIn("SecretRoot", prompt)
        self.assertNotIn("SecretLeaf", prompt)
        self.assertNotIn("SecretLeaf\", \"SecretRoot", prompt)
        self.assertNotIn("http://example.org/SecretRoot", prompt)
        self.assertNotIn("secret_ontology", prompt)
        self.assertNotIn("class_depths", prompt)


class TaxonomyFeedbackLoopTests(unittest.TestCase):
    def setUp(self):
        self.task = TaxonomyTask()

    def test_first_generation_has_no_feedback_second_receives_feedback(self):
        gen = FakeGenerator([
            _response(["A", "B", "C", "D"], []),
            _response(["R", "A", "B", "C"], [["A", "R"], ["B", "R"], ["C", "A"]]),
        ])
        out = _run_generation(
            gen, self.task,
            {"generation": {"sample_size": 1, "feedback": {"enabled": True, "max_rounds": 1}}},
            real_data=[], error_dist=None, judge_call=None, class_prob=0.5, profile=PROFILE,
        )
        self.assertNotIn("Feedback from previous generation", gen.prompts[0])
        self.assertIn("Feedback from previous generation", gen.prompts[1])
        self.assertEqual(len(out[0]["generation_feedback"]["rounds"]), 2)
        self.assertTrue(out[0]["generation_feedback"]["final_taxonomy_feedback_informed"])

    def test_feedback_disabled_matches_phase4_single_generation(self):
        gen = FakeGenerator([_response(["A", "B"], [["B", "A"]])])
        with mock.patch.object(
            self.task, "build_structural_feedback", wraps=self.task.build_structural_feedback
        ) as build_feedback:
            out = _run_generation(
                gen, self.task,
                {"generation": {"sample_size": 1, "feedback": {"enabled": False}}},
                real_data=[], error_dist=None, judge_call=None, class_prob=0.5, profile=PROFILE,
            )
        build_feedback.assert_not_called()
        self.assertEqual(len(gen.prompts), 1)
        self.assertNotIn("Feedback from previous generation", gen.prompts[0])
        self.assertFalse(out[0]["generation_feedback"]["feedback_enabled"])
        self.assertEqual(out[0]["generation_feedback"]["rounds"], [])
        self.assertEqual(out[0]["generation_feedback"]["final_round_selected"], 0)
        self.assertIn("classes", out[0])
        self.assertIn("subclass_axioms", out[0])

    def test_sample_size_two_feedback_state_is_independent(self):
        gen = FakeGenerator([
            _response(["A", "B", "C", "D"], []),
            _response(["R", "A", "B", "C"], [["A", "R"], ["B", "R"], ["C", "A"]]),
            _response(["W", "X", "Y", "Z"], []),
            _response(["Root", "X", "Y", "Z"], [["X", "Root"], ["Y", "Root"], ["Z", "X"]]),
        ])
        out = _run_generation(
            gen, self.task,
            {"generation": {"sample_size": 2, "feedback": {"enabled": True, "max_rounds": 1}}},
            real_data=[], error_dist=None, judge_call=None, class_prob=0.5, profile=PROFILE,
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(len(gen.prompts), 4)
        self.assertNotIn("Feedback from previous generation", gen.prompts[0])
        self.assertIn("Feedback from previous generation", gen.prompts[1])
        self.assertNotIn("Feedback from previous generation", gen.prompts[2])
        self.assertIn("Feedback from previous generation", gen.prompts[3])
        self.assertEqual(len(out[0]["generation_feedback"]["rounds"]), 2)
        self.assertEqual(len(out[1]["generation_feedback"]["rounds"]), 2)
        self.assertTrue(out[0]["generation_feedback"]["rounds"][1]["feedback_informed"])
        self.assertFalse(out[1]["generation_feedback"]["rounds"][0]["feedback_informed"])

    def test_early_stopping_when_tolerance_passes(self):
        gen = FakeGenerator([
            _response(["R", "A", "B", "C"], [["A", "R"], ["B", "R"], ["C", "A"]])
        ])
        out = _run_generation(
            gen, self.task,
            {"generation": {"sample_size": 1, "feedback": {"enabled": True, "max_rounds": 1}}},
            real_data=[], error_dist=None, judge_call=None, class_prob=0.5, profile=PROFILE,
        )
        self.assertEqual(len(gen.prompts), 1)
        self.assertTrue(out[0]["generation_feedback"]["early_stopped"])

    def test_bounded_maximum_rounds(self):
        gen = FakeGenerator([
            _response(["A", "B", "C", "D"], []),
            _response(["E", "F", "G", "H"], []),
        ])
        _run_generation(
            gen, self.task,
            {"generation": {"sample_size": 1, "feedback": {"enabled": True, "max_rounds": 1}}},
            real_data=[], error_dist=None, judge_call=None, class_prob=0.5, profile=PROFILE,
        )
        self.assertEqual(len(gen.prompts), 2)

    def test_malformed_generation_uses_existing_retry_behavior(self):
        gen = FakeGenerator([
            "not json",
            _response(["R", "A", "B", "C"], [["A", "R"], ["B", "R"], ["C", "A"]]),
        ])
        out = _run_generation(
            gen, self.task,
            {"generation": {"sample_size": 1, "max_parse_attempts": 2,
                            "feedback": {"enabled": True, "max_rounds": 1}}},
            real_data=[], error_dist=None, judge_call=None, class_prob=0.5, profile=PROFILE,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(len(gen.prompts), 2)
        attempts = out[0]["generation_feedback"]["attempts"]
        self.assertEqual(attempts[0]["rejection_reason"], "malformed_json")
        self.assertFalse(attempts[0]["valid"])
        self.assertTrue(attempts[1]["valid"])

    def test_diagnostic_data_never_enters_next_generation_prompt(self):
        gen = FakeGenerator([
            "not json with DiagnosticSecret",
            _response(["R", "A", "B", "C"], [["A", "R"], ["B", "R"], ["C", "A"]]),
        ])
        _run_generation(
            gen, self.task,
            {"generation": {"sample_size": 1, "max_parse_attempts": 2,
                            "feedback": {"enabled": False}}},
            real_data=[], error_dist=None, judge_call=None, class_prob=0.5, profile=PROFILE,
        )
        self.assertEqual(len(gen.prompts), 2)
        self.assertNotIn("DiagnosticSecret", gen.prompts[1])
        self.assertNotIn("malformed_json", gen.prompts[1])

    def test_failed_feedback_round_preserves_previous_valid_taxonomy(self):
        gen = FakeGenerator([
            _response(["A", "B", "C", "D"], []),
            "not json",
        ])
        out = _run_generation(
            gen, self.task,
            {"generation": {"sample_size": 1, "max_parse_attempts": 1,
                            "feedback": {"enabled": True, "max_rounds": 1}}},
            real_data=[], error_dist=None, judge_call=None, class_prob=0.5, profile=PROFILE,
        )
        self.assertEqual(out[0]["classes"], ["A", "B", "C", "D"])
        self.assertTrue(out[0]["generation_feedback"]["failed_feedback_round_preserved_previous"])

    def test_final_evaluator_still_receives_only_domain_and_classes(self):
        generated = self.task.parse_structured_generation(
            _response(["R", "A"], [["A", "R"]])
        )
        generated["generation_feedback"] = {
            "feedback": {"messages": ["secret structural feedback"]},
            "attempts": [{"rejection_reason": "malformed_json", "raw_preview": "DiagnosticSecret"}],
        }
        sample = self.task.get_eval_samples([generated])[0]
        payload = json.loads(sample["text"])
        self.assertEqual(sorted(payload), ["classes", "domain"])
        self.assertNotIn("subclass_axioms", sample["text"])
        self.assertNotIn("secret structural feedback", sample["text"])
        self.assertNotIn("DiagnosticSecret", sample["text"])


if __name__ == "__main__":
    unittest.main()
