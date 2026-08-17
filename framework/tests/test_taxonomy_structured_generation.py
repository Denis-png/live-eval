import json
import os
import tempfile
import unittest
from unittest import mock

from framework.pipeline import _run_generation, _should_load_error_distribution, run_pipeline
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
        }
    ],
}


def _response(domain="demo", classes=None, axioms=None):
    return json.dumps({
        "domain": domain,
        "classes": classes or ["GeneratedRoot", "GeneratedChild"],
        "subclass_axioms": axioms if axioms is not None else [["GeneratedChild", "GeneratedRoot"]],
    })


class TaxonomyStructuredGenerationTests(unittest.TestCase):
    def setUp(self):
        self.task = TaxonomyTask()

    def test_taxonomy_strategy_is_structured(self):
        self.assertEqual(self.task.get_generation_strategy(), "structured")

    def test_profile_to_prompt_uses_structural_targets_only(self):
        prompt = self.task.build_structured_generation_prompt(PROFILE)
        self.assertIn('"n_classes": 4', prompt)
        self.assertIn('"n_subclass_axioms": 3', prompt)
        self.assertIn('"domain": "demo"', prompt)
        self.assertNotIn("SecretRoot", prompt)
        self.assertNotIn("SecretLeaf", prompt)
        self.assertNotIn("http://example.org/SecretRoot", prompt)
        self.assertNotIn("SecretLeaf\", \"SecretRoot", prompt)

    def test_valid_structured_response(self):
        parsed = self.task.parse_structured_generation(_response())
        self.assertEqual(parsed["domain"], "demo")
        self.assertEqual(parsed["classes"], ["GeneratedRoot", "GeneratedChild"])
        self.assertEqual(parsed["subclass_axioms"], [["GeneratedChild", "GeneratedRoot"]])

    def test_malformed_json_rejected(self):
        self.assertIsNone(self.task.parse_structured_generation("not json"))

    def test_unknown_edge_endpoint_rejected(self):
        self.assertIsNone(self.task.parse_structured_generation(
            _response(axioms=[["GeneratedChild", "MissingRoot"]])
        ))

    def test_duplicate_classes_rejected(self):
        self.assertIsNone(self.task.parse_structured_generation(
            _response(classes=["GeneratedRoot", "GeneratedRoot"], axioms=[])
        ))

    def test_duplicate_edges_are_normalized(self):
        parsed = self.task.parse_structured_generation(
            _response(axioms=[
                ["GeneratedChild", "GeneratedRoot"],
                ["GeneratedChild", "GeneratedRoot"],
            ])
        )
        self.assertEqual(parsed["subclass_axioms"], [["GeneratedChild", "GeneratedRoot"]])

    def test_self_loop_rejected(self):
        self.assertIsNone(self.task.parse_structured_generation(
            _response(axioms=[["GeneratedRoot", "GeneratedRoot"]])
        ))

    def test_cycle_rejected(self):
        self.assertIsNone(self.task.parse_structured_generation(
            _response(
                classes=["A", "B"],
                axioms=[["A", "B"], ["B", "A"]],
            )
        ))

    def test_multiple_inheritance_accepted(self):
        parsed = self.task.parse_structured_generation(
            _response(
                classes=["RootA", "RootB", "Child"],
                axioms=[["Child", "RootA"], ["Child", "RootB"]],
            )
        )
        self.assertEqual(parsed["subclass_axioms"], [["Child", "RootA"], ["Child", "RootB"]])

    def test_run_generation_structured_one_sample_is_one_taxonomy(self):
        gen = FakeGenerator([_response()])
        config = {"generation": {"sample_size": 1, "feedback": {"enabled": False}}}
        out = _run_generation(
            gen, self.task, config, real_data=[], error_dist=None,
            judge_call=None, class_prob=0.5, profile=PROFILE,
        )
        self.assertEqual(len(out), 1)
        self.assertIn("classes", out[0])
        self.assertIn("subclass_axioms", out[0])

    def test_unsupported_mode_fails_clearly(self):
        gen = FakeGenerator([])
        config = {"generation": {"sample_size": 1, "mode": "forward"}}
        with self.assertRaises(RuntimeError) as ctx:
            _run_generation(
                gen, self.task, config, real_data=[], error_dist=None,
                judge_call=None, class_prob=0.5, profile=PROFILE,
            )
        self.assertIn("mode", str(ctx.exception))

    def test_taxonomy_does_not_load_error_distribution(self):
        self.assertFalse(_should_load_error_distribution("structured", None, True))

    def test_existing_error_distribution_strategy_decisions(self):
        self.assertTrue(_should_load_error_distribution("corruption", "inverse", False))
        self.assertFalse(_should_load_error_distribution("corruption", "forward", False))
        self.assertTrue(_should_load_error_distribution("class_conditional", "inverse", False))

    def test_generated_gold_never_enters_evaluator_prompt(self):
        generated = self.task.parse_structured_generation(_response())
        sample = self.task.get_eval_samples([generated])[0]
        self.assertNotIn("subclass_axioms", json.loads(sample["text"]))
        self.assertNotIn("GeneratedChild\", \"GeneratedRoot", sample["text"])

    def test_mocked_pipeline_generates_and_evaluates_one_taxonomy(self):
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
                json.dump(PROFILE, f)

            generation_generator = FakeGenerator([_response()])
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
                    "feedback": {"enabled": False},
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
                results = run_pipeline(config)

        generated_scores = results["evaluator"]["generated"]
        self.assertEqual(generated_scores["f1"]["mean"], 1.0)
        self.assertIn("GeneratedChild", model_generator.prompts[0])
        self.assertNotIn("GeneratedChild\", \"GeneratedRoot", model_generator.prompts[0])


if __name__ == "__main__":
    unittest.main()
