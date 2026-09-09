import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from framework.evaluate_generated import evaluate_generated_file
from framework.generate_dataset import generate_dataset_once
from framework.profile_dataset import _profile_gec


class FakeGenerationTask:
    def get_generation_strategy(self):
        return "corruption"

    def get_real_eval_samples(self, config, real_data):
        return [{"text": row["text"], "label": "ok"} for row in real_data]

    def get_evaluator_fns(self):
        raise AssertionError("generation-only must not load evaluator functions")

    def get_model(self, model_config):
        raise AssertionError("generation-only must not load task models")

    def build_fidelity_profile(self, rows):
        raise AssertionError("generation-only must not build fidelity profiles")


class FakeEvalModel:
    def predict(self, texts):
        return [text.upper() for text in texts]


class FakeEvalTask:
    def get_task_name(self):
        return "fake"

    def get_eval_samples(self, generated):
        return [{"text": row["text"], "gold": row["gold"]} for row in generated]

    def get_model(self, model_config):
        if model_config["name"] == "remote":
            assert model_config["api_key"] == "eval-key"
        return FakeEvalModel()

    def get_evaluators(self):
        return ("exact",)

    def get_evaluator_fns(self):
        return {"exact": lambda rows: sum(
            1 for row in rows if row["prediction"] == row["gold"]
        ) / len(rows)}

    def build_fidelity_profile(self, rows):
        raise AssertionError("evaluation-only must not build fidelity profiles")


class ComponentEntrypointTests(unittest.TestCase):
    def test_generation_only_writes_three_samples_without_evaluation_or_profiling(self):
        task = FakeGenerationTask()
        context = {
            "generator": object(),
            "task": task,
            "real_data": [{"text": "seed"}],
            "error_dist": None,
            "judge_call": None,
            "class_prob": {},
            "profile": None,
        }
        synthetic = [{"text": f"sample {i}"} for i in range(3)]
        config = {
            "api_keys": {"fake": "test-key"},
            "task": {"name": "fake"},
            "dataset": {"source": "local", "local": {"path": "unused.csv"}},
            "generation": {"provider": "fake", "model": "fake", "sample_size": 99},
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "generated.json"
            with mock.patch(
                "framework.generate_dataset.pipeline.build_generation_context",
                side_effect=AssertionError("generation-only must not build evaluation context"),
            ), mock.patch(
                "framework.generate_dataset.pipeline.load_task",
                return_value=task,
            ), mock.patch(
                "framework.generate_dataset.pipeline.load_real_data",
                return_value=context["real_data"],
            ), mock.patch(
                "framework.generate_dataset.pipeline.load_generator",
                return_value=context["generator"],
            ), mock.patch(
                "framework.generate_dataset.pipeline._build_judge_call",
                return_value=None,
            ), mock.patch(
                "framework.generate_dataset.pipeline._should_load_error_distribution",
                return_value=False,
            ), mock.patch(
                "framework.generate_dataset.pipeline._load_benchmark_profile",
                return_value=None,
            ), mock.patch(
                "framework.generate_dataset.pipeline._load_seed_weights",
                return_value=None,
            ), mock.patch(
                "framework.generate_dataset.pipeline._resolve_class_prob",
                return_value={},
            ), mock.patch(
                "framework.generate_dataset.pipeline._run_generation",
                return_value=synthetic,
            ) as run_generation:
                result = generate_dataset_once(config, output, sample_size=3)

            self.assertEqual(result, synthetic)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), synthetic)
            self.assertEqual(
                run_generation.call_args.args[2]["generation"]["sample_size"], 3
            )
            run_generation.assert_called_once()

    def test_evaluation_only_scores_existing_samples_without_generation_or_profiling(self):
        config = {
            "api_keys": {"openrouter": "eval-key"},
            "generation": {"provider": "openai", "model": "unused-generator"},
            "task": {"name": "fake"},
            "task_models": [{"name": "remote", "type": "fake", "provider": "openrouter"}],
        }
        generated = [
            {"text": "a", "gold": "A"},
            {"text": "b", "gold": "B"},
            {"text": "c", "gold": "C"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            generated_path = Path(tmp) / "generated.json"
            output_path = Path(tmp) / "results.json"
            generated_path.write_text(json.dumps(generated), encoding="utf-8")
            with mock.patch(
                "framework.evaluate_generated.load_task",
                return_value=FakeEvalTask(),
            ) as load_task, mock.patch(
                "framework.pipeline._run_generation",
                side_effect=AssertionError("evaluation-only must not generate"),
            ):
                payload = evaluate_generated_file(config, generated_path, output_path)

            load_task.assert_called_once_with("fake")
            self.assertEqual(payload["meta"]["num_samples"], 3)
            self.assertEqual(payload["results"]["remote"]["exact"], 1.0)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8"))["results"],
                payload["results"],
            )

    def test_profile_dataset_standalone_profiles_without_generation_or_evaluation(self):
        rows = [
            {"incorrect": "I has apple", "correct": "I have apple"},
            {"incorrect": "She go home", "correct": "She goes home"},
            {"incorrect": "They is here", "correct": "They are here"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "profile.json"
            with mock.patch(
                "framework.pipeline.load_task", return_value=object()
            ), mock.patch(
                "framework.pipeline.load_real_data", return_value=rows
            ), mock.patch(
                "framework.pipeline._run_generation",
                side_effect=AssertionError("profiling-only must not generate"),
            ), mock.patch(
                "framework.profile_dataset._build_topic_call",
                side_effect=AssertionError("topics are opt-in only"),
            ):
                written = _profile_gec(
                    {
                        "task": {"name": "gec"},
                        "dataset": {},
                        "generation": {"sample_size": 3},
                    },
                    str(output),
                )

            profile = json.loads(Path(written).read_text(encoding="utf-8"))
            self.assertEqual(profile["num_samples"], 3)
            self.assertIn("incorrect_word_count", profile)


if __name__ == "__main__":
    unittest.main()
