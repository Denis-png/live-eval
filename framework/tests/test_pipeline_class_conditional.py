import json
import os
import tempfile
import unittest
from unittest.mock import patch

import framework.pipeline as pipeline
from framework.tasks.spam.task import SpamTask


class _FakeGen:
    """Stands in for a loaded generator: class-conditional emits alternating
    SPAM/HAM labeled records regardless of prompts."""
    def generate_class_conditional(self, **kw):
        out = []
        for i in range(kw["sample_size"]):
            if i % 2 == 0:
                out.append({"text": f"WIN cash now offer {i} http://x.com", "label": "SPAM", "technique": "phishing_link", "seed": "s"})
            else:
                out.append({"text": f"see you at lunch tomorrow {i}", "label": "HAM", "technique": "paraphrase", "seed": "s"})
        return out
    def call_api(self, prompt): return ""


class _FakeModel:
    def __init__(self, cfg): pass
    def predict(self, texts):
        return ["SPAM" if "WIN" in t else "HAM" for t in texts]


_REAL_REF = [{"text": "win money now free", "label": "SPAM"},
             {"text": "coffee at noon?", "label": "HAM"}]
_CANNED_DIST = {"type_dist": {"phishing_link": 1.0}, "count_dist": {1: 1.0}}


class PipelineClassConditionalTests(unittest.TestCase):
    def test_end_to_end_spam(self):
        with tempfile.TemporaryDirectory() as d:
            config = {
                "task": {"name": "spam"},
                "dataset": {"source": "huggingface", "huggingface": {"name": "x", "split": "train"}},
                "generation": {"provider": "openrouter", "model": "m", "num_runs": 2,
                               "sample_size": 4, "class_balance": 0.5, "seed_field": "incorrect",
                               "inverse": {"profile_size": 10}},
                "evaluation": {"real_baseline": True},
                "task_models": [{"name": "fake", "type": "roberta"}],
                "output": {"base_dir": os.path.join(d, "runs")},
            }
            # Patch every external boundary with context managers (auto-restored):
            with patch.object(pipeline, "load_generator", lambda c: _FakeGen()), \
                 patch.object(pipeline, "load_real_data",
                              lambda config, task: [{"incorrect": f"hi there friend {i}"} for i in range(4)]), \
                 patch.object(SpamTask, "profile_error_distribution",
                              lambda self, real_data, count_max=5, config=None: _CANNED_DIST), \
                 patch.object(SpamTask, "get_real_eval_samples",
                              lambda self, config, real_data: _REAL_REF), \
                 patch.object(SpamTask, "get_model", lambda self, mc: _FakeModel(mc)):
                final = pipeline.run_pipeline(config)

            self.assertIn("fake", final)
            self.assertIn("generated", final["fake"])
            self.assertIn("real", final["fake"])
            task_dir = os.path.join(d, "runs", "spam")
            session = os.listdir(task_dir)[0]
            base = os.path.join(task_dir, session)
            self.assertTrue(os.path.exists(os.path.join(base, "results.json")))
            self.assertTrue(os.path.exists(os.path.join(base, "generated", "run_1.json")))
            self.assertTrue(os.path.exists(os.path.join(base, "real_sample.json")))
            self.assertTrue(os.path.exists(os.path.join(base, "profile.json")))
            self.assertTrue(os.path.isdir(os.path.join(base, "plots")))
            prof = json.load(open(os.path.join(base, "profile.json")))
            self.assertIn("fidelity", prof)
            self.assertIn("type_dist_jsd", prof["fidelity"])


if __name__ == "__main__":
    unittest.main()
