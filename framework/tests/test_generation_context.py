import os
import tempfile
import unittest
from unittest import mock

from framework import pipeline

_ROWS = "label,text\n" + "".join(
    f"ham,let us meet at three tomorrow number {i}\n" for i in range(12)
) + "".join(
    f"spam,WIN a FREE prize now claim $500 http://x{i}.example !\n" for i in range(8)
)


class _StubGenerator:
    def call_api(self, prompt):
        return "Corrupted: stub"


def _config(path):
    return {
        "dataset": {"source": "local", "local": {"path": path, "format": "csv"}},
        "generation": {"provider": "openai", "model": "gpt-x", "num_runs": 1,
                       "sample_size": 5, "mode": "inverse", "seedless": False},
        "task": {"name": "spam"},
        "task_models": [],
    }


class BuildGenerationContextTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "bench.csv")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(_ROWS)

    def _build(self):
        with mock.patch.object(pipeline, "load_generator",
                               return_value=_StubGenerator()):
            return pipeline.build_generation_context(_config(self.path))

    def test_context_exposes_every_key_the_pipeline_and_calibrator_need(self):
        ctx = self._build()
        for key in ("task", "real_data", "generator", "judge_call", "evaluator_fns",
                    "strategy", "mode", "seedless", "error_dist", "profile",
                    "real_reference", "class_prob"):
            self.assertIn(key, ctx)

    def test_resolves_spam_defaults(self):
        ctx = self._build()
        self.assertEqual(ctx["strategy"], "class_conditional")
        self.assertEqual(ctx["mode"], "inverse")
        self.assertFalse(ctx["seedless"])
        self.assertIsNone(ctx["profile"])

    def test_error_distribution_matches_the_standalone_loader(self):
        # Regression guard: the refactor must not change what generation samples
        # from when no calibration artifact exists.
        ctx = self._build()
        expected = pipeline.load_error_distribution(
            _config(self.path), ctx["real_data"], ctx["task"]
        )
        self.assertEqual(ctx["error_dist"], expected)

    def test_class_prob_is_the_empirical_spam_fraction(self):
        ctx = self._build()
        self.assertAlmostEqual(ctx["class_prob"], 8 / 20, places=6)
