"""`mode` means: is the annotation INHERITED from the source, or IMPOSED on it?

Under that definition taxonomy is already inverse — it samples a target
structure from the profile and iterates toward it — so `structured` needs no
exemption from the axis.
"""
import unittest
from unittest import mock

from framework import pipeline
from framework.generators.base_generator import BaseGenerator


class _StubGenerator(BaseGenerator):
    """Implements the generator interface, not just call_api: the structured
    dispatch calls generate_structured(), which lives on BaseGenerator."""

    def call_api(self, prompt):
        return ""


class ResolveModeTests(unittest.TestCase):
    def test_structured_defaults_to_inverse(self):
        # Today's taxonomy imposes a sampled target structure, so its implicit
        # mode is inverse. resolve_mode previously fell through to "forward".
        self.assertEqual(pipeline.resolve_mode({}, "structured"), "inverse")

    def test_explicit_mode_still_wins_for_structured(self):
        cfg = {"generation": {"mode": "forward"}}
        self.assertEqual(pipeline.resolve_mode(cfg, "structured"), "forward")

    def test_other_strategies_keep_their_defaults(self):
        self.assertEqual(pipeline.resolve_mode({}, "corruption"), "forward")
        self.assertEqual(pipeline.resolve_mode({}, "class_conditional"), "inverse")


class ContextAgreementTests(unittest.TestCase):
    """resolve_mode's docstring promises the session name, _build_meta and the
    dispatch always agree. For structured they did not: resolve_mode said
    "forward" while build_generation_context hardcoded None."""

    def _context(self, cfg):
        with mock.patch.object(pipeline, "load_generator",
                               return_value=_StubGenerator()), \
             mock.patch.object(pipeline, "load_task") as load_task, \
             mock.patch.object(pipeline, "load_real_data", return_value=[]):
            task = load_task.return_value
            task.get_generation_strategy.return_value = "structured"
            task.get_task_name.return_value = "taxonomy"
            task.get_evaluator_fns.return_value = {}
            task.get_real_eval_samples.return_value = None
            task.get_class_labels.return_value = None
            with mock.patch.object(pipeline, "_load_benchmark_profile",
                                   return_value={"taxonomies": [{"domain": "x"}]}):
                return pipeline.build_generation_context(cfg)

    def test_context_mode_matches_resolve_mode_for_structured(self):
        cfg = {"generation": {"sample_size": 1}, "task": {"name": "taxonomy"}}
        ctx = self._context(cfg)
        self.assertEqual(ctx["mode"], pipeline.resolve_mode(cfg, "structured"))
        self.assertEqual(ctx["mode"], "inverse")

    def test_context_honours_an_explicit_structured_mode(self):
        cfg = {"generation": {"sample_size": 1, "mode": "forward"},
               "task": {"name": "taxonomy"}}
        self.assertEqual(self._context(cfg)["mode"], "forward")


class CellSlugTests(unittest.TestCase):
    def test_structured_slug_is_uniform_with_other_strategies(self):
        self.assertEqual(
            pipeline.generation_cell_slug({}, "structured"), "inverse_seedless")
        self.assertEqual(
            pipeline.generation_cell_slug(
                {"generation": {"mode": "forward"}}, "structured"),
            "forward_seedless")

    def test_other_strategies_unchanged(self):
        self.assertEqual(
            pipeline.generation_cell_slug({"generation": {"seedless": True}},
                                          "corruption"),
            "forward_seedless")
        self.assertEqual(
            pipeline.generation_cell_slug({}, "class_conditional"),
            "inverse_seeded")


class StructuredRejectionTests(unittest.TestCase):
    """The framework's rule: an unsupported capability says so before any API
    call and names what is missing. It must not assert impossibility for
    something that is merely unimplemented."""

    def _task(self):
        from framework.tasks.base_task import BaseTask

        class _T(BaseTask):
            def get_generation_strategy(self): return "structured"
            def get_task_name(self): return "structured_stub"
            def build_structured_generation_prompt(self, profile, rng=None,
                                                   feedback=None, mode="inverse"):
                return "make one"
            def parse_structured_generation(self, text):
                return {"classes": ["A"]}
            def get_error_types(self): return []
            def get_prompt_instruction(self): return ""
            def get_evaluators(self): return []
            def get_evaluator_fns(self): return {}
            def get_model(self, model_config): return None
            def parse_row(self, row): return row
        return _T()

    def _run(self, gen_cfg):
        cfg = {"generation": {"sample_size": 1, **gen_cfg},
               "task": {"name": "structured_stub"}}
        return pipeline._run_generation(
            _StubGenerator(), self._task(), cfg, [], None, None, 0.5,
            profile={"taxonomies": [{"domain": "x"}]},
        )

    def test_mode_is_no_longer_rejected(self):
        # Both values must dispatch; forward's own behaviour is Task 3. At this point
        # in the plan, mode="forward" still runs the inverse prompt path, so both
        # modes should produce one artifact from the stub task's parser.
        for mode in ("inverse", "forward"):
            with self.subTest(mode=mode):
                out = self._run({"mode": mode})
                self.assertEqual(len(out), 1)
                self.assertIn("classes", out[0])

    def test_seeded_structured_reads_as_unimplemented_not_impossible(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run({"seedless": False})
        message = str(ctx.exception)
        self.assertIn("not implemented", message)
        self.assertIn("structured_stub", message)
        self.assertNotIn("not supported", message)
