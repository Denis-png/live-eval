"""Every shipped config must be runnable by the entry point that reads it.

framework/configs/taxonomy/config.yaml carried no `generation:` or
`task_models:` block for months. Nothing caught it because no test ever put a
shipped config through validate_config -- the task's own 132 tests all built
their config inline. A task can be fully implemented and still not runnable.
"""
import glob
import os
import unittest

import yaml

from framework.main import _expand_env_vars, validate_config

_CONFIGS = sorted(glob.glob("framework/configs/*/config.yaml"))


class ShippedConfigTests(unittest.TestCase):
    def test_there_is_a_config_per_task(self):
        # Guards the glob itself: if the layout moves, the loop below would
        # silently test nothing and still pass.
        found = {os.path.basename(os.path.dirname(p)) for p in _CONFIGS}
        self.assertEqual(found, {"gec", "spam", "sentiment", "taxonomy"})

    def test_every_shipped_config_validates(self):
        for path in _CONFIGS:
            with self.subTest(config=path):
                config = _expand_env_vars(yaml.safe_load(open(path)))
                try:
                    problems = validate_config(config)
                except ValueError as e:            # raises rather than returns
                    self.fail(f"{path} is not runnable:\n{e}")
                self.assertFalse(problems, f"{path}: {problems}")

    def test_every_shipped_config_names_a_known_task(self):
        for path in _CONFIGS:
            with self.subTest(config=path):
                config = yaml.safe_load(open(path))
                name = (config.get("task") or {}).get("name")
                self.assertEqual(name, os.path.basename(os.path.dirname(path)),
                                 "config lives in a directory naming a different task")

    def test_generation_token_budgets_clear_the_truncation_floor(self):
        # A reasoning model spends its budget on <think> before emitting an
        # answer; at 1024 it returns finish_reason=length and the sample is lost.
        # That cost a full GEC run and every artifact of taxonomy's smoke run.
        for path in _CONFIGS:
            gen = (yaml.safe_load(open(path)) or {}).get("generation") or {}
            budget = gen.get("max_tokens")
            if budget is None:
                continue
            with self.subTest(config=path):
                self.assertGreaterEqual(
                    budget, 2048,
                    f"{path}: generation.max_tokens={budget} truncates reasoning models")


if __name__ == "__main__":
    unittest.main()


class TaxonomyCellsRunFromTheShippedConfigTests(unittest.TestCase):
    """Every taxonomy cell must run from the one shipped config, by CLI flags alone.

    The config once set `feedback.enabled: true` explicitly. taxonomy.json already
    defaults it on, and the guards deliberately tolerate that default while
    refusing an EXPLICIT request in a cell with no imposed target. So writing it
    into the run config made forward+seeded, forward+seedless and inverse+seeded
    all refuse to start — the config validated, and only one of four cells ran.
    validate_config cannot see this; only dispatching a cell can.
    """

    def test_no_cell_is_refused_over_the_feedback_loop(self):
        import yaml
        from framework import pipeline
        from framework.generators.base_generator import BaseGenerator
        from framework.main import _expand_env_vars
        from framework.tasks.taxonomy.task import TaxonomyTask

        class _Refuse(BaseGenerator):
            def call_api(self, prompt):
                return "{}"

        base = _expand_env_vars(yaml.safe_load(open("framework/configs/taxonomy/config.yaml")))
        real = [{"domain": "d",
                 "classes": ["A", "B", "C", "D", "E", "F"],
                 "subclass_axioms": [["B", "A"], ["C", "A"], ["D", "B"],
                                     ["E", "B"], ["F", "C"]]}]
        profile = {"taxonomies": [{"domain": "d", "n_classes": 6, "max_depth": 2,
                                   "depth_distribution": {"0": .2, "1": .4, "2": .4}}]}
        for mode in ("forward", "inverse"):
            for seedless in (False, True):
                with self.subTest(mode=mode, seedless=seedless):
                    cfg = dict(base)
                    cfg["generation"] = {**base["generation"], "mode": mode,
                                         "seedless": seedless, "sample_size": 1,
                                         "seed_pool": {"max_depth": 4, "min_classes": 3}}
                    try:
                        pipeline._run_generation(_Refuse(), TaxonomyTask(), cfg, real,
                                                 None, None, None, profile=profile)
                    except RuntimeError as e:
                        # "0 usable samples" is expected from a model returning
                        # "{}" — it means the cell DISPATCHED. A feedback refusal
                        # means it never started.
                        self.assertNotIn("feedback", str(e).lower(),
                                         f"{mode}+{'seedless' if seedless else 'seeded'} "
                                         f"refused to start: {e}")
