import json
import os
import tempfile
import unittest
from unittest.mock import patch

from scripts.rescore_session import rescore_session


class FakeModel:
    """Predicts SPAM for everything."""
    def predict(self, texts):
        return ["SPAM"] * len(texts)


def _write_session(d):
    os.makedirs(os.path.join(d, "generated"))
    meta = {
        "created": "2026-07-08T00:00:00", "task": "spam", "mode": "inverse",
        "provider": "openrouter", "model": "m3", "num_runs": 2, "runs_completed": 2,
        "partial": False, "dataset": {"source": "local", "path": "x.csv",
                                      "format": "csv", "sample_size": 2},
        "effective_samples_per_run": [2, 2], "judge": None,
        "real_baseline": True, "class_balance": "empirical",
    }
    # Old results: only old_model, no per-run scores.
    results = {"old_model": {"generated": {"accuracy": {"mean": 0.5, "std": 0.0}},
                             "real": {"accuracy": 1.0}}}
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f)
    for i in (1, 2):
        with open(os.path.join(d, "generated", f"run_{i}.json"), "w") as f:
            json.dump([{"text": "WIN a FREE prize now!!", "label": "SPAM",
                        "technique": "spam_keywords", "seed": "hi"},
                       {"text": "see you at lunch", "label": "HAM",
                        "technique": "paraphrase", "seed": "lunch?"}], f)
    with open(os.path.join(d, "real_sample.json"), "w") as f:
        json.dump([{"text": "free money!!!", "label": "SPAM"},
                   {"text": "meeting at 3", "label": "HAM"}], f)


def _config():
    return {
        "task": {"name": "spam"},
        "task_models": [{"name": "old_model", "type": "roberta"},
                        {"name": "new_model", "type": "bert"}],
        "evaluation": {"real_baseline": True},
    }


# One ontology whose seed pool (max_depth 3, min_classes 3) is three subtrees.
_TAXONOMY_ROW = {
    "ontology_id": "onto",
    "domain": "cuisine",
    "classes": ["Food", "Pizza", "Dessert", "Margherita", "Napoletana", "Gelato",
                "Sorbet", "Tiramisu", "DessertPizza", "Nutella"],
    "subclass_axioms": [["Pizza", "Food"], ["Dessert", "Food"],
                        ["Margherita", "Pizza"], ["Napoletana", "Pizza"],
                        ["Gelato", "Dessert"], ["Sorbet", "Dessert"],
                        ["Tiramisu", "Dessert"], ["DessertPizza", "Pizza"],
                        ["DessertPizza", "Dessert"], ["Nutella", "DessertPizza"]],
}


def _write_seeded_taxonomy_session(d):
    """A forward+seeded taxonomy session as run_pipeline leaves it: the real
    side is the seed pool's subtrees, the generated side those subtrees renamed.
    Returns the number of real subtrees."""
    from framework.tasks.taxonomy.task import TaxonomyTask

    cfg = {"generation": {"mode": "forward", "seedless": False,
                          "seed_pool": {"max_depth": 3, "min_classes": 3}}}
    real = TaxonomyTask().get_real_eval_samples(cfg, [_TAXONOMY_ROW])
    generated = []
    for item in real:
        names = {c: f"Species{i}" for i, c in enumerate(item["classes"])}
        generated.append({"domain": "marine biology",
                          "classes": [names[c] for c in item["classes"]],
                          "subclass_axioms": [[names[c], names[p]]
                                              for c, p in item["subclass_axioms"]]})
    os.makedirs(os.path.join(d, "generated"))
    meta = {"created": "2026-09-10T00:00:00", "task": "taxonomy", "mode": "forward",
            "seedless": False, "provider": "stub", "model": "stub", "num_runs": 1,
            "runs_completed": 1, "partial": False,
            "effective_samples_per_run": [len(generated)], "real_baseline": True}
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump({"meta": meta, "results": {}}, f)
    with open(os.path.join(d, "generated", "run_1.json"), "w") as f:
        json.dump(generated, f)
    with open(os.path.join(d, "real_sample.json"), "w") as f:
        json.dump(real, f)
    return len(real)


class RescoreSessionTests(unittest.TestCase):
    def _run(self, d, **kwargs):
        with patch("framework.tasks.spam.task.SpamTask.get_model",
                   lambda self, cfg: FakeModel()):
            return rescore_session(d, _config(), **kwargs)

    def test_restores_per_run_scores_and_adds_new_model(self):
        with tempfile.TemporaryDirectory() as d:
            _write_session(d)
            self._run(d)
            out = json.load(open(os.path.join(d, "results.json")))
            for name in ("old_model", "new_model"):
                self.assertEqual(len(out["results"][name]["runs"]), 2)
                # all-SPAM predictions on a 50/50 sample -> accuracy 0.5, recall 1.0
                self.assertEqual(out["results"][name]["generated"]["accuracy"]["mean"], 0.5)
                self.assertEqual(out["results"][name]["runs"][0]["recall"], 1.0)
                self.assertEqual(out["results"][name]["real"]["accuracy"], 0.5)
            self.assertIn("rescored", out["meta"])
            self.assertEqual(out["meta"]["created"], "2026-07-08T00:00:00")
            self.assertFalse(out["meta"]["partial"])

    def test_writes_profile_json(self):
        with tempfile.TemporaryDirectory() as d:
            _write_session(d)
            self._run(d)
            prof = json.load(open(os.path.join(d, "profile.json")))
            self.assertEqual(set(prof), {"real", "generated", "fidelity"})
            self.assertEqual(prof["generated"]["n"], 4)  # 2 runs x 2 rows

    def test_skip_eval_leaves_results_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            _write_session(d)
            before = json.load(open(os.path.join(d, "results.json")))
            self._run(d, skip_eval=True)
            after = json.load(open(os.path.join(d, "results.json")))
            self.assertEqual(before, after)
            self.assertTrue(os.path.exists(os.path.join(d, "profile.json")))

    def test_num_runs_recomputed_from_actual_run_files(self):
        # Stale meta claims 5 runs even though only 2 generated/run_*.json exist
        # (e.g. after a manual merge) — rescoring must trust the files, not meta.
        with tempfile.TemporaryDirectory() as d:
            _write_session(d)
            stale = json.load(open(os.path.join(d, "results.json")))
            stale["meta"]["num_runs"] = 5
            stale["meta"]["runs_completed"] = 3
            stale["meta"]["partial"] = True
            json.dump(stale, open(os.path.join(d, "results.json"), "w"))
            self._run(d)
            out = json.load(open(os.path.join(d, "results.json")))
            self.assertEqual(out["meta"]["num_runs"], 2)
            self.assertEqual(out["meta"]["runs_completed"], 2)
            self.assertFalse(out["meta"]["partial"])

    def test_mode_and_strategy_recomputed_from_task_not_stale_meta(self):
        # _write_session seeds "mode": "inverse" — spam is class_conditional
        # and now has a real mode, so it must be preserved (not nulled) while
        # strategy is filled in from the task, regardless of the seed value.
        with tempfile.TemporaryDirectory() as d:
            _write_session(d)
            self._run(d)
            out = json.load(open(os.path.join(d, "results.json")))
            self.assertEqual(out["meta"]["strategy"], "class_conditional")
            self.assertEqual(out["meta"]["mode"], "inverse")

    def test_mode_defaults_per_strategy_when_absent_from_stale_meta(self):
        # A session from before mode was ever recorded has no "mode" key at
        # all. class_conditional (spam) must default to "inverse" — the same
        # default _build_meta/_run_generation use — not GEC's "forward", and
        # not None.
        with tempfile.TemporaryDirectory() as d:
            _write_session(d)
            stale = json.load(open(os.path.join(d, "results.json")))
            del stale["meta"]["mode"]
            json.dump(stale, open(os.path.join(d, "results.json"), "w"))
            self._run(d)
            out = json.load(open(os.path.join(d, "results.json")))
            self.assertEqual(out["meta"]["mode"], "inverse")

    def test_seedless_survives_rescore(self):
        # Task 9: profile provenance is whatever the original run recorded —
        # rescoring has no generation config to rebuild it from, so it must
        # ride through the meta spread untouched.
        with tempfile.TemporaryDirectory() as d:
            _write_session(d)
            stale = json.load(open(os.path.join(d, "results.json")))
            stale["meta"]["seedless"] = True
            stale["meta"]["profile_path"] = "profile.json"
            json.dump(stale, open(os.path.join(d, "results.json"), "w"))
            self._run(d)
            out = json.load(open(os.path.join(d, "results.json")))
            self.assertTrue(out["meta"]["seedless"])
            self.assertEqual(out["meta"]["profile_path"], "profile.json")

    def test_seedless_defaults_false_for_legacy_meta_without_key(self):
        # _write_session's meta predates the seedless key entirely.
        with tempfile.TemporaryDirectory() as d:
            _write_session(d)
            self._run(d)
            out = json.load(open(os.path.join(d, "results.json")))
            self.assertFalse(out["meta"]["seedless"])

    def test_a_seeded_taxonomy_session_rescores_to_a_profile(self):
        # Its real side is several subtrees of one ontology. The fidelity step
        # used to raise on more than one real taxonomy, so rescoring a seeded
        # session died exactly as the live run did.
        cfg = {"task": {"name": "taxonomy"},
               "task_models": [{"name": "lexical", "type": "lexical"},
                               {"name": "star", "type": "star"}],
               "evaluation": {"real_baseline": True}}
        with tempfile.TemporaryDirectory() as d:
            n_real = _write_seeded_taxonomy_session(d)
            self.assertGreater(n_real, 1)
            rescore_session(d, cfg)
            prof = json.load(open(os.path.join(d, "profile.json")))
            self.assertEqual(prof["fidelity"]["real_profile"]["pooled_taxonomies"], n_real)
            out = json.load(open(os.path.join(d, "results.json")))
            self.assertEqual(set(out["results"]), {"lexical", "star"})

    def test_task_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            _write_session(d)
            cfg = _config()
            cfg["task"]["name"] = "gec"
            with self.assertRaises(ValueError):
                rescore_session(d, cfg)


if __name__ == "__main__":
    unittest.main()
