"""The seeded cells: gold is computed, only the surface is generated."""
import unittest
from random import Random

from framework.tasks.base_task import BaseTask
from framework.tasks.taxonomy.task import TaxonomyTask

_REAL = [{
    "domain": "pizza",
    "classes": ["Food", "Pizza", "Margherita", "Napoletana", "Dessert", "Gelato"],
    "subclass_axioms": [["Pizza", "Food"], ["Margherita", "Pizza"],
                        ["Napoletana", "Pizza"], ["Dessert", "Food"],
                        ["Gelato", "Dessert"]],
}]

_CONFIG = {"generation": {"seed_pool": {"max_depth": 3, "min_classes": 3,
                                        "domains": ["marine biology"]}}}
_PROFILE = {"taxonomies": [{"domain": "pizza", "n_classes": 6, "max_depth": 2,
                            "depth_distribution": {"0": 0.2, "1": 0.4, "2": 0.4}}]}


class ContractTests(unittest.TestCase):
    def test_base_task_declares_the_seeded_hooks(self):
        for hook in ("build_seeded_artifact", "build_seeded_generation_prompt",
                     "verify_structured_match"):
            self.assertTrue(hasattr(BaseTask, hook), hook)

    def test_the_base_defaults_fail_loudly(self):
        # Duck-typed hooks that silently no-op are what let taxonomy's feedback
        # contract degrade unnoticed before. A task that does not implement
        # these must say so, not generate nothing.
        class _Bare(TaxonomyTask):
            pass
        with self.assertRaises(NotImplementedError):
            BaseTask.build_seeded_artifact(_Bare(), {}, "forward", None, _CONFIG, Random(0))


class SeedPoolTests(unittest.TestCase):
    def test_the_pool_holds_subtrees_with_a_recorded_depth(self):
        pool = TaxonomyTask().get_seed_pool(_CONFIG, _REAL, "forward", rng=Random(0))
        self.assertTrue(pool)
        for seed in pool:
            self.assertIn("max_depth", seed)
            self.assertGreaterEqual(len(seed["classes"]), 3)


class ForwardSeededTests(unittest.TestCase):
    def test_forward_inherits_the_seed_structure_exactly(self):
        task = TaxonomyTask()
        seed = task.get_seed_pool(_CONFIG, _REAL, "forward", rng=Random(0))[0]
        gold = task.build_seeded_artifact(seed, "forward", None, _CONFIG, Random(0))
        # Edits preserve class NAMES, so inheritance is plain equality -- no
        # isomorphism needed to compare a seed with the gold derived from it.
        self.assertEqual(sorted(gold["classes"]), sorted(seed["classes"]))
        self.assertEqual(sorted(gold["subclass_axioms"]),
                         sorted(seed["subclass_axioms"]))

    def test_forward_needs_no_profile(self):
        # The only taxonomy cell that runs without one, which makes it the
        # cheapest end-to-end smoke test the task has.
        task = TaxonomyTask()
        seed = task.get_seed_pool(_CONFIG, _REAL, "forward", rng=Random(0))[0]
        self.assertIsNotNone(task.build_seeded_artifact(seed, "forward", None, _CONFIG, Random(0)))

    def test_the_gold_records_the_seed_depth_bucket(self):
        task = TaxonomyTask()
        seed = task.get_seed_pool(_CONFIG, _REAL, "forward", rng=Random(0))[0]
        gold = task.build_seeded_artifact(seed, "forward", None, _CONFIG, Random(0))
        self.assertEqual(gold["source_max_depth"], seed["max_depth"])


class InverseSeededTests(unittest.TestCase):
    def test_inverse_imposes_a_change_on_the_seed(self):
        # THE assertion separating the two cells. Without it they could silently
        # converge and every other test would stay green.
        task = TaxonomyTask()
        seed = task.get_seed_pool(_CONFIG, _REAL, "forward", rng=Random(0))[0]
        gold = task.build_seeded_artifact(seed, "inverse", _PROFILE, _CONFIG, Random(0))
        self.assertNotEqual(
            (sorted(gold["classes"]), sorted(gold["subclass_axioms"])),
            (sorted(seed["classes"]), sorted(seed["subclass_axioms"])))

    def test_inverse_without_a_profile_raises(self):
        task = TaxonomyTask()
        seed = task.get_seed_pool(_CONFIG, _REAL, "forward", rng=Random(0))[0]
        with self.assertRaises(RuntimeError) as ctx:
            task.build_seeded_artifact(seed, "inverse", None, _CONFIG, Random(0))
        self.assertIn("profile", str(ctx.exception).lower())


class PromptAndVerifyTests(unittest.TestCase):
    def test_the_prompt_carries_the_structure_and_the_target_domain(self):
        task = TaxonomyTask()
        seed = task.get_seed_pool(_CONFIG, _REAL, "forward", rng=Random(0))[0]
        gold = task.build_seeded_artifact(seed, "forward", None, _CONFIG, Random(0))
        prompt = task.build_seeded_generation_prompt(gold)
        self.assertIn(gold["domain"], prompt)
        self.assertIn("subclass_axioms", prompt)

    def test_the_prompt_never_leaks_the_source_ontology_names(self):
        # Re-verbalisation exists partly to break memorisation of a canonical
        # tutorial ontology. Leaking the original names would defeat that.
        task = TaxonomyTask()
        seed = task.get_seed_pool(_CONFIG, _REAL, "forward", rng=Random(0))[0]
        gold = task.build_seeded_artifact(seed, "forward", None, _CONFIG, Random(0))
        prompt = task.build_seeded_generation_prompt(gold)
        for name in ("Margherita", "Napoletana", "Gelato"):
            self.assertNotIn(name, prompt)

    def test_the_prompt_states_the_ordering_contract_the_verifier_enforces(self):
        # matches_structure maps positionally (dict(zip(gold, other))), so an
        # answer whose class order differs is discarded however faithful its
        # shape. The prompt's rules never said so: the code asserted the prompt
        # stated the contract, the docs told the reader it did, and every
        # round-trip test happened to answer in order, so nothing could see the
        # gap. Assert the rendered prompt actually carries the rule.
        task = TaxonomyTask()
        seed = task.get_seed_pool(_CONFIG, _REAL, "forward", rng=Random(0))[0]
        gold = task.build_seeded_artifact(seed, "forward", None, _CONFIG, Random(0))
        prompt = task.build_seeded_generation_prompt(gold)
        rules = prompt[prompt.index("Rules:"):]
        self.assertIn("order", rules.lower())
        self.assertRegex(rules, r"EXACTLY the order")

    def test_an_answer_in_a_different_class_order_is_discarded(self):
        # Why the rule above has to be in the prompt: this answer is a perfect
        # structural copy under its own naming, and the verifier still rejects
        # it, because position is the correspondence.
        task = TaxonomyTask()
        gold = {"domain": "d", "classes": ["A", "B", "C"],
                "subclass_axioms": [["B", "A"], ["C", "A"]]}
        in_order = {"classes": ["X", "Y", "Z"],
                    "subclass_axioms": [["Y", "X"], ["Z", "X"]]}
        # Same shape -- one root, two children -- but the root is listed last.
        reordered = {"classes": ["Y", "Z", "X"],
                     "subclass_axioms": [["Y", "X"], ["Z", "X"]]}
        self.assertTrue(task.verify_structured_match(gold, in_order))
        self.assertFalse(task.verify_structured_match(gold, reordered))

    def test_verify_accepts_a_renaming_and_rejects_a_reshape(self):
        task = TaxonomyTask()
        gold = {"domain": "d", "classes": ["A", "B", "C"],
                "subclass_axioms": [["B", "A"], ["C", "A"]]}
        renamed = {"classes": ["X", "Y", "Z"],
                   "subclass_axioms": [["Y", "X"], ["Z", "X"]]}
        reshaped = {"classes": ["X", "Y", "Z"],
                    "subclass_axioms": [["Y", "X"], ["Z", "Y"]]}
        self.assertTrue(task.verify_structured_match(gold, renamed))
        self.assertFalse(task.verify_structured_match(gold, reshaped))


class RoundTripTests(unittest.TestCase):
    """A model that answers the prompt perfectly must be ACCEPTED.

    The prompt anonymises classes and the verifier maps positionally, so those
    two orderings have to agree. Nothing tested them together before, and they
    disagreed: a perfect answer was rejected whenever gold's classes were not
    already sorted. The failure mode is silent -- zero artifacts, no error from
    this code.
    """

    def _answer(self, prompt):
        import json
        struct = json.loads(prompt[prompt.index('{\n  "classes"'):].split("\n\nReturn")[0])
        mapping = {c: f"M{i}" for i, c in enumerate(struct["classes"])}
        return {"classes": [mapping[c] for c in struct["classes"]],
                "subclass_axioms": [[mapping[c], mapping[p]]
                                    for c, p in struct["subclass_axioms"]]}

    def test_a_perfect_answer_is_accepted(self):
        task = TaxonomyTask()
        seed = task.get_seed_pool(_CONFIG, _REAL, "forward", rng=Random(0))[0]
        gold = task.build_seeded_artifact(seed, "forward", None, _CONFIG, Random(0))
        prompt = task.build_seeded_generation_prompt(gold)
        self.assertTrue(task.verify_structured_match(gold, self._answer(prompt)))

    def test_a_perfect_answer_is_accepted_when_gold_is_unsorted(self):
        # The regression itself. Real seeds arrive sorted today, so only an
        # explicitly unsorted gold exercises the disagreement.
        task = TaxonomyTask()
        gold = {"domain": "d",
                "classes": ["Food", "Pizza", "Apple"],
                "subclass_axioms": [["Pizza", "Food"], ["Apple", "Food"]],
                "source_max_depth": 1}
        prompt = task.build_seeded_generation_prompt(gold)
        self.assertTrue(task.verify_structured_match(gold, self._answer(prompt)))

    def test_a_wrong_answer_is_still_rejected(self):
        # Guards the obvious over-correction: making everything match.
        task = TaxonomyTask()
        seed = task.get_seed_pool(_CONFIG, _REAL, "forward", rng=Random(0))[0]
        gold = task.build_seeded_artifact(seed, "forward", None, _CONFIG, Random(0))
        prompt = task.build_seeded_generation_prompt(gold)
        answer = self._answer(prompt)
        answer["subclass_axioms"] = answer["subclass_axioms"][:-1]   # drop an edge
        self.assertFalse(task.verify_structured_match(gold, answer))


if __name__ == "__main__":
    unittest.main()
