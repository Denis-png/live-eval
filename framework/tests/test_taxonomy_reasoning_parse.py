"""Structured responses from reasoning models must parse.

The first live taxonomy run failed every sample with malformed_json. The cause was
not the model: minimax-m3 opened a <think> block, never closed it, and glued a
correct fenced JSON answer onto the end of its reasoning. _extract_json_object ran
json.loads on the WHOLE response and only stripped a fence at the very START, so a
correct answer was rejected every time.

_strip_reasoning's own docstring describes exactly this shape, and the sentence
parsers already recover from it by taking the last occurrence. The structured JSON
parser was never given the same treatment -- in either structured cell, seeded or
seedless.
"""
import unittest

from framework.tasks.taxonomy.task import TaxonomyTask

# Verbatim minimax-m3 output from the first live seeded run: an UNCLOSED <think>
# whose reasoning runs straight into a fenced JSON answer. Do not tidy it -- its
# exact shape is the regression.
REAL_MINIMAX_RESPONSE = '<think>\nThe user wants me to rewrite a taxonomy into the textile manufacturing domain, preserving the exact structure.\n\nThe original structure has:\n- 5 classes: C0, C1, C2, C3, C4\n- Subclass axioms:\n  - C0 ⊆ C3 (C0 is child of C3)\n  - C1 ⊆ C3 (C1 is child of C3)\n  - C2 ⊆ C3 (C2 is child of C3)\n  - C3 ⊆ C4 (C3 is child of C4)\n\nSo the hierarchy is:\n- C4 (root)\n  - C3\n    - C0\n    - C1\n    - C2\n\nI need to map these to textile manufacturing domain, keeping the same structure.\n\nLet me think about a textile manufacturing taxonomy:\n- C4 could be "TextileProduct" (the root)\n- C3 could be "Fabric" (intermediate category)\n  - C0 could be "WovenFabric" (child of Fabric)\n  - C1 could be "KnittedFabric" (child of Fabric)\n  - C2 could be "NonwovenFabric" (child of Fabric)\n\nLet me verify:\n- C0 ⊆ C3: WovenFabric ⊆ Fabric ✓\n- C1 ⊆ C3: KnittedFabric ⊆ Fabric ✓\n- C2 ⊆ C3: NonwovenFabric ⊆ Fabric ✓\n- C3 ⊆ C4: Fabric ⊆ TextileProduct ✓\n\nThe order should be:\n- C0 → WovenFabric\n- C1 → KnittedFabric\n- C2 → NonwovenFabric\n- C3 → Fabric\n- C4 → TextileProduct\n\nLet me finalizethe JSON.```json\n{\n  "domain": "textile manufacturing",\n  "classes": [\n    "WovenFabric",\n    "KnittedFabric",\n    "NonwovenFabric",\n    "Fabric",\n    "TextileProduct"\n  ],\n  "subclass_axioms": [\n    ["WovenFabric", "Fabric"],\n    ["KnittedFabric", "Fabric"],\n    ["NonwovenFabric", "Fabric"],\n    ["Fabric", "TextileProduct"]\n  ]\n}\n```'

_ANSWER = {"domain": "textile manufacturing",
           "classes": ["WovenFabric", "KnittedFabric", "NonwovenFabric",
                       "Fabric", "TextileProduct"]}


def _parse(text):
    return TaxonomyTask().parse_structured_generation_with_diagnostics(text)


class ReasoningModelResponseTests(unittest.TestCase):
    def test_the_real_unclosed_think_response_parses(self):
        result = _parse(REAL_MINIMAX_RESPONSE)
        self.assertIsNotNone(result["artifact"], result["diagnostic"])
        self.assertEqual(result["artifact"]["classes"], _ANSWER["classes"])
        self.assertEqual(result["artifact"]["domain"], _ANSWER["domain"])

    def test_the_real_response_would_have_been_accepted_end_to_end(self):
        # Not just parsed: it carries the gold structure, so the smoke run's
        # sample was a correct answer the parser threw away.
        gold = {"domain": "textile manufacturing",
                "classes": ["C0", "C1", "C2", "C3", "C4"],
                "subclass_axioms": [["C0", "C3"], ["C1", "C3"], ["C2", "C3"],
                                    ["C3", "C4"]]}
        parsed = _parse(REAL_MINIMAX_RESPONSE)["artifact"]
        self.assertTrue(TaxonomyTask().verify_structured_match(gold, parsed))

    def test_a_closed_think_block_is_stripped(self):
        text = '<think>reasoning {not json}</think>\n{"domain": "d", ' \
               '"classes": ["A", "B"], "subclass_axioms": [["B", "A"]]}'
        self.assertEqual(_parse(text)["artifact"]["classes"], ["A", "B"])

    def test_the_last_object_wins_over_a_draft_inside_the_reasoning(self):
        # Reasoning models often write a draft answer before the final one.
        # The final answer is last; a draft must never become the artifact.
        text = ('<think>draft: {"domain": "d", "classes": ["Draft"], '
                '"subclass_axioms": []} -- no, revise.\n'
                '```json\n{"domain": "d", "classes": ["A", "B"], '
                '"subclass_axioms": [["B", "A"]]}\n```')
        self.assertEqual(_parse(text)["artifact"]["classes"], ["A", "B"])


class NoRegressionTests(unittest.TestCase):
    def test_plain_json_still_parses(self):
        text = '{"domain": "d", "classes": ["A", "B"], "subclass_axioms": [["B", "A"]]}'
        self.assertEqual(_parse(text)["artifact"]["classes"], ["A", "B"])

    def test_a_leading_fence_still_parses(self):
        text = '```json\n{"domain": "d", "classes": ["A", "B"], ' \
               '"subclass_axioms": [["B", "A"]]}\n```'
        self.assertEqual(_parse(text)["artifact"]["classes"], ["A", "B"])

    def test_prose_with_no_json_is_still_rejected(self):
        # Recovering an answer must never mean inventing one.
        result = _parse("<think>I could not decide on a structure.")
        self.assertIsNone(result["artifact"])
        self.assertEqual(result["diagnostic"]["rejection_reason"], "malformed_json")

    def test_a_json_array_is_still_rejected(self):
        self.assertIsNone(_parse('["A", "B"]')["artifact"])


if __name__ == "__main__":
    unittest.main()
