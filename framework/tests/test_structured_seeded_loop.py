"""The seeded structured loop: verify against computed gold, never adopt output."""
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from framework.generators.base_generator import BaseGenerator, TruncatedResponse

_GOLDS = [{"domain": "d", "classes": ["A", "B"], "subclass_axioms": [["B", "A"]],
           "source_max_depth": 1} for _ in range(3)]


def _parse_ok(raw):
    return {"artifact": {"domain": "d", "classes": ["X", "Y"],
                         "subclass_axioms": [["Y", "X"]]},
            "diagnostic": {}}


def _parse_fail(raw):
    return {"artifact": None, "diagnostic": {"rejection_reason": "cycle"}}


class _Gen(BaseGenerator):
    def __init__(self, script=None):
        self.calls = 0
        self.script = script or {}

    def call_api(self, prompt):
        self.calls += 1
        exc = self.script.get(self.calls)
        if exc:
            raise exc
        return "{}"


def _run(gen, parse=_parse_ok, verify=lambda g, p: True, golds=None, attempts=1):
    with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
        result = gen.generate_structured_seeded(
            golds if golds is not None else _GOLDS,
            build_prompt=lambda gold: "p",
            parse=parse, verify=verify,
            max_parse_attempts=attempts,
        )
    return result, out.getvalue() + err.getvalue()


class SeededLoopTests(unittest.TestCase):
    def test_one_artifact_per_gold(self):
        out, _ = _run(_Gen())
        self.assertEqual(len(out), 3)

    def test_the_artifact_is_self_consistent_and_gold_shaped(self):
        # THE assertion of this whole spec, in two halves.
        # (a) self-consistent: every axiom names classes the artifact contains.
        #     Carrying gold's axioms onto the model's class list would not be.
        # (b) gold-shaped: the structure is the one that was computed, because
        #     verify() gated it -- a model that drifts loses its sample rather
        #     than redefining the reference.
        from framework.tasks.taxonomy.graph_ops import matches_structure
        out, _ = _run(_Gen())
        art = out[0]
        names = set(art["classes"])
        for child, parent in art["subclass_axioms"]:
            self.assertIn(child, names)
            self.assertIn(parent, names)
        self.assertTrue(matches_structure(
            _GOLDS[0]["classes"], _GOLDS[0]["subclass_axioms"],
            art["classes"], art["subclass_axioms"]))

    def test_a_failed_verification_is_a_skip_not_an_artifact(self):
        out, text = _run(_Gen(), verify=lambda g, p: False)
        self.assertEqual(out, [])
        self.assertIn("structure", text.lower())

    def test_garbage_output_produces_no_artifacts_and_no_corrupt_gold(self):
        out, _ = _run(_Gen(), parse=_parse_fail)
        self.assertEqual(out, [])

    def test_a_truncated_response_costs_one_gold_not_the_run(self):
        gen = _Gen({2: TruncatedResponse("truncated at max_tokens=4096")})
        out, _ = _run(gen)
        self.assertEqual(len(out), 2)

    def test_retries_are_bounded_by_max_parse_attempts(self):
        gen = _Gen()
        _run(gen, verify=lambda g, p: False, golds=_GOLDS[:1], attempts=3)
        self.assertEqual(gen.calls, 3)

    def test_the_source_depth_bucket_survives_onto_the_record(self):
        # Provenance: without it the cell's behaviour is invisible in the
        # archive and a later calibration spec has no attrition to measure.
        out, _ = _run(_Gen())
        self.assertEqual(out[0]["source_max_depth"], 1)

    def test_no_feedback_metadata_is_emitted(self):
        # Seeded cells run no feedback loop: gold is exact, so there is nothing
        # to iterate toward. Emitting empty rounds would imply one ran.
        out, _ = _run(_Gen())
        self.assertNotIn("generation_feedback", out[0])


if __name__ == "__main__":
    unittest.main()
