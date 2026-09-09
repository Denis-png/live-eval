"""One bad response must cost one artifact, not the whole run.

generate_structured was the only generation loop that called call_api outside a
try/except. TruncatedResponse's own docstring claimed "every generation loop
already catches per-sample exceptions and continues" — true of the other four,
false here: a reasoning model that spent its budget on chain-of-thought aborted
the run and discarded every artifact already built. That is the failure
docs/taxonomy_induction.md records as taxonomy's blocker.
"""
import io
import unittest
from contextlib import redirect_stdout, redirect_stderr

from framework.generators.base_generator import BaseGenerator, TruncatedResponse

_ARTIFACT = '{"classes": ["A", "B"]}'


def _parse_ok(raw):
    return {"artifact": {"classes": ["A", "B"]}, "diagnostic": {}}


class _Scripted(BaseGenerator):
    """Raises the scripted exception on the Nth call, else returns an artifact."""

    def __init__(self, failures: dict):
        self.failures = failures     # {call_index: exception}
        self.calls = 0

    def call_api(self, prompt):
        self.calls += 1
        exc = self.failures.get(self.calls)
        if exc is not None:
            raise exc
        return _ARTIFACT


def _run(gen, sample_size=4, max_parse_attempts=1):
    with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
        result = gen.generate_structured(
            build_prompt=lambda fb: "p",
            parse=_parse_ok,
            sample_size=sample_size,
            max_parse_attempts=max_parse_attempts,
        )
    return result, out.getvalue() + err.getvalue()


class TruncationResilienceTests(unittest.TestCase):
    def test_one_truncated_response_costs_one_artifact_not_the_run(self):
        gen = _Scripted({2: TruncatedResponse("truncated at max_tokens=4096")})
        out, _ = _run(gen, sample_size=4)
        # Sample 2 is lost; 1, 3 and 4 survive. Before the fix this raised and
        # every artifact already generated went with it.
        self.assertEqual(len(out), 3)

    def test_the_truncation_reason_reaches_the_artifact_diagnostics(self):
        # A run that silently produces fewer artifacts teaches nothing. The
        # reason has to be recorded where the run's own metadata is read, so
        # attempt 1 truncates and attempt 2 succeeds -- the surviving artifact
        # must still carry the record of what went wrong first.
        gen = _Scripted({1: TruncatedResponse("truncated at max_tokens=4096")})
        out, text = _run(gen, sample_size=1, max_parse_attempts=2)
        self.assertIn("truncated", text.lower())
        self.assertEqual(len(out), 1)
        attempts = out[0]["generation_feedback"]["attempts"]
        self.assertTrue(any("truncat" in str(a).lower() for a in attempts),
                        f"no truncation diagnostic recorded: {attempts}")

    def test_retries_are_bounded_and_the_run_continues(self):
        # Every attempt for sample 1 truncates. It must exhaust max_parse_attempts,
        # be skipped, and leave the remaining samples untouched.
        gen = _Scripted({i: TruncatedResponse("truncated") for i in (1, 2, 3)})
        out, _ = _run(gen, sample_size=3, max_parse_attempts=3)
        self.assertEqual(len(out), 2)     # samples 2 and 3 survive
        self.assertEqual(gen.calls, 5)    # 3 burned on sample 1, then 1 each

    def test_an_unexpected_error_also_costs_only_its_own_sample(self):
        # Not just truncation: any provider-side failure (timeout, 5xx, a task
        # callable raising) must degrade the same way.
        gen = _Scripted({2: RuntimeError("connection reset")})
        out, text = _run(gen, sample_size=3)
        self.assertEqual(len(out), 2)
        self.assertIn("connection reset", text)


class FeedbackResilienceTests(unittest.TestCase):
    """The feedback loop is a refinement, not a precondition for the artifact.

    build_feedback runs AFTER a valid artifact has been parsed and paid for. If
    it raises -- an odd-but-parseable taxonomy that divides by zero in the
    profile comparison, say -- losing the run would throw away work that already
    succeeded, for a failure in the step that only tries to improve it.
    """

    def test_a_failing_feedback_round_keeps_the_artifact_it_already_had(self):
        gen = _Scripted({})

        def _explode(artifact):
            raise ZeroDivisionError("empty taxonomy has no mean depth")

        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            result = gen.generate_structured(
                build_prompt=lambda fb: "p",
                parse=_parse_ok,
                build_feedback=_explode,
                sample_size=2,
                max_feedback_rounds=2,
            )
        text = out.getvalue() + err.getvalue()
        self.assertEqual(len(result), 2, "artifacts lost to a feedback failure")
        self.assertIn("empty taxonomy has no mean depth", text)


if __name__ == "__main__":
    unittest.main()
