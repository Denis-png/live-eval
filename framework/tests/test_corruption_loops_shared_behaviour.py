"""Two behaviours of the shared corruption loops, pinned because a change made
for one task once altered them for every task.

  * An empty response is one call and one skipped sample. A sleep-and-retry
    (30 s in the forward and seedless loops) multiplied wall-clock time for GEC,
    spam and sentiment alike, and hid a model that answers with nothing.
  * A seedless sample records the error type the MODEL reports, falling back to
    the requested one only when it reports none -- as generate_forward does.
"""
import io
import random
import unittest
from contextlib import redirect_stdout
from unittest import mock

from framework.generators import base_generator
from framework.generators.base_generator import BaseGenerator

_PAIR = ("Error type: {etype}\n"
         "Generated: the cats sits on mat\n"
         "Ground truth: the cat sits on the mat")


class _Script(BaseGenerator):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def call_api(self, prompt):
        self.calls += 1
        return self.responses.pop(0) if self.responses else ""


def _quiet(fn):
    with mock.patch.object(base_generator.time, "sleep") as sleep, \
            redirect_stdout(io.StringIO()):
        out = fn()
    return out, sleep


class EmptyResponseTests(unittest.TestCase):
    def test_forward_skips_an_empty_response_without_retrying(self):
        gen = _Script([""])
        out, sleep = _quiet(lambda: gen.generate_forward(
            [{"incorrect": "a b c", "correct": "a b c"}], ["t"], "{sentence}",
            sample_size=1))
        self.assertEqual((out, gen.calls), ([], 1))
        sleep.assert_not_called()

    def test_inverse_skips_an_empty_response_without_retrying(self):
        gen = _Script([""])
        out, sleep = _quiet(lambda: gen.generate_inverse(
            [{"correct": "a b c"}], "{sentence} {error_spec}", {"t": "desc"},
            {"t": 1.0}, {1: 1.0}, sample_size=1, rng=random.Random(0)))
        self.assertEqual((out, gen.calls), ([], 1))
        sleep.assert_not_called()

    def test_seedless_skips_an_empty_response_without_retrying(self):
        gen = _Script([""])
        out, sleep = _quiet(lambda: gen.generate_seedless_pairs(
            ["spec"], "{spec} {error_spec}", {"t": "desc"}, {"t": 1.0}, {1: 1.0},
            rng=random.Random(0)))
        self.assertEqual((out, gen.calls), ([], 1))
        sleep.assert_not_called()


class SeedlessErrorTypeTests(unittest.TestCase):
    def _one(self, response):
        gen = _Script([response])
        out, _ = _quiet(lambda: gen.generate_seedless_pairs(
            ["spec"], "{spec} {error_spec}", {"requested": "desc"},
            {"requested": 1.0}, {1: 1.0}, rng=random.Random(0)))
        self.assertEqual(len(out), 1)
        return out[0]["error_type"]

    def test_the_type_the_model_reports_is_recorded(self):
        self.assertEqual(self._one(_PAIR.format(etype="reported")), "reported")

    def test_the_requested_type_fills_in_when_the_model_reports_none(self):
        response = _PAIR.split("\n", 1)[1]            # no "Error type:" line
        self.assertEqual(self._one(response), "requested")


if __name__ == "__main__":
    unittest.main()
