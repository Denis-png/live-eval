"""A failed generation run must leave its evidence behind.

The first live taxonomy run rejected every sample and wrote an EMPTY session
directory. The "0 usable samples" guard raised inside _run_generation before
anything was saved, and even on success only accepted samples were archived --
every rejection's diagnostics were dropped. Diagnosing it took a live API call
just to see what the model had said.

Two fixes, both pinned here:
  * the structured loops record their rejections, and the pipeline persists them
    from a `finally`, so a run that aborts still writes them;
  * a preview keeps the START and the END of a response. For a reasoning model
    the answer -- and whatever broke -- is at the end; a head-only preview of a
    45,000-character response shows nothing but the opening of its <think> block.
"""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from framework import pipeline
from framework.generators.base_generator import BaseGenerator, preview_response

_GOLDS = [{"domain": "d", "classes": ["A", "B"], "subclass_axioms": [["B", "A"]],
           "source_max_depth": 1} for _ in range(3)]


def _ok(raw):
    return {"artifact": {"domain": "d", "classes": ["X", "Y"],
                         "subclass_axioms": [["Y", "X"]]}, "diagnostic": {}}


class _Script(BaseGenerator):
    def __init__(self, responses):
        self.responses = list(responses)

    def call_api(self, prompt):
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _seeded(gen, verify, golds=_GOLDS, attempts=1):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return gen.generate_structured_seeded(golds, build_prompt=lambda g: "p",
                                              parse=_ok, verify=verify,
                                              max_parse_attempts=attempts)


class PreviewTests(unittest.TestCase):
    def test_a_short_response_is_kept_whole(self):
        self.assertEqual(preview_response("short", limit=800), "short")

    def test_a_long_response_is_bounded_exactly(self):
        self.assertEqual(len(preview_response("x" * 5000, limit=800)), 800)

    def test_it_keeps_both_the_start_and_the_end(self):
        text = "START" + "m" * 5000 + "END"
        preview = preview_response(text, limit=800)
        self.assertTrue(preview.startswith("START"))
        self.assertTrue(preview.endswith("END"))

    def test_a_reasoning_models_answer_survives(self):
        # The shape that actually fails in practice: a long unclosed <think>
        # with the answer glued onto the end.
        text = "<think>" + "reasoning " * 4000 + '```json\n{"domain": "d"}\n```'
        self.assertIn('{"domain": "d"}', preview_response(text, limit=800))

    def test_none_and_non_strings_are_safe(self):
        self.assertIsNone(preview_response(None))
        self.assertEqual(preview_response(123), "123")


class SeededRejectionTests(unittest.TestCase):
    def test_a_gold_that_never_verifies_is_recorded(self):
        gen = _Script(["answer-1", "answer-2", "answer-3"])
        out = _seeded(gen, verify=lambda g, p: False)
        self.assertEqual(out, [])
        self.assertEqual(len(gen.last_rejections), 3)
        rej = gen.last_rejections[0]
        self.assertEqual(rej["attempts"][0]["rejection_reason"],
                         "structure does not match gold")
        # what the model actually said, not just that it was wrong
        self.assertEqual(rej["attempts"][0]["raw_preview"], "answer-1")

    def test_accepted_golds_are_not_recorded_as_rejections(self):
        gen = _Script(["a", "b", "c"])
        _seeded(gen, verify=lambda g, p: True)
        self.assertEqual(gen.last_rejections, [])

    def test_rejections_do_not_leak_from_one_run_into_the_next(self):
        gen = _Script(["a", "b", "c", "d", "e", "f"])
        _seeded(gen, verify=lambda g, p: False)
        _seeded(gen, verify=lambda g, p: True)
        self.assertEqual(gen.last_rejections, [])

    def test_a_failed_call_never_carries_a_stale_preview(self):
        # `raw` is a loop variable. If attempt 2 raises, attempt 1's response is
        # still bound -- attaching it would misattribute an old answer.
        gen = _Script(["first-answer", TimeoutError("read timed out")])
        _seeded(gen, verify=lambda g, p: False, golds=_GOLDS[:1], attempts=2)
        failed_attempt = gen.last_rejections[0]["attempts"][1]
        self.assertIn("TimeoutError", failed_attempt["rejection_reason"])
        self.assertNotIn("raw_preview", failed_attempt)

    def test_the_gold_it_failed_on_is_identified(self):
        gen = _Script(["a", "b", "c"])
        _seeded(gen, verify=lambda g, p: False)
        self.assertEqual([r["index"] for r in gen.last_rejections], [1, 2, 3])
        self.assertEqual(gen.last_rejections[0]["gold_classes"], ["A", "B"])


class SeedlessRejectionTests(unittest.TestCase):
    def test_a_sample_that_never_parses_is_recorded(self):
        gen = _Script(["garbage-1", "garbage-2"])

        def never(raw):
            return {"artifact": None, "diagnostic": {"rejection_reason": "malformed_json"}}

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            out = gen.generate_structured(build_prompt=lambda fb: "p", parse=never,
                                          sample_size=2, max_parse_attempts=1)
        self.assertEqual(out, [])
        self.assertEqual(len(gen.last_rejections), 2)
        self.assertEqual(gen.last_rejections[0]["attempts"][0]["raw_preview"], "garbage-1")


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_rejections_are_written_next_to_the_run(self):
        gen = _Script([])
        gen.last_rejections = [{"index": 1, "attempts": [{"rejection_reason": "x"}]}]
        path = pipeline.save_rejections(gen, self.dir.name, run_idx=0)
        self.assertTrue(path.endswith("run_1_rejected.json"))
        self.assertEqual(json.load(open(path))[0]["index"], 1)

    def test_nothing_is_written_when_nothing_was_rejected(self):
        gen = _Script([])
        gen.last_rejections = []
        self.assertIsNone(pipeline.save_rejections(gen, self.dir.name, run_idx=0))
        self.assertEqual(os.listdir(self.dir.name), [])

    def test_a_generator_that_records_nothing_is_a_no_op(self):
        # Only the structured loops record rejections so far; the others must
        # not break the pipeline for lacking the attribute.
        self.assertIsNone(pipeline.save_rejections(object(), self.dir.name, run_idx=0))

    def test_a_run_that_aborts_still_leaves_its_rejections(self):
        # THE regression: every sample rejected -> "0 usable samples" raises ->
        # before this, the session directory stayed empty.
        gen = _Script(["bad-1", "bad-2", "bad-3"])

        def run():
            _seeded(gen, verify=lambda g, p: False)
            raise RuntimeError("Generation produced 0 usable samples")

        with self.assertRaises(RuntimeError):
            pipeline.run_recording_rejections(run, gen, self.dir.name, run_idx=0)
        written = json.load(open(os.path.join(self.dir.name, "run_1_rejected.json")))
        self.assertEqual(len(written), 3)


if __name__ == "__main__":
    unittest.main()
