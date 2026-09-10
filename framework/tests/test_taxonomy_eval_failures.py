"""A harness failure must never be silently scored as a wrong answer.

The first complete taxonomy run scored minimax-m3 at 0.000 on the real ontology.
The model had truncated at its token cap; _predict_one caught the exception and
returned an empty prediction marked `malformed` -- exactly what a model that
answered with garbage produces. The score dropped to zero with no visible cause.

A truncation, timeout or API error is not the model's answer. It is now:
  * marked `failed`, distinct from `malformed` (a response that could not be read);
  * warned about loudly, naming the model and the error, as it happens;
  * counted in its own `failed_prediction_count` diagnostic.

A failed item still scores as an empty prediction. That is deliberate: every model
is evaluated on the identical item set, so comparisons between them stay valid.
Dropping failures instead would hand each model a different -- and, since larger
items truncate more, an easier -- set. The count makes the cost visible instead.
"""
import io
import json
import unittest
from contextlib import redirect_stderr
from unittest import mock

from framework.evaluators.taxonomy._taxonomy_shared import (
    compute_taxonomy_scores,
    reset_cache,
    score_taxonomy_result,
)
from framework.generators.base_generator import TruncatedResponse
from framework.models.taxonomy import TaxonomyLLMModel
from framework.tasks.taxonomy import TaxonomyTask

_CLASSES = ["Root", "A", "B"]
_GOLD = [["A", "Root"], ["B", "Root"]]


class _Scripted:
    """Returns or raises per call, in order."""

    def __init__(self, script):
        self.script = list(script)

    def call_api(self, prompt):
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _model(script, name="mock-model"):
    with mock.patch("framework.generators.factory.load_generator",
                    return_value=_Scripted(script)):
        return TaxonomyLLMModel({"type": "llm", "name": name,
                                 "provider": "openrouter", "api_key": "k"})


def _text():
    return json.dumps({"domain": "d", "classes": _CLASSES})


def _predict(model, n=1):
    err = io.StringIO()
    with redirect_stderr(err):
        out = model.predict([_text()] * n)
    return out, err.getvalue()


class PredictionFailureTests(unittest.TestCase):
    def test_a_truncation_is_marked_failed_not_malformed(self):
        out, _ = _predict(_model([TruncatedResponse("truncated at max_tokens=8192")]))
        d = out[0]["diagnostics"]
        self.assertTrue(d["failed"])
        self.assertFalse(d["malformed"], "a failure is not an unreadable answer")
        self.assertEqual(d["error_type"], "TruncatedResponse")
        self.assertIn("max_tokens=8192", d["error"])
        self.assertEqual(out[0]["subclass_axioms"], [])

    def test_the_failure_is_announced_as_it_happens(self):
        _, stderr = _predict(_model([TimeoutError("read timed out")], name="big-llm"))
        self.assertIn("[WARN]", stderr)
        self.assertIn("big-llm", stderr)            # which model
        self.assertIn("read timed out", stderr)     # and why

    def test_a_garbage_answer_is_malformed_not_failed(self):
        # The model DID respond; its answer just could not be read. That is the
        # model's fault, and must stay distinguishable from a harness failure.
        out, stderr = _predict(_model(["this is not json"]))
        d = out[0]["diagnostics"]
        self.assertTrue(d["malformed"])
        self.assertFalse(d.get("failed", False))
        self.assertNotIn("[WARN]", stderr)

    def test_one_failure_does_not_stop_the_other_items(self):
        out, _ = _predict(_model([
            '{"subclass_axioms": [["A", "Root"]]}',
            TruncatedResponse("truncated"),
            '{"subclass_axioms": [["B", "Root"]]}',
        ]), n=3)
        self.assertEqual([o["diagnostics"].get("failed", False) for o in out],
                         [False, True, False])
        self.assertEqual(out[2]["subclass_axioms"], [["B", "Root"]])


class ScoringFailureTests(unittest.TestCase):
    def setUp(self):
        reset_cache()

    def _row(self, diagnostics, axioms=()):
        return {"classes": _CLASSES, "subclass_axioms": _GOLD,
                "prediction": {"subclass_axioms": list(axioms), "raw_output": "",
                               "diagnostics": diagnostics}}

    def test_the_scorer_carries_the_distinction_through(self):
        failed = score_taxonomy_result(self._row({"failed": True, "malformed": False,
                                                  "error": "truncated"}))
        self.assertTrue(failed["prediction_failed"])
        self.assertFalse(failed["malformed_prediction"])

    def test_failed_and_malformed_are_counted_separately(self):
        scores = compute_taxonomy_scores([
            self._row({"failed": True, "malformed": False, "error": "truncated"}),
            self._row({"malformed": True}),
            self._row({}, axioms=_GOLD),
        ])
        self.assertEqual(scores["failed_prediction_count"], 1)
        self.assertEqual(scores["malformed_prediction_count"], 1)

    def test_a_failed_item_still_counts_against_recall(self):
        # Identical item sets across models: the failed item's gold relations
        # are false negatives, and the count above is what flags the cost.
        scores = compute_taxonomy_scores([
            self._row({"failed": True, "malformed": False, "error": "truncated"}),
            self._row({}, axioms=_GOLD),
        ])
        self.assertEqual(scores["fn"], len(_GOLD))
        self.assertEqual(scores["recall"], 0.5)

    def test_the_count_reaches_the_diagnostics_metric(self):
        fns = TaxonomyTask().get_evaluator_fns()
        rows = [self._row({"failed": True, "malformed": False, "error": "x"})]
        self.assertEqual(fns["diagnostics"](rows)["failed_prediction_count"], 1)


if __name__ == "__main__":
    unittest.main()
