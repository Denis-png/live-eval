"""analyze_results keeps a calibrated run apart from the uncalibrated run of the
same cell, and can be limited to one sweep.

dedup_sessions kept one session per (task, cell, generation model), ignoring
calibration. The ablation's phase B (calibrated) and phase A (uncalibrated) runs
of the best cell share that key, so the A-vs-B comparison lost one side --
the older, which is the baseline.
"""
import sys
import unittest

sys.path.insert(0, "scripts")

import analyze_results as ar


def _session(created, calibrated, runs=3, model="minimax-m3"):
    meta = {"task": "gec", "strategy": "corruption", "mode": "inverse",
            "seedless": False, "model": model, "created": created,
            "runs_completed": runs,
            "calibration": {"path": "x_calibration.json"} if calibrated else None}
    return {"meta": meta, "dir": f"{created}-{calibrated}", "results": {}}


class CalibratedSessionTests(unittest.TestCase):
    def test_a_calibrated_session_is_its_own_cell(self):
        self.assertEqual(ar._strategy_of(_session("2026-09-12", True)["meta"]),
                         "inverse+calibrated")
        self.assertEqual(ar._strategy_of(_session("2026-09-12", False)["meta"]), "inverse")

    def test_dedup_keeps_both_sides_of_the_calibration_ablation(self):
        a, b = _session("2026-09-12T01", False), _session("2026-09-12T09", True)
        kept, dropped = ar.dedup_sessions([a, b])
        self.assertEqual((len(kept), dropped), (2, []))

    def test_the_suffix_does_not_disturb_the_semantics_quarantine(self):
        meta = dict(_session("2026-09-12", True)["meta"], task="spam",
                    strategy="class_conditional", class_conditional_semantics="old")
        self.assertEqual(ar._cell_and_semantics(ar._strategy_of(meta)),
                         ("inverse+calibrated", "old"))


class SinceFilterTests(unittest.TestCase):
    def test_sessions_before_the_cutoff_are_left_out(self):
        old, new = _session("2026-08-01T00:00:00", False), _session("2026-09-12T00:00:00", False)
        self.assertEqual(ar.filter_since([old, new], "2026-09-11"), [new])

    def test_no_cutoff_keeps_everything(self):
        sessions = [_session("2026-08-01", False)]
        self.assertEqual(ar.filter_since(sessions, None), sessions)


if __name__ == "__main__":
    unittest.main()
