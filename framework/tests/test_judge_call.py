import unittest

from framework.pipeline import _build_judge_call


class FakeMainGenerator:
    def call_api(self, prompt: str) -> str:
        return "main-generator-response"


class BuildJudgeCallTests(unittest.TestCase):
    """Judging must be OPT-IN: no judge block (or enabled=false) → no judging.
    The config comment promises 'If omitted or enabled=false → judging is
    skipped' — silently falling back to the main generator doubles API spend."""

    def test_no_judge_block_skips_judging(self):
        self.assertIsNone(_build_judge_call({}, FakeMainGenerator()))

    def test_judge_none_skips_judging(self):
        self.assertIsNone(_build_judge_call({"judge": None}, FakeMainGenerator()))

    def test_enabled_false_skips_judging(self):
        cfg = {"judge": {"enabled": False, "provider": "groq", "model": "m"}}
        self.assertIsNone(_build_judge_call(cfg, FakeMainGenerator()))

    def test_enabled_but_missing_model_falls_back_to_main_generator(self):
        # An explicitly enabled judge means the user wants judging — degrade
        # to the main generator with a warning instead of silently skipping.
        main = FakeMainGenerator()
        cfg = {"judge": {"enabled": True}}
        # assertEqual, not assertIs: bound methods are re-created per access.
        self.assertEqual(_build_judge_call(cfg, main), main.call_api)


if __name__ == "__main__":
    unittest.main()
