import unittest

from framework.pipeline import resolve_output_paths


class OutputPathsTests(unittest.TestCase):
    def test_builds_session_tree(self):
        cfg = {"output": {"base_dir": "framework/data/runs"}}
        p = resolve_output_paths(cfg, "spam", "20260707_101010")
        self.assertTrue(p["session_dir"].endswith("framework/data/runs/spam/20260707_101010"))
        self.assertTrue(p["generated_dir"].endswith("/generated"))
        self.assertTrue(p["results"].endswith("/results.json"))
        self.assertTrue(p["real_sample"].endswith("/real_sample.json"))
        self.assertTrue(p["profile"].endswith("/profile.json"))
        self.assertTrue(p["plots_dir"].endswith("/plots"))

    def test_default_base_dir(self):
        p = resolve_output_paths({}, "gec", "s")
        self.assertIn("framework/data/runs/gec/s", p["session_dir"])


if __name__ == "__main__":
    unittest.main()
