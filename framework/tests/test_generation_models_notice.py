import unittest

from framework.main import generation_models_notice


class GenerationModelsNoticeTests(unittest.TestCase):
    def test_notice_when_config_lists_generation_models(self):
        config = {
            "generation": {"provider": "openrouter", "model": "minimax-m3"},
            "generation_models": [
                {"provider": "openrouter", "model": "tencent/hy3:free"},
                {"provider": "openrouter", "model": "z-ai/glm-5.2"},
            ],
        }
        notice = generation_models_notice(config)
        self.assertIsNotNone(notice)
        self.assertIn("2", notice)                        # how many were listed
        self.assertIn("minimax-m3", notice)               # what will ACTUALLY run
        self.assertIn("scripts.compare_models", notice)   # how to run the comparison
        self.assertIn("tencent/hy3:free", notice)         # names the ignored entries

    def test_no_notice_without_generation_models(self):
        config = {"generation": {"provider": "openrouter", "model": "minimax-m3"}}
        self.assertIsNone(generation_models_notice(config))

    def test_no_notice_when_list_is_empty(self):
        self.assertIsNone(generation_models_notice({"generation": {}, "generation_models": []}))


if __name__ == "__main__":
    unittest.main()
