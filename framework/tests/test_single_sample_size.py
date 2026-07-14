import unittest

from framework.main import validate_config


def _base():
    return {
        "task": {"name": "spam"},
        "dataset": {"source": "huggingface",
                    "huggingface": {"name": "d", "split": "train"}},
        "generation": {"provider": "openrouter", "model": "m",
                       "num_runs": 2, "sample_size": 20},
        "task_models": [{"name": "x", "type": "roberta"}],
    }


class SingleSampleSizeTests(unittest.TestCase):
    def test_config_without_dataset_sample_size_is_valid(self):
        validate_config(_base())  # must NOT raise

    def test_still_requires_generation_sample_size(self):
        cfg = _base()
        del cfg["generation"]["sample_size"]
        with self.assertRaises(ValueError):
            validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
