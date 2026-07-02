import json
import os
import functools
from ..base_task import BaseTask
from framework.profiling.spam_profiler import (
    DEFAULT_SPAM_DATASET,
    DEFAULT_SPAM_SPLIT,
)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "tasks", "spam.json")


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


class SpamTask(BaseTask):

    def __init__(self):
        self._config = _load_config()

    def get_error_types(self) -> list[str]:
        return self._config["error_types"]

    def get_prompt_instruction(self) -> str:
        return self._config["prompt"]

    def get_judge_prompt(self) -> str | None:
        return self._config.get("judge_prompt")

    def get_inverse_prompt(self) -> str | None:
        return self._config.get("inverse_prompt")

    def get_inverse_judge_prompt(self) -> str | None:
        return self._config.get("inverse_judge_prompt")

    def get_error_descriptions(self) -> dict[str, str]:
        return self._config.get("inverse_error_descriptions", {})

    def profile_error_distribution(self, real_data, count_max=5, config=None):
        """Empirical inverse-mode distribution over spam signals. real_data is
        HAM-only (parse_row drops SPAM), so load the SPAM subset separately via
        the spam profiler and count which signals fire per SPAM message."""
        from framework.profiling import spam_profiler
        from framework.profiling.spam_distribution import profile_spam_distribution

        cfg = config or {}
        ds = cfg.get("dataset") or {}
        inv = ((cfg.get("generation") or {}).get("inverse") or {})
        hf_token = ds.get("hf_token") or (cfg.get("api_keys") or {}).get("huggingface")

        kwargs = {
            "sample_size": inv.get("profile_size", 200),
            "streaming": ds.get("streaming", False),
            "hf_token": hf_token,
            "dataset_name": ds.get("name") or DEFAULT_SPAM_DATASET,
            "split": ds.get("split") or DEFAULT_SPAM_SPLIT,
        }
        rows = spam_profiler.load_spam_rows(**kwargs)
        spam_rows = [r for r in rows if r["label"] == "SPAM"]
        supported = set(self.get_error_descriptions().keys())
        return profile_spam_distribution(spam_rows, supported, count_max=count_max)

    def get_evaluators(self) -> list[str]:
        return self._config["evaluators"]

    def get_evaluator_fns(self) -> dict:
        from framework.evaluators.classification.accuracy import compute_accuracy
        from framework.evaluators.classification.precision import compute_precision
        from framework.evaluators.classification.recall import compute_recall
        from framework.evaluators.classification.f1 import compute_f1
        from framework.evaluators.classification.fpr import compute_fpr
        return {
            "accuracy":  compute_accuracy,
            "precision": functools.partial(compute_precision, positive_label="SPAM"),
            "recall":    functools.partial(compute_recall, positive_label="SPAM"),
            "f1":        functools.partial(compute_f1, positive_label="SPAM"),
            "fpr":       functools.partial(compute_fpr, positive_label="SPAM"),
        }

    def get_model(self, model_config: dict):
        model_type = model_config["type"]
        params = self._config["models"].get(model_type, {})
        merged = {**model_config, **params}

        if model_type == "roberta":
            from framework.models.spam.roberta import RobertaSpamModel
            return RobertaSpamModel(merged)
        else:
            raise ValueError(
                f"Unsupported spam model type: '{model_type}'. "
                f"Add it to configs/tasks/spam.json and tasks/spam/task.py."
            )
        
    def get_label(self, result: dict) -> str:
        # Paraphrased samples remain legitimate (HAM); all other error types produce SPAM.
        if result.get("error_type") == "paraphrase":
            return "HAM"
        return "SPAM"

    def parse_row(self, row: dict) -> dict | None:
        # Skip spam rows — the generator needs legitimate (HAM) messages as input.
        if str(row.get("label", "")).lower() in ("spam", "1"):
            return None
        text = row.get("text") or row.get("message") or row.get("sms")
        if not text:
            return None
        return {"incorrect": text}

    def get_task_name(self) -> str:
        return "spam"