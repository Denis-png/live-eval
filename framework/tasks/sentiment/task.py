import functools
import json
import os
from ..base_task import BaseTask

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "tasks", "sentiment.json")


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


class SentimentTask(BaseTask):

    def __init__(self):
        self._config = _load_config()

    def get_error_types(self) -> list[str]:
        return self._config["error_types"]

    def get_prompt_instruction(self) -> str:
        return self._config["prompt"]

    def get_judge_prompt(self) -> str | None:
        return self._config.get("judge_prompt")

    def get_evaluators(self) -> list[str]:
        return self._config["evaluators"]

    _CLASSES = ("NEGATIVE", "NEUTRAL", "POSITIVE")

    def get_evaluator_fns(self) -> dict:
        from framework.evaluators.classification.accuracy import compute_accuracy
        from framework.evaluators.classification.macro_precision import compute_macro_precision
        from framework.evaluators.classification.macro_recall import compute_macro_recall
        from framework.evaluators.classification.macro_f1 import compute_macro_f1
        return {
            "accuracy": compute_accuracy,
            "macro_precision": functools.partial(compute_macro_precision, labels=self._CLASSES),
            "macro_recall": functools.partial(compute_macro_recall, labels=self._CLASSES),
            "macro_f1": functools.partial(compute_macro_f1, labels=self._CLASSES),
        }

    def get_model(self, model_config: dict):
        model_type = model_config["type"]
        params = self._config["models"].get(model_type, {})
        merged = {**model_config, **params}

        if model_type in ("bertweet", "multilingual"):
            from framework.models.sentiment.transformer import TransformerSentimentModel
            return TransformerSentimentModel(merged)
        raise ValueError(
            f"Unsupported sentiment model type: '{model_type}'. "
            f"Add it to configs/tasks/sentiment.json and tasks/sentiment/task.py."
        )

    def get_label(self, result: dict) -> str | None:
        error_type = result.get("error_type", "")
        if "negative" in error_type or error_type in ("sarcasm_injection", "negation_insertion"):
            return "NEGATIVE"
        if "positive" in error_type:
            return "POSITIVE"
        if error_type == "intensity_reduction":
            return "NEUTRAL"
        return None  # paraphrase and unknown types: original sentiment unknown, skip

    def get_eval_samples(self, synthetic: list[dict]) -> list[dict]:
        out = []
        for item in synthetic:
            label = self.get_label(item)
            if label is None:
                continue  # skip items without a deterministic ground-truth label
            out.append({**item, "text": item["corrupted"], "label": label})
        return out

    _LABEL_MAP = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

    def parse_row(self, row: dict) -> dict | None:
        text = row.get("text") or row.get("sentence") or row.get("review")
        if not text:
            return None
        raw_label = row.get("label")
        label = self._LABEL_MAP.get(raw_label, str(raw_label))
        return {"incorrect": text, "sentiment_label": label}

    def get_real_eval_samples(self, config: dict, real_data: list[dict]) -> list[dict]:
        return [
            {"text": r["incorrect"], "label": r["sentiment_label"]}
            for r in real_data
            if r.get("incorrect") and r.get("sentiment_label")
        ]

    def get_task_name(self) -> str:
        return "sentiment"
