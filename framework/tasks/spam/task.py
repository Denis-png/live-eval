import json
import os
import functools
from ..base_task import BaseTask

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

    def get_evaluators(self) -> list[str]:
        return self._config["evaluators"]

    def get_evaluator_fns(self) -> dict:
        from framework.evaluators.classification.accuracy import compute_accuracy
        from framework.evaluators.classification.precision import compute_precision
        from framework.evaluators.classification.recall import compute_recall
        from framework.evaluators.classification.f1 import compute_f1
        return {
            "accuracy":  compute_accuracy,
            "precision": functools.partial(compute_precision, positive_label="SPAM"),
            "recall":    functools.partial(compute_recall, positive_label="SPAM"),
            "f1":        functools.partial(compute_f1, positive_label="SPAM"),
        }

    def get_model(self, model_config: dict):
        model_type = model_config["type"]
        params = self._config["models"].get(model_type, {})
        merged = {**model_config, **params}

        if model_type == "bert_tiny":
            from framework.models.spam.bert_tiny import BertTinySpamModel
            return BertTinySpamModel(merged)
        elif model_type == "roberta":
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