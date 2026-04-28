import json
import os
from ..base_task import BaseTask

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "tasks", "gec.json")


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return json.load(f)


class GECTask(BaseTask):

    def __init__(self):
        self._config = _load_config()

    def get_error_types(self) -> list[str]:
        return self._config["error_types"]

    def get_prompt_instruction(self) -> str:
        return self._config["prompt"]

    def get_metrics(self) -> list[str]:
        return self._config["metrics"]

    def get_metric_fns(self) -> dict:
        from framework.metrics.gleu import compute_gleu
        from framework.metrics.gec.errant import compute_errant
        return {"gleu": compute_gleu, "errant": compute_errant}

    def get_evaluator(self, model_config: dict):
        model_type = model_config["type"]
        # Merge inference params from gec.json into model_config
        params = self._config["models"].get(model_type, {})
        merged = {**model_config, **params}

        if model_type == "t5":
            from framework.evaluators.gec.t5 import T5Evaluator
            return T5Evaluator(merged)
        elif model_type == "gec_v1":
            from framework.evaluators.gec.gec_v1 import GecV1Evaluator
            return GecV1Evaluator(merged)
        else:
            raise ValueError(
                f"Unsupported GEC model type: '{model_type}'. "
                f"Add it to configs/tasks/gec.json and tasks/gec/task.py."
            )

    def get_task_name(self) -> str:
        return "gec"
