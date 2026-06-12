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

    def get_evaluators(self) -> list[str]:
        return self._config["evaluators"]

    def get_judge_prompt(self) -> str | None:
        return self._config.get("judge_prompt")

    def get_evaluator_fns(self) -> dict:
        from framework.evaluators.gleu import compute_gleu
        from framework.evaluators.gec.errant import compute_errant
        from framework.evaluators.gec.errant_dist import compute_errant_dist
        from framework.evaluators.gec.cola import compute_cola
        from framework.evaluators.gec.correction_extent import compute_correction_extent
        from framework.evaluators.gec.n_edits import compute_n_edits
        return {
            "gleu":              compute_gleu,
            "errant":            compute_errant,
            "errant_dist":       compute_errant_dist,
            "cola":              compute_cola,
            "correction_extent": compute_correction_extent,
            "n_edits":           compute_n_edits,
        }

    def get_model(self, model_config: dict):
        model_type = model_config["type"]
        # Merge inference params from gec.json into model_config
        params = self._config["models"].get(model_type, {})
        merged = {**model_config, **params}

        if model_type == "t5":
            from framework.models.gec.t5 import T5Model
            return T5Model(merged)
        elif model_type == "gec_v1":
            from framework.models.gec.gec_v1 import GecV1Model
            return GecV1Model(merged)
        elif model_type == "coedit":
            from framework.models.gec.coedit import CoEditModel
            return CoEditModel(merged)
        elif model_type == "claude":
            from framework.models.gec.claude import ClaudeModel
            return ClaudeModel(merged)
        else:
            raise ValueError(
                f"Unsupported GEC model type: '{model_type}'. "
                f"Add it to configs/tasks/gec.json and tasks/gec/task.py."
            )

    def get_task_name(self) -> str:
        return "gec"
