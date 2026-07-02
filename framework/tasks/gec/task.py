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

    def get_inverse_prompt(self) -> str | None:
        return self._config.get("inverse_prompt")

    def get_inverse_judge_prompt(self) -> str | None:
        return self._config.get("inverse_judge_prompt")

    def get_error_descriptions(self) -> dict[str, str]:
        """Build human phrases for each supported ERRANT type by composing the
        operation prefix (M/U/R) with the category. E.g. 'R:VERB:TENSE' ->
        'use a wrong verb tense'."""
        ops  = self._config["errant_operations"]
        cats = self._config["errant_categories"]
        out = {}
        for t in self._config["errant_supported_types"]:
            op, _, cat = t.partition(":")  # 'R:VERB:TENSE' -> ('R', ':', 'VERB:TENSE')
            out[t] = f"{ops[op]} {cats[cat]}"
        return out

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

        if model_type in ("t5", "gec_v1", "coedit"):
            # All three are prefix-prompted seq2seq models; the prefix and
            # decoding params come from gec.json["models"][model_type].
            from framework.models.gec.seq2seq import Seq2SeqModel
            return Seq2SeqModel(merged)
        elif model_type == "claude":
            from framework.models.gec.claude import ClaudeModel
            return ClaudeModel(merged)
        else:
            raise ValueError(
                f"Unsupported GEC model type: '{model_type}'. "
                f"Add it to configs/tasks/gec.json and tasks/gec/task.py."
            )

    def parse_row(self, row: dict) -> dict | None:
        # Try common field name variants for incorrect and correct sentences.
        incorrect = row.get("input") or row.get("text") or row.get("incorrect")
        correct   = row.get("output") or row.get("correct") or row.get("target")
        if not incorrect or not correct:
            return None
        return {"incorrect": incorrect, "correct": correct}

    def profile_error_distribution(self, real_data, count_max=5, annotator=None):
        """Empirical inverse-mode distribution: ERRANT-annotate real
        incorrect->correct pairs, keeping only this task's supported edit types."""
        from framework.profiling.errant_distribution import profile_error_distribution
        supported = set(self.get_error_descriptions().keys())
        return profile_error_distribution(
            real_data, supported, count_max=count_max, annotator=annotator
        )

    def get_task_name(self) -> str:
        return "gec"
