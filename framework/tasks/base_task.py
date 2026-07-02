from abc import ABC, abstractmethod


class BaseTask(ABC):

    @abstractmethod
    def get_error_types(self) -> list[str]:
        """
        List of corruption types the generator LLM will introduce.

        GEC example:   ["subject-verb disagreement", "wrong tense"]
        Spam example:  ["add urgency phrase", "add fake offer"]
        """
        pass

    @abstractmethod
    def get_prompt_instruction(self) -> str:
        """
        Prompt template for the generator LLM.
        Must contain a {sentence} placeholder; may optionally use {error_type}
        (the generator always passes both to str.format).
        """
        pass

    @abstractmethod
    def get_evaluators(self) -> list[str]:
        """
        Evaluator names to compute for this task.
        Must match keys returned by get_evaluator_fns().
        """
        pass

    @abstractmethod
    def get_evaluator_fns(self) -> dict:
        """
        Map of evaluator name → callable(results: list[dict]) → score.
        results dicts contain: original, corrupted, prediction.
        """
        pass

    @abstractmethod
    def get_model(self, model_config: dict):
        """
        Return a BaseModel instance for the given model config.
        The task owns the mapping from model type to model class.
        """
        pass

    def get_judge_prompt(self) -> str | None:
        """
        Optional LLM-as-judge prompt for filtering bad generations.
        Must contain {sentence} and {correction} placeholders.
        Return None to disable the judge step.
        """
        return None

    def get_inverse_prompt(self) -> str | None:
        """Inverse-mode prompt template. Must contain {sentence} (the clean source
        text) and {error_spec} (human description of the errors to inject).
        Return None if the task does not support inverse generation."""
        return None

    def get_inverse_judge_prompt(self) -> str | None:
        """Optional inverse-mode judge template with {sentence} and {correction}
        placeholders. Return None to disable judging in inverse mode."""
        return None

    def get_error_descriptions(self) -> dict[str, str]:
        """Map of corruption category key -> short human phrase, used to render
        {error_spec} for the inverse prompt. The keys also define the category
        vocabulary the (placeholder) error distribution samples over.
        Return {} if the task does not support inverse generation."""
        return {}

    def profile_error_distribution(self, real_data: list[dict],
                                   count_max: int = 5, config: dict | None = None) -> dict | None:
        """Empirical inverse-mode error distribution derived from real_data (and
        optionally the run `config`, e.g. to load an auxiliary subset), keyed on
        get_error_descriptions() vocabulary. Return None to fall back to the
        placeholder distribution (default: no empirical profiler)."""
        return None

    @abstractmethod
    def get_task_name(self) -> str:
        """
        Short lowercase task identifier, matching configs/tasks/<name>.json and
        the data/generated/<name>/ archive dir (e.g. "gec"). Must be overridden
        — do NOT derive from the class name, which would drift from the config.
        """
        pass

    def get_label(self, result: dict) -> str | None:
        """
        Return the ground-truth label for a result dict.
        Override in classification tasks to supply the correct label.
        Return None for tasks that don't require a label (e.g. GEC).
        """
        return None

    def get_eval_samples(self, synthetic: list[dict]) -> list[dict]:
        """Expand generated items into rows to classify/score. Each row carries a
        "text" field (the model input). Default: one row per item scoring the
        corrupted text, with the ground-truth "label" from get_label when present.
        Classification tasks may override to add negatives (see SpamTask)."""
        out = []
        for item in synthetic:
            sample = {**item, "text": item["corrupted"]}
            label = self.get_label(sample)
            if label is not None:
                sample["label"] = label
            out.append(sample)
        return out

    @abstractmethod
    def parse_row(self, row: dict) -> dict | None:
        """
        Parse a single raw dataset row into a sample dict.
        Return None to skip the row (e.g. wrong label, missing fields).
        The pipeline collects non-None results up to sample_size.

        GEC example:  maps to {"incorrect": ..., "correct": ...}
        Spam example: filters HAM rows, maps to {"incorrect": ...}
        """
        pass
