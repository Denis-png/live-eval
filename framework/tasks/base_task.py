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

    @abstractmethod
    def get_task_name(self) -> str:
        """
        Short lowercase task identifier, matching configs/tasks/<name>.json and
        the data/generated/<name>/ archive dir (e.g. "gec"). Must be overridden
        — do NOT derive from the class name, which would drift from the config.
        """
        pass
