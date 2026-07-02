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
        Must contain {error_type} and {sentence} placeholders.
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

    def get_task_name(self) -> str:
        return self.__class__.__name__
    
    def get_label(self, result: dict) -> str | None:
        """
        Return the ground-truth label for a result dict.
        Override in classification tasks to supply the correct label.
        Return None for tasks that don't require a label (e.g. GEC).
        """
        return None

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
