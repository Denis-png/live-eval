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
    def get_metrics(self) -> list[str]:
        """
        Metric names to compute for this task.
        Must match keys returned by get_metric_fns().
        """
        pass

    @abstractmethod
    def get_metric_fns(self) -> dict:
        """
        Map of metric name → callable(results: list[dict]) → score.
        results dicts contain: original, corrupted, prediction.
        """
        pass

    @abstractmethod
    def get_evaluator(self, model_config: dict):
        """
        Return a BaseEvaluator instance for the given model config.
        The task owns the mapping from model type to evaluator class.
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
