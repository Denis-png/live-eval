# ============================================================
# Base Task — Abstract Template
# ============================================================
# This is the base class for all tasks in the framework.
# Every new task (GEC, Hate Speech, Spam, etc.) must inherit
# from this class and implement the required methods.
#
# If you want to add a new task,
# you MUST implement these methods. This ensures all tasks
# work the same way inside the pipeline.
# ============================================================

from abc import ABC, abstractmethod


class BaseTask(ABC):

    @abstractmethod
    def get_error_types(self) -> list[str]:
        """
        Return a list of error types for this task.
        The generator LLM will use these to corrupt sentences.

        Example for GEC:
            ["subject-verb disagreement", "wrong tense", "missing article"]

        Example for Hate Speech:
            ["add racial slur", "add gender-based insult"]
        """
        pass

    @abstractmethod
    def get_prompt_instruction(self) -> str:
        """
        Return the instruction for the generator LLM.
        This tells the LLM what kind of corruption to introduce.

        Example for GEC:
            "Introduce exactly one grammatical error of type: {error_type}"

        Example for Hate Speech:
            "Rewrite this sentence to contain hate speech of type: {error_type}"
        """
        pass

    @abstractmethod
    def get_metrics(self) -> list[str]:
        """
        Return the list of metrics to use for this task.

        Example for GEC:     ["gleu", "errant"]
        Example for Hate Speech: ["accuracy", "f1"]
        """
        pass

    def get_task_name(self) -> str:
        """Return the name of this task."""
        return self.__class__.__name__