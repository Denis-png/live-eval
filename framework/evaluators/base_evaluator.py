# ============================================================
# Base Evaluator — Abstract Template
# ============================================================
# This is the base class for all evaluators in the framework.
# An evaluator takes synthetic data, runs a task model on it,
# and returns the model's predictions.
#
# Currently implemented:
#   - GECEvaluator (runs T5 and Gramformer)
#
# Future additions:
#   - HateSpeechEvaluator
#   - SpamEvaluator
# ============================================================

from abc import ABC, abstractmethod


class BaseEvaluator(ABC):

    @abstractmethod
    def load_model(self, model_config: dict):
        """
        Load the task model.

        Args:
            model_config: {"name": "...", "type": "..."}
        """
        pass

    @abstractmethod
    def predict(self, sentences: list[str]) -> list[str]:
        """
        Run the model on a list of sentences.

        Args:
            sentences: list of corrupted (incorrect) sentences

        Returns:
            list of model predictions (corrected sentences)
        """
        pass