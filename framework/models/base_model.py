# ============================================================
# Base Model — Abstract Template
# ============================================================
# This is the base class for all task models in the framework.
# A task model takes synthetic (corrupted) sentences and returns
# its predictions, which the evaluators then score.
#
# Currently implemented:
#   - GEC models (T5, GEC v1, CoEdit, Claude)
#
# Future additions:
#   - Hate speech models
#   - Spam models
# ============================================================

from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self, model_config: dict):
        self.load_model(model_config)

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