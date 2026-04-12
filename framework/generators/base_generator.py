# ============================================================
# Base Generator — Abstract Template
# ============================================================
# This is the base class for all generators in the framework.
# A generator takes real sentences and produces synthetic
# corrupted sentences using an LLM or other method.
#
# Currently implemented:
#   - LLMGenerator (uses Groq/OpenAI API)
#
# Future additions could be:
#   - RuleBasedGenerator (no LLM, uses rules to corrupt text)
#   - LocalModelGenerator (uses a local model via vLLM)
# ============================================================

from abc import ABC, abstractmethod


class BaseGenerator(ABC):

    @abstractmethod
    def generate(
        self,
        real_samples: list[dict],
        error_types: list[str],
        prompt_instruction: str,
        sample_size: int,
    ) -> list[dict]:
        """
        Generate synthetic corrupted sentences.

        Args:
            real_samples:       list of {"incorrect": ..., "correct": ...}
            error_types:        list of error types to introduce
            prompt_instruction: instruction template for the LLM
            sample_size:        how many synthetic samples to generate

        Returns:
            list of {
                "original":   correct sentence (ground truth),
                "corrupted":  sentence with error (model input),
                "error_type": type of error introduced
            }
        """
        pass