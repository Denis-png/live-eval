import random
from abc import ABC, abstractmethod


class BaseGenerator(ABC):

    def generate(
        self,
        real_samples: list[dict],
        error_types: list[str],
        prompt_instruction: str,
        sample_size: int,
    ) -> list[dict]:
        """
        Generate synthetic corrupted sentences.
        Loop is shared across all providers — only _call_api() differs.

        Args:
            real_samples:        list of {"incorrect": ..., "correct": ...}
            error_types:         corruption types to introduce
            prompt_instruction:  template with {sentence} and {error_type} placeholders
            sample_size:         number of synthetic samples to produce

        Returns:
            list of {"original": ..., "corrupted": ..., "error_type": ...}
        """
        synthetic = []
        samples = real_samples[:sample_size]

        for item in samples:
            error_type = random.choice(error_types)
            prompt = prompt_instruction.format(
                sentence=item["correct"], error_type=error_type
            )
            try:
                corrupted = self._call_api(prompt).strip()
                if corrupted == item["correct"].strip():
                    print("[SKIP] LLM returned unchanged sentence.")
                    continue
                synthetic.append({
                    "original":   item["correct"],
                    "corrupted":  corrupted,
                    "error_type": error_type,
                })
            except Exception as e:
                print(f"[WARN] Generation failed: {e}")

        print(f"Generated {len(synthetic)} synthetic samples.")
        return synthetic

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Send a single prompt string, return the response string."""
        pass
