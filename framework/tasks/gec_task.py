# ============================================================
# GEC Task — Grammatical Error Correction
# ============================================================
# This is the concrete implementation of BaseTask for GEC.
# It defines the specific error types, prompt instruction,
# and metrics for grammar error correction.
#
# When someone wants to add a new task (e.g. Hate Speech),
# they create a new file just like this one.
# ============================================================

from .base_task import BaseTask


class GECTask(BaseTask):

    def get_error_types(self) -> list[str]:
        """
        List of grammatical error types the LLM will introduce
        into correct sentences to create synthetic test data.
        """
        return [
            "subject-verb disagreement",
            "wrong tense",
            "missing article",
            "spelling mistake",
            "wrong preposition",
        ]

    def get_prompt_instruction(self) -> str:
        """
        Instruction for the generator LLM.
        The LLM receives a correct sentence and must introduce
        exactly one grammatical error of the given type.
        """
        return (
            "You are a grammar error generator for NLP research.\n"
            "Given a correct English sentence, introduce exactly one "
            "grammatical error of type: {error_type}\n"
            "Return ONLY the corrupted sentence. No explanation.\n\n"
            "Correct sentence: {sentence}\n"
            "Corrupted sentence:"
        )

    def get_metrics(self) -> list[str]:
        """
        Metrics used to evaluate GEC models.
        - gleu: measures fluency of correction
        - errant: measures error detection accuracy
        """
        return ["gleu", "errant"]