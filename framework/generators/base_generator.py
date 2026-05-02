import random
import re
from abc import ABC, abstractmethod

_ERROR_TYPE_RE   = re.compile(r"(?im)^\s*Error\s*type:\s*(.+?)\s*$")
_GENERATED_RE    = re.compile(r"(?im)^\s*Generated:\s*(.+?)\s*$")
_GROUND_TRUTH_RE = re.compile(r"(?im)^\s*Ground\s*truth:\s*(.+?)\s*$")
_REDUNDANCY_RE   = re.compile(r"(?im)^\s*Redundancy:\s*(trivial|valid)\b")
_CORRECTION_RE   = re.compile(r"(?im)^\s*Correction:\s*(correct|incorrect)\b")


def _parse_generation(raw: str) -> tuple[str | None, str | None, str | None]:
    """Pull (error_type, corrupted, gold) out of the 3-step CoT response."""
    et = _ERROR_TYPE_RE.search(raw)
    g  = _GENERATED_RE.search(raw)
    gt = _GROUND_TRUTH_RE.search(raw)
    return (
        et.group(1).strip() if et else None,
        g.group(1).strip()  if g  else None,
        gt.group(1).strip() if gt else None,
    )


def _judgement_passes(raw: str) -> bool:
    """True iff the judge marks the sample valid AND the correction correct.
    Missing fields default to True (keep) to mirror Denis's parse_judge_output."""
    r = _REDUNDANCY_RE.search(raw)
    c = _CORRECTION_RE.search(raw)
    valid   = (r.group(1).lower() == "valid")   if r else True
    correct = (c.group(1).lower() == "correct") if c else True
    return valid and correct


class BaseGenerator(ABC):

    def generate(
        self,
        real_samples: list[dict],
        error_types: list[str],
        prompt_instruction: str,
        sample_size: int,
        judge_prompt: str | None = None,
    ) -> list[dict]:
        """
        Generate synthetic corrupted sentences with optional LLM-as-judge filter.
        Loop is shared across all providers — only _call_api() differs.

        Args:
            real_samples:        list of {"incorrect": ..., "correct": ...}
            error_types:         legacy hint list; the model identifies the type itself
            prompt_instruction:  CoT template with a {sentence} placeholder
            sample_size:         number of synthetic samples to produce
            judge_prompt:        optional LLM-as-judge template with {sentence}, {correction}

        Returns:
            list of {"original": <LLM gold>, "corrupted": ..., "error_type": ...}
        """
        synthetic = []
        samples = real_samples[:sample_size]
        judge_dropped = 0
        parse_failed = 0

        for item in samples:
            fallback_type = random.choice(error_types) if error_types else None
            prompt = prompt_instruction.format(
                sentence=item["incorrect"], error_type=fallback_type
            )
            try:
                raw = self._call_api(prompt)
                error_type, corrupted, gold = _parse_generation(raw)
                if not corrupted or not gold:
                    print(f"[SKIP] Failed to parse generation: {raw[:80]!r}")
                    parse_failed += 1
                    continue
                if corrupted.strip() == gold.strip():
                    print("[SKIP] LLM produced identical corrupted/gold pair.")
                    continue

                if judge_prompt:
                    judge_raw = self._call_api(
                        judge_prompt.format(sentence=corrupted, correction=gold)
                    )
                    if not _judgement_passes(judge_raw):
                        print(f"[JUDGE] dropped: {corrupted[:60]}")
                        judge_dropped += 1
                        continue

                synthetic.append({
                    "original":   gold,
                    "corrupted":  corrupted,
                    "error_type": error_type or fallback_type,
                })
            except Exception as e:
                print(f"[WARN] Generation failed: {e}")

        print(
            f"Generated {len(synthetic)} synthetic samples "
            f"(judge dropped: {judge_dropped}, parse failed: {parse_failed})."
        )
        return synthetic

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Send a single prompt string, return the response string."""
        pass
