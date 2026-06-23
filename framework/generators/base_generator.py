import random
import re
import time
from abc import ABC, abstractmethod
from typing import Callable

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
        judge_call: Callable[[str], str] | None = None,
    ) -> list[dict]:
        """
        Generate synthetic corrupted sentences with optional LLM-as-judge filter.
        Loop is shared across all providers — only call_api() differs.

        Args:
            real_samples:        list of {"incorrect": ..., "correct": ...}
            error_types:         fallback labels used only when the model's CoT
                                 output omits an "Error type:" line. Also offered
                                 to templates via the optional {error_type} field.
            prompt_instruction:  CoT template with a {sentence} placeholder
            sample_size:         number of synthetic samples to produce
            judge_prompt:        optional LLM-as-judge template with {sentence}, {correction}
            judge_call:          callable(prompt) → str for the judge. If None and
                                 judge_prompt is set, falls back to self.call_api.

        Returns:
            list of {"original": <LLM gold>, "corrupted": ..., "error_type": ...}
        """
        synthetic = []
        samples = real_samples[:sample_size]
        judge_dropped = 0
        parse_failed = 0
        judge_fn = judge_call or self.call_api

        total = len(samples)
        run_start = time.monotonic()
        print(f"Generating {total} samples ...", flush=True)

        for i, item in enumerate(samples, 1):
            fallback_type = random.choice(error_types) if error_types else None
            prompt = prompt_instruction.format(
                sentence=item["incorrect"], error_type=fallback_type
            )
            t0 = time.monotonic()
            try:
                raw = self.call_api(prompt)
                gen_dt = time.monotonic() - t0
                error_type, corrupted, gold = _parse_generation(raw)
                if not corrupted or not gold:
                    print(f"[{i}/{total}] gen {gen_dt:.1f}s — [SKIP] parse failed: {raw[:60]!r}", flush=True)
                    parse_failed += 1
                    continue
                if corrupted.strip() == gold.strip():
                    print(f"[{i}/{total}] gen {gen_dt:.1f}s — [SKIP] identical corrupted/gold", flush=True)
                    continue
                if len(corrupted.split()) < 3:
                    print(f"[{i}/{total}] gen {gen_dt:.1f}s — [SKIP] too short: {corrupted!r}", flush=True)
                    continue

                judge_dt = 0.0
                if judge_prompt:
                    t1 = time.monotonic()
                    judge_raw = judge_fn(
                        judge_prompt.format(sentence=corrupted, correction=gold)
                    )
                    judge_dt = time.monotonic() - t1
                    if not _judgement_passes(judge_raw):
                        print(f"[{i}/{total}] gen {gen_dt:.1f}s + judge {judge_dt:.1f}s — [JUDGE] dropped: {corrupted[:50]}", flush=True)
                        judge_dropped += 1
                        continue

                synthetic.append({
                    "original":   gold,
                    "corrupted":  corrupted,
                    "error_type": error_type or fallback_type,
                })
                suffix = f" + judge {judge_dt:.1f}s" if judge_prompt else ""
                print(f"[{i}/{total}] gen {gen_dt:.1f}s{suffix} ✓ ({error_type or fallback_type})", flush=True)
            except Exception as e:
                dt = time.monotonic() - t0
                print(f"[{i}/{total}] failed after {dt:.1f}s: {e}", flush=True)

        total_dt = time.monotonic() - run_start
        print(f"Generation phase done in {total_dt:.1f}s.")

        print(
            f"Generated {len(synthetic)} synthetic samples "
            f"(judge dropped: {judge_dropped}, parse failed: {parse_failed})."
        )
        return synthetic

    @abstractmethod
    def call_api(self, prompt: str) -> str:
        """Send a single prompt string, return the response string.

        Public on purpose: the pipeline reuses a generator's call_api as the
        LLM-as-judge callable (see pipeline._build_judge_call)."""
        pass
