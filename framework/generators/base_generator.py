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
_CORRUPTED_RE = re.compile(r"(?im)^\s*Corrupted:\s*(.+?)\s*$")

# Consulted only when structured parsing found no answer field — so a false
# positive merely relabels a skip, it never drops a good sample.
_REFUSAL_RE = re.compile(
    r"(?i)\b(?:I\s+can(?:no|')t|I\s+cannot|I(?:'m| am)\s+(?:not able|unable)|"
    r"I\s+won'?t|I\s+will\s+not|I(?:'m| am)\s+sorry|I\s+apologi[sz]e|as an AI)\b"
)


def _looks_like_refusal(raw: str) -> bool:
    """Heuristic: does this response read as a safety refusal rather than an
    answer? Checked only after the structured fields failed to parse."""
    return bool(_REFUSAL_RE.search(raw[:400]))


def _parse_inverse(raw: str) -> str | None:
    """Pull the corrupted sentence out of an inverse-mode `Corrupted:` response.

    Falls back to accepting a bare single-line response: real models often obey
    the prompt's "respond with exactly one line" but drop the "Corrupted:"
    prefix. Multiline output without the field (reasoning dumps, prose) is
    still rejected."""
    m = _CORRUPTED_RE.search(raw)
    if m:
        return m.group(1).strip()
    text = (raw or "").strip()
    if text and "\n" not in text:
        return text
    return None


def _sample_categories(
    type_dist: dict[str, float],
    count_dist: dict[int, float],
    rng,
) -> list[str]:
    """Sample a count n ~ count_dist, then n category keys ~ type_dist.

    Sampling is without replacement when n <= len(type_dist) (distinct
    categories), and with replacement otherwise (n exceeds available keys).
    `rng` is an injected random.Random for deterministic tests."""
    counts = list(count_dist.keys())
    n = rng.choices(counts, weights=[count_dist[c] for c in counts], k=1)[0]

    keys = list(type_dist.keys())
    weights = [type_dist[k] for k in keys]
    if n > len(keys):
        return rng.choices(keys, weights=weights, k=n)

    chosen: list[str] = []
    pool, pool_w = keys[:], weights[:]
    for _ in range(n):
        idx = rng.choices(range(len(pool)), weights=pool_w, k=1)[0]
        chosen.append(pool.pop(idx))
        pool_w.pop(idx)
    return chosen


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
        request_delay: float = 0.0,
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
            request_delay:       seconds to sleep after each successful request to
                                 respect provider TPM rate limits (default: 0).

        Returns:
            list of {"original": <LLM gold>, "corrupted": ..., "error_type": ...}
        """
        synthetic = []
        samples = real_samples[:sample_size]
        judge_dropped = 0
        parse_failed = 0
        refused = 0
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
                    if _looks_like_refusal(raw):
                        print(f"[{i}/{total}] gen {gen_dt:.1f}s — [SKIP] model refused: {raw[:60]!r}", flush=True)
                        refused += 1
                    else:
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

                # Throttle requests to stay within provider rate limits (e.g. Groq TPM cap).
                if request_delay > 0:
                    time.sleep(request_delay)
            except Exception as e:
                dt = time.monotonic() - t0
                print(f"[{i}/{total}] failed after {dt:.1f}s: {e}", flush=True)

        total_dt = time.monotonic() - run_start
        print(f"Generation phase done in {total_dt:.1f}s.")

        print(
            f"Generated {len(synthetic)} synthetic samples "
            f"(judge dropped: {judge_dropped}, parse failed: {parse_failed}, "
            f"refused: {refused})."
        )
        return synthetic

    def generate_inverse(
        self,
        real_samples: list[dict],
        inverse_prompt: str,
        error_descriptions: dict[str, str],
        type_dist: dict[str, float],
        count_dist: dict[int, float],
        sample_size: int,
        source_field: str = "correct",
        judge_prompt: str | None = None,
        judge_call: Callable[[str], str] | None = None,
        rng=None,
        request_delay: float = 0.0,
    ) -> list[dict]:
        """Inverse generation: corrupt a known-clean source sentence according to
        an injected error distribution. Task-agnostic — `type_dist` keys are opaque
        strings rendered to human text via `error_descriptions`.

        Args:
            real_samples:       list of dicts; `source_field` holds the clean text.
            inverse_prompt:     template with {sentence} (clean text) and {error_spec}.
            error_descriptions: category_key -> human phrase, for building {error_spec}.
            type_dist:          {category_key: prob}, injected (placeholder for now).
            count_dist:         {n: prob}, errors-per-sentence, injected.
            sample_size:        number of source sentences to process.
            source_field:       which dataset field is the clean source (default "correct").
            judge_prompt:       optional inverse judge template with {sentence}, {correction}.
            judge_call:         callable(prompt) -> str for the judge.
            rng:                injected random.Random for deterministic sampling.
            request_delay:      seconds to sleep after each successful request to
                                respect provider TPM rate limits (default: 0).

        Returns:
            list of {"original": <clean source>, "corrupted": ..., "error_type": ...}
        """
        rng = rng or random.Random()
        synthetic = []
        samples = real_samples[:sample_size]
        judge_dropped = 0
        parse_failed = 0
        refused = 0
        judge_fn = judge_call or self.call_api

        total = len(samples)
        run_start = time.monotonic()
        print(f"Generating {total} samples (inverse) ...", flush=True)

        for i, item in enumerate(samples, 1):
            gold = item.get(source_field)
            if not gold:
                print(f"[{i}/{total}] [SKIP] missing source field {source_field!r}", flush=True)
                parse_failed += 1
                continue

            keys = _sample_categories(type_dist, count_dist, rng)
            error_spec = "; ".join(error_descriptions.get(k, k) for k in keys)
            prompt = inverse_prompt.format(sentence=gold, error_spec=error_spec)

            t0 = time.monotonic()
            try:
                raw = self.call_api(prompt)
                gen_dt = time.monotonic() - t0
                # Refusal check BEFORE _parse_inverse: its bare single-line
                # fallback would otherwise accept a one-line refusal as the
                # corrupted text. An explicit Corrupted: field always wins.
                if _CORRUPTED_RE.search(raw) is None and _looks_like_refusal(raw):
                    print(f"[{i}/{total}] gen {gen_dt:.1f}s — [SKIP] model refused: {raw[:60]!r}", flush=True)
                    refused += 1
                    continue
                corrupted = _parse_inverse(raw)
                if not corrupted:
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
                    "error_type": ", ".join(keys),
                })
                suffix = f" + judge {judge_dt:.1f}s" if judge_prompt else ""
                print(f"[{i}/{total}] gen {gen_dt:.1f}s{suffix} ✓ ({', '.join(keys)})", flush=True)

                # Throttle requests to stay within provider rate limits (e.g. Groq TPM cap).
                if request_delay > 0:
                    time.sleep(request_delay)
            except Exception as e:
                dt = time.monotonic() - t0
                print(f"[{i}/{total}] failed after {dt:.1f}s: {e}", flush=True)

        total_dt = time.monotonic() - run_start
        print(f"Generation phase done in {total_dt:.1f}s.")
        print(
            f"Generated {len(synthetic)} synthetic samples (inverse) "
            f"(judge dropped: {judge_dropped}, parse failed: {parse_failed}, "
            f"refused: {refused})."
        )
        return synthetic

    @abstractmethod
    def call_api(self, prompt: str) -> str:
        """Send a single prompt string, return the response string.

        Public on purpose: the pipeline reuses a generator's call_api as the
        LLM-as-judge callable (see pipeline._build_judge_call)."""
        pass
