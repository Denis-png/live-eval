import random
import re
import sys
import time
from abc import ABC, abstractmethod
from typing import Callable

_REDUNDANCY_RE   = re.compile(r"(?im)^\s*Redundancy:\s*(trivial|valid)\b")
_CORRECTION_RE   = re.compile(r"(?im)^\s*Correction:\s*(correct|incorrect)\b")
_CORRUPTED_RE = re.compile(r"(?im)^\s*Corrupted:\s*(.+?)\s*$")

# Consulted only when structured parsing found no answer field — so a false
# positive merely relabels a skip, it never drops a good sample.
_REFUSAL_RE = re.compile(
    r"(?i)\b(?:I\s+can(?:no|')t|I\s+cannot|I(?:'m| am)\s+(?:not able|unable)|"
    r"I\s+won'?t|I\s+will\s+not|I(?:'m| am)\s+sorry|I\s+apologi[sz]e|as an AI)\b"
)

# Reasoning models (e.g. minimax-m3) wrap chain-of-thought in <think>…</think>.
_THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>")


class TruncatedResponse(RuntimeError):
    """The provider stopped because it hit max_tokens — the response is
    incomplete and must never be parsed.

    A truncated reasoning-model response frequently still contains the answer
    field names, with a half-written value after the last one (e.g.
    "Ground truth: I am"). Parsing that yields a sample whose gold reference is
    a fragment, which is worse than no sample: it silently corrupts the
    benchmark instead of being skipped. Every generation loop already catches
    per-sample exceptions and continues, so raising here turns truncation into
    a loud, counted skip in every loop at once.
    """


def _strip_reasoning(raw: str) -> str:
    """Drop closed <think>…</think> chain-of-thought blocks. An UNCLOSED <think>
    (the model opened a reasoning block, never closed it, and glued the answer on)
    is left intact — the tag parser recovers the answer from it."""
    return _THINK_BLOCK_RE.sub("", raw or "").strip()


def _is_reasoning_dump(text: str) -> bool:
    """True when `text` (already reasoning-stripped) still opens with an unclosed
    <think> — i.e. the model reasoned rather than refused."""
    return text.lstrip().lower().startswith("<think>")


def _looks_like_refusal(raw: str) -> bool:
    """Heuristic: does this response read as a safety refusal rather than an
    answer? Checked only after the structured fields failed to parse. Reasoning
    chatter is ignored: a model that emits a <think> block engaged with the task,
    so its incidental 'I can't…' musings are not treated as a refusal."""
    text = _strip_reasoning(raw)
    if _is_reasoning_dump(text):
        return False
    return bool(_REFUSAL_RE.search(text[:400]))


def _extract_field(text: str, field_pattern: str) -> str | None:
    """Extract the value of a `Field: value` line from reasoning-stripped `text`.

    Line-anchored normally; for an unclosed reasoning dump (the answer glued onto
    the chain-of-thought with no line break) the field is mid-line, so take the
    LAST occurrence's rest-of-line. `field_pattern` is a regex fragment, e.g.
    re.escape("Corrupted") or r"Ground\\s*truth"."""
    if _is_reasoning_dump(text):
        matches = list(re.finditer(rf"(?i){field_pattern}:[ \t]*(.+)", text))
        return matches[-1].group(1).strip() or None if matches else None
    m = re.search(rf"(?im)^\s*{field_pattern}:\s*(.+?)\s*$", text)
    return m.group(1).strip() if m else None


def _parse_tagged(raw: str, tag: str) -> str | None:
    """Pull the payload out of a single-field `<Tag>: <text>` response.

    Handles reasoning models (e.g. minimax-m3): closed <think>…</think> blocks are
    stripped and, for an unclosed reasoning dump, the last `Tag:` occurrence is
    taken as the answer (see _extract_field). A bare single-line response is
    accepted as a last resort (models often obey 'one line' but drop the prefix).
    Multiline output with no tag is rejected."""
    if not raw:
        return None
    text = _strip_reasoning(raw)
    field = _extract_field(text, re.escape(tag))
    if field is not None:
        return field
    if text and "\n" not in text:
        return text
    return None


def _parse_inverse(raw: str) -> str | None:
    """Pull the corrupted sentence out of an inverse-mode `Corrupted:` response.
    See _parse_tagged for the bare-single-line fallback behaviour."""
    return _parse_tagged(raw, "Corrupted")


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
    """Pull (error_type, corrupted, gold) out of the 3-step CoT response.
    Reasoning-model safe (see _extract_field): <think> blocks are stripped and a
    field glued onto the chain-of-thought is still recovered."""
    text = _strip_reasoning(raw or "")
    return (
        _extract_field(text, r"Error\s*type"),
        _extract_field(text, r"Generated"),
        _extract_field(text, r"Ground\s*truth"),
    )


def _accept_pair(corrupted: str | None, gold: str | None) -> str | None:
    """Return a rejection reason for a generated (corrupted, gold) pair, or None
    when the pair is acceptable. Shared by seeded and seedless forward loops."""
    if not corrupted or not gold:
        return "parse failed"
    if corrupted.strip() == gold.strip():
        return "identical corrupted/gold"
    if len(corrupted.split()) < 3:
        return f"too short: {corrupted!r}"
    return None


def _judgement_passes(raw: str) -> bool:
    """True iff the judge marks the sample valid AND the correction correct.
    Missing fields default to True (keep) to mirror Denis's parse_judge_output."""
    r = _REDUNDANCY_RE.search(raw)
    c = _CORRECTION_RE.search(raw)
    valid   = (r.group(1).lower() == "valid")   if r else True
    correct = (c.group(1).lower() == "correct") if c else True
    return valid and correct


class BaseGenerator(ABC):

    def generate_forward(
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
                reason = _accept_pair(corrupted, gold)
                if reason:
                    if reason == "parse failed":
                        if _looks_like_refusal(raw):
                            print(f"[{i}/{total}] gen {gen_dt:.1f}s — [SKIP] model refused: {raw[:60]!r}", flush=True)
                            refused += 1
                        else:
                            print(f"[{i}/{total}] gen {gen_dt:.1f}s — [SKIP] parse failed: {raw[:60]!r}", flush=True)
                            parse_failed += 1
                    else:
                        print(f"[{i}/{total}] gen {gen_dt:.1f}s — [SKIP] {reason}", flush=True)
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
            type_dist:          {category_key: prob}, empirical, injected by the pipeline.
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

    def generate_seedless_pairs(
        self,
        specs: list[str],
        prompt: str,
        error_descriptions: dict[str, str],
        type_dist: dict[str, float],
        count_dist: dict[int, float],
        judge_prompt: str | None = None,
        judge_call: Callable[[str], str] | None = None,
        rng=None,
        request_delay: float = 0.0,
    ) -> list[dict]:
        """Forward generation with no real seed: the content comes from a
        profile spec and the error type from the empirical distribution.

        Returns the same records as generate(): {"original", "corrupted",
        "error_type"}."""
        rng = rng or random.Random()
        synthetic: list[dict] = []
        judge_fn = judge_call or self.call_api
        judge_dropped = parse_failed = refused = 0

        total = len(specs)
        run_start = time.monotonic()
        print(f"Generating {total} samples (seedless forward) ...", flush=True)

        for i, spec in enumerate(specs, 1):
            keys = _sample_categories(type_dist, count_dist, rng)
            error_spec = "; ".join(error_descriptions.get(k, k) for k in keys)
            t0 = time.monotonic()
            try:
                raw = self.call_api(prompt.format(spec=spec, error_spec=error_spec))
                gen_dt = time.monotonic() - t0
                error_type, corrupted, gold = _parse_generation(raw)
                reason = _accept_pair(corrupted, gold)
                if reason:
                    if reason == "parse failed" and _looks_like_refusal(raw):
                        print(f"[{i}/{total}] gen {gen_dt:.1f}s — [SKIP] model refused: {raw[:60]!r}", flush=True)
                        refused += 1
                    else:
                        print(f"[{i}/{total}] gen {gen_dt:.1f}s — [SKIP] {reason}", flush=True)
                        parse_failed += reason == "parse failed"
                    continue

                judge_dt = 0.0
                if judge_prompt:
                    t1 = time.monotonic()
                    judge_raw = judge_fn(judge_prompt.format(sentence=corrupted, correction=gold))
                    judge_dt = time.monotonic() - t1
                    if not _judgement_passes(judge_raw):
                        print(f"[{i}/{total}] gen {gen_dt:.1f}s + judge {judge_dt:.1f}s — [JUDGE] dropped: {corrupted[:50]}", flush=True)
                        judge_dropped += 1
                        continue

                synthetic.append({
                    "original": gold,
                    "corrupted": corrupted,
                    "error_type": error_type or ", ".join(keys),
                })
                suffix = f" + judge {judge_dt:.1f}s" if judge_prompt else ""
                print(f"[{i}/{total}] gen {gen_dt:.1f}s{suffix} ✓ ({error_type or ', '.join(keys)})", flush=True)
                if request_delay > 0:
                    time.sleep(request_delay)
            except Exception as e:
                print(f"[{i}/{total}] failed after {time.monotonic() - t0:.1f}s: {e}", flush=True)

        print(f"Generation phase done in {time.monotonic() - run_start:.1f}s.")
        print(
            f"Generated {len(synthetic)} synthetic samples (seedless forward) "
            f"(judge dropped: {judge_dropped}, parse failed: {parse_failed}, "
            f"refused: {refused})."
        )
        return synthetic

    def generate_carriers(
        self,
        specs: list[str],
        carrier_prompt: str,
        tag: str,
        request_delay: float = 0.0,
    ) -> list[str]:
        """Synthesize seed texts from profile-derived content specs.

        Seedless *inverse* generation needs clean source texts but no real
        benchmark text: these carriers are drop-in replacements for real seeds
        in generate_inverse() / generate_class_conditional(). Fail-soft like the
        other loops — a refused or unparseable carrier is skipped, not raised.

        Args:
            specs:          already-rendered spec strings (see
                            profiling.spec_sampler.render_spec).
            carrier_prompt: template with a {spec} placeholder.
            tag:            expected response field, e.g. "Sentence".
            request_delay:  seconds to sleep after each successful request.
        """
        carriers: list[str] = []
        total = len(specs)
        refused = parse_failed = 0
        run_start = time.monotonic()
        if total:
            print(f"Synthesizing {total} carriers ...", flush=True)

        for i, spec in enumerate(specs, 1):
            prompt = carrier_prompt.format(spec=spec)
            t0 = time.monotonic()
            try:
                raw = self.call_api(prompt)
                dt = time.monotonic() - t0
                # Refusal check BEFORE _parse_tagged: its bare single-line
                # fallback would otherwise accept a one-line refusal as the
                # carrier text. An explicit tag field always wins.
                tag_re = re.compile(rf"(?im)^\s*{re.escape(tag)}:\s*(.+?)\s*$")
                if tag_re.search(raw) is None and _looks_like_refusal(raw):
                    print(f"[{i}/{total}] carrier {dt:.1f}s — [SKIP] model refused: {raw[:60]!r}", flush=True)
                    refused += 1
                    continue
                text = _parse_tagged(raw, tag)
                if not text:
                    print(f"[{i}/{total}] carrier {dt:.1f}s — [SKIP] parse failed: {raw[:60]!r}", flush=True)
                    parse_failed += 1
                    continue
                if len(text.split()) < 3:
                    print(f"[{i}/{total}] carrier {dt:.1f}s — [SKIP] too short: {text!r}", flush=True)
                    continue
                carriers.append(text)
                print(f"[{i}/{total}] carrier {dt:.1f}s ✓", flush=True)
                if request_delay > 0:
                    time.sleep(request_delay)
            except Exception as e:
                print(f"[{i}/{total}] carrier failed after {time.monotonic() - t0:.1f}s: {e}", flush=True)

        if total:
            print(
                f"Synthesized {len(carriers)} carriers in "
                f"{time.monotonic() - run_start:.1f}s "
                f"(parse failed: {parse_failed}, refused: {refused})."
            )
        return carriers

    def generate_class_conditional(
        self,
        *,
        class_prob: float,
        type_dist: dict[str, float],
        count_dist: dict[int, float],
        error_descriptions: dict[str, str],
        inject_prompt: str,
        negative_prompt: str,
        positive_label: str,
        negative_label: str,
        sample_size: int,
        seed_policy: str = "cross_class",
        real_seeds: list[dict] | None = None,
        seed_field: str | None = None,
        forward_prompts: dict[str, str] | None = None,
        seedless_prompts: dict[str, str] | None = None,
        specs_by_label: dict[str, list[str]] | None = None,
        label_field: str = "label",
        judge_prompt: str | None = None,
        judge_call: Callable[[str], str] | None = None,
        request_delay: float = 0.0,
        rng=None,
    ) -> list[dict]:
        """Symmetric class-conditional generation for classification tasks. One
        loop serves all seeding strategies, selected by `seed_policy`:

        - "cross_class" (default, today's behavior): every draw seeds from the
          same `real_seeds` pool regardless of which class was drawn. Positive →
          inject the sampled signal mix into the seed via `inject_prompt` (a
          `Corrupted:` line). Negative → paraphrase the seed via `negative_prompt`
          (a `Rewritten:` line).
        - "same_class": the seed is drawn from the subset of `real_seeds` whose
          `label_field` matches the drawn class, and each class is produced via
          `forward_prompts[label]` (a `Rewritten:` line). Raises RuntimeError if
          that subset is empty. The positive class rewrites its seed emphasising
          the sampled signal mix, so forward generation targets the empirical
          distribution instead of inheriting its seed's signals; the negative
          class is a plain same-class rewrite, recorded as "imitation".
        - "none": no real seed at all — content comes from a per-label spec pool
          (`specs_by_label`) rendered through `seedless_prompts[label]` (a
          `Message:` line). The record's "seed" field is "".

        Both classes are LLM-authored, so a classifier cannot separate them on
        authorship artifacts. Seeds/specs are cycled if sample_size exceeds their
        count.

        Returns records {"text", "label", "technique", "seed"}."""
        rng = rng or random.Random()
        synthetic = []
        # Per-class attempted/survived counts. Drops are class-asymmetric — a
        # model refuses phishing text far more than a benign paraphrase, and a
        # paraphrase trips "identical to seed" far more than an injection — so
        # the surviving balance drifts from class_prob systematically. Exposed
        # like last_response_diagnostic for the calibrator to read.
        attrition = {positive_label: {"attempted": 0, "survived": 0},
                     negative_label: {"attempted": 0, "survived": 0}}
        self.last_class_attrition = attrition
        judge_fn = judge_call or self.call_api
        if seed_policy in ("cross_class", "same_class") and not real_seeds:
            return synthetic
        if seed_policy == "same_class":
            missing = [lbl for lbl in (positive_label, negative_label)
                       if not (forward_prompts or {}).get(lbl)]
            if missing:
                raise RuntimeError(
                    f"seed_policy='same_class' needs forward_prompts entries for every "
                    f"class; missing or empty: {', '.join(missing)}."
                )
        if seed_policy == "none":
            # Both dicts are indexed by the drawn label inside the loop. Check
            # them here so a task that supplies only one class fails before the
            # first API call instead of KeyError-ing partway through a paid run.
            for name, mapping in (("seedless_prompts", seedless_prompts or {}),
                                  ("specs_by_label", specs_by_label or {})):
                missing = [lbl for lbl in (positive_label, negative_label)
                           if not mapping.get(lbl)]
                if missing:
                    raise RuntimeError(
                        f"seed_policy='none' needs {name} entries for every class; "
                        f"missing or empty: {', '.join(missing)}."
                    )

        run_start = time.monotonic()
        print(f"Generating {sample_size} samples (class-conditional) ...", flush=True)
        judge_dropped = parse_failed = refused = 0
        judge_skip_notice_printed = False

        def _missing_seed(i: int, source) -> bool:
            """True (and accounted for) iff `source` is falsy. Shared by the
            cross_class and same_class branches, which both cycle through a
            seed pool and must skip identically on a missing seed field."""
            if source:
                return False
            print(f"[{i}/{sample_size}] [SKIP] missing seed field {seed_field!r}", flush=True)
            nonlocal parse_failed
            parse_failed += 1
            return True

        for i in range(1, sample_size + 1):
            # cross_class's seed is purely index-based ((i-1) % len(real_seeds)) —
            # it does not depend on the class drawn below. Resolving it (and
            # skipping on a missing field) BEFORE any rng draw keeps this
            # policy's rng-consumption sequence identical to the pre-restructure
            # implementation: a skipped iteration must burn zero rng draws, or
            # every later draw in the run shifts. same_class can't do this — its
            # seed choice depends on the drawn label — but it's new code with no
            # equivalence requirement to preserve.
            if seed_policy == "cross_class":
                seed = real_seeds[(i - 1) % len(real_seeds)]
                source = seed.get(seed_field)
                if _missing_seed(i, source):
                    continue

            is_positive = rng.random() < class_prob
            label = positive_label if is_positive else negative_label
            attrition[label]["attempted"] += 1

            # Every policy targets the empirical signal mix for the positive
            # class — same_class emphasises the sampled signals in its rewrite
            # rather than inheriting whatever its seed happened to carry, so
            # forward generation is steerable the way inverse already is. The
            # negative class never carries signals under any policy.
            keys: list[str] = []
            error_spec = ""
            if is_positive:
                keys = _sample_categories(type_dist, count_dist, rng)
                error_spec = "; ".join(error_descriptions.get(k, k) for k in keys)
            if is_positive:
                technique = ", ".join(keys)
            else:
                technique = "imitation" if seed_policy == "same_class" else "paraphrase"

            if seed_policy == "cross_class":
                if is_positive:
                    prompt = inject_prompt.format(sentence=source, error_spec=error_spec)
                    tag = "Corrupted"
                else:
                    prompt = negative_prompt.format(sentence=source)
                    tag = "Rewritten"
            elif seed_policy == "same_class":
                pool = [row for row in real_seeds if row.get(label_field) == label]
                if not pool:
                    raise RuntimeError(
                        f"seed_policy='same_class' needs seeds labeled {label!r}; the pool has none."
                    )
                seed = pool[(i - 1) % len(pool)]
                source = seed.get(seed_field)
                if _missing_seed(i, source):
                    continue
                template = forward_prompts[label]
                prompt = (template.format(sentence=source, error_spec=error_spec)
                          if is_positive else template.format(sentence=source))
                tag = "Rewritten"
            elif seed_policy == "none":
                source = None
                spec = specs_by_label[label][(i - 1) % len(specs_by_label[label])]
                if is_positive:
                    prompt = seedless_prompts[label].format(spec=spec, error_spec=error_spec)
                else:
                    prompt = seedless_prompts[label].format(spec=spec)
                tag = "Message"
            else:
                raise ValueError(f"unknown seed_policy: {seed_policy!r}")

            t0 = time.monotonic()
            try:
                raw = self.call_api(prompt)
                gen_dt = time.monotonic() - t0
                tag_re = re.compile(rf"(?im)^\s*{re.escape(tag)}:\s*(.+?)\s*$")
                if tag_re.search(raw) is None and _looks_like_refusal(raw):
                    print(f"[{i}/{sample_size}] gen {gen_dt:.1f}s — [SKIP] model refused: {raw[:60]!r}", flush=True)
                    refused += 1
                    continue
                text = _parse_tagged(raw, tag)
                if not text:
                    print(f"[{i}/{sample_size}] gen {gen_dt:.1f}s — [SKIP] parse failed: {raw[:60]!r}", flush=True)
                    parse_failed += 1
                    continue
                if source and text.strip() == source.strip():
                    print(f"[{i}/{sample_size}] gen {gen_dt:.1f}s — [SKIP] identical to seed", flush=True)
                    continue
                if len(text.split()) < 3:
                    print(f"[{i}/{sample_size}] gen {gen_dt:.1f}s — [SKIP] too short: {text!r}", flush=True)
                    continue

                judge_dt = 0.0
                if judge_prompt and source:
                    t1 = time.monotonic()
                    judge_raw = judge_fn(judge_prompt.format(sentence=text, correction=source))
                    judge_dt = time.monotonic() - t1
                    if not _judgement_passes(judge_raw):
                        print(f"[{i}/{sample_size}] gen {gen_dt:.1f}s + judge {judge_dt:.1f}s — [JUDGE] dropped: {text[:50]}", flush=True)
                        judge_dropped += 1
                        continue
                elif judge_prompt and not judge_skip_notice_printed:
                    # seed_policy="none" has no seed to compare against — the judge
                    # prompts ask whether `text` matches/relates to a counterpart
                    # seed, so calling it with correction=None (rendered "None")
                    # would make every sample look wrong and get dropped. Skip the
                    # judge step entirely for this policy; note it once so a run's
                    # log doesn't look like judging silently never happened.
                    print(f"[{i}/{sample_size}] judge configured but skipped: no seed "
                          f"to compare against (seed_policy='none').", flush=True)
                    judge_skip_notice_printed = True

                synthetic.append({"text": text, "label": label, "technique": technique, "seed": source or ""})
                attrition[label]["survived"] += 1
                suffix = f" + judge {judge_dt:.1f}s" if judge_prompt else ""
                print(f"[{i}/{sample_size}] gen {gen_dt:.1f}s{suffix} ✓ ({label}: {technique})", flush=True)
                if request_delay > 0:
                    time.sleep(request_delay)
            except Exception as e:
                dt = time.monotonic() - t0
                print(f"[{i}/{sample_size}] failed after {dt:.1f}s: {e}", flush=True)

        total_dt = time.monotonic() - run_start
        print(f"Generation phase done in {total_dt:.1f}s.")
        print(
            f"Generated {len(synthetic)} synthetic samples (class-conditional) "
            f"(judge dropped: {judge_dropped}, parse failed: {parse_failed}, refused: {refused})."
        )
        return synthetic

    def generate_structured(
        self,
        build_prompt,
        parse,
        build_feedback=None,
        *,
        sample_size: int,
        max_parse_attempts: int = 3,
        max_feedback_rounds: int = 0,
        request_delay: float = 0.0,
    ) -> list[dict]:
        """Generate whole structured artifacts, one per sample, with an optional
        per-artifact feedback loop.

        Unlike corruption and class_conditional, where a sample is one sentence
        and fidelity only means something across a distribution, a structured
        artifact has its own measurable shape — so it can be compared against
        the reference and regenerated on its own.

        Stays ontology-agnostic like every other loop here: it never sees the
        task, the profile, or what the artifact means. Callers bind those.

            build_prompt(feedback) -> str       feedback is None on round 0
            parse(raw)             -> {"artifact": dict | None, "diagnostic": dict}
            build_feedback(artifact) -> {"feedback", "comparison",
                                         "synthetic_profile"}, or None to
                                        disable the loop entirely

        Each returned artifact carries its own `generation_feedback` metadata:
        the rounds it took, why attempts were rejected, and whether it stopped
        early inside tolerance.
        """
        feedback_enabled = build_feedback is not None
        rounds_budget = max(0, max_feedback_rounds) if feedback_enabled else 0
        max_parse_attempts = max(1, max_parse_attempts)
        synthetic: list[dict] = []

        for _ in range(sample_size):
            selected = None
            metadata: dict = {
                "feedback_enabled": feedback_enabled,
                "max_feedback_rounds": rounds_budget,
                "rounds": [],
                "early_stopped": False,
                "final_round_selected": None,
                "final_feedback_informed": False,
            }
            feedback = None
            for round_idx in range(rounds_budget + 1):
                parsed = None
                attempts = 0
                attempt_diagnostics: list[dict] = []
                while parsed is None and attempts < max_parse_attempts:
                    attempts += 1
                    raw = self.call_api(build_prompt(feedback))
                    parse_result = parse(raw)
                    parsed = parse_result["artifact"]
                    diagnostic = {"attempt": attempts, **parse_result["diagnostic"]}
                    provider_diagnostic = getattr(self, "last_response_diagnostic", None)
                    if parsed is None and provider_diagnostic:
                        diagnostic["provider_response"] = provider_diagnostic
                    attempt_diagnostics.append(diagnostic)
                    if parsed is None:
                        reason = diagnostic.get("rejection_reason", "unknown")
                        print(
                            "[SKIP] structured generation returned invalid "
                            f"JSON/artifact ({reason})."
                        )
                    if request_delay > 0:
                        time.sleep(request_delay)

                if parsed is None:
                    metadata["rounds"].append({
                        "round": round_idx,
                        "feedback_informed": feedback is not None,
                        "parse_attempts": attempts,
                        "attempts": attempt_diagnostics,
                        "valid": False,
                    })
                    if selected is not None:
                        metadata["failed_feedback_round_preserved_previous"] = True
                    break

                selected = parsed
                metadata.setdefault("attempts", []).extend(attempt_diagnostics)
                if not feedback_enabled:
                    metadata["final_round_selected"] = round_idx
                    break
                round_info = build_feedback(selected)
                feedback_result = round_info["feedback"]
                metadata["rounds"].append({
                    "round": round_idx,
                    "feedback_informed": feedback is not None,
                    "parse_attempts": attempts,
                    "attempts": attempt_diagnostics,
                    "valid": True,
                    "within_tolerance": feedback_result["within_tolerance"],
                    "feedback": feedback_result,
                    "comparison": round_info["comparison"],
                    "synthetic_profile": round_info["synthetic_profile"],
                })
                metadata["final_round_selected"] = round_idx
                metadata["final_feedback_informed"] = feedback is not None
                if feedback_result["within_tolerance"]:
                    metadata["early_stopped"] = True
                    break
                if round_idx >= rounds_budget:
                    break
                feedback = feedback_result

            if selected is not None:
                selected["generation_feedback"] = metadata
                synthetic.append(selected)

        if len(synthetic) < sample_size:
            print(
                f"[WARN] structured generation produced {len(synthetic)} valid "
                f"artifacts for {sample_size} requested.",
                file=sys.stderr,
            )
        return synthetic

    @abstractmethod
    def call_api(self, prompt: str) -> str:
        """Send a single prompt string, return the response string.

        Public on purpose: the pipeline reuses a generator's call_api as the
        LLM-as-judge callable (see pipeline._build_judge_call)."""
        pass
