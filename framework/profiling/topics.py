"""LLM topic profiling: two-pass open labeling (label -> consolidate).

Provider-agnostic: all LLM access goes through an injected
call_api(prompt) -> str callable — the same seam the pipeline uses for
judge_call. Fractions are always computed in code from label counts;
LLM arithmetic is never trusted.
"""

from __future__ import annotations

import re

LABEL_PROMPT = """You are profiling a text dataset. For each numbered text below, give a short topic label (1-3 lowercase words) describing what the text is about.

Reply with exactly one line per text, formatted as:
<number>: <topic label>

No other output.

Texts:
{items}"""

_LABEL_LINE_RE = re.compile(r"^\s*(\d+)\s*[:.)\-]\s*(.+?)\s*$")
_TEXT_TRUNCATE = 300


def _parse_label_lines(raw: str, expected: int) -> dict[int, str]:
    """Parse `N: label` lines into {0-based index: normalized label}.

    Unparseable lines and out-of-range indices are silently skipped —
    a partially usable batch keeps its usable lines.
    """
    labels: dict[int, str] = {}
    for line in (raw or "").splitlines():
        match = _LABEL_LINE_RE.match(line)
        if not match:
            continue
        index = int(match.group(1))
        if 1 <= index <= expected:
            labels[index - 1] = match.group(2).strip().lower()
    return labels


def _label_texts(texts: list[str], call_api, batch_size: int) -> list[str | None]:
    """Pass 1: label every text via batched prompts. Returns one entry per
    input text — a normalized label, or None where the batch/line failed."""
    labels: list[str | None] = [None] * len(texts)
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        items = "\n".join(
            f"{i + 1}. {text[:_TEXT_TRUNCATE]}" for i, text in enumerate(batch)
        )
        try:
            raw = call_api(LABEL_PROMPT.format(items=items))
        except Exception as exc:  # fail-soft, mirrors generation-loop [SKIP] policy
            print(f"[topics] label batch failed: {exc}", flush=True)
            continue
        for index, label in _parse_label_lines(raw, expected=len(batch)).items():
            labels[start + index] = label
    return labels
