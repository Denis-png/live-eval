"""LLM topic profiling: two-pass open labeling (label -> consolidate).

Provider-agnostic: all LLM access goes through an injected
call_api(prompt) -> str callable — the same seam the pipeline uses for
judge_call. Fractions are always computed in code from label counts;
LLM arithmetic is never trusted.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter

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


CONSOLIDATE_PROMPT = """You are consolidating raw topic labels from a text dataset into canonical topics.

Below are raw labels with their counts. Merge them into {min_topics}-{max_topics} canonical topics. Reply with ONLY a JSON object, no code fences, shaped as:
{{"<canonical topic name>": {{"description": "<one sentence>", "members": ["<raw label>", ...]}}}}

Every raw label should appear in exactly one canonical topic's members.

Raw labels:
{labels}"""

_EXAMPLE_TRUNCATE = 200
_EXAMPLES_PER_TOPIC = 3


def _parse_consolidation(raw: str) -> dict | None:
    """Parse the consolidation JSON. Strips code fences and leading/trailing
    prose. Returns {canonical: {"description": str, "members": [str]}} or None."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not data:
        return None
    clean: dict = {}
    for name, block in data.items():
        if not isinstance(block, dict) or not isinstance(block.get("members"), list):
            return None
        clean[str(name)] = {
            "description": str(block.get("description", "")),
            "members": [str(m).strip().lower() for m in block["members"]],
        }
    return clean


def profile_topics(
    texts: list[str],
    call_api,
    sample_size: int = 200,
    batch_size: int = 20,
    rng=None,
) -> dict:
    """Two-pass topic profile of `texts`. See module docstring.

    Consolidation is retried once on a parse failure; if both attempts fail
    the normalized raw-label distribution is returned as the topics (with a
    note). Raw labels no canonical topic claimed become their own topic so
    fraction mass is preserved."""
    rng = rng or random.Random(0)
    sample = list(texts)
    if len(sample) > sample_size:
        sample = rng.sample(sample, sample_size)

    labels = _label_texts(sample, call_api, batch_size)
    labeled = [(text, label) for text, label in zip(sample, labels) if label]
    raw_counts = Counter(label for _, label in labeled)
    if not labeled:
        return {"n_sampled": len(sample), "n_labeled": 0, "topics": {},
                "raw_labels": {}, "note": "no batches labeled successfully"}

    label_lines = "\n".join(
        f"{label} — {count}" for label, count in raw_counts.most_common()
    )
    note = None
    consolidated = None
    for _ in range(2):  # one retry
        try:
            raw = call_api(CONSOLIDATE_PROMPT.format(
                min_topics=min(5, len(raw_counts)),
                max_topics=max(5, min(15, len(raw_counts))),
                labels=label_lines,
            ))
        except Exception as exc:
            print(f"[topics] consolidation call failed: {exc}", flush=True)
            continue
        consolidated = _parse_consolidation(raw)
        if consolidated:
            break
    if not consolidated:
        note = "consolidation failed; topics are normalized raw labels"
        consolidated = {
            label: {"description": "", "members": [label]} for label in raw_counts
        }

    claimed = {m for block in consolidated.values() for m in block["members"]}
    for label in raw_counts:
        if label not in claimed:
            consolidated[label] = {"description": "", "members": [label]}

    n_labeled = len(labeled)
    topics = {}
    for name, block in consolidated.items():
        members = set(block["members"])
        count = sum(raw_counts.get(member, 0) for member in members)
        if count == 0:
            continue
        examples = [
            text[:_EXAMPLE_TRUNCATE] for text, label in labeled if label in members
        ][:_EXAMPLES_PER_TOPIC]
        topics[name] = {
            "fraction": round(count / n_labeled, 4),
            "description": block["description"],
            "examples": examples,
        }
    return {
        "n_sampled": len(sample),
        "n_labeled": n_labeled,
        "topics": topics,
        "raw_labels": dict(raw_counts),
        "note": note,
    }
