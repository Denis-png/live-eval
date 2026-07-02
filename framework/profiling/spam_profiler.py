"""Standalone spam dataset profiling helpers.

This module intentionally loads the raw spam dataset directly instead of using
SpamTask.parse_row(), because the generation pipeline filters to HAM inputs and
drops labels. Profiling needs the original text/label distribution.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

from framework.profiling.dataset_profiler import profile_classification_rows

DEFAULT_SPAM_DATASET = "deysi/spam-detection-dataset"
DEFAULT_SPAM_SPLIT = "train"
TEXT_FIELDS = ("text", "message", "sms")

# Spam signal patterns
_URL_RE       = re.compile(r"https?://|www\.", re.IGNORECASE)
_CURRENCY_RE  = re.compile(r"[\$€£]")
_CAPS_RE      = re.compile(r"\b[A-Z]{2,}\b")
_EXCLAIM_RE   = re.compile(r"!")
_SPAM_KEYWORDS = {
    "free", "win", "winner", "won", "urgent", "click", "prize", "offer",
    "congratulations", "claim", "limited", "exclusive", "guaranteed",
    "cash", "money", "reward", "bonus", "deal", "discount",
}


def normalize_spam_label(label: Any) -> str:
    """Normalize raw dataset labels into HAM/SPAM."""
    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized in {"spam", "1"}:
            return "SPAM"
        if normalized in {"ham", "not_spam", "0"}:
            return "HAM"
    if label == 1:
        return "SPAM"
    if label == 0:
        return "HAM"
    return "HAM"


def normalize_spam_row(row: dict[str, Any]) -> dict[str, str] | None:
    """Convert a raw spam dataset row into {"text": ..., "label": ...}."""
    text = None
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            text = value
            break
    if text is None:
        return None
    return {"text": text, "label": normalize_spam_label(row.get("label"))}


def load_spam_rows(
    dataset_name: str = DEFAULT_SPAM_DATASET,
    split: str = DEFAULT_SPAM_SPLIT,
    streaming: bool = False,
    sample_size: int | None = None,
    hf_token: str | None = None,
) -> list[dict[str, str]]:
    """Load and normalize raw spam rows from Hugging Face."""
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        token=hf_token or os.getenv("HF_TOKEN") or None,
    )

    rows = []
    for raw_row in dataset:
        parsed = normalize_spam_row(raw_row)
        if parsed is not None:
            rows.append(parsed)
        if sample_size is not None and len(rows) >= sample_size:
            break
    return rows


def analyze_spam_signals(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """
    Compute spam signal rates per label (HAM/SPAM).

    For each label, returns the fraction of texts containing:
    - url_rate:      a URL (http:// or www.)
    - currency_rate: a currency symbol ($, €, £)
    - caps_rate:     at least one ALL CAPS word (2+ letters)
    - exclaim_rate:  at least one exclamation mark
    - keyword_rate:  at least one known spam keyword
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row["text"])

    signals: dict[str, dict[str, float]] = {}
    for label, texts in sorted(grouped.items()):
        total = len(texts)
        if total == 0:
            continue

        def rate(pattern: re.Pattern) -> float:
            return round(sum(1 for t in texts if pattern.search(t)) / total, 4)

        keyword_hits = sum(
            1 for t in texts
            if any(kw in t.lower().split() for kw in _SPAM_KEYWORDS)
        )

        signals[label] = {
            "url_rate":      rate(_URL_RE),
            "currency_rate": rate(_CURRENCY_RE),
            "caps_rate":     rate(_CAPS_RE),
            "exclaim_rate":  rate(_EXCLAIM_RE),
            "keyword_rate":  round(keyword_hits / total, 4),
        }

    return signals


def profile_spam_dataset(
    dataset_name: str = DEFAULT_SPAM_DATASET,
    split: str = DEFAULT_SPAM_SPLIT,
    streaming: bool = False,
    sample_size: int | None = None,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Load the raw spam dataset and return classification profile stats."""
    rows = load_spam_rows(
        dataset_name=dataset_name,
        split=split,
        streaming=streaming,
        sample_size=sample_size,
        hf_token=hf_token,
    )
    profile = profile_classification_rows(rows, text_field="text", label_field="label")
    profile["spam_signals"] = analyze_spam_signals(rows)
    return profile
