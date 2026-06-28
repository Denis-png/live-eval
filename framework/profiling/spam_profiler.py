"""Standalone spam dataset profiling helpers.

This module intentionally loads the raw spam dataset directly instead of using
SpamTask.parse_row(), because the generation pipeline filters to HAM inputs and
drops labels. Profiling needs the original text/label distribution.
"""

from __future__ import annotations

import os
from typing import Any

from framework.profiling.dataset_profiler import profile_classification_rows

DEFAULT_SPAM_DATASET = "deysi/spam-detection-dataset"
DEFAULT_SPAM_SPLIT = "train"
TEXT_FIELDS = ("text", "message", "sms")


def normalize_spam_label(label: Any) -> str:
    """Normalize raw dataset labels into HAM/SPAM."""
    if label == 1:
        return "SPAM"
    if isinstance(label, str) and label.strip().lower() in {"spam", "1"}:
        return "SPAM"
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
    return profile_classification_rows(rows, text_field="text", label_field="label")
