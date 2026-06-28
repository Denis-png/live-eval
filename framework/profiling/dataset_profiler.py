"""General lightweight dataset profiling utilities.

The helpers in this module operate on in-memory rows represented as dictionaries.
They intentionally use only the Python standard library so profiling can run as a
standalone preprocessing step without adding dependencies to the GET pipeline.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "were",
    "with",
    "you",
}


def count_samples(rows: list[dict[str, Any]]) -> int:
    """Return the number of rows in the dataset."""
    return len(rows)


def detect_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Return sorted keys observed across all rows."""
    columns: set[str] = set()
    for row in rows:
        columns.update(row.keys())
    return sorted(columns)


def missing_or_empty_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count rows where each observed column is missing, None, or empty text."""
    columns = detect_columns(rows)
    counts = {column: 0 for column in columns}
    for row in rows:
        for column in columns:
            value = row.get(column)
            if value is None or (isinstance(value, str) and not value.strip()):
                counts[column] += 1
    return counts


def numeric_stats(values: list[int | float], include_std: bool = False) -> dict[str, float | int]:
    """Return count, mean, median, min, max, and optional population std dev."""
    if not values:
        stats: dict[str, float | int] = {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0,
            "max": 0,
        }
        if include_std:
            stats["std"] = 0.0
        return stats

    stats = {
        "count": len(values),
        "mean": round(float(mean(values)), 4),
        "median": round(float(median(values)), 4),
        "min": min(values),
        "max": max(values),
    }
    if include_std:
        stats["std"] = round(float(pstdev(values)), 4)
    return stats


def _basic_stats(values: list[int | float]) -> dict[str, float | int]:
    """Return the legacy compact summary shape used by the generic row profile."""
    stats = numeric_stats(values)
    return {
        "count": stats["count"],
        "avg": stats["mean"],
        "min": stats["min"],
        "max": stats["max"],
    }


def text_length_stats(rows: list[dict[str, Any]], columns: Iterable[str]) -> dict[str, dict[str, float | int]]:
    """Calculate character length stats for text values in selected columns."""
    stats = {}
    for column in columns:
        lengths = [
            len(value)
            for row in rows
            if isinstance((value := row.get(column)), str) and value.strip()
        ]
        stats[column] = _basic_stats(lengths)
    return stats


def tokenize(text: str, stopwords: set[str] | None = None) -> list[str]:
    """Tokenize text into simple lowercase word-like tokens."""
    tokens = [match.group(0).lower() for match in _WORD_RE.finditer(text)]
    if stopwords is None:
        return tokens
    return [token for token in tokens if token not in stopwords]


def word_count_stats(rows: list[dict[str, Any]], columns: Iterable[str]) -> dict[str, dict[str, float | int]]:
    """Calculate word count stats for text values in selected columns."""
    stats = {}
    for column in columns:
        counts = [
            len(tokenize(value))
            for row in rows
            if isinstance((value := row.get(column)), str) and value.strip()
        ]
        stats[column] = _basic_stats(counts)
    return stats


def top_frequent_words(
    rows: list[dict[str, Any]],
    columns: Iterable[str],
    limit: int = 20,
    stopwords: set[str] | None = None,
) -> dict[str, list[dict[str, int | str]]]:
    """Return the most common lowercase words for each selected text column."""
    output = {}
    for column in columns:
        counter: Counter[str] = Counter()
        for row in rows:
            value = row.get(column)
            if isinstance(value, str):
                counter.update(tokenize(value, stopwords=stopwords))
        output[column] = [
            {"word": word, "count": count}
            for word, count in counter.most_common(limit)
        ]
    return output


def simple_examples(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Return the first few rows as examples."""
    return rows[:limit]


def save_profile_json(profile: dict[str, Any], output_path: str | Path) -> str:
    """Save profile data as pretty JSON and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    return str(path)


def profile_rows(
    rows: list[dict[str, Any]],
    text_columns: Iterable[str] | None = None,
    top_word_limit: int = 20,
    example_limit: int = 5,
) -> dict[str, Any]:
    """Build a general-purpose profile for dictionary-shaped dataset rows."""
    columns = detect_columns(rows)
    selected_columns = list(text_columns) if text_columns is not None else columns
    return {
        "num_samples": count_samples(rows),
        "columns": columns,
        "missing_or_empty_counts": missing_or_empty_counts(rows),
        "text_length_stats": text_length_stats(rows, selected_columns),
        "word_count_stats": word_count_stats(rows, selected_columns),
        "top_frequent_words": top_frequent_words(rows, selected_columns, limit=top_word_limit),
        "examples": simple_examples(rows, limit=example_limit),
    }


def profile_classification_rows(
    rows: list[dict[str, Any]],
    text_field: str,
    label_field: str,
    top_word_limit: int = 20,
    example_limit: int = 3,
    stopwords: set[str] | None = DEFAULT_STOPWORDS,
) -> dict[str, Any]:
    """Profile text classification rows for future spam/hate-speech tasks.

    Args:
        rows: Dataset rows represented as dictionaries.
        text_field: Field containing the input text.
        label_field: Field containing the class label.
        top_word_limit: Number of frequent words to keep per label.
        example_limit: Number of example texts to keep per label.
        stopwords: Optional words to exclude from top-word counts.
    """
    grouped: dict[str, list[str]] = {}
    for row in rows:
        text = row.get(text_field)
        label = row.get(label_field)
        if not isinstance(text, str) or not text.strip() or label is None:
            continue
        grouped.setdefault(str(label), []).append(text)

    total = count_samples(rows)
    valid = sum(len(texts) for texts in grouped.values())
    label_distribution = {label: len(texts) for label, texts in sorted(grouped.items())}
    label_percentages = {
        label: round((count / valid) * 100, 4) if valid else 0.0
        for label, count in label_distribution.items()
    }

    text_length_by_label = {}
    word_count_by_label = {}
    top_words_by_label = {}
    examples_by_label = {}
    for label, texts in sorted(grouped.items()):
        text_length_by_label[label] = numeric_stats([len(text) for text in texts], include_std=True)
        word_count_by_label[label] = numeric_stats(
            [len(tokenize(text)) for text in texts],
            include_std=True,
        )
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text, stopwords=stopwords))
        top_words_by_label[label] = [
            {"word": word, "count": count}
            for word, count in counter.most_common(top_word_limit)
        ]
        examples_by_label[label] = texts[:example_limit]

    return {
        "num_samples": total,
        "num_valid_rows": valid,
        "text_field": text_field,
        "label_field": label_field,
        "label_distribution": label_distribution,
        "label_percentages": label_percentages,
        "text_length_stats_per_label": text_length_by_label,
        "word_count_stats_per_label": word_count_by_label,
        "top_frequent_words_per_label": top_words_by_label,
        "examples_per_label": examples_by_label,
    }
