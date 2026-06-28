"""GEC-specific profiling for original benchmark examples.

Expected input rows are normalized dictionaries shaped as:
    {"incorrect": "...", "correct": "..."}
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from framework.profiling.dataset_profiler import (
    DEFAULT_STOPWORDS,
    count_samples,
    numeric_stats,
    tokenize,
    top_frequent_words,
)


def sequence_similarity(source: str, reference: str) -> float:
    """Return character-level similarity in [0, 1].

    This uses difflib.SequenceMatcher from the standard library. A score near
    1.0 means the incorrect and correct sentences are very similar, while lower
    scores indicate larger corrections or rewrites.
    """
    if not source and not reference:
        return 1.0
    return SequenceMatcher(None, source, reference).ratio()


def complexity_category(similarity: float) -> str:
    """Bucket correction complexity using a transparent similarity rule.

    The similarity score is character-level SequenceMatcher ratio:
    - low complexity: mostly local edits, similarity >= 0.90
    - medium complexity: moderate changes, similarity >= 0.75 and < 0.90
    - high complexity: larger rewrite or many edits, similarity < 0.75
    """
    if similarity >= 0.9:
        return "low"
    if similarity >= 0.75:
        return "medium"
    return "high"


def similarity_bucket(similarity: float) -> str:
    """Bucket source/reference similarity for distribution matching."""
    if similarity >= 0.9:
        return "very_similar"
    if similarity >= 0.75:
        return "moderately_changed"
    return "strongly_changed"


def _examples_by_complexity(
    rows_with_similarity: list[dict[str, Any]],
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    """Collect a few example pairs from each complexity category."""
    examples = {"low": [], "medium": [], "high": []}
    for item in rows_with_similarity:
        bucket = item["complexity"]
        if len(examples[bucket]) >= limit:
            continue
        examples[bucket].append(
            {
                "incorrect": item["incorrect"],
                "correct": item["correct"],
                "similarity": item["similarity"],
            }
        )
    return examples


def profile_gec_rows(
    rows: list[dict[str, Any]],
    top_word_limit: int = 20,
    example_limit: int = 3,
) -> dict[str, Any]:
    """Compute GEC-specific statistics for normalized original dataset rows."""
    rows_with_similarity = []
    for row in rows:
        incorrect = row.get("incorrect")
        correct = row.get("correct")
        if not isinstance(incorrect, str) or not isinstance(correct, str):
            continue
        similarity = round(sequence_similarity(incorrect, correct), 4)
        rows_with_similarity.append(
            {
                "incorrect": incorrect,
                "correct": correct,
                "similarity": similarity,
                "similarity_bucket": similarity_bucket(similarity),
                "complexity": complexity_category(similarity),
            }
        )

    pairs = [(item["incorrect"], item["correct"]) for item in rows_with_similarity]

    incorrect_texts = [incorrect for incorrect, _ in pairs]
    correct_texts = [correct for _, correct in pairs]
    incorrect_lengths = [len(text) for text in incorrect_texts]
    correct_lengths = [len(text) for text in correct_texts]
    incorrect_word_counts = [len(tokenize(text)) for text in incorrect_texts]
    correct_word_counts = [len(tokenize(text)) for text in correct_texts]
    similarities = [item["similarity"] for item in rows_with_similarity]

    complexity_counts = {"low": 0, "medium": 0, "high": 0}
    similarity_buckets = {
        "very_similar": 0,
        "moderately_changed": 0,
        "strongly_changed": 0,
    }
    for similarity in similarities:
        complexity_counts[complexity_category(similarity)] += 1
        similarity_buckets[similarity_bucket(similarity)] += 1

    return {
        "num_samples": count_samples(rows),
        "num_valid_pairs": len(pairs),
        "incorrect_char_length": numeric_stats(incorrect_lengths),
        "correct_char_length": numeric_stats(correct_lengths),
        "incorrect_word_count": numeric_stats(incorrect_word_counts, include_std=True),
        "correct_word_count": numeric_stats(correct_word_counts, include_std=True),
        "similarity": {
            "metric": "difflib.SequenceMatcher character ratio",
            "stats": numeric_stats(similarities),
            "buckets": similarity_buckets,
            "bucket_rule": {
                "very_similar": "similarity >= 0.90",
                "moderately_changed": "0.75 <= similarity < 0.90",
                "strongly_changed": "similarity < 0.75",
            },
        },
        "correction_complexity": {
            "description": "low >= 0.90 similarity, medium >= 0.75, high < 0.75",
            "counts": complexity_counts,
        },
        "top_frequent_words": top_frequent_words(
            rows,
            columns=("incorrect", "correct"),
            limit=top_word_limit,
            stopwords=DEFAULT_STOPWORDS,
        ),
        "example_pairs_by_complexity": _examples_by_complexity(
            rows_with_similarity,
            limit=example_limit,
        ),
    }
