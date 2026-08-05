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
from framework.profiling.text_stats import (
    CHAR_BINS,
    WORD_BINS,
    length_distribution,
    style_profile,
    vocab_profile,
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
        "length_distributions": {
            "incorrect": {
                "words": length_distribution(incorrect_word_counts, WORD_BINS),
                "chars": length_distribution(incorrect_lengths, CHAR_BINS),
            },
            "correct": {
                "words": length_distribution(correct_word_counts, WORD_BINS),
                "chars": length_distribution(correct_lengths, CHAR_BINS),
            },
        },
        "style": {
            "incorrect": style_profile(incorrect_texts),
            "correct": style_profile(correct_texts),
        },
        "vocabulary": {
            "incorrect": vocab_profile(incorrect_texts),
            "correct": vocab_profile(correct_texts),
        },
        "profile_version": 2,
    }


def profile_gec_edit_types(
    rows: list[dict[str, Any]],
    *,
    supported_types=(),
    annotator=None,
    count_max: int = 5,
) -> dict[str, Any]:
    """Edit-type profile of (corrupted -> original) pairs via ERRANT re-annotation.

    Accepts both real-reference rows ({"corrupted", "original", "text"}) and
    generated rows ({"original", "corrupted", "error_type"}); a generated row's
    own error_type claim is deliberately ignored so real and generated are
    measured with the same instrument. supported_types is the inverse-mode
    vocabulary; supported_fraction reports how much of the observed edit mass
    that vocabulary can express.
    """
    from collections import Counter

    if annotator is None:
        from framework.evaluators.gec._errant_shared import get_annotator
        annotator = get_annotator()

    supported = set(supported_types)
    type_counter: Counter = Counter()
    per_pair: list[int] = []
    for row in rows:
        incorrect = row.get("corrupted") or row.get("incorrect")
        correct = row.get("original") or row.get("correct")
        if not incorrect or not correct:
            continue
        try:
            edits = annotator.annotate(annotator.parse(incorrect), annotator.parse(correct))
        except Exception:
            continue
        types = [e.type for e in edits if e.type and e.type != "noop"]
        type_counter.update(types)
        per_pair.append(min(len(types), count_max))

    n_annotated = len(per_pair)
    total_edits = sum(type_counter.values())
    count_counter = Counter(per_pair)
    return {
        "n": len(rows),
        "n_annotated": n_annotated,
        "edits_per_pair_mean": (
            round(sum(per_pair) / n_annotated, 4) if n_annotated else 0.0
        ),
        "error_type_dist": {
            t: type_counter[t] / total_edits for t in sorted(type_counter)
        } if total_edits else {},
        "error_count_dist": {
            n: count_counter[n] / n_annotated for n in sorted(count_counter)
        } if n_annotated else {},
        "supported_fraction": (
            sum(c for t, c in type_counter.items() if t in supported) / total_edits
            if total_edits else 0.0
        ),
    }
