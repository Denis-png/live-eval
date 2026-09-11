"""Sentiment dataset profiling helpers.

Loads cardiffnlp/tweet_eval (sentiment) and builds a profile suitable for
seedless generation (profile_version 2). The profile is keyed on the tweet
text side ("incorrect") since SentimentTask.get_profile_side() returns that
for both forward and inverse modes.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

from framework.profiling.dataset_profiler import (
    profile_classification_rows,
    tokenize,
)
from framework.profiling.text_stats import (
    WORD_BINS,
    length_distribution,
    style_profile,
    vocab_profile,
)

DEFAULT_SENTIMENT_DATASET = "cardiffnlp/tweet_eval"
DEFAULT_SENTIMENT_SUBSET = "sentiment"
DEFAULT_SENTIMENT_SPLIT = "test"

_LABEL_MAP = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}


def load_sentiment_rows(
    dataset_name: str = DEFAULT_SENTIMENT_DATASET,
    subset: str = DEFAULT_SENTIMENT_SUBSET,
    split: str = DEFAULT_SENTIMENT_SPLIT,
    streaming: bool = True,
    sample_size: int | None = None,
    hf_token: str | None = None,
) -> list[dict[str, str]]:
    """Load and normalize TweetEval sentiment rows into {"text", "label"}."""
    from datasets import load_dataset

    ds = load_dataset(
        dataset_name,
        subset,
        split=split,
        streaming=streaming,
        token=hf_token or os.getenv("HF_TOKEN") or None,
    )
    rows = []
    for raw in ds:
        text = raw.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        label = _LABEL_MAP.get(raw.get("label"), "NEUTRAL")
        rows.append({"text": text, "label": label})
        if sample_size is not None and len(rows) >= sample_size:
            break
    return rows


def profile_sentiment_dataset(
    dataset_name: str = DEFAULT_SENTIMENT_DATASET,
    subset: str = DEFAULT_SENTIMENT_SUBSET,
    split: str = DEFAULT_SENTIMENT_SPLIT,
    streaming: bool = True,
    sample_size: int | None = None,
    hf_token: str | None = None,
    topic_call_api=None,
    topic_sample_size: int = 200,
) -> dict[str, Any]:
    """Build a generation-ready profile for TweetEval sentiment.

    Returns a profile_version=2 dict with:
      - classification stats per label (label_distribution, length, style, vocab)
      - length_distributions["incorrect"]["words"] — sampled by spec_sampler
      - style["incorrect"]                          — sampled by spec_sampler
      - topics                                      — when topic_call_api given
    """
    rows = load_sentiment_rows(
        dataset_name=dataset_name,
        subset=subset,
        split=split,
        streaming=streaming,
        sample_size=sample_size,
        hf_token=hf_token,
    )

    profile = profile_classification_rows(rows, text_field="text", label_field="label")

    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row["text"])

    all_texts = [row["text"] for row in rows]

    # Per-label length, style, vocab
    profile["length_distributions_per_label"] = {
        label: {"words": length_distribution([len(tokenize(t)) for t in texts], WORD_BINS)}
        for label, texts in sorted(grouped.items())
    }
    profile["style_per_label"] = {
        label: style_profile(texts) for label, texts in sorted(grouped.items())
    }
    profile["vocabulary_per_label"] = {
        label: vocab_profile(texts) for label, texts in sorted(grouped.items())
    }

    # Global "incorrect" side — what spec_sampler reads for corruption tasks
    profile["length_distributions"] = {
        "incorrect": {"words": length_distribution([len(tokenize(t)) for t in all_texts], WORD_BINS)}
    }
    profile["style"] = {"incorrect": style_profile(all_texts)}

    profile["profile_version"] = 2

    if topic_call_api is not None:
        from framework.profiling.topics import profile_topics
        profile["topics"] = profile_topics(all_texts, topic_call_api, sample_size=topic_sample_size)

    return profile
