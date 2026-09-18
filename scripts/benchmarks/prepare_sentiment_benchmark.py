"""Build the local sentiment benchmark: the first N tweets of TweetEval's
sentiment split (cardiffnlp/tweet_eval), as a text,label CSV.

framework/data/** is gitignored, so the CSV the sentiment config points at does
not ship with the repo; this script is how it is rebuilt, and it reports whether
the result matches the final sweep's file byte for byte. Labels are written as
TweetEval delivers them (0 = negative, 1 = neutral, 2 = positive), the same
values the task's parse_row reads when the source is HuggingFace.

Usage:
    python -m scripts.benchmarks.prepare_sentiment_benchmark [--n 150] [--split test] \\
        [--output framework/data/benchmarks/sentiment/tweeteval_test_150.csv]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

DATASET = "cardiffnlp/tweet_eval"
SUBSET = "sentiment"
DEFAULT_OUTPUT = "framework/data/benchmarks/sentiment/tweeteval_test_150.csv"
SWEEP_ARGS, SWEEP_MD5 = ("test", 150), "5e1cccd2a6c9427d185466a90fdeed01"


def load_rows(split: str) -> Iterable[dict]:
    """TweetEval's sentiment rows in split order. The import is deferred because
    only this step needs the network and the datasets library."""
    from datasets import load_dataset

    return load_dataset(DATASET, SUBSET, split=split)


def write_benchmark(rows: Iterable[dict], output: str | Path, n: int) -> Counter:
    """Write the first n rows that carry a tweet and a label to output as
    text,label, and return the label counts written.

    A row without both is skipped rather than written, because parse_row would
    drop it on load and the benchmark would silently hold fewer than n. For the
    same reason a source with fewer than n usable rows raises ValueError."""
    kept = []
    for row in rows:
        if len(kept) == n:
            break
        text, label = row.get("text"), row.get("label")
        if text and text.strip() and label is not None:
            kept.append((text, label))
    if len(kept) < n:
        raise ValueError(f"Only {len(kept)} usable rows in the source; asked for {n}.")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(kept)
    return Counter(label for _, label in kept)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=150, help="tweets to keep (default 150)")
    parser.add_argument("--split", default="test", help="TweetEval split (default test)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"CSV to write (default {DEFAULT_OUTPUT})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = write_benchmark(load_rows(args.split), args.output, args.n)
    print("Sentiment benchmark preparation summary")
    print("=" * 39)
    print(f"Source            : {DATASET} ({SUBSET}, {args.split}, first {args.n})")
    print(f"Labels            : {dict(sorted(counts.items()))} (0 neg, 1 neu, 2 pos)")
    print(f"Output            : {args.output}")
    digest = hashlib.md5(Path(args.output).read_bytes()).hexdigest()
    same = (args.split, args.n) == SWEEP_ARGS and digest == SWEEP_MD5
    print(f"Matches the sweep : {'yes' if same else 'NO -- md5 ' + digest}")


if __name__ == "__main__":
    main()
