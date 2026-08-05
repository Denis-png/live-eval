"""Samplable text statistics for benchmark profiles (stdlib only).

Length histograms use fixed canonical bins (module constants) so real and
generated datasets are always histogrammed identically — a JSD between two
profiles is only meaningful when both sides used the same bins.
"""

from __future__ import annotations

import re
import string
from statistics import mean, quantiles

from framework.profiling.dataset_profiler import tokenize

WORD_BINS: tuple[tuple[int, int | None], ...] = (
    (1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 50), (51, None),
)
CHAR_BINS: tuple[tuple[int, int | None], ...] = (
    (1, 25), (26, 50), (51, 100), (101, 200), (201, 400), (401, None),
)

_QUANTILE_POINTS = (10, 25, 50, 75, 90)


def bin_label(lo: int, hi: int | None) -> str:
    """Human-readable histogram bin label: "1-5" for (1, 5), "51+" for (51, None)."""
    return f"{lo}+" if hi is None else f"{lo}-{hi}"


def length_distribution(
    values: list[int],
    bins: tuple[tuple[int, int | None], ...],
) -> dict:
    """Histogram `values` over fixed `bins` plus p10..p90 quantiles.

    Fractions are normalized over the input; out-of-range values clamp into
    the boundary bins (an open-ended last bin catches everything above it).
    Empty input returns all-zero bins and zero quantiles.
    """
    labels = [bin_label(lo, hi) for lo, hi in bins]
    counts = {label: 0 for label in labels}
    for value in values:
        value = max(value, bins[0][0])  # clamp below-range into the first bin
        for (lo, hi), label in zip(bins, labels):
            if hi is None or value <= hi:
                counts[label] += 1
                break

    n = len(values)
    if n == 0:
        quantile_stats: dict[str, float | int] = {f"p{q}": 0 for q in _QUANTILE_POINTS}
    elif n == 1:
        quantile_stats = {f"p{q}": values[0] for q in _QUANTILE_POINTS}
    else:
        cut = quantiles(values, n=100, method="inclusive")
        quantile_stats = {f"p{q}": round(cut[q - 1], 4) for q in _QUANTILE_POINTS}

    return {
        "bins": {
            label: round(counts[label] / n, 4) if n else 0.0 for label in labels
        },
        "quantiles": quantile_stats,
        "count": n,
    }


_CONTRACTION_RE = re.compile(r"[A-Za-z]+'[A-Za-z]+")
_ALLCAPS_RE = re.compile(r"\b[A-Z]{2,}\b")
_DIGIT_RE = re.compile(r"\d")
_PUNCT_CHARS = set(string.punctuation)

# Contracted forms are listed explicitly because tokenize() keeps internal
# apostrophes ("i'm" is one token, not "i").
_FIRST_PERSON = {
    "i", "me", "my", "mine", "we", "us", "our", "ours",
    "i'm", "i've", "i'll", "i'd", "im", "we're", "we've", "we'll",
}
_SECOND_PERSON = {"you", "your", "yours", "u", "ur", "you're", "you've", "you'll", "you'd"}

# Bare digits ("2", "4") are deliberately excluded: they fire on genuine
# numbers far more often than on "to"/"for" shorthand.
TEXTING_SLANG = {
    "u", "ur", "r", "2day", "2moro", "2nite", "gr8", "thx", "plz", "pls",
    "wat", "wanna", "gonna", "luv", "lol", "tho", "cuz", "msg", "txt",
    "b4", "l8r", "w8", "omg", "btw", "idk",
}


def style_profile(texts: list[str]) -> dict:
    """Fraction-of-texts style rates plus mean punctuation density."""
    n = len(texts)

    def rate(predicate) -> float:
        return round(sum(1 for t in texts if predicate(t)) / n, 4) if n else 0.0

    token_sets = [set(tokenize(t)) for t in texts]

    def token_rate(vocabulary: set[str]) -> float:
        return (
            round(sum(1 for tokens in token_sets if tokens & vocabulary) / n, 4)
            if n else 0.0
        )

    densities = [
        sum(1 for ch in t if ch in _PUNCT_CHARS) / len(t) for t in texts if t
    ]
    return {
        "question_rate": rate(lambda t: "?" in t),
        "exclaim_rate": rate(lambda t: "!" in t),
        "first_person_rate": token_rate(_FIRST_PERSON),
        "second_person_rate": token_rate(_SECOND_PERSON),
        "contraction_rate": rate(lambda t: bool(_CONTRACTION_RE.search(t))),
        "digit_rate": rate(lambda t: bool(_DIGIT_RE.search(t))),
        "uppercase_word_rate": rate(lambda t: bool(_ALLCAPS_RE.search(t))),
        "texting_slang_rate": token_rate(TEXTING_SLANG),
        "punctuation_density": round(mean(densities), 4) if densities else 0.0,
    }
