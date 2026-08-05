"""Samplable text statistics for benchmark profiles (stdlib only).

Length histograms use fixed canonical bins (module constants) so real and
generated datasets are always histogrammed identically — a JSD between two
profiles is only meaningful when both sides used the same bins.
"""

from __future__ import annotations

from statistics import quantiles

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
