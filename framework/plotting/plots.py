"""Pure figure builders: plain dicts in, matplotlib Figure out. No I/O.

Importing this module pulls in matplotlib, so it is imported lazily by
session.render_session — the core pipeline never imports it.
"""
import matplotlib

matplotlib.use("Agg")  # headless: no display/GUI backend on the server

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

from framework.plotting.style import (  # noqa: E402
    INK,
    INK_MUTED,
    SERIES_GENERATED,
    SERIES_REAL,
    SURFACE,
    apply_axes_style,
)


def flatten_point(block: dict) -> dict[str, float]:
    """Flatten a scores block whose leaves are numbers.
    {"errant": {"precision": 0.8}} -> {"errant.precision": 0.8}"""
    out: dict[str, float] = {}
    for name, value in (block or {}).items():
        if isinstance(value, dict):
            for sub, v in value.items():
                if isinstance(v, (int, float)):
                    out[f"{name}.{sub}"] = float(v)
        elif isinstance(value, (int, float)):
            out[name] = float(value)
    return out


def flatten_mean_std(block: dict) -> dict[str, tuple[float, float]]:
    """Flatten a `generated` block whose leaves are {"mean", "std"}.
    {"errant": {"precision": {"mean": .7, "std": .05}}} -> {"errant.precision": (.7, .05)}"""
    out: dict[str, tuple[float, float]] = {}
    for name, value in (block or {}).items():
        if not isinstance(value, dict):
            continue
        if "mean" in value:
            out[name] = (float(value["mean"]), float(value.get("std", 0.0) or 0.0))
            continue
        for sub, v in value.items():
            if isinstance(v, dict) and "mean" in v:
                out[f"{name}.{sub}"] = (float(v["mean"]), float(v.get("std", 0.0) or 0.0))
    return out


# Evaluators that carry no signal in the figures — fpr reads a flat 0.00 against a
# 0.00 baseline on any decent classifier, so it is dead space in the chart. They are
# still computed and written to results.json; this only hides them from the plots.
# Lives here (not in config) because the standalone CLI renders from the session
# artifacts alone and must produce the same figures the pipeline did.
HIDDEN_METRICS = frozenset({"fpr"})


def _visible(names) -> list[str]:
    """Metric names to draw, dropping the ones with no signal (see HIDDEN_METRICS)."""
    return [n for n in names if n not in HIDDEN_METRICS]


def _split_by_scale(values: dict[str, float]) -> list[list[str]]:
    """Group metric names so that each group shares one y-scale.

    0-1 scores and unbounded counts (GEC n_edits) must never share an axis — that
    would either dwarf the scores or force a dual axis. Returns 1 group when every
    metric is in [0, 1], otherwise 2 groups (unit scores, then the rest)."""
    unit = [n for n, v in values.items() if v <= 1.0]
    other = [n for n, v in values.items() if v > 1.0]
    groups = [sorted(unit), sorted(other)]
    return [g for g in groups if g]


def _subtitle(meta: dict | None) -> str:
    if not meta:
        return ""
    # `seedless` belongs here: without it two sessions from different generation
    # cells (same task/mode/model, seeded vs profile-driven) render identically.
    cell = meta.get("mode")
    if meta.get("seedless"):
        cell = f"{cell}+seedless" if cell else "seedless"
    bits = [meta.get("task"), cell, meta.get("model")]
    return "  ·  ".join(str(b) for b in bits if b)


def plot_generated_vs_real(model: str, generated: dict, real: dict | None,
                           meta: dict | None = None):
    """Grouped bars per evaluator: generated (mean, ±std) beside the real baseline.
    Metrics of different scale are split into small multiples, never a dual axis."""
    gen = flatten_mean_std(generated)
    real_flat = flatten_point(real or {})
    peak = {n: max(gen.get(n, (0.0, 0.0))[0], real_flat.get(n, 0.0))
            for n in _visible(set(gen) | set(real_flat))}
    groups = _split_by_scale(peak) or [[]]

    fig, axes = plt.subplots(
        1, len(groups), figsize=(max(6.0, 1.6 * len(peak) + 2), 4.4), squeeze=False,
        gridspec_kw={"width_ratios": [max(len(g), 1) for g in groups]},
    )
    fig.patch.set_facecolor(SURFACE)
    width = 0.38  # leaves a visible surface gap between the paired bars

    for ax, names in zip(axes[0], groups):
        apply_axes_style(ax)
        x = range(len(names))
        gen_x = [i - width / 2 - 0.01 for i in x]
        gen_vals = [gen.get(n, (0.0, 0.0))[0] for n in names]
        # Draw the bars first, then layer the error bars on top via a separate
        # call: ax.bar(..., yerr=...) would register its internal errorbar
        # container in ax.containers *before* the BarContainer itself, which
        # shifts containers[0] away from the "generated" bars callers expect.
        gen_stds = [gen.get(n, (0.0, 0.0))[1] for n in names]
        ax.bar(gen_x, gen_vals, width,
              label="generated (mean ± std)", color=SERIES_GENERATED)
        ax.errorbar(gen_x, gen_vals, yerr=gen_stds,
                    fmt="none", capsize=3, ecolor=INK_MUTED, elinewidth=1.2,
                    label="_nolegend_")
        # Anchor each value label above the error-bar whisker (val + std), not just
        # the bar top — bar_label's point-based padding sits inside the whisker
        # whenever std is non-trivial, drawing the cap straight through the digits.
        for xi, val, std in zip(gen_x, gen_vals, gen_stds):
            ax.annotate(f"{val:.2f}", xy=(xi, val + std), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        color=INK_MUTED, fontsize=8)
        if real_flat:
            rbars = ax.bar([i + width / 2 + 0.01 for i in x],
                           [real_flat.get(n, 0.0) for n in names], width,
                           label="real benchmark", color=SERIES_REAL)
            ax.bar_label(rbars, fmt="%.2f", padding=2, color=INK_MUTED, fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.margins(y=0.18)

    axes[0][0].set_ylabel("score", color=INK_MUTED)
    axes[0][0].legend(frameon=False, labelcolor=INK_MUTED, fontsize=9)
    fig.suptitle(f"{model} — generated vs real\n{_subtitle(meta)}".strip(),
                 color=INK, fontsize=11)
    fig.tight_layout()
    return fig


def plot_run_variance(model: str, runs: list[dict], generated: dict | None = None,
                      meta: dict | None = None):
    """One dot per run per evaluator (x-jittered) with the mean as a wide marker.
    Shows run-to-run instability — the core GET signal. Deliberately unlabelled:
    a number on every dot would be chaos; the y-axis carries the values."""
    per_run = [flatten_point(r) for r in (runs or [])]
    names = sorted(_visible({n for r in per_run for n in r}))
    peak = {n: max([r[n] for r in per_run if n in r] or [0.0]) for n in names}
    groups = _split_by_scale(peak) or [[]]

    fig, axes = plt.subplots(
        1, len(groups), figsize=(max(6.0, 1.5 * len(names) + 2), 4.4), squeeze=False,
        gridspec_kw={"width_ratios": [max(len(g), 1) for g in groups]},
    )
    fig.patch.set_facecolor(SURFACE)

    for ax, group in zip(axes[0], groups):
        apply_axes_style(ax)
        for i, name in enumerate(group):
            values = [r[name] for r in per_run if name in r]
            # spread the dots so overlapping runs stay countable
            offsets = [(k - (len(values) - 1) / 2) * 0.08 for k in range(len(values))]
            ax.scatter([i + o for o in offsets], values, s=64, zorder=3,
                       color=SERIES_GENERATED, edgecolors=SURFACE, linewidths=1.5,
                       label="run score" if i == 0 else None)
            if values:
                mean = sum(values) / len(values)
                ax.plot([i - 0.22, i + 0.22], [mean, mean], color=INK_MUTED,
                        linewidth=2, zorder=4, label="mean" if i == 0 else None)
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels(group, rotation=20, ha="right")
        ax.margins(x=0.15, y=0.2)

    axes[0][0].set_ylabel("score", color=INK_MUTED)
    if names:
        axes[0][0].legend(frameon=False, labelcolor=INK_MUTED, fontsize=9)
    fig.suptitle(f"{model} — score per run (n={len(per_run)})\n{_subtitle(meta)}".strip(),
                 color=INK, fontsize=11)
    fig.tight_layout()
    return fig


def plot_error_type_distribution(
    synthetic_counts: dict[str, int],
    real_rates: dict[str, float] | None = None,
    meta: dict | None = None,
):
    """Bar chart of synthetic error-type counts with optional real signal-rate overlay.

    synthetic_counts: {error_type: count} tallied from the run JSONs.
    real_rates: {signal_type: rate} from profile.real.signal_rate — drawn as a
        second bar group when present, scaled to the synthetic total so both axes
        share the same unit (count).  When absent, only the synthetic bars are shown.
    """
    labels = sorted(synthetic_counts)
    synth_vals = [synthetic_counts[l] for l in labels]
    total = sum(synth_vals) or 1

    has_real = bool(real_rates)
    if has_real:
        real_vals = [real_rates.get(l, 0.0) * total for l in labels]

    width = 0.38 if has_real else 0.55
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(labels) + 2), 4.4))
    fig.patch.set_facecolor(SURFACE)
    apply_axes_style(ax)

    if has_real:
        rb = ax.bar([i - width / 2 - 0.01 for i in x], real_vals, width,
                    label="real (scaled)", color=SERIES_REAL)
        ax.bar_label(rb, fmt="%.1f", padding=2, color=INK_MUTED, fontsize=8)
        gb = ax.bar([i + width / 2 + 0.01 for i in x], synth_vals, width,
                    label="synthetic", color=SERIES_GENERATED)
        ax.bar_label(gb, fmt="%d", padding=2, color=INK_MUTED, fontsize=8)
        ax.legend(frameon=False, labelcolor=INK_MUTED, fontsize=9)
    else:
        gb = ax.bar(list(x), synth_vals, width, label="synthetic", color=SERIES_GENERATED)
        ax.bar_label(gb, fmt="%d", padding=2, color=INK_MUTED, fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("count", color=INK_MUTED)
    ax.margins(y=0.18)

    fig.suptitle(
        f"error-type distribution — synthetic{'  vs  real (scaled)' if has_real else ''}"
        f"\n{_subtitle(meta)}".strip(),
        color=INK, fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_fidelity(profile: dict, meta: dict | None = None):
    """Real vs generated signal rates + class balance, with the JSD scores in the
    title. Two panels (each one scale) rather than one mixed axis."""
    real = (profile or {}).get("real") or {}
    gen = (profile or {}).get("generated") or {}
    fid = (profile or {}).get("fidelity") or {}

    signals = sorted(set(real.get("signal_rate", {})) | set(gen.get("signal_rate", {})))
    fig, (ax_sig, ax_bal) = plt.subplots(
        1, 2, figsize=(max(8.0, 1.5 * len(signals) + 4), 4.4),
        gridspec_kw={"width_ratios": [max(len(signals), 1), 1]},
    )
    fig.patch.set_facecolor(SURFACE)
    width = 0.38

    apply_axes_style(ax_sig)
    x = range(len(signals))
    rb = ax_sig.bar([i - width / 2 - 0.01 for i in x],
                    [real.get("signal_rate", {}).get(s, 0.0) for s in signals], width,
                    label="real", color=SERIES_REAL)
    gb = ax_sig.bar([i + width / 2 + 0.01 for i in x],
                    [gen.get("signal_rate", {}).get(s, 0.0) for s in signals], width,
                    label="generated", color=SERIES_GENERATED)
    ax_sig.bar_label(rb, fmt="%.2f", padding=2, color=INK_MUTED, fontsize=8)
    ax_sig.bar_label(gb, fmt="%.2f", padding=2, color=INK_MUTED, fontsize=8)
    ax_sig.set_xticks(list(x))
    ax_sig.set_xticklabels(signals, rotation=20, ha="right")
    ax_sig.set_ylabel("signal rate (of SPAM messages)", color=INK_MUTED)
    ax_sig.set_title("spam signals", color=INK, fontsize=10)
    ax_sig.legend(frameon=False, labelcolor=INK_MUTED, fontsize=9)
    ax_sig.margins(y=0.18)

    apply_axes_style(ax_bal)
    bal = [real.get("class_balance", {}).get("spam_fraction", 0.0),
           gen.get("class_balance", {}).get("spam_fraction", 0.0)]
    bb = ax_bal.bar([0, 1], bal, 0.5, color=[SERIES_REAL, SERIES_GENERATED])
    ax_bal.bar_label(bb, fmt="%.2f", padding=2, color=INK_MUTED, fontsize=8)
    ax_bal.set_xticks([0, 1])
    ax_bal.set_xticklabels(["real", "generated"])
    ax_bal.set_ylabel("SPAM fraction", color=INK_MUTED)
    ax_bal.set_title("class balance", color=INK, fontsize=10)
    ax_bal.margins(y=0.18)

    jsd = (f"JSD type {fid.get('type_dist_jsd', float('nan')):.3f}  ·  "
           f"JSD count {fid.get('count_dist_jsd', float('nan')):.3f}  "
           f"(0 = identical, 1 = disjoint)")
    fig.suptitle(f"real vs generated fidelity — {jsd}\n{_subtitle(meta)}".strip(),
                 color=INK, fontsize=11)
    fig.tight_layout()
    return fig


def plot_taxonomy_fidelity(profile: dict, meta: dict | None = None):
    """Structural real-vs-synthetic taxonomy fidelity.

    Scalars are plotted as real value vs synthetic mean with min/max variation.
    Distribution panels show Jensen-Shannon divergence; lower means closer to
    the real benchmark distribution.
    """
    fidelity = (profile or {}).get("fidelity") or {}
    real = fidelity.get("real_profile") or {}
    aggregate = fidelity.get("aggregate") or {}
    scalar_agg = aggregate.get("scalar_characteristics") or {}
    dist_agg = aggregate.get("distribution_characteristics") or {}

    scalar_keys = [
        "n_classes",
        "n_subclass_axioms",
        "n_roots",
        "n_leaves",
        "max_depth",
    ]
    dist_keys = [
        "depth_distribution",
        "parent_count_distribution",
        "child_count_distribution",
    ]

    fig, (ax_scalar, ax_dist) = plt.subplots(
        1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [5, 3]},
    )
    fig.patch.set_facecolor(SURFACE)
    width = 0.38

    apply_axes_style(ax_scalar)
    x = range(len(scalar_keys))
    real_values = [real.get(key, 0) or 0 for key in scalar_keys]
    synthetic_mean = [
        (scalar_agg.get(key, {}).get("synthetic") or {}).get("mean") or 0
        for key in scalar_keys
    ]
    synthetic_min = [
        (scalar_agg.get(key, {}).get("synthetic") or {}).get("min")
        for key in scalar_keys
    ]
    synthetic_max = [
        (scalar_agg.get(key, {}).get("synthetic") or {}).get("max")
        for key in scalar_keys
    ]
    rb = ax_scalar.bar(
        [i - width / 2 - 0.01 for i in x], real_values, width,
        label="real", color=SERIES_REAL,
    )
    sx = [i + width / 2 + 0.01 for i in x]
    gb = ax_scalar.bar(sx, synthetic_mean, width, label="synthetic mean",
                       color=SERIES_GENERATED)
    yerr_low, yerr_high = [], []
    for mean_value, min_value, max_value in zip(synthetic_mean, synthetic_min, synthetic_max):
        if min_value is None or max_value is None:
            yerr_low.append(0)
            yerr_high.append(0)
        else:
            yerr_low.append(max(0, mean_value - min_value))
            yerr_high.append(max(0, max_value - mean_value))
    ax_scalar.errorbar(
        sx, synthetic_mean, yerr=[yerr_low, yerr_high], fmt="none",
        capsize=3, ecolor=INK_MUTED, elinewidth=1.2,
    )
    ax_scalar.bar_label(rb, fmt="%.1f", padding=2, color=INK_MUTED, fontsize=8)
    ax_scalar.bar_label(gb, fmt="%.1f", padding=2, color=INK_MUTED, fontsize=8)
    ax_scalar.set_xticks(list(x))
    ax_scalar.set_xticklabels(scalar_keys, rotation=20, ha="right")
    ax_scalar.set_ylabel("count / value", color=INK_MUTED)
    ax_scalar.set_title("structural scalars", color=INK, fontsize=10)
    ax_scalar.legend(frameon=False, labelcolor=INK_MUTED, fontsize=9)
    ax_scalar.margins(y=0.2)

    apply_axes_style(ax_dist)
    dx = range(len(dist_keys))
    divergences = [
        ((dist_agg.get(key, {}).get("jensen_shannon_divergence") or {}).get("mean") or 0)
        for key in dist_keys
    ]
    db = ax_dist.bar(list(dx), divergences, 0.55, color=SERIES_GENERATED)
    ax_dist.bar_label(db, fmt="%.3f", padding=2, color=INK_MUTED, fontsize=8)
    ax_dist.set_xticks(list(dx))
    ax_dist.set_xticklabels(dist_keys, rotation=20, ha="right")
    ax_dist.set_ylabel("JSD (0 = identical)", color=INK_MUTED)
    ax_dist.set_title("distribution divergence", color=INK, fontsize=10)
    ax_dist.set_ylim(0, max(1.0, max(divergences or [0]) * 1.15))

    n = aggregate.get("n_synthetic_taxonomies", 0)
    fig.suptitle(
        f"taxonomy structural fidelity — real vs synthetic (n={n})\n{_subtitle(meta)}".strip(),
        color=INK, fontsize=11,
    )
    fig.tight_layout()
    return fig


def _sort_distribution_bins(bins) -> list[str]:
    """Sort stringified histogram bins numerically when possible."""
    def key(value):
        text = str(value)
        try:
            return (0, int(text))
        except ValueError:
            try:
                return (0, float(text))
            except ValueError:
                return (1, text)
    return [str(value) for value in sorted({str(v) for v in bins}, key=key)]


def _normalize_distribution(dist: dict) -> dict[str, float]:
    """Normalize count-like distribution values into probabilities."""
    if not isinstance(dist, dict):
        return {}
    total = sum(float(v) for v in dist.values() if isinstance(v, (int, float)))
    if total <= 0:
        return {str(k): 0.0 for k in dist}
    return {
        str(k): (float(v) / total if isinstance(v, (int, float)) else 0.0)
        for k, v in dist.items()
    }


def _taxonomy_distribution_series(fidelity: dict, key: str) -> dict:
    """Aligned real/synthetic probability series for one taxonomy distribution.

    Missing bins are zero. Synthetic mean/min/max are computed over independently
    normalized synthetic taxonomies, not over raw counts pooled together.
    """
    real = fidelity.get("real_profile") or {}
    synthetics = fidelity.get("synthetic_profiles") or []
    real_norm = _normalize_distribution(real.get(key, {}))
    synth_norms = [
        _normalize_distribution((synthetic or {}).get(key, {}))
        for synthetic in synthetics
        if isinstance(synthetic, dict)
    ]
    labels = _sort_distribution_bins(
        set(real_norm) | {bin_key for dist in synth_norms for bin_key in dist}
    )

    real_values = [real_norm.get(label, 0.0) for label in labels]
    per_synth = [
        [dist.get(label, 0.0) for label in labels]
        for dist in synth_norms
    ]
    if per_synth:
        mean_values = [
            sum(row[i] for row in per_synth) / len(per_synth)
            for i in range(len(labels))
        ]
        min_values = [min(row[i] for row in per_synth) for i in range(len(labels))]
        max_values = [max(row[i] for row in per_synth) for i in range(len(labels))]
    else:
        mean_values = [0.0 for _ in labels]
        min_values = [0.0 for _ in labels]
        max_values = [0.0 for _ in labels]

    return {
        "labels": labels,
        "real": real_values,
        "synthetic_mean": mean_values,
        "synthetic_min": min_values,
        "synthetic_max": max_values,
    }


def plot_taxonomy_fidelity_distributions(profile: dict, meta: dict | None = None):
    """Real vs synthetic taxonomy structure distributions.

    Each taxonomy's counts are normalized to proportions before aggregation so
    differently sized taxonomies can be compared honestly. The synthetic band is
    min/max run-to-run variation, not a confidence interval.
    """
    fidelity = (profile or {}).get("fidelity") or {}
    panels = [
        ("depth_distribution", "hierarchy depth", "depth"),
        ("parent_count_distribution", "parent count", "number of parents"),
        ("child_count_distribution", "child / branching count", "number of children"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    for ax, (key, title, xlabel) in zip(axes[0], panels):
        apply_axes_style(ax)
        series = _taxonomy_distribution_series(fidelity, key)
        labels = series["labels"]
        x = list(range(len(labels)))
        if not labels:
            ax.text(
                0.5, 0.5, "no distribution data",
                ha="center", va="center", color=INK_MUTED, transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_ylim(0, 1)
        else:
            ax.fill_between(
                x, series["synthetic_min"], series["synthetic_max"],
                color=SERIES_GENERATED, alpha=0.18,
                label="synthetic min/max" if key == panels[0][0] else None,
            )
            ax.plot(
                x, series["synthetic_mean"], marker="o", linewidth=2,
                color=SERIES_GENERATED,
                label="synthetic mean" if key == panels[0][0] else None,
            )
            ax.plot(
                x, series["real"], marker="o", linewidth=2,
                color=SERIES_REAL,
                label="real" if key == panels[0][0] else None,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            peak = max(
                series["real"] + series["synthetic_max"] + series["synthetic_mean"] + [0.0]
            )
            ax.set_ylim(0, min(1.0, max(0.05, peak) * 1.18))
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlabel(xlabel, color=INK_MUTED)
        ax.set_ylabel("proportion of classes", color=INK_MUTED)

    axes[0][0].legend(frameon=False, labelcolor=INK_MUTED, fontsize=9)
    fig.suptitle(
        f"taxonomy structural distributions — real vs synthetic\n{_subtitle(meta)}".strip(),
        color=INK, fontsize=11,
    )
    fig.tight_layout()
    return fig
