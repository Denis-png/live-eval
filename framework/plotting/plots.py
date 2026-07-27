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
    bits = [meta.get("task"), meta.get("mode"), meta.get("model")]
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
