"""Cross-session analysis of generated-vs-real benchmark results.

Scans one or more results roots for session dirs (anything holding a
results.json), keeps the best session per (task, strategy, generation model)
— strategy being mode plus the seedless flag, so a seedless run never
shadows or merges with a seeded run of the same mode — and argues the three
questions the archive exists to answer:

  1. fidelity   — how close are generated scores to real ones, and do generated
                  benchmarks rank the evaluated models the way the real one does
                  (Kendall tau-b), while staying at least as challenging?
  2. model impact — how much of the per-run score variance is explained by the
                  generation model (eta^2, between/within decomposition)?
  3. mode impact  — paired forward-vs-inverse deltas for generation models that
                  ran in both modes.

Emits figures + analysis.md + analysis.json into --out
(default: <first-root>/analysis).

Usage:
    python -m scripts.analyze_results [ROOT ...] [--out DIR]
"""
import argparse
import glob
import json
import math
import os
import textwrap
from collections import defaultdict
from datetime import datetime

from framework.plotting.plots import flatten_mean_std, flatten_point, _visible

# ── Fixed entity colors (validated: Okabe-Ito subset, all-pairs CVD-checked;
#    pink/amber/sky carry a contrast WARN whose relief is the direct labels on
#    the figures and the tables in analysis.md). Real stays the repo's orange
#    and only ever appears as a dashed, directly-labeled reference. ──
MODEL_COLORS = {
    "minimax-m3": "#0072B2",
    "z-ai/glm-5.2": "#009E73",
    "xiaomi/mimo-v2.5": "#CC79A7",
    "tencent/hy3:free": "#E69F00",
    "deepseek/deepseek-v4-flash": "#56B4E9",
}
FALLBACK_COLOR = "#52514e"
MODE_MARKERS = {"forward": "o", "inverse": "^",
                "forward+seedless": "D", "inverse+seedless": "v"}
HEADLINE = {"spam": "f1", "gec": "errant.f0.5"}
IDENTITY_METRICS = {
    "spam": ["accuracy", "precision", "recall", "f1"],
    "gec": ["gleu", "errant.f0.5", "errant.precision", "errant.recall", "errant_dist"],
}


# ── Statistics ────────────────────────────────────────────────

def eta_squared(groups):
    """Between-group share of total variance (eta^2) for lists of per-run
    values grouped by generation model. None when undefined (fewer than two
    groups, or zero total variance)."""
    groups = [g for g in groups if g]
    if len(groups) < 2:
        return None
    values = [v for g in groups for v in g]
    grand = sum(values) / len(values)
    ss_total = sum((v - grand) ** 2 for v in values)
    if ss_total == 0:
        return None
    ss_between = sum(len(g) * ((sum(g) / len(g)) - grand) ** 2 for g in groups)
    return ss_between / ss_total


def kendall_tau_b(x, y):
    """Kendall tau-b (tie-corrected) between two equally long score lists.
    None when either list is constant."""
    n = len(x)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = (x[i] - x[j]) * (y[i] - y[j])
            if a > 0:
                concordant += 1
            elif a < 0:
                discordant += 1

    def _tie_term(vals):
        counts = defaultdict(int)
        for v in vals:
            counts[v] += 1
        return sum(c * (c - 1) / 2 for c in counts.values())

    n0 = n * (n - 1) / 2
    n1, n2 = _tie_term(x), _tie_term(y)
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    if denom == 0:
        return None
    return (concordant - discordant) / denom


# ── Strategy grouping ─────────────────────────────────────────

def _strategy_of(meta: dict) -> str:
    """Generation-cell label for grouping: mode plus the seedless flag.
    Derived rather than stored, so sessions written before seedless existed
    (no `seedless` key) group correctly as their plain mode.

    `mode` defaults the same way the writers (`_build_meta` / `_run_generation`)
    do: a session that has no `mode` at all — including archived spam sessions,
    where the old `_build_meta` forced `"mode": null` — resolves per its task
    shape (`meta["strategy"]`): "inverse" for class_conditional, "forward"
    otherwise. This is required so a legacy null-mode session lines up with a
    freshly re-run session of the identical config (which now records its
    mode explicitly) instead of forming its own stray "-" bucket that dedup
    never supersedes and that drops out of single-model-group plots. Legacy
    sessions lacking BOTH `mode` and `strategy` fall back to "forward", the
    historical default.

    Note: this is NOT `meta["strategy"]` — that key already means the task's
    generation shape ("corruption" / "class_conditional") and is untouched
    here."""
    mode = meta.get("mode") or ("inverse" if meta.get("strategy") == "class_conditional" else "forward")
    return f"{mode}+seedless" if meta.get("seedless") else mode


# ── Session discovery ─────────────────────────────────────────

def discover_sessions(roots):
    sessions = []
    for root in roots:
        for results_path in sorted(glob.glob(os.path.join(root, "**", "results.json"),
                                             recursive=True)):
            session_dir = os.path.dirname(results_path)
            if os.path.basename(session_dir) == "comparison":
                continue
            with open(results_path, encoding="utf-8") as f:
                data = json.load(f)
            if "meta" not in data or "results" not in data:
                continue
            profile_path = os.path.join(session_dir, "profile.json")
            profile = None
            if os.path.exists(profile_path):
                with open(profile_path, encoding="utf-8") as f:
                    profile = json.load(f)
            sessions.append({"dir": session_dir, "meta": data["meta"],
                             "results": data["results"], "profile": profile})
    return sessions


def dedup_sessions(sessions):
    """One session per (task, strategy, generation model): most completed runs
    wins, then the newest. Returns (kept, dropped)."""
    best = {}
    for s in sessions:
        m = s["meta"]
        key = (m["task"], _strategy_of(m), m["model"])
        rank = (m.get("runs_completed", 0), m.get("created", ""))
        if key not in best or rank > (best[key]["meta"].get("runs_completed", 0),
                                      best[key]["meta"].get("created", "")):
            best[key] = s
    kept = sorted(best.values(), key=lambda s: (s["meta"]["task"],
                                                _strategy_of(s["meta"]),
                                                s["meta"]["model"]))
    dropped = [s for s in sessions if s not in kept]
    return kept, dropped


# ── Tidy extraction ───────────────────────────────────────────

def session_rows(session):
    """Flatten one session into rows:
    {task, strategy, gen_model, eval_model, metric, gen_mean, gen_std, real, runs}."""
    meta = session["meta"]
    rows = []
    for eval_model, blocks in session["results"].items():
        gen = flatten_mean_std(blocks.get("generated") or {})
        real = flatten_point(blocks.get("real") or {})
        per_run = [flatten_point(r) for r in blocks.get("runs") or []]
        for metric in _visible(gen.keys()):
            mean, std = gen[metric]
            rows.append({
                "task": meta["task"], "strategy": _strategy_of(meta),
                "gen_model": meta["model"], "eval_model": eval_model,
                "metric": metric, "gen_mean": mean, "gen_std": std,
                "real": real.get(metric),
                "runs": [r[metric] for r in per_run if metric in r],
            })
    return rows


def _short(name):
    return name.split("/")[-1]


def _color(gen_model):
    return MODEL_COLORS.get(gen_model, FALLBACK_COLOR)


# ── Figures ───────────────────────────────────────────────────

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_identity(rows, task, out_dir):
    """Generated vs real scatter with the y=x fidelity diagonal, one panel per
    metric. Color = generation model, marker = strategy, error bar = run std."""
    from framework.plotting.style import INK_MUTED, SURFACE, apply_axes_style
    plt = _plt()
    metrics = [m for m in IDENTITY_METRICS[task]
               if any(r["metric"] == m and r["real"] is not None for r in rows)]
    if not metrics:
        return None
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.8),
                             sharex=True, sharey=True)
    axes = [axes] if len(metrics) == 1 else list(axes)
    fig.patch.set_facecolor(SURFACE)
    for ax, metric in zip(axes, metrics):
        apply_axes_style(ax)
        ax.plot([0, 1], [0, 1], linestyle="--", color=INK_MUTED, linewidth=1, zorder=1)
        for r in rows:
            if r["metric"] != metric or r["real"] is None:
                continue
            ax.errorbar(r["real"], r["gen_mean"], yerr=r["gen_std"],
                        marker=MODE_MARKERS.get(r["strategy"], "s"), linestyle="none",
                        markersize=6, color=_color(r["gen_model"]),
                        markeredgecolor="white", markeredgewidth=0.5,
                        elinewidth=1, capsize=2, zorder=3)
        ax.set_title(metric, fontsize=10)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("real score", fontsize=9)
    axes[0].set_ylabel("generated score", fontsize=9)
    # Only show the strategy legend/marker split when the task actually has
    # more than one distinct strategy present (a single-strategy task gets one
    # marker shape, encoded by color alone — same treatment spam gets today).
    has_strategy_axis = len({r["strategy"] for r in rows}) > 1
    _legend_models_modes(fig, rows, modes=has_strategy_axis)
    fig.suptitle(f"{task}: generated vs real (diagonal = perfect fidelity; "
                 f"below = generated harder)", fontsize=11, y=1.04)
    return _save(fig, os.path.join(out_dir, f"identity_{task}.png"))


def plot_model_impact(rows, task, strategy, out_dir):
    """Per-run headline scores by generation model (position encodes the model),
    real baseline as dashed reference, eta^2 annotated per evaluated model."""
    from framework.plotting.style import (INK, SERIES_REAL, SURFACE,
                                          apply_axes_style)
    plt = _plt()
    metric = HEADLINE[task]
    sel = [r for r in rows if r["task"] == task and r["strategy"] == strategy
           and r["metric"] == metric and r["runs"]]
    eval_models = sorted({r["eval_model"] for r in sel})
    gen_models = sorted({r["gen_model"] for r in sel})
    if len(gen_models) < 2:
        return None
    fig, axes = plt.subplots(1, len(eval_models),
                             figsize=(0.62 * len(gen_models) * len(eval_models) + 2.5, 3.6),
                             sharey=True)
    axes = [axes] if len(eval_models) == 1 else list(axes)
    fig.patch.set_facecolor(SURFACE)
    for ax, ev in zip(axes, eval_models):
        apply_axes_style(ax)
        by_model = {r["gen_model"]: r for r in sel if r["eval_model"] == ev}
        groups = []
        for i, gm in enumerate(gen_models):
            r = by_model.get(gm)
            if r is None:
                continue
            groups.append(r["runs"])
            ax.scatter([i] * len(r["runs"]), r["runs"], s=28, zorder=3,
                       color=_color(gm), edgecolor="white", linewidth=0.5)
            mean = sum(r["runs"]) / len(r["runs"])
            ax.hlines(mean, i - 0.22, i + 0.22, color=_color(gm), linewidth=2, zorder=4)
            if r["real"] is not None:
                ax.axhline(r["real"], linestyle="--", color=SERIES_REAL,
                           linewidth=1, zorder=2)
        e2 = eta_squared(groups)
        title = "\n".join(textwrap.wrap(_short(ev), 24))
        if e2 is not None:
            title += f"\nη² = {e2:.2f}"
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xticks(range(len(gen_models)))
        ax.set_xticklabels([_short(m) for m in gen_models], rotation=30,
                           ha="right", fontsize=8)
    axes[0].set_ylabel(metric, fontsize=9)
    _legend_models_modes(fig, sel, modes=False, real_line=True)
    label = f"{task} {strategy}" if strategy != "-" else task
    suffix = f"_{strategy}" if strategy != "-" else ""
    fig.suptitle(f"{label}: per-run {metric} by generation model "
                 f"(dashed = real benchmark)", fontsize=11, y=1.04)
    return _save(fig, os.path.join(out_dir, f"model_impact_{task}{suffix}.png"))


def plot_mode_effect(rows, task, out_dir):
    """Paired forward -> inverse slopes per generation model that ran both modes."""
    from framework.plotting.style import SERIES_REAL, SURFACE, apply_axes_style
    plt = _plt()
    metric = HEADLINE[task]
    sel = [r for r in rows if r["task"] == task and r["metric"] == metric]
    both = sorted({m for m in {r["gen_model"] for r in sel}
                   if {"forward", "inverse"} <=
                   {r["strategy"] for r in sel if r["gen_model"] == m}})
    if not both:
        return None
    eval_models = sorted({r["eval_model"] for r in sel})
    fig, axes = plt.subplots(1, len(eval_models),
                             figsize=(2.6 * len(eval_models) + 1.5, 3.4), sharey=True)
    axes = [axes] if len(eval_models) == 1 else list(axes)
    fig.patch.set_facecolor(SURFACE)
    for ax, ev in zip(axes, eval_models):
        apply_axes_style(ax)
        for gm in both:
            pts = {r["strategy"]: r for r in sel
                   if r["eval_model"] == ev and r["gen_model"] == gm}
            # Membership, not len(pts) < 2: with seedless variants in the mix,
            # pts can hold >=2 keys (e.g. "inverse" + "inverse+seedless")
            # without actually pairing forward with inverse.
            if not {"forward", "inverse"} <= set(pts):
                continue
            xs, ys = [0, 1], [pts["forward"]["gen_mean"], pts["inverse"]["gen_mean"]]
            ax.errorbar(xs, ys, yerr=[pts["forward"]["gen_std"], pts["inverse"]["gen_std"]],
                        marker="o", markersize=5, linewidth=2, capsize=2,
                        color=_color(gm), markeredgecolor="white", markeredgewidth=0.5)
            real = pts["forward"]["real"]
            if real is not None:
                ax.axhline(real, linestyle="--", color=SERIES_REAL, linewidth=1, zorder=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["forward", "inverse"], fontsize=9)
        ax.set_xlim(-0.35, 1.35)
        ax.set_title("\n".join(textwrap.wrap(_short(ev), 24)), fontsize=8)
    axes[0].set_ylabel(metric, fontsize=9)
    _legend_models_modes(fig, [r for r in sel if r["gen_model"] in both],
                         modes=False, real_line=True)
    fig.suptitle(f"{task}: mode effect on {metric} (dashed = real benchmark)",
                 fontsize=11, y=1.04)
    return _save(fig, os.path.join(out_dir, f"mode_effect_{task}.png"))


def plot_fidelity_jsd(sessions, task, out_dir):
    """Distribution fidelity per session: error-type and count JSDs (0 = the
    generated dataset's error mix matches the real benchmark exactly)."""
    from framework.plotting.style import SURFACE, apply_axes_style
    plt = _plt()
    keys = ("type_dist_jsd", "count_dist_jsd")
    sel = [s for s in sessions if s["meta"]["task"] == task
           and (s.get("profile") or {}).get("fidelity")]
    sel = [s for s in sel if all(k in s["profile"]["fidelity"] for k in keys)]
    if not sel:
        return None
    sel.sort(key=lambda s: (_strategy_of(s["meta"]), s["meta"]["model"]))
    labels = [
        (f"{_strategy_of(s['meta'])}\n{_short(s['meta']['model'])}"
         if _strategy_of(s["meta"]) != "-" else _short(s["meta"]["model"]))
        for s in sel
    ]
    fig, axes = plt.subplots(1, 2, figsize=(1.1 * len(sel) + 5, 3.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, key in zip(axes, keys):
        apply_axes_style(ax)
        for i, s in enumerate(sel):
            ax.bar(i, s["profile"]["fidelity"][key], width=0.62,
                   color=_color(s["meta"]["model"]), edgecolor=SURFACE, linewidth=1)
        ax.set_xticks(range(len(sel)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(key.replace("_", " "), fontsize=10)
    axes[0].set_ylabel("Jensen-Shannon divergence", fontsize=9)
    fig.suptitle(f"{task}: distribution fidelity (lower = generated errors "
                 f"distributed like real ones)", fontsize=11, y=1.04)
    return _save(fig, os.path.join(out_dir, f"fidelity_{task}.png"))


def _legend_models_modes(fig, rows, modes=True, real_line=False):
    from matplotlib.lines import Line2D
    from framework.plotting.style import INK_MUTED, SERIES_REAL
    handles = []
    for gm in sorted({r["gen_model"] for r in rows}):
        handles.append(Line2D([], [], marker="s", linestyle="none", markersize=7,
                              color=_color(gm), label=_short(gm)))
    if modes:
        for mode, mk in MODE_MARKERS.items():
            handles.append(Line2D([], [], marker=mk, linestyle="none", markersize=6,
                                  color=INK_MUTED, label=mode))
    if real_line:
        handles.append(Line2D([], [], linestyle="--", linewidth=1,
                              color=SERIES_REAL, label="real benchmark"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 0.92),
               frameon=False, fontsize=8)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    _plt().close(fig)
    return path


# ── Report ────────────────────────────────────────────────────

def rank_preservation(rows, task, strategy, gen_model):
    """Kendall tau-b between real and generated eval-model orderings on the
    task's headline metric, plus the orderings themselves."""
    metric = HEADLINE[task]
    sel = [r for r in rows if (r["task"], r["strategy"], r["gen_model"], r["metric"])
           == (task, strategy, gen_model, metric) and r["real"] is not None]
    if len(sel) < 2:
        return None
    sel.sort(key=lambda r: r["eval_model"])
    tau = kendall_tau_b([r["real"] for r in sel], [r["gen_mean"] for r in sel])
    return {
        "tau_b": tau,
        "n_eval_models": len(sel),
        "real_order": [_short(r["eval_model"])
                       for r in sorted(sel, key=lambda r: -r["real"])],
        "generated_order": [_short(r["eval_model"])
                            for r in sorted(sel, key=lambda r: -r["gen_mean"])],
    }


def build_summary(sessions, rows):
    summary = {"created": datetime.now().isoformat(timespec="seconds"),
               "sessions": [], "fidelity_gap": {}, "rank_preservation": {},
               "model_impact": {}, "mode_effect": {}}
    for s in sessions:
        m = s["meta"]
        summary["sessions"].append({
            "dir": s["dir"], "task": m["task"], "strategy": _strategy_of(m),
            "gen_model": m["model"], "runs": m.get("runs_completed"),
            "has_per_run_scores": any(b.get("runs") for b in s["results"].values()),
            "has_profile": s.get("profile") is not None,
        })

    configs = sorted({(r["task"], r["strategy"], r["gen_model"]) for r in rows})
    for task, strategy, gm in configs:
        key = f"{task}/{strategy}/{gm}"
        headline = HEADLINE[task]
        sel = [r for r in rows if (r["task"], r["strategy"], r["gen_model"]) ==
               (task, strategy, gm) and r["real"] is not None]
        if sel:
            gaps = [abs(r["gen_mean"] - r["real"]) for r in sel]
            head = [r["gen_mean"] - r["real"] for r in sel if r["metric"] == headline]
            summary["fidelity_gap"][key] = {
                "mean_abs_gap_all_metrics": round(sum(gaps) / len(gaps), 4),
                "headline_gap_mean": (round(sum(head) / len(head), 4) if head else None),
                "harder_than_real": (sum(head) / len(head) < 0) if head else None,
            }
        rp = rank_preservation(rows, task, strategy, gm)
        if rp:
            summary["rank_preservation"][key] = rp

    for task in sorted({r["task"] for r in rows}):
        for strategy in sorted({r["strategy"] for r in rows if r["task"] == task}):
            metric = HEADLINE[task]
            sel = [r for r in rows if (r["task"], r["strategy"], r["metric"]) ==
                   (task, strategy, metric) and r["runs"]]
            per_eval = {}
            for ev in sorted({r["eval_model"] for r in sel}):
                groups = [r["runs"] for r in sel if r["eval_model"] == ev]
                e2 = eta_squared(groups)
                if e2 is None:
                    continue
                means = [sum(g) / len(g) for g in groups]
                stds = [(sum((v - sum(g) / len(g)) ** 2 for v in g) / len(g)) ** 0.5
                        for g in groups]
                per_eval[_short(ev)] = {
                    "eta_sq": round(e2, 3),
                    "between_model_range": round(max(means) - min(means), 4),
                    "mean_within_model_std": round(sum(stds) / len(stds), 4),
                    "n_gen_models": len(groups),
                }
            if per_eval:
                summary["model_impact"][f"{task}/{strategy}"] = per_eval

        both = sorted({m for m in {r["gen_model"] for r in rows if r["task"] == task}
                       if {"forward", "inverse"} <=
                       {r["strategy"] for r in rows
                        if r["task"] == task and r["gen_model"] == m}})
        for gm in both:
            metric = HEADLINE[task]
            deltas = {}
            for ev in sorted({r["eval_model"] for r in rows if r["task"] == task}):
                pts = {r["strategy"]: r["gen_mean"] for r in rows
                       if (r["task"], r["gen_model"], r["eval_model"], r["metric"]) ==
                       (task, gm, ev, metric)}
                if {"forward", "inverse"} <= set(pts):
                    deltas[_short(ev)] = round(pts["inverse"] - pts["forward"], 4)
            if deltas:
                summary["mode_effect"][f"{task}/{gm}"] = {
                    "headline_metric": metric,
                    "inverse_minus_forward": deltas,
                }
    return summary


def write_markdown(summary, sessions, rows, figures, out_path):
    lines = ["# Generated-vs-real benchmark analysis",
             f"_Generated {summary['created']} by scripts/analyze_results.py_", ""]

    lines += ["## Run inventory", "",
              "| task | strategy | generation model | runs | per-run scores | profile |",
              "|---|---|---|---|---|---|"]
    for s in summary["sessions"]:
        lines.append(f"| {s['task']} | {s['strategy']} | {s['gen_model']} | {s['runs']} "
                     f"| {'yes' if s['has_per_run_scores'] else 'no'} "
                     f"| {'yes' if s['has_profile'] else 'no'} |")

    lines += ["", "## Headline scores (generated vs real)", ""]
    for task in sorted({r["task"] for r in rows}):
        metric = HEADLINE[task]
        lines += [f"### {task} — {metric}", "",
                  "| strategy | generation model | evaluated model | generated | real | gap |",
                  "|---|---|---|---|---|---|"]
        sel = [r for r in rows if r["task"] == task and r["metric"] == metric]
        for r in sorted(sel, key=lambda r: (r["strategy"], r["gen_model"], r["eval_model"])):
            real = f"{r['real']:.3f}" if r["real"] is not None else "-"
            gap = (f"{r['gen_mean'] - r['real']:+.3f}" if r["real"] is not None else "-")
            lines.append(f"| {r['strategy']} | {_short(r['gen_model'])} "
                         f"| {_short(r['eval_model'])} "
                         f"| {r['gen_mean']:.3f} ± {r['gen_std']:.3f} | {real} | {gap} |")
        lines.append("")

    lines += ["## Fidelity gaps & rank preservation", "",
              "| config | mean abs gap (all metrics) | headline gap | harder than real | tau-b | generated order |",
              "|---|---|---|---|---|---|"]
    for key in sorted(summary["fidelity_gap"]):
        f = summary["fidelity_gap"][key]
        rp = summary["rank_preservation"].get(key) or {}
        tau = rp.get("tau_b")
        lines.append(f"| {key} | {f['mean_abs_gap_all_metrics']} "
                     f"| {f['headline_gap_mean']} "
                     f"| {'yes' if f['harder_than_real'] else 'no'} "
                     f"| {tau if tau is None else round(tau, 2)} "
                     f"| {' > '.join(rp.get('generated_order', [])) or '-'} |")
    reals = {}
    for key, rp in summary["rank_preservation"].items():
        reals[" > ".join(rp["real_order"])] = rp["n_eval_models"]
    for order in reals:
        lines.append(f"\nReal-benchmark order: **{order}**")

    lines += ["", "## Model impact (eta^2 on per-run headline scores)", ""]
    for cfg, per_eval in summary["model_impact"].items():
        lines += [f"### {cfg}", "",
                  "| evaluated model | eta^2 | between-model range | within-model std | n models |",
                  "|---|---|---|---|---|"]
        for ev, d in per_eval.items():
            lines.append(f"| {ev} | {d['eta_sq']} | {d['between_model_range']} "
                         f"| {d['mean_within_model_std']} | {d['n_gen_models']} |")
        lines.append("")

    lines += ["## Mode effect (inverse - forward, headline metric)", ""]
    for cfg, d in summary["mode_effect"].items():
        deltas = ", ".join(f"{k}: {v:+.3f}" for k, v in d["inverse_minus_forward"].items())
        lines.append(f"- **{cfg}** ({d['headline_metric']}): {deltas}")

    lines += ["", "## Figures", ""]
    lines += [f"- {os.path.basename(p)}" for p in figures if p]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Cross-session generated-vs-real analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("roots", nargs="*",
                        default=["/srv/code/data/team_project/results"],
                        help="Results roots to scan recursively for sessions")
    parser.add_argument("--out", help="Output dir (default: <first-root>/analysis)")
    args = parser.parse_args()

    sessions = discover_sessions(args.roots)
    kept, dropped = dedup_sessions(sessions)
    for s in dropped:
        print(f"[dedup] ignoring {s['dir']} "
              f"(superseded for {s['meta']['task']}/{_strategy_of(s['meta'])}/"
              f"{s['meta']['model']})")
    rows = [r for s in kept for r in session_rows(s)]
    if not rows:
        raise SystemExit("no sessions found under: " + ", ".join(args.roots))

    out_dir = args.out or os.path.join(args.roots[0], "analysis")
    os.makedirs(out_dir, exist_ok=True)

    figures = []
    tasks = sorted({r["task"] for r in rows})
    for task in tasks:
        task_rows = [r for r in rows if r["task"] == task]
        figures.append(plot_identity(task_rows, task, out_dir))
        for strategy in sorted({r["strategy"] for r in task_rows}):
            figures.append(plot_model_impact(task_rows, task, strategy, out_dir))
        figures.append(plot_mode_effect(task_rows, task, out_dir))
        figures.append(plot_fidelity_jsd(kept, task, out_dir))

    summary = build_summary(kept, rows)
    with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    md = write_markdown(summary, kept, rows, figures, os.path.join(out_dir, "analysis.md"))
    print(f"\nAnalysis written to {out_dir}")
    print(f"  {md}")
    for p in figures:
        if p:
            print(f"  {p}")


if __name__ == "__main__":
    main()
