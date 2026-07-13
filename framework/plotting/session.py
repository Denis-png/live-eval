"""Session I/O for plotting: read a run session's artifacts, render every
applicable figure, save PNGs. All fail-soft policy lives here.

matplotlib is imported lazily (via _import_plots) so importing this module — or
the pipeline that calls it — never pulls in a plotting stack.
"""
import json
import os
import re
import sys

_FIG_DPI = 150


def _slug(text: str) -> str:
    """Filesystem-safe lowercase slug. Defined locally on purpose: a framework/
    module must not import from scripts/ (compare_models has the same one-liner)."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()


def _import_plots():
    """Lazy import of the matplotlib-backed builders (patched in tests)."""
    from framework.plotting import plots
    return plots


def load_session(session_dir: str) -> tuple[dict, dict | None]:
    """Return (results payload, profile or None). Raises ValueError naming the path
    when results.json is missing or unreadable — the CLI has no run to protect."""
    results_path = os.path.join(session_dir, "results.json")
    if not os.path.isfile(results_path):
        raise ValueError(
            f"No results.json in '{session_dir}' — point this at a run session "
            f"directory (e.g. framework/data/runs/<task>/<session>/)."
        )
    try:
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse '{results_path}': {e}") from e

    profile = None
    profile_path = os.path.join(session_dir, "profile.json")
    if os.path.isfile(profile_path):
        try:
            with open(profile_path, encoding="utf-8") as f:
                profile = json.load(f)
        except json.JSONDecodeError:
            print(f"[WARN] ignoring unparseable {profile_path}", file=sys.stderr)
    return results, profile


def _save(fig, path: str, plt) -> str:
    fig.savefig(path, dpi=_FIG_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def render_session(session_dir: str, out_dir: str | None = None) -> list[str]:
    """Render every applicable figure into out_dir (default <session_dir>/plots/).

    Fail-soft: a missing matplotlib, or any single figure blowing up, warns and is
    skipped — plotting must never cost a run whose results are already on disk.
    Returns the paths actually written."""
    results, profile = load_session(session_dir)
    try:
        plots = _import_plots()
    except ImportError:
        print("[WARN] matplotlib not installed — skipping plots "
              "(pip install matplotlib)", file=sys.stderr)
        return []
    import matplotlib.pyplot as plt

    out_dir = out_dir or os.path.join(session_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    meta = results.get("meta") or {}
    written: list[str] = []

    for model, blocks in (results.get("results") or {}).items():
        slug = _slug(model)
        jobs = [(
            f"generated_vs_real_{slug}.png",
            lambda m=model, b=blocks: plots.plot_generated_vs_real(
                m, b.get("generated") or {}, b.get("real"), meta),
        )]
        if blocks.get("runs"):
            jobs.append((
                f"run_variance_{slug}.png",
                lambda m=model, b=blocks: plots.plot_run_variance(
                    m, b["runs"], b.get("generated"), meta),
            ))
        for filename, build in jobs:
            try:
                written.append(_save(build(), os.path.join(out_dir, filename), plt))
            except Exception as e:  # one bad figure must not sink the rest
                print(f"[WARN] could not render {filename}: {e}", file=sys.stderr)

    if profile:
        try:
            written.append(_save(plots.plot_fidelity(profile, meta),
                                 os.path.join(out_dir, "fidelity.png"), plt))
        except Exception as e:
            print(f"[WARN] could not render fidelity.png: {e}", file=sys.stderr)

    if written:
        print(f"Plots written to {out_dir} ({len(written)} figures)")
    return written
