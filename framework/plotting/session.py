"""Session I/O for plotting: read a run session's artifacts, render every
applicable figure, save PNGs. All fail-soft policy lives here.

matplotlib is imported lazily (via _import_plots) so importing this module — or
the pipeline that calls it — never pulls in a plotting stack.
"""
import json
import os
import re
import sys

from framework.real_baseline import real_point

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
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"Could not read '{results_path}': {e}") from e

    if not isinstance(results, dict):
        raise ValueError(f"'{results_path}' must contain a JSON object, got {type(results).__name__}.")

    profile = None
    profile_path = os.path.join(session_dir, "profile.json")
    if os.path.isfile(profile_path):
        try:
            with open(profile_path, encoding="utf-8") as f:
                profile = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] ignoring unreadable {profile_path}: {e}", file=sys.stderr)
    return results, profile


def _save(fig, path: str, plt) -> str:
    fig.savefig(path, dpi=_FIG_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# A coded error-type vocabulary is small: the ERRANT codes a run actually uses,
# spam's signals, sentiment's transformations -- at most ~20 in the final sweep.
# A generator that names its own types (gec's forward cells, sentiment's
# forward+seedless) writes a new phrase almost every sample, and one bar per
# phrase is no distribution at all.
MAX_ERROR_TYPES = 30


def _error_type_skip_reason(counts: dict[str, int]) -> str | None:
    """Why error_type_dist.png would say nothing, or None when it can be drawn."""
    if set(counts) == {"unknown"}:
        return "the samples carry no error types"
    if len(counts) > MAX_ERROR_TYPES:
        return (f"{len(counts)} distinct error types -- the generator named them in "
                "free text, so there is no distribution to plot")
    return None


def _load_error_type_counts(session_dir: str) -> dict[str, int]:
    """Tally error types across every run_*.json in <session_dir>/generated/.

    A sample's types are its `error_type`, or `technique` -- the same thing under
    the name class-conditional (spam) records use -- split on commas: one sample
    can carry several (gec's inverse cells join ERRANT codes, spam joins its
    signals), and tallying the joined string made every combination its own
    type."""
    import glob
    counts: dict[str, int] = {}
    # run_<N>_rejected.json shares the directory and the glob, but holds
    # rejected attempts rather than samples.
    run_files = [p for p in glob.glob(os.path.join(session_dir, "generated", "run_*.json"))
                 if re.fullmatch(r"run_\d+\.json", os.path.basename(p))]
    for path in run_files:
        try:
            with open(path, encoding="utf-8") as f:
                items = json.load(f)
            for item in (items or []):
                raw = item.get("error_type") or item.get("technique") or "unknown"
                for et in (t.strip() for t in str(raw).split(",")):
                    if et:
                        counts[et] = counts.get(et, 0) + 1
        except Exception as e:
            print(f"[WARN] could not read {path} for error-type counts: {e}", file=sys.stderr)
    return counts


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
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        print(f"[WARN] could not create output dir '{out_dir}': {e}", file=sys.stderr)
        return []
    meta = results.get("meta") or {}
    written: list[str] = []

    model_results = results.get("results") or {}
    if not isinstance(model_results, dict):
        print(f"[WARN] 'results' is not a dict (got {type(model_results).__name__}) "
              "— skipping per-model figures", file=sys.stderr)
        model_results = {}

    for model, blocks in model_results.items():
        if not isinstance(blocks, dict):
            print(f"[WARN] results for model '{model}' are not a dict "
                  f"(got {type(blocks).__name__}) — skipping", file=sys.stderr)
            continue
        slug = _slug(model)
        jobs = [(
            f"generated_vs_real_{slug}.png",
            lambda m=model, b=blocks: plots.plot_generated_vs_real(
                m, b.get("generated") or {}, real_point(b) or None, meta),
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
        filename = "fidelity.png"
        try:
            fidelity_type = ((profile.get("fidelity") or {}).get("profile_type"))
            if fidelity_type == "taxonomy_structural_fidelity":
                filename = "taxonomy_fidelity.png"
                written.append(_save(
                    plots.plot_taxonomy_fidelity(profile, meta),
                    os.path.join(out_dir, filename), plt,
                ))
                filename = "taxonomy_fidelity_distributions.png"
                written.append(_save(
                    plots.plot_taxonomy_fidelity_distributions(profile, meta),
                    os.path.join(out_dir, filename), plt,
                ))
            elif fidelity_type == "sentiment_fidelity":
                filename = "sentiment_fidelity.png"
                written.append(_save(
                    plots.plot_sentiment_fidelity(profile, meta),
                    os.path.join(out_dir, filename), plt,
                ))
            else:
                written.append(_save(plots.plot_fidelity(profile, meta),
                                     os.path.join(out_dir, filename), plt))
        except Exception as e:
            print(f"[WARN] could not render {filename}: {e}", file=sys.stderr)

    error_counts = _load_error_type_counts(session_dir)
    skip = _error_type_skip_reason(error_counts) if error_counts else None
    if skip:
        print(f"[NOTE] no error_type_dist.png: {skip}", file=sys.stderr)
    elif error_counts:
        # The real signal-type DISTRIBUTION, not signal_rate: a rate is the share
        # of spam messages carrying a signal, multi-label and summing above 1, so
        # scaling it to a count inflated every real bar.
        real_rates = (profile or {}).get("real", {}).get("signal_type_dist") if profile else None
        try:
            written.append(_save(
                plots.plot_error_type_distribution(error_counts, real_rates, meta),
                os.path.join(out_dir, "error_type_dist.png"), plt,
            ))
        except Exception as e:
            print(f"[WARN] could not render error_type_dist.png: {e}", file=sys.stderr)

    if written:
        print(f"Plots written to {out_dir} ({len(written)} figures)")
    return written
