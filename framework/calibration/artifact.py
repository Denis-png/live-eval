"""On-disk format for calibration artifacts.

Lives beside benchmark profiles in framework/data/profiles/<task>/ and is
gitignored the same way: expensive to produce, reusable across every session,
tied to a (benchmark, cell) pair rather than to any one run.
"""

from __future__ import annotations

import json
import os

_INT_KEYED = ("count_dist",)


def calibration_filename(config: dict, task_name: str, strategy: str,
                         num_samples: int) -> str:
    """<benchmark>_<n>_<cell>_calibration.json.

    The "_calibration.json" suffix cannot be matched by _resolve_benchmark_profile_path's
    "*_<task>_profile.json" glob, so the two artifact kinds share a directory
    without colliding.
    """
    from framework.pipeline import benchmark_slug, generation_cell_slug

    cell = generation_cell_slug(config, strategy)
    return f"{benchmark_slug(config)}_{num_samples}_{cell}_calibration.json"


def default_calibration_path(config: dict, task_name: str, strategy: str,
                             num_samples: int) -> str:
    from framework.pipeline import benchmark_profile_dir

    return os.path.join(
        benchmark_profile_dir(task_name),
        calibration_filename(config, task_name, strategy, num_samples),
    )


def resolve_calibration_path(config: dict, task, strategy: str) -> str | None:
    """The calibration artifact a run should use, or None when there is none.

    `generation.calibration_path` wins when set; an explicit null disables the
    lookup entirely (the documented opt-out). Otherwise the task's profile
    directory is searched for this benchmark and cell — the sample size is
    baked into the filename and a run cannot predict it, so this matches rather
    than computes. Several matches resolve to the newest, since a later
    calibration of the same cell supersedes an earlier one.
    """
    import glob

    from framework.pipeline import benchmark_slug, generation_cell_slug, benchmark_profile_dir

    gen = config.get("generation") or {}
    if "calibration_path" in gen:
        return gen["calibration_path"] or None

    pattern = os.path.join(
        benchmark_profile_dir(task.get_task_name()),
        f"{benchmark_slug(config)}_*_{generation_cell_slug(config, strategy)}"
        "_calibration.json",
    )
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return matches[-1] if matches else None


def write_calibration(path: str, payload: dict) -> str:
    """Write the artifact, creating parent directories. Called after EVERY round
    so a killed calibration still leaves a usable best-so-far."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def _coerce_int_keys(block: dict) -> dict:
    """Restore int keys that JSON serialization turned into strings."""
    out = dict(block or {})
    for name in _INT_KEYED:
        dist = out.get(name)
        if isinstance(dist, dict):
            out[name] = {int(k): v for k, v in dist.items()}
    return out


def load_calibration(path: str) -> dict:
    """Load an artifact with `count_dist` keys coerced back to int.

    JSON has no integer keys. Left as strings, `_sample_categories` draws a str
    as the edit count and raises TypeError on `n > len(keys)` deep inside the
    generation loop, and only for calibrated runs.
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    for section in ("target", "calibrated"):
        if isinstance(payload.get(section), dict):
            payload[section] = _coerce_int_keys(payload[section])
    return payload


def targets_match(a: dict, b: dict, *, atol: float = 1e-6) -> bool:
    """Whether two setpoints are the same distribution, within float noise.

    A mismatch means the benchmark or its sample size changed since calibration:
    the artifact is stale and must not be used silently.
    """
    for name in ("type_dist", "count_dist"):
        pa, pb = (a or {}).get(name) or {}, (b or {}).get(name) or {}
        if set(pa) != set(pb):
            return False
        for key, value in pa.items():
            if abs(float(value) - float(pb[key])) > atol:
                return False
    return True
