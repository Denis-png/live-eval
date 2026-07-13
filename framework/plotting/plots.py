"""Pure figure builders: plain dicts in, matplotlib Figure out. No I/O.

Importing this module pulls in matplotlib, so it is imported lazily by
session.render_session — the core pipeline never imports it.
"""
import matplotlib

matplotlib.use("Agg")  # headless: no display/GUI backend on the server

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

from framework.plotting.style import (  # noqa: E402
    GRID,
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
