"""Validated chart palette + shared matplotlib chrome.

The two-series palette was selected by running the data-viz validator on the
light surface: all six checks PASS, worst adjacent CVD ΔE 96.7, both hues >= 3:1
contrast (so no relief rule is owed). Colour follows the entity — generated is
always blue, real is always orange, in every figure.

PNG is a static medium and cannot be theme-aware, so we deliberately commit to
the light surface only.
"""
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e2"
SERIES_GENERATED = "#2a78d6"  # blue
SERIES_REAL = "#eb6834"       # orange


def apply_axes_style(ax) -> None:
    """Recessive chrome: solid hairline y-grid behind the marks, no top/right
    spines, ticks and labels in muted ink (never a series colour)."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, length=0)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_color(INK_MUTED)
