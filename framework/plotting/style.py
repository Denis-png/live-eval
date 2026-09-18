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


LEGEND_KW = dict(frameon=False, labelcolor=INK_MUTED, fontsize=9)


def new_figure(nrows: int = 1, ncols: int = 1, *, figsize, **subplot_kw):
    """(fig, axes) on the chart surface, laid out by matplotlib's constrained
    layout with room around and between the plots.

    Constrained layout, not tight_layout: it is the one that can reserve space
    for a figure-level legend OUTSIDE the axes (figure_legend), which is what
    keeps a legend off the data. pyplot is imported here, not at module level,
    so importing the palette never pulls in a plotting stack."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, layout="constrained",
                             **subplot_kw)
    fig.patch.set_facecolor(SURFACE)
    fig.get_layout_engine().set(w_pad=0.12, h_pad=0.12, wspace=0.06, hspace=0.06)
    return fig, axes


def figure_legend(fig, *axes) -> None:
    """One legend for the whole figure, in a row below the plots, so it can
    never sit on a bar, a point or a value label. Below, not above: constrained
    layout does not stack an outside legend under a two-line suptitle, and the
    two overlapped. Entries come from `axes` in order, one per label. Needs a
    new_figure figure."""
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=len(labels),
                   **LEGEND_KW)


def apply_axes_style(ax, grid_axis: str = "y") -> None:
    """Recessive chrome: solid hairline grid behind the marks (along y, or x for
    horizontal bars), no top/right spines, ticks and labels in muted ink (never
    a series colour)."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, length=0)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_color(INK_MUTED)
