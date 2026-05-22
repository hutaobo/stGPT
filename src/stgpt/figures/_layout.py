"""Shared layout primitives for stGPT figures.

Centralises the house style the user iterated on so every figure (F1, F2, …)
places panel labels and legends identically:

* panel labels are bold, top-left, left-aligned with the y-axis label;
* legends are single-column, with the title left-aligned to the entry *text*;
* everything sits in a compact, near-zero-pad footprint.

Keeping these here (rather than duplicated per figure) is what guarantees the
"looks like one figure family" consistency the user asked for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.legend import Legend

# Panel labels (A, B, C…) are the one element allowed to exceed the 7pt body size.
PANEL_LABEL_FONTSIZE = 9
# Empirical x-offset (points) that indents a legend title to align with the
# entry text, i.e. past the marker + handletextpad.
_LEGEND_TITLE_INDENT = 17


def place_panel_label(ax: plt.Axes, tag: str) -> None:
    """Put a bold panel label at the top-left, aligned with the y-axis label.

    The label's left edge matches the y-axis label ("UMAP 2", a metric name, …)
    and it sits just inside the top of the axes — never floating above the frame.
    """
    fig = ax.get_figure()
    if fig is None:  # pragma: no cover - an Axes always has a figure
        return
    canvas = fig.canvas
    canvas.draw()  # force layout so the ylabel's position is known
    renderer = canvas.get_renderer()  # type: ignore[attr-defined]
    ylabel_bb = ax.yaxis.label.get_window_extent(renderer)
    ylabel_x = ax.transAxes.inverted().transform((ylabel_bb.x0, 0))[0]
    ax.text(
        ylabel_x, 0.99, tag, transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE, fontweight="bold", va="top", ha="left",
    )


def compact_legend(
    ax: plt.Axes,
    color_map: dict[Any, str],
    *,
    title: str | None = None,
    bbox_to_anchor: tuple[float, float] = (1.01, 1.0),
    max_entries: int | None = 12,
) -> Legend:
    """Draw a single-column, left-aligned legend hugging the axes.

    Very large biological label sets are truncated to keep panel geometry
    stable. The title aligns with the entry text, not the marker. Returns the
    legend so callers can reposition it if needed.
    """
    items = list(color_map.items())
    truncated = max_entries is not None and len(items) > max_entries
    if truncated and max_entries is not None:
        items = items[:max_entries]
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=3, markerfacecolor=color, markeredgewidth=0.0)
        for _, color in items
    ]
    labels = [str(label) for label, _ in items]
    if truncated and max_entries is not None and labels:
        labels[-1] = f"{labels[-1]} (+{len(color_map) - max_entries} more)"
    legend = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=bbox_to_anchor,
        borderaxespad=0.0,
        borderpad=0.0,
        handletextpad=0.3,
        labelspacing=0.15,
        ncol=1,
        title=title,
        alignment="left",
    )
    if title:
        legend.get_title().set_fontsize(7)
        legend.get_title().set_fontweight("normal")
        legend.get_title().set_position((_LEGEND_TITLE_INDENT, 0))
    return legend
