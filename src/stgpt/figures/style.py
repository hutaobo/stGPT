"""Publication style for stGPT figures.

A self-contained, colorblind-safe (Okabe-Ito) Nature-leaning matplotlib style.
Vendored into the package so figure generation never depends on an external
authoring skill being installed. Adapted from the Okabe-Ito (2008) palette and
Nature single/double-column sizing conventions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

# Okabe-Ito (2008): the most widely recommended colorblind-friendly categorical
# palette. Black is kept last so it is only used when categories exceed 7.
OKABE_ITO: tuple[str, ...] = (
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
)

# Nature column widths in inches (89 mm single, 183 mm double).
SINGLE_COLUMN_IN: float = 89.0 / 25.4
DOUBLE_COLUMN_IN: float = 183.0 / 25.4

# rcParams adapted from the scientific-visualization nature.mplstyle. Kept as a
# dict (not a .mplstyle file) so no package-data plumbing is required.
STGPT_RCPARAMS: dict[str, Any] = {
    "figure.dpi": 100,
    "figure.facecolor": "white",
    "figure.constrained_layout.use": True,
    "font.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.5,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "black",
    "axes.axisbelow": True,
    "axes.grid": False,
    "axes.prop_cycle": None,  # set below from OKABE_ITO
    "xtick.major.size": 2.5,
    "xtick.major.width": 0.5,
    "xtick.labelsize": 7,
    "xtick.direction": "out",
    "ytick.major.size": 2.5,
    "ytick.major.width": 0.5,
    "ytick.labelsize": 7,
    "ytick.direction": "out",
    "lines.linewidth": 1.2,
    "lines.markersize": 3,
    "lines.markeredgewidth": 0.4,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.facecolor": "white",
    "image.cmap": "viridis",
    # Type 42 (TrueType) keeps text as editable text in Adobe Illustrator.
    # The matplotlib default (Type 3) rasterizes glyphs into path outlines,
    # which AI cannot edit as text. Always use 42 for both PDF and EPS.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style() -> None:
    """Apply the stGPT publication style to the active matplotlib session.

    Resets to the matplotlib default first so the style is deterministic
    regardless of any user rcParams or previously applied styles.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.style.use("default")
    params = {key: value for key, value in STGPT_RCPARAMS.items() if value is not None}
    params["axes.prop_cycle"] = cycler(color=list(OKABE_ITO))
    mpl.rcParams.update(params)


def categorical_color_map(categories: Iterable[Any], palette: Sequence[str] = OKABE_ITO) -> dict[Any, str]:
    """Map an ordered, de-duplicated set of categories to palette colors.

    Categories beyond the palette length cycle through it again, so the figure
    still renders; the caller is responsible for warning when this happens.
    """
    seen: list[Any] = []
    for category in categories:
        if category not in seen:
            seen.append(category)
    return {category: palette[index % len(palette)] for index, category in enumerate(seen)}
