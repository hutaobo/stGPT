"""F2 figure: the ablation / condition comparison.

stGPT's evidence suite trains a small grid of conditions per tissue — a random-
init control, a gene+spatial ablation (no image-gene alignment), the full M6
contour-store model, and a zero-shot cross-tissue transfer. This figure reads
the evidence layer's ``evidence_summary.csv`` and lays the conditions side by
side across the headline evaluation metrics, one panel per metric.

It is artifact-first like F1: it never recomputes a metric, only reads the
exported summary, and writes a ``.provenance.json`` sidecar recording exactly
which rows and columns produced each bar.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # figure generation writes files; never needs a display

import re  # noqa: E402
from collections.abc import Mapping, Sequence  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import _io  # noqa: E402
from ._layout import place_panel_label  # noqa: E402
from .export import save_figure  # noqa: E402
from .style import DOUBLE_COLUMN_IN, apply_style, categorical_color_map  # noqa: E402

# Default metrics: (summary column, axis label). These are the headline
# evaluation outputs of the evidence suite.
DEFAULT_METRICS: tuple[tuple[str, str], ...] = (
    ("eval_gene_correlation", "Gene correlation"),
    ("eval_label_retrieval_top5", "Label retrieval (top-5)"),
    ("eval_silhouette_mean", "Silhouette"),
    ("eval_image_to_gene_top5", "Image→gene (top-5)"),
)

# Long evidence-suite condition names -> short bar labels. Unknown conditions
# fall back to a cleaned version of the raw string.
DEFAULT_CONDITION_LABELS: dict[str, str] = {
    "Full M6 contour-store random init": "Random init",
    "Contour-unit Gene+Spatial 500-step": "Gene+spatial",
    "Full M6 Zarr contour store": "Full M6",
    "Zero-shot Cervical→Breast Full M6": "Zero-shot",
    "Zero-shot Breast→Cervical Full M6": "Zero-shot",
    "Gene-only baseline": "Gene-only",
    "Full M6 PNG fallback": "M6 PNG",
    "L3 Full M6 contour-store lambda=0.01, slide-holdout, 20k steps": "Full M6",
    "L3 Gene+Spatial contour-unit baseline, slide-holdout, 20k steps": "Gene+spatial",
    "L3 Structure-context M6 contour-store, slide-holdout, 20k steps": "Structure context",
}

# Logical ablation reading order (floor -> full model -> transfer).
DEFAULT_CONDITION_ORDER: tuple[str, ...] = (
    "Gene-only",
    "M6 PNG",
    "Random init",
    "Gene+spatial",
    "Full M6",
    "Structure context",
    "Zero-shot",
)


def plot_ablation_comparison(
    summary: str | Path | pd.DataFrame,
    output_dir: str | Path,
    *,
    name: str = "f2_ablation_comparison",
    metrics: Sequence[tuple[str, str]] | None = None,
    condition_key: str = "condition",
    group_key: str = "tissue",
    run_ids: Sequence[str] | None = None,
    condition_labels: Mapping[str, str] | None = None,
    condition_order: Sequence[str] | None = None,
    formats: Sequence[str] = ("pdf", "png"),
    title: str | None = None,
) -> dict[str, Any]:
    """Render the F2 ablation comparison from an evidence summary table.

    Parameters mirror F1's artifact-first contract: pass the evidence layer's
    ``evidence_summary.csv`` (or a DataFrame). Each metric becomes one bar
    panel; bars are grouped by ``group_key`` (tissue) and coloured by the
    training ``condition``.

    Returns a summary dict with ``artifacts`` (figure + provenance paths),
    ``status`` (``pass``/``warning``), ``warnings``, and the resolved keys.
    """
    frame = _io.load_table(summary)
    source = str(frame.attrs.get("source", "<dataframe>"))
    warnings: list[str] = []

    requested = list(metrics) if metrics is not None else list(DEFAULT_METRICS)
    available = [(col, label) for col, label in requested if col in frame.columns]
    missing = [col for col, _ in requested if col not in frame.columns]
    if missing:
        warnings.append(f"missing_metrics: {missing}")
    if not available:
        raise ValueError(
            "none of the requested metric columns are present in the summary; "
            f"looked for {[c for c, _ in requested]}"
        )

    if condition_key not in frame.columns:
        raise ValueError(f"condition_key '{condition_key}' is not a column in the summary")
    if group_key not in frame.columns:
        raise ValueError(f"group_key '{group_key}' is not a column in the summary")
    if run_ids is not None:
        if "run_id" not in frame.columns:
            raise ValueError("run_ids filter requested but the summary has no 'run_id' column")
        requested_ids = [str(item) for item in run_ids]
        frame = frame[frame["run_id"].astype(str).isin(requested_ids)].copy()
        if frame.empty:
            raise ValueError("no rows in the summary match the requested run_ids")

    label_map = {**DEFAULT_CONDITION_LABELS, **(condition_labels or {})}
    frame = frame.copy()
    frame["_condition_short"] = frame[condition_key].map(lambda value: label_map.get(value, _clean_condition(value)))

    # Keep only rows that carry at least one finite value among the metrics —
    # this drops baselines without an evaluation block, so the x-axis stays honest.
    metric_cols = [col for col, _ in available]
    has_value = frame[metric_cols].apply(lambda row: row.notna().any(), axis=1)
    dropped = int((~has_value).sum())
    frame = frame[has_value]
    if dropped:
        warnings.append(f"dropped_{dropped}_rows_without_eval_metrics")
    if frame.empty:
        raise ValueError("no rows in the summary carry finite values for the requested metrics")

    conditions = _order_conditions(frame["_condition_short"].unique().tolist(), condition_order)
    color_map = categorical_color_map(conditions)

    apply_style()
    n_metrics = len(available)
    ncols = min(n_metrics, 2)
    nrows = (n_metrics + ncols - 1) // ncols
    width = DOUBLE_COLUMN_IN * 0.78
    panel = width / ncols
    height = panel * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    flat_axes = [ax for row in axes for ax in row]

    legend_handles: list[Any] = []
    for index, (col, label) in enumerate(available):
        ax = flat_axes[index]
        tag = chr(ord("A") + index)
        handles = _grouped_bars(ax, frame, metric=col, group_key=group_key, conditions=conditions, color_map=color_map)
        ax.set_ylabel(label)
        place_panel_label(ax, tag)
        if not legend_handles:
            legend_handles = handles

    for ax in flat_axes[n_metrics:]:
        ax.set_axis_off()

    # One shared condition legend to the right of the whole grid.
    if legend_handles:
        fig.legend(
            legend_handles,
            conditions,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            title="Condition",
            alignment="left",
            handletextpad=0.3,
            labelspacing=0.3,
        )

    if title:
        fig.suptitle(title, fontsize=9, fontweight="bold")

    status = "warning" if any(_is_blocking(item) for item in warnings) else "pass"
    provenance = {
        "figure": "F2_ablation_comparison",
        "source_summary": source,
        "metrics": [col for col, _ in available],
        "missing_metrics": missing,
        "condition_key": condition_key,
        "group_key": group_key,
        "run_ids": [str(item) for item in run_ids] if run_ids is not None else None,
        "conditions": conditions,
        "groups": sorted(str(value) for value in frame[group_key].dropna().unique()),
        "n_rows": int(len(frame)),
        "palette": "okabe_ito",
        "warnings": warnings,
        "status": status,
    }
    artifacts = save_figure(fig, output_dir, name, formats=formats, provenance=provenance)
    plt.close(fig)

    return {
        "status": status,
        "n_rows": int(len(frame)),
        "metrics": [col for col, _ in available],
        "conditions": conditions,
        "warnings": warnings,
        "artifacts": artifacts,
    }


def _grouped_bars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    metric: str,
    group_key: str,
    conditions: Sequence[str],
    color_map: dict[Any, str],
) -> list[Any]:
    """Draw bars grouped by ``group_key`` (x) and coloured by condition.

    Returns one proxy handle per condition (for a shared legend).
    """
    pivot = frame.pivot_table(index=group_key, columns="_condition_short", values=metric, aggfunc="mean")
    pivot = pivot.reindex(columns=list(conditions))
    groups = [str(value) for value in pivot.index]
    x = np.arange(len(groups), dtype=float)
    n_cond = len(conditions)
    total_width = 0.8
    bar_width = total_width / max(n_cond, 1)

    handles: list[Any] = []
    for cond_index, condition in enumerate(conditions):
        offset = (cond_index - (n_cond - 1) / 2.0) * bar_width
        values = pivot[condition].to_numpy(dtype=float) if condition in pivot.columns else np.full(len(groups), np.nan)
        bars = ax.bar(x + offset, np.nan_to_num(values, nan=0.0), width=bar_width * 0.9, color=color_map[condition])
        handles.append(bars[0])

    ax.axhline(0.0, color="black", linewidth=0.5)  # silhouette can be negative
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.margins(x=0.05)
    return handles


def _clean_condition(value: Any) -> str:
    """Fallback short label for an unknown condition string."""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if len(text) <= 16 else text[:15] + "…"


def _order_conditions(found: Sequence[str], condition_order: Sequence[str] | None) -> list[str]:
    """Order conditions by the requested/canonical order; append unknowns last."""
    order = list(condition_order) if condition_order is not None else list(DEFAULT_CONDITION_ORDER)
    found_set = set(found)
    ordered = [item for item in order if item in found_set]
    ordered.extend(sorted(item for item in found_set if item not in set(order)))
    return ordered


def _is_blocking(warning: str) -> bool:
    # Dropping eval-less baseline rows is expected, not a failure.
    return not warning.startswith("dropped_")
