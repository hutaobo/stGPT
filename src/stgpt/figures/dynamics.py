"""F3 figure: learning dynamics for the 43-case structure-context comparison."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from collections.abc import Mapping, Sequence  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import _io  # noqa: E402
from ._layout import place_panel_label  # noqa: E402
from .export import save_figure  # noqa: E402
from .style import DOUBLE_COLUMN_IN, OKABE_ITO, apply_style  # noqa: E402

DEFAULT_43CASE_RUN_IDS: tuple[str, ...] = (
    "gene_spatial_contour_unit_20k",
    "full_m6_contour_store_lambda_0_01_20k",
    "structure_context_m6_20k",
)

DEFAULT_RUN_LABELS: dict[str, str] = {
    "gene_spatial_contour_unit_20k": "Gene+spatial",
    "full_m6_contour_store_lambda_0_01_20k": "Full M6",
    "structure_context_m6_20k": "Structure context",
}

DEFAULT_DYNAMICS_METRICS: tuple[tuple[str, str], ...] = (
    ("val_gene_loss", "Validation gene loss"),
    ("alignment_score", "Alignment score"),
    ("image_to_gene_top5", "Image→gene top-5"),
    ("gene_to_image_top5", "Gene→image top-5"),
)


def plot_learning_dynamics(
    learning_dynamics: str | Path | pd.DataFrame,
    output_dir: str | Path,
    *,
    name: str = "f3_learning_dynamics",
    run_ids: Sequence[str] = DEFAULT_43CASE_RUN_IDS,
    run_labels: Mapping[str, str] | None = None,
    metrics: Sequence[tuple[str, str]] | None = None,
    formats: Sequence[str] = ("pdf", "png"),
    title: str | None = None,
) -> dict[str, Any]:
    """Render 43-case learning dynamics from ``learning_dynamics.csv``.

    The function is artifact-first: it reads only the evidence-summary dynamics
    table and does not recompute metrics or touch checkpoints.
    """
    frame = _io.load_table(learning_dynamics)
    source = str(frame.attrs.get("source", "<dataframe>"))
    warnings: list[str] = []
    required = {"run_id", "step"}
    missing_required = sorted(required - set(frame.columns))
    if missing_required:
        raise ValueError(f"learning dynamics table is missing required columns: {missing_required}")

    selected = frame[frame["run_id"].astype(str).isin([str(item) for item in run_ids])].copy()
    missing_runs = [str(item) for item in run_ids if str(item) not in set(selected["run_id"].astype(str))]
    if missing_runs:
        warnings.append(f"missing_runs: {missing_runs}")
    if selected.empty:
        raise ValueError("no learning-dynamics rows match the requested run_ids")

    requested = list(metrics) if metrics is not None else list(DEFAULT_DYNAMICS_METRICS)
    available = [(col, label) for col, label in requested if col in selected.columns]
    missing_metrics = [col for col, _ in requested if col not in selected.columns]
    if missing_metrics:
        warnings.append(f"missing_metrics: {missing_metrics}")
    if not available:
        raise ValueError(f"none of the requested metrics are present: {[col for col, _ in requested]}")

    labels = {**DEFAULT_RUN_LABELS, **(run_labels or {})}
    ordered_run_ids = [str(item) for item in run_ids if str(item) in set(selected["run_id"].astype(str))]
    color_map = {run_id: OKABE_ITO[index % len(OKABE_ITO)] for index, run_id in enumerate(ordered_run_ids)}

    apply_style()
    n_metrics = len(available)
    ncols = min(n_metrics, 2)
    nrows = (n_metrics + ncols - 1) // ncols
    width = DOUBLE_COLUMN_IN * 0.78
    height = (width / ncols) * nrows * 0.82
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    flat_axes = [ax for row in axes for ax in row]

    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    for index, (metric, ylabel) in enumerate(available):
        ax = flat_axes[index]
        for run_id in ordered_run_ids:
            subset = selected[selected["run_id"].astype(str) == run_id].sort_values("step")
            values = pd.to_numeric(subset[metric], errors="coerce")
            line = ax.plot(
                pd.to_numeric(subset["step"], errors="coerce"),
                values,
                color=color_map[run_id],
                linewidth=1.2,
                label=labels.get(run_id, run_id),
            )[0]
            if index == 0:
                legend_handles.append(line)
                legend_labels.append(labels.get(run_id, run_id))
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        if metric == "val_gene_loss":
            ax.set_yscale("log")
        place_panel_label(ax, chr(ord("A") + index))

    for ax in flat_axes[n_metrics:]:
        ax.set_axis_off()
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            title="Run",
            alignment="left",
        )
    if title:
        fig.suptitle(title, fontsize=9, fontweight="bold")

    status = "warning" if missing_runs or missing_metrics else "pass"
    provenance = {
        "figure": "F3_learning_dynamics",
        "source_learning_dynamics": source,
        "run_ids": ordered_run_ids,
        "missing_runs": missing_runs,
        "metrics": [col for col, _ in available],
        "missing_metrics": missing_metrics,
        "step_min": int(pd.to_numeric(selected["step"], errors="coerce").min()),
        "step_max": int(pd.to_numeric(selected["step"], errors="coerce").max()),
        "n_rows": int(len(selected)),
        "palette": "okabe_ito",
        "warnings": warnings,
        "status": status,
    }
    artifacts = save_figure(fig, output_dir, name, formats=formats, provenance=provenance)
    plt.close(fig)

    return {
        "status": status,
        "n_rows": int(len(selected)),
        "run_ids": ordered_run_ids,
        "metrics": [col for col, _ in available],
        "warnings": warnings,
        "artifacts": artifacts,
    }
