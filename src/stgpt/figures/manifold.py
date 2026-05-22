"""F1 hero figure: the cross-platform / cross-modal aligned region space.

The central stGPT claim is that contour/region embeddings are reusable across
measured ST platforms. This figure documents that with three panels:

* A: the latent manifold coloured by a batch/platform axis -> points should mix.
* B: the same manifold coloured by structure label -> biology should separate.
* C: quantitative support -- batch-mixing entropy (higher = better mixing) and
  structure silhouette (higher = better separation).

A good result is *both* high mixing (A, C) and clear structure (B, C). Panels A
and B carry the message even when the optional metric CSVs for panel C are
absent.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # figure generation writes files; never needs a display

from collections.abc import Sequence  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import _io  # noqa: E402
from ._layout import compact_legend, place_panel_label  # noqa: E402
from .export import save_figure  # noqa: E402
from .style import DOUBLE_COLUMN_IN, OKABE_ITO, apply_style, categorical_color_map  # noqa: E402


def plot_cross_platform_manifold(
    manifold: str | Path | pd.DataFrame,
    output_dir: str | Path,
    *,
    name: str = "f1_cross_platform_manifold",
    batch_key: str = "auto",
    structure_key: str = "structure_label",
    run_id: str | None = None,
    batch_mixing_csv: str | Path | None = None,
    embedding_qc_csv: str | Path | None = None,
    formats: Sequence[str] = ("pdf", "png"),
    point_size: float = 4.0,
    max_points: int = 50000,
    seed: int = 0,
    title: str | None = None,
) -> dict[str, Any]:
    """Render the F1 cross-platform manifold figure from evidence artifacts.

    Parameters mirror the evidence layer's artifact-first contract: pass a
    projected ``latent_manifold.csv`` (or DataFrame) plus, optionally, the
    ``batch_mixing_metrics.csv`` and ``embedding_qc.csv`` that quantify panel C.

    Returns a summary dict with ``artifacts`` (figure + provenance paths),
    ``status`` (``pass``/``warning``), ``warnings``, and the resolved keys.
    """
    frame = _io.load_manifold_frame(manifold)
    source = str(frame.attrs.get("source", "<dataframe>"))
    warnings: list[str] = []

    if run_id is not None:
        if "run_id" not in frame.columns:
            raise ValueError("run_id filter requested but the manifold frame has no 'run_id' column")
        frame = frame[frame["run_id"].astype(str) == str(run_id)].copy()
        if frame.empty:
            raise ValueError(f"no rows match run_id='{run_id}'")

    resolved_batch_key, batch_warnings = _io.resolve_batch_key(frame, batch_key)
    warnings.extend(batch_warnings)
    checkpoint_hashes, hash_warnings = _io.checkpoint_guardrail(frame, run_id)
    warnings.extend(hash_warnings)

    if structure_key not in frame.columns:
        warnings.append(f"missing_structure_key: '{structure_key}' not found; panel B coloured uniformly")

    frame, was_sampled = _io.subsample(frame, max_points=max_points, seed=seed)
    if was_sampled:
        warnings.append(f"subsampled_to_{max_points}_points")

    mixing_metrics = _read_optional_csv(batch_mixing_csv)
    qc_metrics = _read_optional_csv(embedding_qc_csv)
    has_panel_c = mixing_metrics is not None or qc_metrics is not None

    apply_style()
    n_panels = 3 if has_panel_c else 2
    width = DOUBLE_COLUMN_IN * 0.78
    height = width / (2.8 if has_panel_c else 2.2)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(width, height),
        constrained_layout={"w_pad": 0.01, "h_pad": 0.01, "wspace": 0.02},
    )
    axes = list(axes)

    _scatter_panel(axes[0], frame, resolved_batch_key, "A", point_size)
    structure_col = structure_key if structure_key in frame.columns else None
    _scatter_panel(axes[1], frame, structure_col, "B", point_size)
    if has_panel_c:
        _metric_panel(axes[2], mixing_metrics, qc_metrics, "C")

    if title:
        fig.suptitle(title, fontsize=9, fontweight="bold")

    status = "warning" if any(_is_blocking(item) for item in warnings) else "pass"
    provenance = {
        "figure": "F1_cross_platform_manifold",
        "source_manifold": source,
        "batch_mixing_csv": str(batch_mixing_csv) if batch_mixing_csv else None,
        "embedding_qc_csv": str(embedding_qc_csv) if embedding_qc_csv else None,
        "n_points": int(len(frame)),
        "batch_key": resolved_batch_key,
        "structure_key": structure_key,
        "run_id": run_id,
        "checkpoint_hashes": checkpoint_hashes,
        "palette": "okabe_ito",
        "warnings": warnings,
        "status": status,
    }
    artifacts = save_figure(fig, output_dir, name, formats=formats, provenance=provenance)
    plt.close(fig)

    return {
        "status": status,
        "n_points": int(len(frame)),
        "batch_key": resolved_batch_key,
        "structure_key": structure_key,
        "checkpoint_hashes": checkpoint_hashes,
        "warnings": warnings,
        "artifacts": artifacts,
    }


def _pretty_label(key: str) -> str:
    """Turn a snake_case column name into a readable legend title."""
    return key.replace("_", " ").title()


def _scatter_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    color_key: str | None,
    tag: str,
    point_size: float,
) -> None:
    x = frame["manifold_x"].to_numpy()
    y = frame["manifold_y"].to_numpy()
    legend_title = _pretty_label(color_key) if color_key else None
    if color_key and color_key in frame.columns:
        categories = frame[color_key].fillna("unknown").astype(str)
        color_map = categorical_color_map(categories.tolist())
        if len(color_map) > len(OKABE_ITO):
            # Cycling colors reuses hues; flag it so the caller can react.
            ax.figure.stale = True
        colors = categories.map(color_map)
        ax.scatter(x, y, s=point_size, c=list(colors), linewidths=0.0, alpha=0.7, rasterized=True)
        compact_legend(ax, color_map, title=legend_title)
    else:
        ax.scatter(x, y, s=point_size, c="#4b5563", linewidths=0.0, alpha=0.7, rasterized=True)

    ax.set_box_aspect(1)  # force square axes box
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    place_panel_label(ax, tag)


def _metric_panel(
    ax: plt.Axes,
    mixing_metrics: pd.DataFrame | None,
    qc_metrics: pd.DataFrame | None,
    tag: str,
) -> None:
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    if mixing_metrics is not None and "batch_mixing_entropy" in mixing_metrics.columns:
        value = float(mixing_metrics["batch_mixing_entropy"].astype(float).mean())
        labels.append("batch mixing\nentropy")
        values.append(value)
        colors.append(OKABE_ITO[4])  # blue
    if qc_metrics is not None and "silhouette" in qc_metrics.columns:
        value = float(qc_metrics["silhouette"].astype(float).mean())
        labels.append("structure\nsilhouette")
        values.append(value)
        colors.append(OKABE_ITO[2])  # bluish green

    if not values:
        ax.text(0.5, 0.5, "no metric CSV", ha="center", va="center", fontsize=7)
        ax.set_axis_off()
        return

    positions = range(len(values))
    ax.bar(positions, values, color=colors, width=0.6)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("metric value")
    place_panel_label(ax, tag)
    for pos, value in zip(positions, values, strict=False):
        offset = 0.02 if value >= 0 else -0.04
        ax.text(pos, value + offset, f"{value:.3g}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)


def _read_optional_csv(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.exists():
        return None
    try:
        return pd.read_csv(resolved)
    except Exception:
        return None


def _is_blocking(warning: str) -> bool:
    # Subsampling and trivial single-group panels are informational, not failures.
    return not warning.startswith(("subsampled_to_", "no_multi_value_batch_key"))
