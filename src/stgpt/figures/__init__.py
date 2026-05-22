"""Publication figures for stGPT (optional ``figures`` extra).

This subpackage turns the evidence layer's exported artifacts into journal-ready
figures. It depends on matplotlib, which is an optional extra; install with::

    pip install -e ".[figures]"

Importing this package without matplotlib raises ImportError by design -- the
core stGPT runtime never imports it, so the foundation/evidence/runtime layers
stay dependency-light.

Design contract (mirrors ``stgpt.evidence``):

* artifact-first -- read parquet/csv, never train or mutate checkpoints;
* provenance-carrying -- every figure writes a ``.provenance.json`` sidecar;
* reproducible -- coordinates and hashes are reused from the evidence layer,
  not re-derived here.
"""

from __future__ import annotations

from .manifold import plot_cross_platform_manifold
from .style import OKABE_ITO, apply_style, categorical_color_map

__all__ = [
    "OKABE_ITO",
    "apply_style",
    "categorical_color_map",
    "plot_cross_platform_manifold",
]
