"""Figure export with a provenance sidecar.

Every saved figure is accompanied by a ``<name>.provenance.json`` recording the
source artifact, checkpoint hashes, columns used, and stGPT version, so a
published figure remains auditable and reproducible.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._io import json_safe

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# DPI is irrelevant for pure vector formats but caps embedded rasters.
_VECTOR_FORMATS = {"pdf", "eps", "svg"}


def save_figure(
    fig: Figure,
    output_dir: str | Path,
    name: str,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 600,
    provenance: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save ``fig`` in each requested format plus a provenance sidecar.

    Returns a mapping of artifact key -> path string (e.g. ``{"pdf": ...,
    "png": ..., "provenance": ...}``).
    """
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for fmt in formats:
        target = out / f"{name}.{fmt}"
        save_dpi = min(dpi, 300) if fmt in _VECTOR_FORMATS else dpi
        fig.savefig(target, format=fmt, dpi=save_dpi)
        artifacts[fmt] = str(target)

    payload = dict(provenance or {})
    payload.setdefault("name", name)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["stgpt_version"] = _stgpt_version()
    payload["formats"] = list(formats)
    sidecar = out / f"{name}.provenance.json"
    sidecar.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    artifacts["provenance"] = str(sidecar)
    return artifacts


def _stgpt_version() -> str:
    try:
        from .. import __version__

        return str(__version__)
    except Exception:
        return "unknown"
