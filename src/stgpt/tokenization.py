from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anndata as ad
import numpy as np


@dataclass(frozen=True)
class GeneVocab:
    genes: tuple[str, ...]
    pad_id: int = 0

    @classmethod
    def from_adata(cls, adata: ad.AnnData, *, gene_name_key: str = "feature_name") -> GeneVocab:
        if gene_name_key in adata.var.columns:
            names = adata.var[gene_name_key].astype(str).tolist()
        elif "gene_name" in adata.var.columns:
            names = adata.var["gene_name"].astype(str).tolist()
        elif "name" in adata.var.columns:
            names = adata.var["name"].astype(str).tolist()
        else:
            names = adata.var_names.astype(str).tolist()
        return cls(tuple(_make_unique(names)))

    @property
    def size(self) -> int:
        return len(self.genes) + 1

    def ids_for_positions(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray(positions, dtype=np.int64) + 1

    def to_dict(self) -> dict[str, Any]:
        return {"genes": list(self.genes), "pad_id": self.pad_id}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GeneVocab:
        return cls(tuple(str(item) for item in payload["genes"]), pad_id=int(payload.get("pad_id", 0)))


def _make_unique(values: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for value in values:
        text = str(value)
        count = seen.get(text, 0)
        seen[text] = count + 1
        out.append(text if count == 0 else f"{text}.{count}")
    return out


@dataclass(frozen=True)
class ExpressionBinner:
    n_bins: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        numeric = np.asarray(values, dtype=np.float32)
        clipped = np.clip(np.log1p(np.maximum(numeric, 0.0)), 0.0, None)
        if clipped.size == 0:
            return np.zeros(0, dtype=np.int64)
        max_value = float(np.nanmax(clipped))
        if max_value <= 0:
            return np.zeros(clipped.shape, dtype=np.int64)
        bins = np.floor((clipped / max_value) * (self.n_bins - 1)).astype(np.int64)
        return np.clip(bins, 0, self.n_bins - 1)
