from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return payload


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["synthetic", "xenium", "anndata"] = "synthetic"
    output_dir: str = "outputs/stgpt"
    dataset_root: str | None = None
    input_h5ad: str | None = None
    spatho_run_root: str | None = None
    patch_manifest: str | None = None
    structure_assignments_csv: str | None = None
    cluster_key: str = "cluster"
    structure_key: str = "structure_id"
    gene_name_key: str = "feature_name"
    spatial_key: str = "spatial"
    include_structure_context: bool = False
    n_cells: int = Field(default=32, ge=2)
    n_genes: int = Field(default=64, ge=4)
    n_structures: int = Field(default=4, ge=1)
    image_size: int = Field(default=64, ge=16)
    seed: int = 0

    def path_or_none(self, value: str | None) -> Path | None:
        if not value:
            return None
        expanded = os.path.expandvars(value)
        if "$" in expanded or "%" in expanded:
            return None
        return Path(expanded).expanduser()

    @property
    def output_path(self) -> Path:
        return Path(os.path.expandvars(self.output_dir)).expanduser()


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    d_model: int = Field(default=128, ge=16)
    n_heads: int = Field(default=4, ge=1)
    n_layers: int = Field(default=2, ge=1)
    dim_feedforward: int | None = None
    max_genes: int = Field(default=1200, ge=4)
    n_expression_bins: int = Field(default=51, ge=2)
    image_size: int = Field(default=224, ge=16)
    image_channels: int = Field(default=3, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, le=0.9)

    @field_validator("n_heads")
    @classmethod
    def _validate_heads(cls, value: int) -> int:
        if value < 1:
            raise ValueError("n_heads must be positive")
        return value


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    max_steps: int = Field(default=1000, ge=1)
    mask_probability: float = Field(default=0.15, gt=0.0, lt=1.0)
    neighborhood_k: int = Field(default=8, ge=1)
    image_gene_loss_weight: float = Field(default=0.1, ge=0.0)
    neighborhood_loss_weight: float = Field(default=0.25, ge=0.0)
    structure_loss_weight: float = Field(default=0.1, ge=0.0)
    output_dir: str = "outputs/stgpt/train"
    device: str = "auto"
    num_workers: int = Field(default=0, ge=0)
    seed: int = 0

    @property
    def output_path(self) -> Path:
        return Path(os.path.expandvars(self.output_dir)).expanduser()


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["spatial_block"] = "spatial_block"
    train_fraction: float = Field(default=0.70, ge=0.0, le=1.0)
    val_fraction: float = Field(default=0.15, ge=0.0, le=1.0)
    test_fraction: float = Field(default=0.15, ge=0.0, le=1.0)
    seed: int | None = None

    @model_validator(mode="after")
    def _validate_fraction_sum(self) -> SplitConfig:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError("split fractions must sum to 1.0")
        if self.train_fraction <= 0.0:
            raise ValueError("split.train_fraction must be positive")
        return self


class StGPTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_name: str = "stgpt_case"
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)

    @classmethod
    def from_file(cls, path: str | Path, *, preset: str | None = None) -> StGPTConfig:
        config_path = Path(path).expanduser().resolve()
        payload = _expand_env(_load_mapping(config_path))
        cfg = cls.model_validate(payload)
        return cfg.apply_preset(preset)

    def apply_preset(self, preset: str | None) -> StGPTConfig:
        if preset is None:
            return self
        normalized = str(preset).strip().lower()
        payload = self.model_dump()
        if normalized == "smoke":
            payload["training"]["device"] = "cpu"
            payload["training"]["num_workers"] = 0
            payload["training"]["max_steps"] = min(int(payload["training"]["max_steps"]), 2)
            payload["model"]["d_model"] = min(int(payload["model"]["d_model"]), 64)
            payload["model"]["n_layers"] = min(int(payload["model"]["n_layers"]), 2)
        elif normalized == "pdc":
            payload["training"]["device"] = "cuda"
            payload["training"]["num_workers"] = max(int(payload["training"]["num_workers"]), 2)
        else:
            raise ValueError("preset must be one of: smoke, pdc")
        return type(self).model_validate(payload)

    def resolved_output_dir(self) -> Path:
        return self.data.output_path

    def to_json_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())
