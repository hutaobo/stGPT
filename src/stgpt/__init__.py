from __future__ import annotations

from .config import AblationMode, DataConfig, ModelConfig, SplitConfig, StGPTConfig, TrainingConfig
from .curated_spatial import (
    CuratedSpatialPrior,
    audit_curated_structures,
    predict_curated_spatial_prior,
    train_curated_spatial_prior,
)
from .data import build_training_manifest, load_xenium_case
from .evaluation import evaluate
from .foundation import package_model
from .inference import embed_anndata
from .models import ImageGeneSTGPT
from .pseudo_spatial import PseudoSpatialPrior, predict_pseudo_spatial, train_pseudo_spatial_prior
from .qc import validate_data, validate_training_case
from .runtime import embed_cells, embed_regions, evaluate_checkpoint, export_spatho_artifacts
from .spatho import (
    CELL_EMBEDDING_REQUIRED_COLUMNS,
    REGION_EMBEDDING_REQUIRED_COLUMNS,
    STRUCTURE_SUMMARY_REQUIRED_COLUMNS,
    PatchManifestRow,
    SpathoExportResult,
    run_spatho_export,
)
from .training import train

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AblationMode",
    "CELL_EMBEDDING_REQUIRED_COLUMNS",
    "CuratedSpatialPrior",
    "DataConfig",
    "ImageGeneSTGPT",
    "ModelConfig",
    "PatchManifestRow",
    "PseudoSpatialPrior",
    "REGION_EMBEDDING_REQUIRED_COLUMNS",
    "SplitConfig",
    "SpathoExportResult",
    "STRUCTURE_SUMMARY_REQUIRED_COLUMNS",
    "StGPTConfig",
    "TrainingConfig",
    "audit_curated_structures",
    "build_training_manifest",
    "embed_anndata",
    "embed_cells",
    "embed_regions",
    "evaluate",
    "evaluate_checkpoint",
    "export_spatho_artifacts",
    "load_xenium_case",
    "package_model",
    "predict_pseudo_spatial",
    "predict_curated_spatial_prior",
    "run_spatho_export",
    "train",
    "train_curated_spatial_prior",
    "train_pseudo_spatial_prior",
    "validate_data",
    "validate_training_case",
]
