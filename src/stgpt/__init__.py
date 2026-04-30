from __future__ import annotations

from .config import AblationMode, DataConfig, ModelConfig, SplitConfig, StGPTConfig, TrainingConfig
from .data import build_training_manifest, load_xenium_case
from .evaluation import evaluate
from .foundation import package_model
from .inference import embed_anndata
from .models import ImageGeneSTGPT
from .qc import validate_data, validate_training_case
from .runtime import embed_cells, evaluate_checkpoint, export_spatho_artifacts
from .spatho import (
    CELL_EMBEDDING_REQUIRED_COLUMNS,
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
    "DataConfig",
    "ImageGeneSTGPT",
    "ModelConfig",
    "PatchManifestRow",
    "SplitConfig",
    "SpathoExportResult",
    "STRUCTURE_SUMMARY_REQUIRED_COLUMNS",
    "StGPTConfig",
    "TrainingConfig",
    "build_training_manifest",
    "embed_anndata",
    "embed_cells",
    "evaluate",
    "evaluate_checkpoint",
    "export_spatho_artifacts",
    "load_xenium_case",
    "package_model",
    "run_spatho_export",
    "train",
    "validate_data",
    "validate_training_case",
]
