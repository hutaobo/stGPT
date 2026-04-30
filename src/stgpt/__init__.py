from __future__ import annotations

from .config import AblationMode, DataConfig, ModelConfig, SplitConfig, StGPTConfig, TrainingConfig
from .data import build_training_manifest, load_xenium_case
from .evaluation import evaluate
from .inference import embed_anndata
from .models import ImageGeneSTGPT
from .qc import validate_data, validate_training_case
from .spatho import SpathoStGPTModel, embed_spatho_case, package_model, write_spatho_artifacts
from .training import train

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AblationMode",
    "DataConfig",
    "ImageGeneSTGPT",
    "ModelConfig",
    "SpathoStGPTModel",
    "SplitConfig",
    "StGPTConfig",
    "TrainingConfig",
    "build_training_manifest",
    "embed_anndata",
    "evaluate",
    "embed_spatho_case",
    "load_xenium_case",
    "package_model",
    "train",
    "validate_data",
    "validate_training_case",
    "write_spatho_artifacts",
]
