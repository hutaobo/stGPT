from __future__ import annotations

import json
from pathlib import Path

from stgpt.config import DataConfig, ModelConfig, SplitConfig, StGPTConfig, TrainingConfig
from stgpt.evidence import evaluate, validate_data
from stgpt.foundation import ImageGeneSTGPT, embed_anndata, package_model, train
from stgpt.runtime import embed_cells, embed_regions, evaluate_checkpoint, export_spatho_artifacts


def _config(tmp_path: Path) -> StGPTConfig:
    return StGPTConfig(
        case_name="runtime_case",
        data=DataConfig(
            mode="synthetic",
            output_dir=str(tmp_path / "case"),
            n_cells=10,
            n_genes=14,
            n_structures=2,
            image_size=32,
            seed=22,
        ),
        model=ModelConfig(d_model=32, n_heads=4, n_layers=1, max_genes=8, image_size=32, n_expression_bins=8),
        training=TrainingConfig(batch_size=5, max_steps=1, output_dir=str(tmp_path / "train"), device="cpu", seed=23),
        split=SplitConfig(seed=24),
    )


def test_namespace_imports_are_public() -> None:
    assert ImageGeneSTGPT is not None
    assert train is not None
    assert embed_anndata is not None
    assert package_model is not None
    assert validate_data is not None
    assert evaluate is not None
    assert embed_cells is not None
    assert embed_regions is not None
    assert evaluate_checkpoint is not None
    assert export_spatho_artifacts is not None


def test_runtime_wraps_synthetic_evidence_pipeline(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    qc = validate_data(cfg, output_dir=tmp_path / "qc")
    checkpoint = Path(train(cfg, preset="smoke", max_steps=1)["checkpoint"])
    evaluation = evaluate_checkpoint(
        checkpoint=checkpoint,
        config=cfg,
        splits=qc["splits"],
        output_dir=tmp_path / "eval",
        batch_size=5,
        device="cpu",
    )
    package = package_model(
        checkpoint=checkpoint,
        evaluation=evaluation["artifacts"]["evaluation_metrics"],
        output_dir=tmp_path / "model",
        model_name="runtime-synthetic-stgpt",
    )
    assert Path(package["manifest"]).exists()
    manifest = json.loads(Path(package["manifest"]).read_text(encoding="utf-8"))
    assert manifest["training_provenance"]["split"]["counts"]["train"] > 0
    assert "domain_counts" in manifest["training_provenance"]
    embedded = export_spatho_artifacts(
        config=cfg,
        checkpoint=tmp_path / "model",
        output_dir=tmp_path / "spatho",
        batch_size=5,
        device="cpu",
    )
    assert Path(embedded["cell_embeddings"]).exists()
    assert Path(embedded["region_embeddings"]).exists()
    assert Path(embedded["structure_summary"]).exists()
    assert Path(embedded["structure_embedding_summary"]).exists()
    assert Path(embedded["qc_report"]).exists()
    deprecated = embed_cells(
        config=cfg,
        checkpoint=tmp_path / "model",
        output_dir=tmp_path / "spatho_deprecated",
        batch_size=5,
        device="cpu",
    )
    assert "deprecated" in deprecated
    assert Path(deprecated["region_embeddings"]).exists()
