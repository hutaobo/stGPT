"""Smoke tests for the region auto-annotation runtime."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from stgpt.annotation import ABSTAIN_LABEL, annotate_regions
from stgpt.cli import app
from stgpt.config import DataConfig, ModelConfig, SplitConfig, StGPTConfig, TrainingConfig
from stgpt.runtime import annotate_regions as runtime_annotate_regions
from stgpt.spatho import run_spatho_export
from stgpt.training import train


def _small_config(tmp_path: Path) -> StGPTConfig:
    return StGPTConfig(
        case_name="annotation_smoke",
        data=DataConfig(
            mode="synthetic",
            output_dir=str(tmp_path / "case"),
            n_cells=24,
            n_genes=16,
            n_structures=3,
            image_size=32,
            seed=11,
        ),
        model=ModelConfig(
            d_model=32,
            n_heads=4,
            n_layers=1,
            max_genes=12,
            image_size=32,
            n_expression_bins=8,
            n_prototypes=3,
        ),
        training=TrainingConfig(batch_size=4, max_steps=2, output_dir=str(tmp_path / "train"), device="cpu", seed=7),
        split=SplitConfig(seed=3),
    )


def _train_and_seed(tmp_path: Path) -> tuple[StGPTConfig, Path, Path]:
    cfg = _small_config(tmp_path)
    checkpoint = Path(train(cfg, preset="smoke", max_steps=2)["checkpoint"])
    spatho_out = tmp_path / "spatho"
    export = run_spatho_export(cfg, checkpoint=checkpoint, output_dir=spatho_out, batch_size=4, device="cpu")
    regions = pd.read_parquet(export.region_embeddings)
    # Pick 60% of regions per class as seeds so each class has >= 2 seeds for k-fold.
    seed_rows_list = []
    for _, group in regions.groupby("structure_label", sort=True):
        keep = max(2, int(len(group) * 0.6))
        seed_rows_list.append(group.head(keep)[["region_id", "structure_label"]])
    seeds = pd.concat(seed_rows_list, ignore_index=True)
    seeds["confidence"] = 1.0
    seeds_path = tmp_path / "seeds.csv"
    seeds.to_csv(seeds_path, index=False)
    return cfg, checkpoint, seeds_path


def test_annotate_regions_writes_predictions(tmp_path: Path) -> None:
    cfg, checkpoint, seeds_path = _train_and_seed(tmp_path)
    out_dir = tmp_path / "annotate"
    result = annotate_regions(
        config=cfg,
        checkpoint=checkpoint,
        seed_labels=seeds_path,
        output_dir=out_dir,
        write_probabilities=True,
        batch_size=4,
        device="cpu",
    )
    predictions_path = Path(result["predictions"])
    report_path = Path(result["report"])
    probs_path = Path(result["probabilities"])
    agreement_path = Path(result["path_agreement"])
    assert predictions_path.exists()
    assert report_path.exists()
    assert probs_path.exists()
    assert agreement_path.exists()

    predictions = pd.read_parquet(predictions_path)
    expected_cols = {
        "region_id",
        "predicted_label",
        "predicted_prob",
        "entropy",
        "nearest_seed_region_id",
        "nearest_seed_distance",
        "qc_flag",
        "classifier",
        "evidence_id",
        "propagation_kind",
        "expression_present",
    }
    assert expected_cols.issubset(set(predictions.columns))

    seeds = pd.read_csv(seeds_path)
    seed_rows = predictions[predictions["region_id"].isin(seeds["region_id"])]
    assert (seed_rows["qc_flag"] == "seed").all()
    assert set(seed_rows["predicted_label"].tolist()) == set(seeds["structure_label"].tolist())
    assert (seed_rows["predicted_prob"] == 1.0).all()

    pool_rows = predictions[predictions["qc_flag"] != "seed"]
    assert len(pool_rows) > 0
    valid = pool_rows["predicted_label"]
    allowed_labels = set(seeds["structure_label"].tolist()) | {ABSTAIN_LABEL}
    assert set(valid.tolist()).issubset(allowed_labels)
    assert (pool_rows["predicted_prob"] >= 0.0).all()
    assert (pool_rows["predicted_prob"] <= 1.0 + 1e-6).all()
    assert (pool_rows["propagation_kind"] == "same_slide").all()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"].startswith("stgpt.region_auto_annotation")
    assert report["n_seed_regions"] == len(seeds)
    assert report["n_pool_regions"] == len(pool_rows)
    assert report["primary_classifier"] in {"structure_head", "prototype_knn"}
    assert "label_vocab" in report
    assert "seed_cross_validation" in report
    assert isinstance(report["warnings"], list)


def test_annotate_regions_unknown_region_id_raises(tmp_path: Path) -> None:
    cfg, checkpoint, seeds_path = _train_and_seed(tmp_path)
    bad_seeds = tmp_path / "bad_seeds.csv"
    pd.DataFrame({
        "region_id": ["definitely_not_a_real_region"],
        "structure_label": ["structure_0"],
        "confidence": [1.0],
    }).to_csv(bad_seeds, index=False)
    with pytest.raises(ValueError, match="unknown region_ids"):
        annotate_regions(
            config=cfg,
            checkpoint=checkpoint,
            seed_labels=bad_seeds,
            output_dir=tmp_path / "bad",
            batch_size=4,
            device="cpu",
        )


def test_annotate_regions_runtime_alias_matches(tmp_path: Path) -> None:
    cfg, checkpoint, seeds_path = _train_and_seed(tmp_path)
    direct = annotate_regions(
        config=cfg,
        checkpoint=checkpoint,
        seed_labels=seeds_path,
        output_dir=tmp_path / "direct",
        batch_size=4,
        device="cpu",
    )
    runtime_result = runtime_annotate_regions(
        config=cfg,
        checkpoint=checkpoint,
        seed_labels=seeds_path,
        output_dir=tmp_path / "runtime",
        batch_size=4,
        device="cpu",
    )
    assert direct.keys() == runtime_result.keys()
    direct_pred = pd.read_parquet(direct["predictions"]).sort_values("region_id").reset_index(drop=True)
    runtime_pred = pd.read_parquet(runtime_result["predictions"]).sort_values("region_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(direct_pred, runtime_pred)


def test_annotate_regions_cli(tmp_path: Path) -> None:
    cfg, checkpoint, seeds_path = _train_and_seed(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
case_name: {cfg.case_name}
data:
  mode: synthetic
  output_dir: {cfg.data.output_dir}
  n_cells: {cfg.data.n_cells}
  n_genes: {cfg.data.n_genes}
  n_structures: {cfg.data.n_structures}
  image_size: {cfg.data.image_size}
  seed: {cfg.data.seed}
model:
  d_model: {cfg.model.d_model}
  n_heads: {cfg.model.n_heads}
  n_layers: {cfg.model.n_layers}
  max_genes: {cfg.model.max_genes}
  image_size: {cfg.model.image_size}
  n_expression_bins: {cfg.model.n_expression_bins}
  n_prototypes: {cfg.model.n_prototypes}
training:
  batch_size: {cfg.training.batch_size}
  max_steps: {cfg.training.max_steps}
  output_dir: {cfg.training.output_dir}
  device: cpu
  seed: {cfg.training.seed}
split:
  seed: {cfg.split.seed}
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "annotate-regions",
            "--config", str(config_path),
            "--checkpoint", str(checkpoint),
            "--seed-labels", str(seeds_path),
            "--output", str(tmp_path / "cli"),
            "--classifier", "both",
            "--batch-size", "4",
            "--device", "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert Path(payload["predictions"]).exists()
    assert Path(payload["report"]).exists()
