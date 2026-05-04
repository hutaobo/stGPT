from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from typer.testing import CliRunner

from stgpt.cli import app
from stgpt.config import DataConfig, ModelConfig, StGPTConfig, TrainingConfig
from stgpt.data import ImageGeneDataset, build_training_case, make_synthetic_case
from stgpt.image_qc import inspect_images, precompute_image_embeddings
from stgpt.images import write_synthetic_patch
from stgpt.models import ContourEvidenceEncoder, ImageGeneSTGPT, resolve_image_encoder_spec
from stgpt.training import train


def _write_synthetic_config(
    tmp_path: Path,
    *,
    image_embedding_store: Path | None = None,
    image_encoder_backend: str = "cnn",
    image_encoder_preset: str | None = None,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "image.yaml"
    store_line = f"  image_embedding_store: {image_embedding_store.as_posix()}\n" if image_embedding_store else ""
    preset_line = f"  image_encoder_preset: {image_encoder_preset}\n" if image_encoder_preset else ""
    config.write_text(
        f"""
case_name: image_case
data:
  mode: synthetic
  output_dir: {tmp_path.as_posix()}/case
  n_cells: 12
  n_genes: 20
  n_structures: 2
  image_size: 32
{store_line}model:
  d_model: 32
  n_heads: 4
  n_layers: 1
  max_genes: 12
  n_expression_bins: 8
  image_size: 32
  image_encoder_backend: {image_encoder_backend}
{preset_line}  image_embedding_dim: 32
training:
  batch_size: 4
  max_steps: 1
  output_dir: {tmp_path.as_posix()}/train
  device: cpu
  num_workers: 0
""",
        encoding="utf-8",
    )
    return config


def test_model_accepts_precomputed_image_embeddings() -> None:
    model = ImageGeneSTGPT(
        n_genes=12,
        d_model=32,
        n_heads=4,
        n_layers=1,
        n_expression_bins=8,
        image_encoder_backend="precomputed",
        image_embedding_dim=16,
    )
    output = model(
        gene_ids=torch.randint(1, 12, (2, 6)),
        expr_values=torch.rand(2, 6),
        expr_bins=torch.randint(0, 8, (2, 6)),
        image=torch.zeros(2, 3, 32, 32),
        spatial=torch.rand(2, 2),
        precomputed_image_embedding=torch.rand(2, 16),
    )
    assert output.image_emb.shape == (2, 32)
    assert output.region_emb.shape == (2, 32)


def test_mocked_timm_and_hf_image_encoder_backends(monkeypatch) -> None:
    class FakeTimm(torch.nn.Module):
        num_features = 7

        def forward(self, image):
            return torch.ones(image.shape[0], self.num_features, device=image.device)

    timm = types.ModuleType("timm")
    timm.create_model = lambda *args, **kwargs: FakeTimm()
    monkeypatch.setitem(sys.modules, "timm", timm)

    timm_encoder = ContourEvidenceEncoder(3, 16, image_encoder_backend="timm", image_encoder_name="fake")
    _, timm_emb = timm_encoder(object_image=torch.rand(2, 3, 32, 32))
    assert timm_emb.shape == (2, 16)

    class FakeHF(torch.nn.Module):
        config = types.SimpleNamespace(hidden_size=7)

        def forward(self, pixel_values):
            return types.SimpleNamespace(last_hidden_state=torch.ones(pixel_values.shape[0], 1, 7))

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, name):
            return FakeHF()

    transformers = types.ModuleType("transformers")
    transformers.AutoModel = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    hf_encoder = ContourEvidenceEncoder(3, 16, image_encoder_backend="hf", image_encoder_name="fake")
    _, hf_emb = hf_encoder(object_image=torch.rand(2, 3, 32, 32))
    assert hf_emb.shape == (2, 16)


def test_virchow_preset_uses_paige_timm_kwargs(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeTimm(torch.nn.Module):
        num_features = 7

        def forward(self, image):
            return torch.ones(image.shape[0], 257, self.num_features, device=image.device)

    class FakeSwiGLUPacked:
        pass

    timm = types.ModuleType("timm")

    def create_model(name, **kwargs):
        calls.append({"name": name, **kwargs})
        return FakeTimm()

    timm.create_model = create_model
    timm_layers = types.ModuleType("timm.layers")
    timm_layers.SwiGLUPacked = FakeSwiGLUPacked
    monkeypatch.setitem(sys.modules, "timm", timm)
    monkeypatch.setitem(sys.modules, "timm.layers", timm_layers)

    encoder = ContourEvidenceEncoder(
        3,
        16,
        image_encoder_backend="timm",
        image_encoder_preset="virchow",
    )
    _, embedding = encoder(object_image=torch.rand(2, 3, 224, 224))

    assert embedding.shape == (2, 16)
    assert calls[0]["name"] == "hf-hub:paige-ai/Virchow"
    assert calls[0]["mlp_layer"] is FakeSwiGLUPacked
    assert calls[0]["act_layer"] is torch.nn.SiLU
    assert "num_classes" not in calls[0]


def test_virchow2_preset_skips_register_tokens() -> None:
    spec = resolve_image_encoder_spec(backend="timm", preset="virchow2")
    assert spec.name == "hf-hub:paige-ai/Virchow2"
    assert spec.embedding_strategy == "class_token_plus_mean_patch_tokens"
    assert spec.gated_access


def test_image_qc_reports_valid_and_missing_patches(tmp_path: Path) -> None:
    patch = write_synthetic_patch(tmp_path / "patch.png", image_size=32, structure_id=1, intensity=0.6, seed=1)
    missing = tmp_path / "missing.png"
    manifest = tmp_path / "patches.json"
    manifest.write_text(
        json.dumps(
            [
                {"contour_id": "contour_a", "structure_id": 1, "structure_label": "a", "image_path": patch.as_posix()},
                {"contour_id": "contour_b", "structure_id": 2, "structure_label": "b", "image_path": missing.as_posix()},
            ]
        ),
        encoding="utf-8",
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[1, 0], [0, 1]], dtype=np.float32)),
        obs=pd.DataFrame(
            {"cell_id": ["cell_a", "cell_b"], "contour_id": ["contour_a", "contour_b"]},
            index=["cell_a", "cell_b"],
        ),
        var=pd.DataFrame({"feature_name": ["GeneA", "GeneB"]}, index=["GeneA", "GeneB"]),
    )
    adata.obsm["spatial"] = np.asarray([[0, 0], [1, 1]], dtype=np.float32)
    h5ad = tmp_path / "cells.h5ad"
    adata.write_h5ad(h5ad)
    cfg = StGPTConfig(
        case_name="image_qc",
        data=DataConfig(mode="anndata", input_h5ad=str(h5ad), patch_manifest=str(manifest), output_dir=str(tmp_path / "case")),
        model=ModelConfig(d_model=32, n_heads=4, n_layers=1, max_genes=4, image_size=32, n_expression_bins=8),
        training=TrainingConfig(batch_size=2, max_steps=1, output_dir=str(tmp_path / "train"), device="cpu"),
    )

    result = inspect_images(cfg, output_dir=tmp_path / "qc")
    assert result["summary"]["status"] == "fail"
    assert result["summary"]["missing_image_count"] == 1
    assert Path(result["artifacts"]["image_qc_summary_csv"]).exists()


def test_precompute_images_and_training_with_store(tmp_path: Path) -> None:
    config_path = _write_synthetic_config(tmp_path)
    store = tmp_path / "image_embeddings.parquet"
    summary = precompute_image_embeddings(config_path, output=store, encoder_backend="cnn", batch_size=4, device="cpu")
    assert summary["embedding_dim"] == 32
    assert store.exists()

    train_config = _write_synthetic_config(
        tmp_path / "with_store",
        image_embedding_store=store,
        image_encoder_backend="precomputed",
        image_encoder_preset="virchow",
    )
    result = train(train_config, preset="smoke", max_steps=1)
    payload = torch.load(result["checkpoint"], map_location="cpu")
    assert payload["image_encoder"]["backend"] == "precomputed"
    assert payload["image_encoder"]["preset"] == "virchow"
    assert payload["image_encoder"]["name"] == "hf-hub:paige-ai/Virchow"
    assert payload["image_encoder"]["normalization_source"] == "timm.resolve_data_config(pretrained_cfg)"


def test_precompute_images_with_mocked_virchow_preset(tmp_path: Path, monkeypatch) -> None:
    class FakeTimm(torch.nn.Module):
        num_features = 7

        def forward(self, image):
            return torch.ones(image.shape[0], 257, self.num_features, device=image.device)

    class FakeSwiGLUPacked:
        pass

    timm = types.ModuleType("timm")
    timm.create_model = lambda *args, **kwargs: FakeTimm()
    timm_layers = types.ModuleType("timm.layers")
    timm_layers.SwiGLUPacked = FakeSwiGLUPacked
    monkeypatch.setitem(sys.modules, "timm", timm)
    monkeypatch.setitem(sys.modules, "timm.layers", timm_layers)

    config_path = _write_synthetic_config(tmp_path)
    store = tmp_path / "virchow_embeddings.parquet"
    summary = precompute_image_embeddings(
        config_path,
        output=store,
        encoder_backend="timm",
        encoder_preset="virchow",
        batch_size=4,
        device="cpu",
    )
    manifest = pd.read_csv(tmp_path / "image_embedding_manifest.csv")

    assert summary["encoder_preset"] == "virchow"
    assert summary["encoder_name"] == "hf-hub:paige-ai/Virchow"
    assert summary["normalization_source"] == "timm.resolve_data_config(pretrained_cfg)"
    assert store.exists()
    assert manifest["encoder_preset"].iloc[0] == "virchow"
    assert bool(manifest["gated_access"].iloc[0])


def test_cli_inspect_and_precompute_images(tmp_path: Path) -> None:
    config_path = _write_synthetic_config(tmp_path)
    runner = CliRunner()
    qc = runner.invoke(app, ["inspect-images", "--config", str(config_path), "--output", str(tmp_path / "qc")])
    assert qc.exit_code == 0, qc.output
    assert (tmp_path / "qc" / "image_qc_summary.csv").exists()

    precompute = runner.invoke(
        app,
        [
            "precompute-images",
            "--config",
            str(config_path),
            "--encoder-backend",
            "cnn",
            "--output",
            str(tmp_path / "embeddings.parquet"),
            "--batch-size",
            "4",
            "--device",
            "cpu",
        ],
    )
    assert precompute.exit_code == 0, precompute.output
    assert (tmp_path / "embeddings.parquet").exists()


def test_dataset_collates_precomputed_embeddings(tmp_path: Path) -> None:
    cfg = StGPTConfig(
        case_name="precomputed_dataset",
        data=DataConfig(mode="synthetic", output_dir=str(tmp_path / "case"), n_cells=8, n_genes=12, image_size=32),
        model=ModelConfig(d_model=32, n_heads=4, n_layers=1, max_genes=8, image_size=32, n_expression_bins=8),
        training=TrainingConfig(batch_size=4, max_steps=1, output_dir=str(tmp_path / "train"), device="cpu"),
    )
    case = make_synthetic_case(cfg.data)
    regions = case.patch_table["contour_id"].astype(str).tolist()
    pd.DataFrame(
        {
            "region_id": regions,
            "emb_0": np.arange(len(regions), dtype=np.float32),
            "emb_1": np.arange(len(regions), dtype=np.float32) + 1,
        }
    ).to_parquet(tmp_path / "image_embeddings.parquet", index=False)
    payload = cfg.model_dump()
    payload["data"]["image_embedding_store"] = str(tmp_path / "image_embeddings.parquet")
    cfg = StGPTConfig.model_validate(payload)
    dataset = ImageGeneDataset(build_training_case(cfg), cfg)
    batch = dataset.collate([dataset[0], dataset[1]])
    assert batch["precomputed_image_embedding"].shape == (2, 2)
    assert batch["has_precomputed_image_embedding"].tolist() == [True, True]
