# stGPT

`stGPT` is a Xenium-first morpho-molecular foundation-model backend in development for spatial transcriptomics. It is designed as a clean, independent package inspired by [`bowang-lab/scGPT-spatial`](https://github.com/bowang-lab/scGPT-spatial), while adding trainable H&E image context and hooks for pyXenium and spatho evidence. The target claim is reusable Xenium-centered spatial pathology embeddings, not just another H&E-to-expression regressor.

The first implementation target is deliberately practical:

- load small synthetic fixtures for CPU smoke tests
- load real Xenium data through optional `pyXenium`
- consume spatho H&E contour patch manifests when present
- train a lightweight contour/region-level image-gene Transformer prototype
- export region embeddings and provenance artifacts for downstream spatial pathology workflows

No scGPT-spatial source code or model weights are vendored in this repository.

## Architecture Narrative

The platform story is:

> stGPT learns reusable contour/region morpho-molecular representations; spatho plans, validates, and turns them into auditable spatial pathology evidence.

The package is organized around three public layers while preserving the older imports:

- `stgpt.foundation`: model, training, embedding, checkpoint loading, and model packaging
- `stgpt.evidence`: QC, deterministic splits, evaluation, ablations, and failure analysis
- `stgpt.runtime`: callable tool API for downstream systems such as spatho

The runtime tool surface is intentionally conservative today: `embed_regions`, `evaluate_checkpoint`, `package_model`, and `export_spatho_artifacts` are implemented. `embed_cells` remains as a deprecated compatibility wrapper that returns region-first artifacts. Region retrieval, panel imputation, niche scoring, region comparison, and structure explanation remain planned capabilities until backed by tested outputs.

## Install

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

For real Xenium adapters:

```bash
python -m pip install -e ".[dev,xenium,spatho]"
```

When using the checked-out development environment on Windows, run tests through the project virtual environment:

```bash
.\.venv\Scripts\python.exe -m pytest
```

The system Python must have the editable package and dependencies installed before `python -m pytest` will work.

## Documentation

The stGPT Read the Docs source lives under [`docs/`](docs/). The first real-data reproducibility record is the Atera XeniumSlide data-foundation notebook:

- [`docs/tutorials/atera_xeniumslide_data_foundation.ipynb`](docs/tutorials/atera_xeniumslide_data_foundation.ipynb)

Build the documentation locally with:

```bash
.\.venv\Scripts\python.exe -m sphinx -b html docs docs\_build\html
```

## spatho Adapter

`stGPT` is designed to serve as the morpho-molecular embedding backend for spatial pathology tools such as `spatho`.
The stable entry points are `stgpt embed-regions` and `stgpt export-spatho`, which run the full region embed pipeline and write versioned artifacts:

| Artifact | Contents |
|---|---|
| `region_embeddings.parquet` | One row per contour/region: `region_id`, `x`, `y`, `structure_label`, `n_cells`, `qc_flag`, `emb_0` … `emb_{d-1}` |
| `region_cell_membership.parquet` | Region-to-cell membership used as molecular evidence |
| `region_molecular_summary.parquet` | Raw mean measured expression per region |
| `region_image_manifest.json` | H&E patch, crop, and registration provenance |
| `region_qc_report.json` | Region counts, image coverage, cell assignment coverage, and per-structure breakdown |
| `evidence_manifest.json` | Paths and provenance for the exported evidence bundle |
| `structure_summary.parquet` | One row per structure: `structure_label`, summed member `n_cells`, mean region `emb_*` |
| `structure_embedding_summary.csv` | CSV mirror of the structure summary for lightweight workbench consumption |

```bash
stgpt embed-regions \
  --config configs/atera_wta_breast_slide.yaml \
  --model outputs/xenium_slides/atera/stgpt/breast/train/checkpoints/last.pt \
  --output outputs/xenium_slides/atera/stgpt/breast/spatho_export
```

The same pipeline is available from Python:

```python
from stgpt.spatho import run_spatho_export

result = run_spatho_export(
    "configs/atera_wta_breast_slide.yaml",
    checkpoint="outputs/xenium_slides/atera/stgpt/breast/train/checkpoints/last.pt",
    output_dir="outputs/xenium_slides/atera/stgpt/breast/spatho_export",
)
print(result.region_embeddings)   # Path to region_embeddings.parquet
print(result.n_cells)             # Deprecated field name; value is n_regions
```

The `qc_flag` column in `region_embeddings.parquet` records `"ok"` when a contour H&E patch was loaded from the patch manifest, and `"no_image"` when the model fell back to a zero image tensor.

## Loading a Pretrained Model

```python
from stgpt.models import ImageGeneSTGPT

model = ImageGeneSTGPT.from_pretrained(
    "outputs/atera_wta_breast/train/checkpoints/last.pt",
    device="auto",   # "auto" | "cpu" | "cuda"
)
# model is in eval mode on the requested device
```

To inspect the raw checkpoint payload (config, vocab, training metrics):

```python
payload = ImageGeneSTGPT.load_checkpoint("outputs/.../last.pt")
print(payload["config"])
print(payload["vocab"]["genes"][:10])
```

## Data/QC Validation

Before training on real data, validate the data contract and inspect the generated QC report:

```bash
stgpt validate-data --config configs/atera_wta_breast.yaml --output outputs/atera_wta_breast/qc
```

This writes `case_manifest.json`, `qc_report.json`, `qc_report.md`, and a region-level `splits.csv` with `region_id`, `split`, `split_strategy`, and `block_id`. Training should only proceed after fatal QC errors are resolved.

After training, evaluate against the QC split file instead of creating a new split:

```bash
stgpt evaluate --checkpoint outputs/atera_wta_breast/train/checkpoints/last.pt --config configs/atera_wta_breast.yaml --splits outputs/atera_wta_breast/qc/splits.csv --output outputs/atera_wta_breast/eval
```

This writes `evaluation_metrics.json`, `prediction_summary.csv`, `retrieval_metrics.csv`, `embedding_qc.csv`, `label_retrieval_metrics.csv`, `batch_mixing_metrics.csv`, and `failure_analysis.csv`.

For multi-case development, use `data.mode: corpus` with `input_h5ad_list` or `dataset_roots`, then use `split.strategy: slide_holdout` to keep slide or patient groups from leaking across train, validation, and test splits.

For Atera-style real data, the preferred first step is to build canonical `XeniumSlide` stores with pyXenium. This creates one auditable learning object per case and uses contour-segmented H&E crops as image context:

```bash
pyxenium slide build-atera --atera-root Y:\long\10X_datasets\Xenium\Atera --output-root D:\GitHub\stGPT\outputs\xenium_slides\atera
```

Each case writes `xenium_slide.zarr`, `slide_manifest.json`, `qc_report.json`, `cell_to_contour.parquet`, `structure_assignments.csv`, `contour_patches_manifest.json`, and `contour_patches/*.png`. Raw data under `Y:\...` is read only; stGPT never assumes per-cell H&E crops for this mode. Point `data.mode: xenium_slide` at the generated `xenium_slide.zarr`, then run `stgpt validate-data` before training.

To package a trained checkpoint as a reusable stGPT model backend and emit spatho-compatible artifacts:

```bash
stgpt package-model --checkpoint outputs/atera_wta_breast/train/checkpoints/last.pt --eval outputs/atera_wta_breast/eval/evaluation_metrics.json --output outputs/atera_wta_breast/model
stgpt spatho-embed --model outputs/atera_wta_breast/model --config configs/atera_wta_breast.yaml --output outputs/atera_wta_breast/spatho
```

The spatho export writes `region_embeddings.parquet`, `region_cell_membership.parquet`, `region_molecular_summary.parquet`, `region_image_manifest.json`, `region_qc_report.json`, `evidence_manifest.json`, `structure_summary.parquet`, and `structure_embedding_summary.csv`.

## Smoke Training

```bash
stgpt doctor
stgpt train --config configs/smoke.yaml --preset smoke --max-steps 2
```

The smoke config generates a tiny synthetic AnnData object and synthetic H&E-like image patches at runtime under `outputs/smoke/`.
Training now writes both `checkpoints/last.pt` and `checkpoints/best.pt`; `best.pt` is selected from the validation split when available. Cosine/one-cycle learning-rate schedules, checkpoint intervals, and loss-weight warmups are configured under `training`.

For paper-facing baselines, train explicit ablations from the same config:

```bash
stgpt train --config configs/smoke.yaml --preset smoke --ablation gene_only
stgpt train --config configs/smoke.yaml --preset smoke --ablation image_only
stgpt train --config configs/smoke.yaml --preset smoke --ablation spatial_only
stgpt train --config configs/smoke.yaml --preset smoke --ablation image_gene
stgpt train --config configs/smoke.yaml --preset smoke --ablation image_gene_spatial
stgpt train --config configs/smoke.yaml --preset smoke --ablation full
```

## Atera / PDC Training

Copy the example config and point it at the generated Atera XeniumSlide directories:

```bash
copy configs\atera_wta_breast_slide.yaml.example configs\atera_wta_breast_slide.yaml
stgpt validate-data --config configs\atera_wta_breast_slide.yaml --output outputs\xenium_slides\atera\stgpt\breast\qc
stgpt train --config configs\atera_wta_breast_slide.yaml --preset pdc
stgpt evaluate --checkpoint outputs\xenium_slides\atera\stgpt\breast\train\checkpoints\last.pt --config configs\atera_wta_breast_slide.yaml --splits outputs\xenium_slides\atera\stgpt\breast\qc\splits.csv --output outputs\xenium_slides\atera\stgpt\breast\eval
stgpt package-model --checkpoint outputs\xenium_slides\atera\stgpt\breast\train\checkpoints\last.pt --eval outputs\xenium_slides\atera\stgpt\breast\eval\evaluation_metrics.json --output outputs\xenium_slides\atera\stgpt\breast\model
stgpt embed-regions --model outputs\xenium_slides\atera\stgpt\breast\model --config configs\atera_wta_breast_slide.yaml --output outputs\xenium_slides\atera\stgpt\breast\spatho
```

On PDC, use:

```bash
sbatch scripts/pdc/train_atera.slurm
```

The public repo intentionally stores only templates. Real data, generated `.h5ad` files, patches, checkpoints, and logs are ignored.

## Model v0

`ImageGeneSTGPT` combines:

- gene tokens and expression-value/bin embeddings
- a trainable lightweight, optionally multi-scale image patch encoder
- region spatial coordinate and optional structure-context tokens
- sampled member-cell expression tokens as region context
- Transformer fusion over image, spatial, context, cell, and gene tokens
- masked region gene reconstruction, region-neighborhood reconstruction, image-region contrastive loss, and optional structure classification
- modality switches for gene-only, image-only, spatial-only, image-gene, image-gene-spatial, and full ablations

The API entry points are:

```python
from stgpt.config import StGPTConfig
from stgpt.data import build_training_manifest, load_xenium_case
from stgpt.foundation import ImageGeneSTGPT, embed_anndata, embed_regions, package_model, train
from stgpt.evidence import evaluate, validate_data
from stgpt.runtime import embed_regions as runtime_embed_regions, evaluate_checkpoint, export_spatho_artifacts
from stgpt.spatho import PatchManifestRow, SpathoExportResult, run_spatho_export
```

Legacy imports such as `stgpt.models.ImageGeneSTGPT`, `stgpt.training.train`, `stgpt.qc.validate_data`, and `stgpt.evaluation.evaluate` remain supported.

Development strategy: [`docs/strategy.md`](docs/strategy.md)

## Attribution

This project is inspired by scGPT-spatial:

> Wang et al. scGPT-spatial: Continual Pretraining of Single-Cell Foundation Model for Spatial Transcriptomics. bioRxiv, 2025.

See `NOTICE` and `CITATION.cff`.
