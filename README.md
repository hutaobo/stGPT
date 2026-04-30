# stGPT

`stGPT` is a Xenium-first image-gene GPT prototype for spatial transcriptomics. It is designed as a clean, independent package inspired by [`bowang-lab/scGPT-spatial`](https://github.com/bowang-lab/scGPT-spatial), while adding trainable H&E image context and hooks for pyXenium and spatho evidence.

The first implementation target is deliberately practical:

- load small synthetic fixtures for CPU smoke tests
- load real Xenium data through optional `pyXenium`
- consume spatho H&E contour patch manifests when present
- train a lightweight image-gene Transformer prototype
- export compact embeddings for downstream spatial pathology workflows

No scGPT-spatial source code or model weights are vendored in this repository.

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

## Data/QC Validation

Before training on real data, validate the data contract and inspect the generated QC report:

```bash
stgpt validate-data --config configs/atera_wta_breast.yaml --output outputs/atera_wta_breast/qc
```

This writes `case_manifest.json`, `qc_report.json`, `qc_report.md`, and `splits.csv`. Training should only proceed after fatal QC errors are resolved.

After training, evaluate against the QC split file instead of creating a new split:

```bash
stgpt evaluate --checkpoint outputs/atera_wta_breast/train/checkpoints/last.pt --config configs/atera_wta_breast.yaml --splits outputs/atera_wta_breast/qc/splits.csv --output outputs/atera_wta_breast/eval
```

This writes `evaluation_metrics.json`, `prediction_summary.csv`, `retrieval_metrics.csv`, and `embedding_qc.csv`.

## Smoke Training

```bash
stgpt doctor
stgpt train --config configs/smoke.yaml --preset smoke --max-steps 2
```

The smoke config generates a tiny synthetic AnnData object and synthetic H&E-like image patches at runtime under `outputs/smoke/`.

## Atera / PDC Training

Copy the example config and point it at real data through environment variables:

```bash
copy configs\atera_wta_breast.yaml.example configs\atera_wta_breast.yaml
set STGPT_XENIUM_ROOT=Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs
set STGPT_SPATHO_RUN_ROOT=D:\path\to\spatho_run
set STGPT_OUTPUT_ROOT=D:\path\to\stgpt_outputs
stgpt prepare-xenium --config configs\atera_wta_breast.yaml
stgpt validate-data --config configs\atera_wta_breast.yaml --output %STGPT_OUTPUT_ROOT%\atera_wta_breast\qc
stgpt train --config configs\atera_wta_breast.yaml --preset pdc
stgpt evaluate --checkpoint %STGPT_OUTPUT_ROOT%\atera_wta_breast\train\checkpoints\last.pt --config configs\atera_wta_breast.yaml --splits %STGPT_OUTPUT_ROOT%\atera_wta_breast\qc\splits.csv --output %STGPT_OUTPUT_ROOT%\atera_wta_breast\eval
```

On PDC, use:

```bash
sbatch scripts/pdc/train_atera.slurm
```

The public repo intentionally stores only templates. Real data, generated `.h5ad` files, patches, checkpoints, and logs are ignored.

## Model v0

`ImageGeneSTGPT` combines:

- gene tokens and expression-value/bin embeddings
- a trainable lightweight image patch encoder
- spatial coordinate and optional context tokens
- Transformer fusion over image, spatial, context, and gene tokens
- masked gene reconstruction, neighborhood reconstruction, image-gene contrastive loss, and optional structure classification

The API entry points are:

```python
from stgpt.config import StGPTConfig
from stgpt.data import build_training_manifest, load_xenium_case
from stgpt.inference import embed_anndata
from stgpt.models import ImageGeneSTGPT
from stgpt.training import train
```

Development strategy: [`docs/strategy.md`](docs/strategy.md)

## Attribution

This project is inspired by scGPT-spatial:

> Wang et al. scGPT-spatial: Continual Pretraining of Single-Cell Foundation Model for Spatial Transcriptomics. bioRxiv, 2025.

See `NOTICE` and `CITATION.cff`.
