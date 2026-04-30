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

## spatho Adapter

`stGPT` is designed to serve as the morpho-molecular embedding backend for spatial pathology tools such as `spatho`.
The stable entry point is `stgpt export-spatho`, which runs the full embed pipeline and writes three versioned artifacts:

| Artifact | Contents |
|---|---|
| `cell_embeddings.parquet` | One row per cell: `cell_id`, `x`, `y`, `structure_label`, `qc_flag`, `emb_0` … `emb_{d-1}` |
| `structure_summary.parquet` | One row per structure: `structure_label`, `n_cells`, mean `emb_*` |
| `qc_report.json` | Operational QC: cell counts, image coverage, per-structure breakdown |

```bash
stgpt export-spatho \
  --config  configs/atera_wta_breast.yaml \
  --checkpoint outputs/atera_wta_breast/train/checkpoints/last.pt \
  --output  outputs/atera_wta_breast/spatho_export
```

The same pipeline is available from Python:

```python
from stgpt.spatho import run_spatho_export

result = run_spatho_export(
    "configs/atera_wta_breast.yaml",
    checkpoint="outputs/atera_wta_breast/train/checkpoints/last.pt",
    output_dir="outputs/atera_wta_breast/spatho_export",
)
print(result.cell_embeddings)   # Path to cell_embeddings.parquet
print(result.n_cells_with_image)
```

The `qc_flag` column in `cell_embeddings.parquet` records `"ok"` when an H&E patch was loaded from the patch manifest, and `"no_image"` when the model fell back to a zero image tensor.

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
from stgpt.spatho import PatchManifestRow, SpathoExportResult, run_spatho_export
from stgpt.training import train
```

Development strategy: [`docs/strategy.md`](docs/strategy.md)

## Attribution

This project is inspired by scGPT-spatial:

> Wang et al. scGPT-spatial: Continual Pretraining of Single-Cell Foundation Model for Spatial Transcriptomics. bioRxiv, 2025.

See `NOTICE` and `CITATION.cff`.
