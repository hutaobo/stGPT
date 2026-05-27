# Pseudo-spatial generative prior

This module trains an expression-only prior from spatial transcriptomics regions and then applies it to single-cell expression.

## Biological contract

Input:

- A spatial transcriptomics training corpus with expression, region coordinates, and structure labels.
- A single-cell or single-nucleus `.h5ad` at inference time.

Output per input cell:

- Probability over tissue structure labels.
- Probability over relative spatial `x` and `y` bins.
- Probability over learned tissue niches.
- Optional projection to a concrete reference slice region and coordinate.

Without a reference slice, the model predicts pseudo-space probabilities, not physical coordinates. With a reference slice, the predicted token distribution is matched to reference regions and projected back to `x, y`.

## A100 training command

```bash
ssh taobo.hu@sscb-a100.scilifelab.se
cd /data/taobo.hu/projects/stgpt_l3_20260504/repos/stGPT_figures_main_20260522
git fetch origin main && git checkout main && git pull --ff-only origin main
source /data/taobo.hu/projects/stgpt_l3_20260504/.venv/bin/activate

export STGPT_OUTPUT_ROOT=/data/taobo.hu/projects/stgpt_l3_20260504/runs
export STGPT_XENIUM_SLIDES=/data/taobo.hu/projects/stgpt_l3_20260504/data/xenium_slides
export CUDA_VISIBLE_DEVICES=4,5,6,7

stgpt train-pseudo-spatial \
  --config configs/pilots/l3_43/pseudo_spatial_prior_43case.yaml \
  --output /data/taobo.hu/projects/stgpt_l3_20260504/runs/pilot_runs/l3_20260507_43case/pseudo_spatial_prior_43case/train \
  --max-steps 5000 \
  --n-spatial-bins 32 \
  --n-niches 32 \
  --max-genes 512 \
  --batch-size 512 \
  --device cuda \
  --num-workers 8 \
  --data-parallel
```

The command writes:

- `last.pt` and `best.pt`
- `metrics.json` and `metrics.csv`
- `splits.csv`
- `reference_regions.parquet`

## Single-cell inference

```bash
stgpt predict-pseudo-spatial \
  --model /data/taobo.hu/projects/stgpt_l3_20260504/runs/pilot_runs/l3_20260507_43case/pseudo_spatial_prior_43case/train/best.pt \
  --input /path/to/single_cell.h5ad \
  --output /path/to/pseudo_spatial_predictions.parquet \
  --reference-regions /data/taobo.hu/projects/stgpt_l3_20260504/runs/pilot_runs/l3_20260507_43case/pseudo_spatial_prior_43case/train/reference_regions.parquet \
  --batch-size 2048 \
  --device cuda
```

For publication work, report this as a probabilistic spatial-prior model. The current implementation is a supervised discrete generative prior over pseudo-space tokens; diffusion or autoregressive sampling over tissue token fields can be added after this interface is stable.
