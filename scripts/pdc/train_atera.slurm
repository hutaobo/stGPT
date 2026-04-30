#!/bin/bash
#SBATCH -A naiss2025-22-606
#SBATCH -J stgpt-atera
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o stgpt-atera-%j.out
#SBATCH -e stgpt-atera-%j.err

set -euo pipefail

: "${STGPT_XENIUM_ROOT:?Set STGPT_XENIUM_ROOT to the Xenium outs directory}"
: "${STGPT_SPATHO_RUN_ROOT:?Set STGPT_SPATHO_RUN_ROOT to the spatho run root}"
: "${STGPT_OUTPUT_ROOT:?Set STGPT_OUTPUT_ROOT to the stGPT output root}"

python -m pip install -U pip
python -m pip install -e ".[xenium,spatho]"

stgpt prepare-xenium --config configs/atera_wta_breast.yaml
stgpt train --config configs/atera_wta_breast.yaml --preset pdc
