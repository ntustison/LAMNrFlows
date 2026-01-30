#!/usr/bin/env bash
set -euo pipefail

# Single-view PCA sweep: K × hidden_channels
#
# This script expects:
#   - launch_tabular_lamnr_sweep.py in the same directory
#   - train_lamnr_flows_tabular.py one directory up (../train_lamnr_flows_tabular.py)
#
# It will create per-run directories under:
#   runs/singleview_pca_khc/<PACKAGE>/pca/seed1_K{K}_hc{hc}/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${ROOT_DIR}/launch_tabular_lamnr_sweep.py"

CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
MAX_PROCS="${MAX_PROCS:-1}"

# Stage 1

## Sweep grids
K_SWEEP="${K_SWEEP:-1,4,12,32}"
HC_SWEEP="${HC_SWEEP:-96,192,320}"

## Extra trainer args (optional). Example:
EXTRA_ARGS=(--val-interval 100 --max-iter 1500 --verbose --early-stop-patience 200)

## Output root for this sweep
OUTROOT="${OUTROOT:-runs/singleview_stage1}"

## Manifest (one row per seed / baseline setting)
## Need to change in singleview_sweep.csv
MANIFEST="${MANIFEST:-singleview_sweep.csv}"

##############################################



# Trainer path (relative to this script directory)
TRAINER="${TRAINER:-../train_lamnr_flows_tabular.py}"


run_package () {
  local PKG="$1"
  local VIEWS_CSV="$2"

  echo "============================================================"
  echo "Package: ${PKG}"
  echo "Views:   ${VIEWS_CSV}"
  echo "Outdir:  ${OUTROOT}/${PKG}"
  echo "K:       ${K_SWEEP}"
  echo "Hidden:  ${HC_SWEEP}"
  echo "============================================================"

  python "${LAUNCHER}" \
    --manifest "${MANIFEST}" \
    --trainer "${TRAINER}" \
    --python "${PYTHON:-python}" \
    --views "${VIEWS_CSV}" \
    --cuda-device "${CUDA_DEVICE}" \
    --screening-mode none \
    --outdir "${OUTROOT}/${PKG}" \
    --K-sweep "${K_SWEEP}" \
    --hidden-sweep "${HC_SWEEP}" \
    --max-procs "${MAX_PROCS}" \
    --skip-existing \
    --extra "${EXTRA_ARGS[@]}"
}

# -------------------- packages --------------------
run_package "ANTsX"       "../InputData/antsxukbb_deeplearning_ANTsX.csv"
run_package "FSL"         "../InputData/antsxukbb_deeplearning_FSL.csv"
run_package "FreeSurfer"  "../InputData/antsxukbb_deeplearning_FreeSurfer.csv"
