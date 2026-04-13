#!/usr/bin/env bash
set -euo pipefail

# --- CONFIGURATION ---
PYTHON_EXE="/Users/ntustison/anaconda3/bin/python3"
ROOT_DIR="$(pwd)"
TRAINER="${ROOT_DIR}/train_lamnr_flows_tabular.py"
OUTROOT="${ROOT_DIR}/runs/multiview_production"
DATA_DIR="${ROOT_DIR}/data/processed/trimmed_input"

# Paramètres Fixes (Validés)
K_FIXED=4
HC_FIXED=80
Z_DIM=31
SEEDS=(1337) # Deux graines pour la robustesse
# SEEDS=(42 1337) # Deux graines pour la robustesse

# Arguments communs (Stabilité)
COMMON_ARGS=(
  --base-distribution GaussianPCA --pca-latent-dimension ${Z_DIM}
  --base-sigma 0.1 --base-min-log -2.0 --base-max-log 2.0
  --normalization 0mean --impute mean
  --lr 1e-4 --batch-size 256 --weight-decay 1e-5
  --val-interval 200 --max-iter 8000
  --early-stop-enabled --early-stop-patience 800
  --scale-cap 3.0 --spectral-norm-scales
)

# --- DÉFINITION DES DATASETS ---
# Ordre important: DTI, rsfMRI, T1 (comme UKBB)
NNL_VIEWS=("${DATA_DIR}/clean_input_NNL_DTI.csv" "${DATA_DIR}/clean_input_NNL_rsfMRI.csv" "${DATA_DIR}/clean_input_NNL_T1.csv")
PPMI_VIEWS=("${DATA_DIR}/clean_input_PPMI_DTI.csv" "${DATA_DIR}/clean_input_PPMI_rsfMRI.csv" "${DATA_DIR}/clean_input_PPMI_T1.csv")

# ==========================================
# PARTIE 1 : BASELINE (Penalty = None)
# ==========================================
echo "--- Lancement PARTIE 1 : BASELINE (No Penalty) ---"

for seed in "${SEEDS[@]}"; do

    # 2. PPMI
    "$PYTHON_EXE" "$TRAINER" "${COMMON_ARGS[@]}" \
        --verbose \
        --views "${PPMI_VIEWS[@]}" \
        --output-prefix "${OUTROOT}/PPMI/seed${seed}_baseline" \
        --seed "$seed" --K "$K_FIXED" --hidden-channels "$HC_FIXED" \
        --penalty-type none

    # 1. NNL
    # "$PYTHON_EXE" "$TRAINER" "${COMMON_ARGS[@]}" \
    #     --verbose \
    #     --views "${NNL_VIEWS[@]}" \
    #     --output-prefix "${OUTROOT}/NNL/seed${seed}_baseline" \
    #     --seed "$seed" --K "$K_FIXED" --hidden-channels "$HC_FIXED" \
    #     --penalty-type none


done