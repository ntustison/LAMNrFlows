#!/usr/bin/env bash
set -euo pipefail
PYTHON_EXE="/Users/ntustison/anaconda3/bin/python3"
ROOT_DIR="$(pwd)"
TRAINER="${ROOT_DIR}/train_lamnr_flows_tabular.py"
OUTROOT="${ROOT_DIR}/runs/multiview_production"
DATA_DIR="${ROOT_DIR}/data/processed/trimmed_input"

# Stabilisation pour macOS (évite l'erreur "condition_variable wait failed")
export OMP_NUM_THREADS=1

K_FIXED=4; HC_FIXED=80; Z_DIM=31; SEED=42
COMMON_ARGS=(--base-distribution GaussianPCA --pca-latent-dimension ${Z_DIM} --normalization 0mean --impute mean --lr 1e-4 --batch-size 256 --max-iter 8000 --early-stop-enabled --early-stop-patience 800 --scale-cap 3.0 --spectral-norm-scales)

NNL_VIEWS=("${DATA_DIR}/clean_input_NNL_DTI.csv" "${DATA_DIR}/clean_input_NNL_rsfMRI.csv" "${DATA_DIR}/clean_input_NNL_T1.csv")
PPMI_VIEWS=("${DATA_DIR}/clean_input_PPMI_DTI.csv" "${DATA_DIR}/clean_input_PPMI_rsfMRI.csv" "${DATA_DIR}/clean_input_PPMI_T1.csv")

# Paramètres HSIC
LAMBDAS=(1.0 10.0 50.0)
# Pas d'arguments supplémentaires pour HSIC, donc on supprime la variable de la commande

echo "--- Lancement PARTIE 3 : HSIC (Final) ---"

for lam in "${LAMBDAS[@]}"; do
    tag="lambda${lam}"

    # PPMI
    "$PYTHON_EXE" "$TRAINER" "${COMMON_ARGS[@]}" \
        --verbose \
        --views "${PPMI_VIEWS[@]}" \
        --output-prefix "${OUTROOT}/PPMI/hsic_${tag}" \
        --seed "$SEED" --K "$K_FIXED" --hidden-channels "$HC_FIXED" \
        --penalty-type hsic --lambda-penalty "$lam"

    # NNL
    # --- CORRECTION : Suppression de ${HSIC_ARGS} ---
    "$PYTHON_EXE" "$TRAINER" "${COMMON_ARGS[@]}" \
        --verbose \
        --views "${NNL_VIEWS[@]}" \
        --output-prefix "${OUTROOT}/NNL/hsic_${tag}" \
        --seed "$SEED" --K "$K_FIXED" --hidden-channels "$HC_FIXED" \
        --penalty-type hsic --lambda-penalty "$lam"

done