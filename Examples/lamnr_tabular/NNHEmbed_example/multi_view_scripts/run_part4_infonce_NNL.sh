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

# Paramètres InfoNCE
LAMBDAS=(0.1 1.0)
INFONCE_ARGS=(--info-nce-T 0.1)

echo "--- Lancement PARTIE 4 : InfoNCE ---"

for lam in "${LAMBDAS[@]}"; do
    tag="lambda${lam}"
    
    # # PPMI
    # if [ -f "${OUTROOT}/PPMI/infonce_${tag}_metrics.json" ]; then
    #     echo "[SKIP] PPMI lambda ${tag} (Déjà terminé)"
    # else
    #     "$PYTHON_EXE" "$TRAINER" "${COMMON_ARGS[@]}" "${INFONCE_ARGS[@]}" \
    #         --verbose \
    #         --views "${PPMI_VIEWS[@]}" \
    #         --output-prefix "${OUTROOT}/PPMI/infonce_${tag}" \
    #         --seed "$SEED" --K "$K_FIXED" --hidden-channels "$HC_FIXED" \
    #         --penalty-type info_nce --lambda-penalty "$lam"
    #     fi    

    # NNL

    if [ -f "${OUTROOT}/NNL/infonce_${tag}_metrics.json" ]; then
        echo "[SKIP] NNL lambda ${tag} (Déjà terminé)"
    else
        "$PYTHON_EXE" "$TRAINER" "${COMMON_ARGS[@]}" "${INFONCE_ARGS[@]}" \
        --verbose \
        --views "${NNL_VIEWS[@]}" \
        --output-prefix "${OUTROOT}/NNL/infonce_${tag}" \
        --seed "$SEED" --K "$K_FIXED" --hidden-channels "$HC_FIXED" \
        --penalty-type info_nce --lambda-penalty "$lam"
    fi    

done