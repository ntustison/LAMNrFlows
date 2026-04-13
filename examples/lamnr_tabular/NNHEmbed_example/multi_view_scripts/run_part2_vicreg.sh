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

# --- CORRECTION ICI : Nouveaux noms d'arguments VicReg ---
# w-inv (sim) = 25, w-var (std) = 25, w-cov (cov) = 1
LAMBDAS_NNL=(0.1 0.25 0.3 0.5 1.0 5.0) 
LAMBDAS_PPMI=(0.1 0.3 1.0 2.0 5.0) 
VICREG_ARGS=(--vicreg-w-inv 25.0 --vicreg-w-var 25.0 --vicreg-w-cov 1.0)

echo "--- Lancement PARTIE 2 : VICREG (Corrigé) ---"

for lam in "${LAMBDAS_PPMI[@]}"; do
    tag="lambda${lam}"
    

    if [ -f "${OUTROOT}/PPMI/vicreg_${tag}_metrics.json" ]; then
        echo "[SKIP] PPMI lambda ${tag} (Déjà terminé)"
    else    
        # PPMI
        "$PYTHON_EXE" "$TRAINER" "${COMMON_ARGS[@]}" "${VICREG_ARGS[@]}" \
            --verbose \
            --views "${PPMI_VIEWS[@]}" \
            --output-prefix "${OUTROOT}/PPMI/vicreg_${tag}" \
            --seed "$SEED" --K "$K_FIXED" --hidden-channels "$HC_FIXED" \
            --penalty-type vicreg --lambda-penalty "$lam"
    fi 

done

for lam in "${LAMBDAS_NNL[@]}"; do
    tag="lambda${lam}"

    if [ -f "${OUTROOT}/NNL/vicreg_${tag}_metrics.json" ]; then
        echo "[SKIP] NNL lambda ${tag} (Déjà terminé)"
    else    
        # NNL
        "$PYTHON_EXE" "$TRAINER" "${COMMON_ARGS[@]}" "${VICREG_ARGS[@]}" \
            --verbose \
            --views "${NNL_VIEWS[@]}" \
            --output-prefix "${OUTROOT}/NNL/vicreg_${tag}" \
            --seed "$SEED" --K "$K_FIXED" --hidden-channels "$HC_FIXED" \
            --penalty-type vicreg --lambda-penalty "$lam"
    fi

done