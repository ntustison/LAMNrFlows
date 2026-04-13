#!/usr/bin/env bash
set -euo pipefail

# --- CONFIGURATION PYTHON ---
# On pointe directement vers votre binaire Anaconda pour éviter les problèmes d'alias
PYTHON_EXE="/Users/ntustison/anaconda3/bin/python3"

# --- PHILOSOPHIE : VALIDATION CIBLÉE ---
K_SWEEP="3,4,5"
HC_SWEEP="64,80,96"

EXTRA_ARGS=(--val-interval 200 --max-iter 5000 --verbose \
            --early-stop-enabled --early-stop-patience 600 \
            --early-stop-min-iters 1000 --pca-latent-dimension 31)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${ROOT_DIR}/launch_tabular_lamnr_sweep.py"
TRAINER="${ROOT_DIR}/train_lamnr_flows_tabular.py"

# --- CORRECTION MANIFESTE ---
MANIFEST="${ROOT_DIR}/seeds.csv"
if [ ! -f "$MANIFEST" ]; then
    echo "Re-génération du fichier seeds.csv..."
    echo "seed,output_prefix,alignment_lambda" > "$MANIFEST"
    echo "42,seed42,1.0" >> "$MANIFEST"
    echo "1337,seed1337,1.0" >> "$MANIFEST"
fi

INPUT_DIR="${ROOT_DIR}/data/processed/trimmed_input"
OUTROOT="${ROOT_DIR}/runs/validation_targeted"

CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
MAX_PROCS="${MAX_PROCS:-1}"

# --- BOUCLE AUTOMATIQUE ---
for input_file in "${INPUT_DIR}"/input_*.csv; do
    
    filename=$(basename -- "$input_file")
    package_name="${filename#input_}"
    package_name="${package_name%.*}" 
    
    echo "============================================================"
    echo "Validation Ciblée : ${package_name}"
    echo "============================================================"

    # --- CHANGEMENT ICI : On utilise "$PYTHON_EXE" au lieu de "python" ---
    
    "$PYTHON_EXE" "${LAUNCHER}" \
        --manifest "${MANIFEST}" \
        --trainer "${TRAINER}" \
        --python "$PYTHON_EXE" \
        --views "${input_file}" \
        --cuda-device "${CUDA_DEVICE}" \
        --screening-mode none \
        --outdir "${OUTROOT}/${package_name}" \
        --K-sweep "${K_SWEEP}" \
        --hidden-sweep "${HC_SWEEP}" \
        --max-procs "${MAX_PROCS}" \
        --skip-existing \
        --extra "${EXTRA_ARGS[@]}"

done

echo "--- Terminé ---"