#!/usr/bin/env bash
set -euo pipefail

DO_PLOTS=1

DATA_DIR=./SiMLR_NNH_Data/UKBB/
RUNS_DIR=./runs/lamnr_ukbb
mkdir -p "${RUNS_DIR}"

# CONDA_BASE=$(conda info --base)
# PY="${CONDA_BASE}/bin/python"
PY="python"

# Set DO_PLOTS=1 to generate marginal plots (requires pandas in $PY).
DO_PLOTS="${DO_PLOTS:-0}"

# Hardware (matches your CLI flag)
CUDA_DEVICE="${CUDA_DEVICE:-cuda:1}"

# -------------------------
# Common arguments (from your original command.sh)
# -------------------------
COMMON_ARGS=(
  --views
    "${DATA_DIR}/ukb672504_list_dt.csv"
    "${DATA_DIR}/ukb672504_list_rsf.csv"
    "${DATA_DIR}/ukb672504_list_t1.csv"

  --base-distribution GaussianPCA
  --pca-latent-dimension 31
  --base-sigma 0.1
  --base-min-log -2.0
  --base-max-log  2.0

  --normalization 0mean
  --add-noise-in normalized
  --impute mean
  --jitter-alpha 0.0
  --jitter-alpha-end 0.0
  --jitter-alpha-mode cosine

  --lr 1e-4
  --batch-size 512
  --weight-decay 0.0
  --max-iter 6000
  --cuda-device "${CUDA_DEVICE}"

  --val-fraction 0.2
  --val-interval 50
  --val-batch-size 2048

  --save-z
  --save-whitened pca
  --dataset-normalizers-json "${RUNS_DIR}/dataset_normalizers.json"
  --verbose
)

# Screening (auto-disables alignment if dependence is too low)
SCREEN_ARGS=(
  --screening-mode cca
  --screening-fraction 0.2
  --screening-max-samples 5000
  --screening-threshold 0.1
)

# Shared tradeoff knobs (used by your existing Barlow setup; harmless if ignored)
TRADEOFF_ARGS=(
  --tradeoff-mode uncertainty
  --target-ratio 4.0
  --scale-penalty-weight 1.0
)

# Penalty-specific defaults
VICREG_ARGS=(
  --vicreg-w-inv 25.0
  --vicreg-w-var 25.0
  --vicreg-w-cov 1.0
  --vicreg-gamma 1.0
)

HSIC_ARGS=(
  --hsic-sigma 0.0   # median heuristic per batch
)

INFONCE_ARGS=(
  --info-nce-T 0.2
)

# Resume helper: skip if outputs already exist
already_done () {
  local prefix="$1"
  local f="${prefix}_whitened_view0.csv"
  echo "[CHECK] ${f}" >&2
  [[ -s "${f}" ]] || [[ -s "${f}.gz" ]]
}

# Plot helper (optional)
plot_views () {
  local prefix="$1"
  local outdir="$2"

  if [[ "${DO_PLOTS}" != "1" ]]; then
    return 0
  fi

  if ! "${PY}" -c "import pandas" >/dev/null 2>&1; then
    echo "[PLOT SKIP] pandas not available in ${PY}" >&2
    return 0
  fi

  mkdir -p "${outdir}"
  "${PY}" plot_csv_distributions.py \
    "${prefix}"_whitened_view?.csv \
    --ncols 4 \
    -o "${outdir}" \
    --secondary-winsorize-quantiles 1 99 \
    --secondary-standardize \
    --secondary-tag winsor_std
}

# -------------------------
# Multi-view sweep (Stage 1 recommandé)
# -------------------------
K_FIXED=4
HC_FIXED=80
SEEDS=(0 1 2)

PENALTIES=(none barlow_twins_align vicreg hsic pearson info_nce)
LAMBDAS=(0.01 0.03 0.1)

already_done () {
  local prefix="$1"
  # Multi-view: on veut au moins les 3 vues
  for v in 0 1 2; do
    local f="${prefix}_whitened_view${v}.csv"
    echo "[CHECK] ${f}" >&2
    [[ -s "${f}" ]] || [[ -s "${f}.gz" ]] || return 1
  done
  return 0
}

for seed in "${SEEDS[@]}"; do

  # ---- baseline (no alignment) ----
  out_prefix="${RUNS_DIR}/ukbb_seed${seed}_K${K_FIXED}_hc${HC_FIXED}_none_lambda0"
  if already_done "${out_prefix}"; then
    echo "[SKIP] ${out_prefix} (already has whitened_view0/1/2)"
  else
    "${PY}" train_lamnr_flows_tabular.py \
      "${COMMON_ARGS[@]}" \
      --output-prefix "${out_prefix}" \
      --seed "${seed}" \
      --K "${K_FIXED}" \
      --hidden-channels "${HC_FIXED}" \
      --scale-cap 3.0 \
      --spectral-norm-scales \
      --penalty-type none \
      --lambda-penalty 0.0 \
      --dataset-normalizers-json "${out_prefix}_dataset_normalizers.json"
  fi
  plot_views "${out_prefix}" "${RUNS_DIR}/figs_seed${seed}_K${K_FIXED}_hc${HC_FIXED}_none_lambda0"

  # ---- aligned penalties ----
  for pen in "${PENALTIES[@]}"; do
    [[ "${pen}" == "none" ]] && continue

    for lam in "${LAMBDAS[@]}"; do
      tag="$(echo "${lam}" | sed 's/\./p/g')"
      out_prefix="${RUNS_DIR}/ukbb_seed${seed}_K${K_FIXED}_hc${HC_FIXED}_${pen}_lambda${tag}"

      declare -a extra
      extra=()

      case "${pen}" in
        vicreg)   extra+=("${VICREG_ARGS[@]}") ;;
        hsic)     extra+=("${HSIC_ARGS[@]}") ;;
        info_nce) extra+=("${INFONCE_ARGS[@]}") ;;
        *)        ;;
      esac

      if already_done "${out_prefix}"; then
        echo "[SKIP] ${out_prefix} (already has whitened_view0/1/2)"
      else
        "${PY}" train_lamnr_flows_tabular.py \
          "${COMMON_ARGS[@]}" \
          --output-prefix "${out_prefix}" \
          --seed "${seed}" \
          --K "${K_FIXED}" \
          --hidden-channels "${HC_FIXED}" \
          --scale-cap 3.0 \
          --spectral-norm-scales \
          --penalty-type "${pen}" \
          --lambda-penalty "${lam}" \
          "${TRADEOFF_ARGS[@]}" \
          "${SCREEN_ARGS[@]}" \
          --dataset-normalizers-json "${out_prefix}_dataset_normalizers.json" \
          ${extra[@]+"${extra[@]}"}
      fi

      plot_views "${out_prefix}" "${RUNS_DIR}/figs_seed${seed}_K${K_FIXED}_hc${HC_FIXED}_${pen}_lambda${tag}"
    done
  done

done
