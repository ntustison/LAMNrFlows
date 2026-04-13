#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Match: run_config.json (128×128 successful run)
# -----------------------------

# total steps (keep the 2-phase horizon for the phase-1 aug schedule, even though we run only phase 1)
iterations=120000               # phase 1 (max_iter)
extra=40000                     # unused here (extra_iters=0), but used to set aug horizon
total=$((iterations + extra))   # 160000

# model / data
H=128; W=128; L=5; K=12; hidden=192
BATCH=64
SLICE_IDX=116

# optimization
LR=1e-4
WARMUP=1000
WEIGHT_DECAY=1e-5

# alignment + screening
align="vicreg"
align_weight=0.01
ALIGN_WARMUP=500

SCREEN_METHOD="cca"
SCREEN_FRAC=0.5
SCREEN_WARMUP=1000
SCREEN_REFRESH=0
CCA_RIDGE=1e-3
PREFILTER_FRAC=0.5

# sampling / eval
SAMPLE_TEMP=1.0
EVAL_INTERVAL=1000
PLOT_INTERVAL=1000

# misc
TRAIN_SAMPLES=3000
VAL_SAMPLES=128
VAL_FRAC=0.0
DEVICES="cuda:1"
OUTDIR="runs/hcp_t1_t2_fa_128x128_vicreg_K12_H192_vicreg_retest"

# base distribution / scale config
SCALE_CAP=2.0
GLOWBASE_MIN_LOG=-5.0
GLOWBASE_MAX_LOG=5.0
GLOWBASE_LOGSCALE_FACTOR=3.0
SCALE_MAP="tanh"

# -----------------------------
# Augmentation schedule (exact string from config)
# -----------------------------
aug_params_phase1="noise_std:cos:0.05->0.004@${total},\
sd_affine:cos:0.05->0.00@$((total*3/5)),\
sd_deformation:linear:12.0->0.6@$((total*7/10)),\
sd_simulated_bias_field:cos:0.20->0.03@${total},\
sd_histogram_warping:cos:0.04->0.008@${total}"

# -----------------------------
# Optional: only add flags if trainer supports them
# -----------------------------
HELP="$(python train_2d.py -h 2>&1 || true)"
extra_args=()

grep -q -- "--weight-decay" <<<"$HELP"            && extra_args+=( --weight-decay "${WEIGHT_DECAY}" )
grep -q -- "--seed" <<<"$HELP"                    && extra_args+=( --seed 0 )
grep -q -- "--num-workers" <<<"$HELP"             && extra_args+=( --num-workers 4 )
grep -q -- "--scale-map" <<<"$HELP"               && extra_args+=( --scale-map "${SCALE_MAP}" )
grep -q -- "--glowbase-logscale-factor" <<<"$HELP"&& extra_args+=( --glowbase-logscale-factor "${GLOWBASE_LOGSCALE_FACTOR}" )

# -----------------------------
# Data (keep if you use it; harmless if already downloaded)
# -----------------------------
python download_hcp_data.py

# -----------------------------
# Phase 1
# -----------------------------
HCP_ROOT="/home/ntustison/Data/HCPTemplates"

python train_2d.py \
  --view ${HCP_ROOT}/{HCP-A,HCP-Intermediate,HCP-YA}/T_template0.nii.gz \
  --view ${HCP_ROOT}/{HCP-A,HCP-Intermediate,HCP-YA}/T_template1.nii.gz \
  --view ${HCP_ROOT}/{HCP-A,HCP-Intermediate,HCP-YA}/T_template2.nii.gz \
  --H ${H} --W ${W} --L ${L} --K ${K} --hidden ${hidden} \
  --batch ${BATCH} \
  --slice-idx ${SLICE_IDX} --val-frac ${VAL_FRAC} \
  --max-iter "${iterations}" \
  --devices ${DEVICES} --precision mixed --amp-dtype bf16 \
  --ema --ema-decay 0.9997 \
  --auto-resume \
  --aug-schedules "${aug_params_phase1}" \
  --lr ${LR} --warmup-iters ${WARMUP} \
  --eval-interval ${EVAL_INTERVAL} --plot-interval ${PLOT_INTERVAL} \
  --grad-clip 1.0 \
  --train-samples ${TRAIN_SAMPLES} --val-samples ${VAL_SAMPLES} \
  --smooth-alpha 0.05 \
  --sample-mode model --sample-temp ${SAMPLE_TEMP} \
  --weighting fixed \
  --align "${align}" --align-weight "${align_weight}" --align-warmup "${ALIGN_WARMUP}" \
  --screen "${SCREEN_METHOD}" \
  --screen-warmup "${SCREEN_WARMUP}" --screen-refresh "${SCREEN_REFRESH}" --screen-frac "${SCREEN_FRAC}" \
  --cca-ridge "${CCA_RIDGE}" --prefilter-frac "${PREFILTER_FRAC}" \
  --scale-cap ${SCALE_CAP} \
  --glowbase-max-log ${GLOWBASE_MAX_LOG} --glowbase-min-log ${GLOWBASE_MIN_LOG} \
  --out-dir "${OUTDIR}" \
  "${extra_args[@]}"
