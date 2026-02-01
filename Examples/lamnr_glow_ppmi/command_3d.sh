#!/usr/bin/env bash
set -euo pipefail

# ---------- total steps ----------
ITERATIONS=120000              # phase 1
EXTRA=0                       # phase 2
TOTAL=$((ITERATIONS + EXTRA))  # horizon for phase-1 aug schedule

# ---------- 3D arch ----------
H=32
W=32
D=32               # depth for 3D Glow
L=3
K=32
HIDDEN=64
BATCH=64             # start conservative in 3D; bump if VRAM allows
GRAD_ACCUM=2
NUM_WORKERS=4

PLATEAU_FACTOR=0.999999
PLATEAU_PATIENCE=100000
PLATEAU_THRESHOLD=1e-3
PLATEAU_COOLDOWN=5

# alignment + screening
ALIGN="vicreg"
ALIGN_WEIGHT=0.01
ALIGN_VICREG_INV=25.0
ALIGN_VICREG_VAR=25.0
ALIGN_VICREG_GAMMA=1.0
ALIGN_VICREG_COV=1.0

ALIGN_WARMUP=1000

OUTDIR="runs3d/hcp_t1_fa_${H}x${W}x${D}_K${K}_L${L}_H${HIDDEN}"

# ---------- screening configuration ----------
SCREEN_METHOD=cca           # none | cca | hsic
SCREEN_FRAC=0.5             # keep top 50% dims for alignment
SCREEN_WARMUP=1000          # start screening after N iters
SCREEN_REFRESH=5000         # 0 = discover once; else refresh cadence
CCA_RIDGE=1e-3              # stability for CCA
PREFILTER_FRAC=0.5          # HSIC Pearson prefilter (ignored for CCA)

# Optimization
LR=2.5e-5      
WARMUP=2000  


# ---------- augmentation schedules ----------

# Phase 1: original decreasing schedule (strong -> weak)
AUG_PARAMS_PHASE1="noise_std:cos:0.05->0.004@${TOTAL},\
sd_affine:cos:0.05->0.00@$((TOTAL)),\
sd_deformation:linear:12.0->0.6@$((TOTAL)),\
sd_simulated_bias_field:cos:0.20->0.03@${TOTAL},\
sd_histogram_warping:cos:0.04->0.008@${TOTAL}"

# Phase 2: template-focused fine-tune (no shape; mild intensity jitter)
AUG_PARAMS_PHASE2="\
noise_std:cos:0.004->0.004@${EXTRA},\
sd_affine:cos:0.00->0.00@${EXTRA},\
sd_deformation:linear:0.0->0.0@${EXTRA},\
sd_simulated_bias_field:cos:0.00->0.00@${EXTRA},\
sd_histogram_warping:cos:0.004->0.004@${EXTRA}"

PPMI_ROOT="/home/ntustison/Data/PPMI_Dataset"

mapfile -t T1 < <(ls -1 ${PPMI_ROOT}/sub*/ses*/*ppmixt1.nii.gz | sort)
mapfile -t FA < <(ls -1 ${PPMI_ROOT}/sub*/ses*/*ppmixfa.nii.gz | sort)

echo "T1: ${#T1[@]}  FA: ${#FA[@]}"

# ---------- Phase 1: strong -> weak aug ----------
python train_3d.py \
  --view "${T1[@]}" \
  --view "${FA[@]}" \
  --H ${H} --W ${W} --D ${D} \
  --spatial-dims 3 \
  --L ${L} --K ${K} --hidden ${HIDDEN} \
  --batch ${BATCH} \
  --val-frac 0.0 \
  --max-iter "${ITERATIONS}" \
  --devices cuda:0 --precision mixed --amp-dtype bf16 \
  --ema --ema-decay 0.9997 \
  --auto-resume \
  --aug-schedules "${AUG_PARAMS_PHASE1}" \
  --lr ${LR} --warmup-iters ${WARMUP} \
  --lr-decay-gamma 1.0 \
  --lr-decay-steps 0 \
  --eval-interval 1000 --plot-interval 1000 \
  --grad-clip 1.0 \
  --plateau-factor ${PLATEAU_FACTOR} --plateau-patience ${PLATEAU_PATIENCE} \
  --plateau-threshold ${PLATEAU_THRESHOLD} --plateau-cooldown ${PLATEAU_COOLDOWN} \
  --grad-accum ${GRAD_ACCUM} \
  --train-samples 3000 --val-samples 128 \
  --smooth-alpha 0.05 \
  --sample-mode model \
  --sample-temp 1.0 \
  --weighting fixed \
  --align "${ALIGN}" \
  --align-weight "${ALIGN_WEIGHT}" \
  --align-warmup "${ALIGN_WARMUP}" \
  --vicreg-inv "${ALIGN_VICREG_INV}" \
  --vicreg-var "${ALIGN_VICREG_VAR}" \
  --vicreg-cov "${ALIGN_VICREG_COV}" \
  --vicreg-gamma "${ALIGN_VICREG_GAMMA}" \
  --proj-hidden 320 --proj-dim 256 \
  --screen "${SCREEN_METHOD}" \
  --screen-warmup "${SCREEN_WARMUP}" \
  --screen-refresh "${SCREEN_REFRESH}" \
  --screen-frac "${SCREEN_FRAC}" \
  --cca-ridge "${CCA_RIDGE}" \
  --prefilter-frac "${PREFILTER_FRAC}" \
  --num-workers ${NUM_WORKERS} \
  --out-dir "${OUTDIR}"

