#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# total steps
iterations=120000

# model / data
H=256; W=256; L=6; K=12; hidden=192
SLICE_IDX=138

# optimization
LR=5e-5
WARMUP=0
WEIGHT_DECAY=1e-6
BATCH=16
GRAD_ACCUM=8
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

ALIGN_WARMUP=500

SCREEN_METHOD="cca"
SCREEN_FRAC=0.5
SCREEN_WARMUP=1000
SCREEN_REFRESH=5000
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
OUTDIR="runs/ppmi_t1_fa_${H}x${W}_K${K}_L${L}_HC${hidden}"

# base distribution / scale config
SCALE_CAP=2.0
GLOWBASE_MIN_LOG=-5.0
GLOWBASE_MAX_LOG=5.0
GLOWBASE_LOGSCALE_FACTOR=3.0
SCALE_MAP="tanh"

# -----------------------------
# Augmentation schedule (exact string from config)
# -----------------------------
aug_params_phase1="noise_std:cos:0.05->0.004@${iterations},\
sd_affine:cos:0.05->0.00@${iterations},\
sd_deformation:linear:12.0->0.6@${iterations},\
sd_simulated_bias_field:cos:0.20->0.03@${iterations},\
sd_histogram_warping:cos:0.04->0.008@${iterations}"


PPMI_ROOT="/home/ntustison/Data/PPMI_Dataset"

mapfile -t T1 < <(ls -1 ${PPMI_ROOT}/sub*/ses*/*ppmixt1.nii.gz | sort)
mapfile -t FA < <(ls -1 ${PPMI_ROOT}/sub*/ses*/*ppmixfa.nii.gz | sort)

echo "T1: ${#T1[@]}  FA: ${#FA[@]}"

python train_2d.py \
  --view "${T1[@]}" \
  --view "${FA[@]}" \
  --H ${H} --W ${W} --L ${L} --K ${K} --hidden ${hidden} \
  --batch ${BATCH} \
  --grad-accum ${GRAD_ACCUM} \
  --slice-idx ${SLICE_IDX} --val-frac ${VAL_FRAC} \
  --max-iter "${iterations}" \
  --devices ${DEVICES} --precision mixed --amp-dtype bf16 \
  --ema --ema-decay 0.9997 \
  --auto-resume \
  --aug-schedules "${aug_params_phase1}" \
  --lr ${LR} --warmup-iters ${WARMUP} \
  --eval-interval ${EVAL_INTERVAL} --plot-interval ${PLOT_INTERVAL} \
  --grad-clip 1.0 \
  --plateau-factor ${PLATEAU_FACTOR} --plateau-patience ${PLATEAU_PATIENCE} \
  --plateau-threshold ${PLATEAU_THRESHOLD} --plateau-cooldown ${PLATEAU_COOLDOWN} \
  --train-samples ${TRAIN_SAMPLES} --val-samples ${VAL_SAMPLES} \
  --smooth-alpha 0.05 \
  --sample-mode model --sample-temp ${SAMPLE_TEMP} \
  --weighting fixed \
  --align "${ALIGN}" --align-weight "${ALIGN_WEIGHT}" --align-warmup "${ALIGN_WARMUP}" \
  --vicreg-inv "${ALIGN_VICREG_INV}" \
  --vicreg-var "${ALIGN_VICREG_VAR}" \
  --vicreg-cov "${ALIGN_VICREG_COV}" \
  --vicreg-gamma "${ALIGN_VICREG_GAMMA}" \
  --screen "${SCREEN_METHOD}" \
  --screen-warmup "${SCREEN_WARMUP}" --screen-refresh "${SCREEN_REFRESH}" --screen-frac "${SCREEN_FRAC}" \
  --cca-ridge "${CCA_RIDGE}" --prefilter-frac "${PREFILTER_FRAC}" \
  --scale-cap ${SCALE_CAP} \
  --glowbase-max-log ${GLOWBASE_MAX_LOG} --glowbase-min-log ${GLOWBASE_MIN_LOG} \
  --out-dir "${OUTDIR}" \
  --num-workers ${NUM_WORKERS} \
  --weight-decay ${WEIGHT_DECAY} \
  --scale-map ${SCALE_MAP} 
