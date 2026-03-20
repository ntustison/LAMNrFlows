#!/usr/bin/env bash
set -euo pipefail

# --- SOLUTION ANTI-BLOCAGE & PERF ---
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ------------------------------------

# total steps
iterations=100000

# model / data
H=96; W=128; L=5; K=12; hidden=192  # K=32 comme en 3D ? À vous de voir (K=12 était votre valeur 2D)
SLICE_IDX=115
# Note: K=12 est bien pour la 2D, mais vous pouvez monter si la VRAM le permet.

# optimization
LR=5e-5
WARMUP=2000
WEIGHT_DECAY=1e-6

# --- CONFIG MULTI-GPU ROBUSTE ---
BATCH=48             # Plus gros batch possible (ajustez selon VRAM)
GRAD_ACCUM=2         # 32 * 4 = 128 (Batch effectif)
NUM_WORKERS=8        # Activé car OMP_NUM_THREADS=1 protège du blocage
# DEVICES="cuda:0,cuda:1"
DEVICES="cuda:1"
PRECISION="float"    # FP32 pour éviter les NaNs
# --------------------------------

OUTDIR="runs2d/dlbs_t1_t2flair_fa_${H}x${W}_K${K}_L${L}_HC${hidden}"

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
VAL_FRAC=0.1

# base distribution / scale config
SCALE_CAP=1.0
GLOWBASE_MIN_LOG=-5.0
GLOWBASE_MAX_LOG=5.0
GLOWBASE_LOGSCALE_FACTOR=1.0
SCALE_MAP="tanh"

# Augmentation schedule
aug_params_phase1="noise_std:cos:0.05->0.004@${iterations},\
sd_affine:cos:0.05->0.00@${iterations},\
sd_deformation:linear:12.0->0.6@${iterations},\
sd_simulated_bias_field:cos:0.20->0.03@${iterations},\
sd_histogram_warping:cos:0.04->0.008@${iterations}"

DLBS_ROOT="/home/ntustison/Data/ds004856/BIDSAlignedToTemplate/"

mapfile -t T1 < <(ls -1 ${DLBS_ROOT}/sub-20*/ses-wave1/anat/*T1w.nii.gz | sort)
mapfile -t T2 < <(ls -1 ${DLBS_ROOT}/sub-20*/ses-wave1/anat/*T2w.nii.gz | sort)
mapfile -t FA < <(ls -1 ${DLBS_ROOT}/sub-20*/ses-wave1/dwi/*fa.nii.gz | sort)

echo "T1: ${#T1[@]}  T2: ${#T2[@]}  FA: ${#FA[@]}"

python train_2d.py \
  --view "${T1[@]}" \
  --view "${T2[@]}" \
  --view "${FA[@]}" \
  --H ${H} --W ${W} --L ${L} --K ${K} --hidden ${hidden} \
  --batch ${BATCH} \
  --grad-accum ${GRAD_ACCUM} \
  --slice-idx ${SLICE_IDX} --val-frac ${VAL_FRAC} \
  --max-iter "${iterations}" \
  --devices ${DEVICES} --precision ${PRECISION} --amp-dtype bf16 \
  --ema --ema-decay 0.9997 \
  --auto-resume \
  --aug-schedules "${aug_params_phase1}" \
  --lr ${LR} --warmup-iters ${WARMUP} \
  --eval-interval ${EVAL_INTERVAL} --plot-interval ${PLOT_INTERVAL} \
  --grad-clip 0.1 \
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