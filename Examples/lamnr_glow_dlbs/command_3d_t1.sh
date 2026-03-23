#!/usr/bin/env bash
set -euo pipefail

# --- SOLUTION ANTI-BLOCAGE ---
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
# -----------------------------

# --- SOLUTION ANTI-FRAGMENTATION (Ajoutez ceci) ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# --------------------------------------------------

# ---------- total steps ----------
ITERATIONS=300000              # phase 1
EXTRA=0                       # phase 2
TOTAL=$((ITERATIONS + EXTRA))  # horizon for phase-1 aug schedule

# ---------- 3D arch ----------
H=64
W=80
D=64               # depth for 3D Glow
L=4
K=32
HIDDEN=64
BATCH=8             # start conservative in 3D; bump if VRAM allows
GRAD_ACCUM=4
NUM_WORKERS=2

PLATEAU_FACTOR=0.999999
PLATEAU_PATIENCE=100000
PLATEAU_THRESHOLD=1e-3
PLATEAU_COOLDOWN=5

# alignment + screening
ALIGN="vicreg"
ALIGN_WEIGHT=0.0
ALIGN_VICREG_INV=25.0
ALIGN_VICREG_VAR=25.0
ALIGN_VICREG_GAMMA=1.0
ALIGN_VICREG_COV=1.0
ALIGN_WARMUP=1000

OUTDIR="runs3d/dlbs_t1_${H}x${W}x${D}_K${K}_L${L}_H${HIDDEN}"

# ---------- screening configuration ----------
SCREEN_METHOD=cca           # none | cca | hsic
SCREEN_FRAC=0.5             # keep top 50% dims for alignment
SCREEN_WARMUP=1000          # start screening after N iters
SCREEN_REFRESH=5000         # 0 = discover once; else refresh cadence
CCA_RIDGE=1e-3              # stability for CCA
PREFILTER_FRAC=0.5          # HSIC Pearson prefilter (ignored for CCA)

# sampling / eval
SAMPLE_TEMP=0.5
EVAL_INTERVAL=1000
PLOT_INTERVAL=1000

# base distribution / scale config
SCALE_CAP=1.0
GLOWBASE_MIN_LOG=-5.0
GLOWBASE_MAX_LOG=5.0
GLOWBASE_LOGSCALE_FACTOR=1.0
SCALE_MAP="tanh"

# Optimization
LR=2.5e-5      
WARMUP=2000 
DEVICES="cuda:0" 
PRECISION="float" # "float" required for multiple GPUs

# ---------- augmentation schedules ----------

# Augmentation schedule - MISE À JOUR : RAMPE VERS UN PALIER BAS, PAS ZÉRO
# noise_std:cos:0.05->0.01 : Maintient un bruit résiduel (0.01) pour la robustesse
# sd_deformation:linear:12.0->0.5 : Maintient une déformation légère résiduelle (0.5)
# sd_simulated_bias_field:cos:0.20->0.05 : Maintient un biais léger résiduel (0.05)
# sd_histogram_warping:cos:0.04->0.01 : Maintient un warping léger résiduel (0.01)
aug_params_phase1="noise_std:cos:0.05->0.01@${ITERATIONS},\
sd_affine:cos:0.05->0.005@${ITERATIONS},\
sd_deformation:linear:12.0->0.5@${ITERATIONS},\
sd_simulated_bias_field:cos:0.20->0.05@${ITERATIONS},\
sd_histogram_warping:cos:0.04->0.01@${ITERATIONS}"


DLBS_ROOT="/home/ntustison/Data/ds004856/BIDSAlignedToTemplate/"

mapfile -t T1 < <(ls -1 ${DLBS_ROOT}/sub-*/ses-wave1/anat/*T1w.nii.gz | sort)

echo "T1: ${#T1[@]}"

# ---------- Phase 1: strong -> weak aug ----------
python train_3d.py \
  --view "${T1[@]}" \
  --H ${H} --W ${W} --D ${D} --L ${L} --K ${K} --hidden ${HIDDEN} \
  --spatial-dims 3 \
  --batch ${BATCH} \
  --grad-accum ${GRAD_ACCUM} \
  --val-frac 0.1 \
  --max-iter ${ITERATIONS} \
  --devices ${DEVICES} --precision ${PRECISION} --amp-dtype bf16 \
  --ema --ema-decay 0.9997 \
  --auto-resume \
  --aug-schedules "${aug_params_phase1}" \
  --lr 2.5e-5 --warmup-iters 5000 \
  --eval-interval ${EVAL_INTERVAL} --plot-interval ${PLOT_INTERVAL} \
  --grad-clip 0.1 \
  --plateau-factor ${PLATEAU_FACTOR} --plateau-patience ${PLATEAU_PATIENCE} \
  --plateau-threshold ${PLATEAU_THRESHOLD} --plateau-cooldown ${PLATEAU_COOLDOWN} \
  --smooth-alpha 0.05 \
  --sample-mode model --sample-temp ${SAMPLE_TEMP} \
  --align "${ALIGN}" --align-weight "${ALIGN_WEIGHT}" \
  --align-warmup 1000 \
  --vicreg-inv "${ALIGN_VICREG_INV}" \
  --vicreg-var "${ALIGN_VICREG_VAR}" \
  --vicreg-cov "${ALIGN_VICREG_COV}" \
  --vicreg-gamma "${ALIGN_VICREG_GAMMA}" \
  --scale-cap ${SCALE_CAP} \
  --glowbase-max-log ${GLOWBASE_MAX_LOG} \
  --glowbase-min-log ${GLOWBASE_MIN_LOG} \
  --out-dir ${OUTDIR} \
  --num-workers ${NUM_WORKERS} \
  --scale-map ${SCALE_MAP} \
  --lr-decay-gamma 0.5 \
  --lr-decay-steps 200000

