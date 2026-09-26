#!/usr/bin/env bash
set -euo pipefail

# --- SOLUTION ANTI-BLOCAGE & ANTI-FRAGMENTATION ---
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
export PYTHONPATH="/home/ntustison/Pkg/ANTsTorch:${PYTHONPATH:-}"
# --------------------------------------------------

# Total steps
ITERATIONS=120000

# Model / Data (3D T1)
H=112; W=128; D=112               
L=4; K="12 16 24 32"; HIDDEN="48 64 96 128"

# --- CONFIG MULTI-GPU & VRAM ROBUSTE ---
BATCH=12
GRAD_ACCUM=1
NUM_WORKERS=4
VAL_SAMPLES=32
VAL_FRAC=0.0
PRECISION="mixed"
# ---------------------------------------

OUTDIR="runs3d/dlbs_t1_${H}x${W}x${D}_L${L}_MultiK_MultiHC"

# Optimization
LR=2.5e-5
WARMUP=5000 # 2000
LR_DECAY_GAMMA=0.5
LR_DECAY_STEPS=120000
WEIGHT_DECAY=1e-5

PLATEAU_FACTOR=0.999999
PLATEAU_PATIENCE=100000
PLATEAU_THRESHOLD=1e-3
PLATEAU_COOLDOWN=5

# One-view experiment: latent alignment and screening are disabled.
ALIGN="none"

# Sampling / Eval
SAMPLE_TEMP=0.8
EVAL_INTERVAL=1000
PLOT_INTERVAL=1000

GRAD_CLIP=0.1

# Base distribution / Scale config
SCALE_CAP=0.1
GLOWBASE_MIN_LOG=-1.0
GLOWBASE_MAX_LOG=1.0
SCALE_MAP="tanh"
GLOWBASE_LOGSCALE_FACTOR=1.0

ACTNORM_SCALE_CAP=0.1
LEGACY_CONV_CAP=2.5

# Augmentation schedule : Aligné sur le JSON (fin à 100k)
AUG_STOP_STEP=100000

AUG_PARAMS="noise_std:cos:0.02->0.001@${AUG_STOP_STEP},\
sd_affine:cos:0.05->0.01@${AUG_STOP_STEP},\
sd_deformation:linear:12.0->2.0@${AUG_STOP_STEP},\
sd_simulated_bias_field:cos:0.20->0.0@${AUG_STOP_STEP},\
sd_histogram_warping:cos:0.04->0.0@${AUG_STOP_STEP}"

DLBS_ROOT="/home/ntustison/Data/ds004856/BIDSAlignedToTemplate/"

mapfile -t T1 < <(find "${DLBS_ROOT}" -path '*/sub-*/ses-wave1/anat/*T1w.nii.gz' -type f | sort)

echo "T1 Volumes trouvés: ${#T1[@]}"
if (( ${#T1[@]} == 0 )); then
  echo "Aucun volume T1 trouvé sous ${DLBS_ROOT}" >&2
  exit 1
fi

torchrun --standalone --nproc_per_node=2 -m antstorch.lamnr_flows.scripts.train_lamnr_glow_3d \
  --view "${T1[@]}" \
  --auto-resume \
  --H ${H} --W ${W} --D ${D} \
  --spatial-dims 3 \
  --L ${L} --K ${K} --hidden ${HIDDEN} \
  --batch ${BATCH} \
  --grad-accum ${GRAD_ACCUM} \
  --val-frac ${VAL_FRAC} \
  --max-iter "${ITERATIONS}" \
  --precision ${PRECISION} \
  --amp-dtype bf16 \
  --ema --ema-decay 0.9997 \
  --aug-schedules "${AUG_PARAMS}" \
  --lr ${LR} --warmup-iters ${WARMUP} \
  --lr-decay-gamma ${LR_DECAY_GAMMA} --lr-decay-steps ${LR_DECAY_STEPS} \
  --weight-decay ${WEIGHT_DECAY} \
  --eval-interval ${EVAL_INTERVAL} --plot-interval ${PLOT_INTERVAL} \
  --grad-clip ${GRAD_CLIP} \
  --plateau-factor ${PLATEAU_FACTOR} --plateau-patience ${PLATEAU_PATIENCE} \
  --plateau-threshold ${PLATEAU_THRESHOLD} --plateau-cooldown ${PLATEAU_COOLDOWN} \
  --train-samples 3000 --val-samples ${VAL_SAMPLES} \
  --smooth-alpha 0.15 \
  --sample-mode model --sample-temp ${SAMPLE_TEMP} \
  --weighting fixed \
  --align "${ALIGN}" \
  --scale-cap ${SCALE_CAP} \
  --legacy-conv-cap ${LEGACY_CONV_CAP} \
  --glowbase-logscale-factor ${GLOWBASE_LOGSCALE_FACTOR} \
  --glowbase-max-log ${GLOWBASE_MAX_LOG} --glowbase-min-log ${GLOWBASE_MIN_LOG} \
  --out-dir "${OUTDIR}" \
  --num-workers ${NUM_WORKERS} \
  --scale-map ${SCALE_MAP} \
  --actnorm-scale-cap ${ACTNORM_SCALE_CAP} --net-actnorm \
  --grad-checkpoint auto
