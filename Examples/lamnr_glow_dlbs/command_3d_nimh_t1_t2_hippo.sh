#!/usr/bin/env bash
set -euo pipefail

# --- SOLUTION ANTI-BLOCAGE & ANTI-FRAGMENTATION ---
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# --------------------------------------------------

# Total steps
ITERATIONS=100000

# Model / Data (3D)
H=50; W=40; D=64               
L=3; K=32; HIDDEN=128

# --- CONFIG MULTI-GPU & VRAM ROBUSTE ---
BATCH=8
GRAD_ACCUM=8        # Batch effectif stable à 128
NUM_WORKERS=4
VAL_SAMPLES=32       # Évite le pic OOM lors de l'évaluation
DEVICES="cuda:0"
PRECISION="float"    
# ---------------------------------------

OUTDIR="runs3d/nimh_t1_t2_hippo_${H}x${W}x${D}_K${K}_L${L}_HC${HIDDEN}"

# Optimization
LR=2.5e-5
WARMUP=5000
LR_DECAY_GAMMA=0.5
LR_DECAY_STEPS=120000

PLATEAU_FACTOR=0.999999
PLATEAU_PATIENCE=100000
PLATEAU_THRESHOLD=1e-3
PLATEAU_COOLDOWN=5

# Alignment (Désactivé car 1 seule vue T1)
ALIGN="vicreg"
ALIGN_WEIGHT=1.0
ALIGN_VICREG_INV=25.0
ALIGN_VICREG_VAR=25.0
ALIGN_VICREG_GAMMA=1.0
ALIGN_VICREG_COV=1.0

# Sampling / Eval
SAMPLE_TEMP=1.0
EVAL_INTERVAL=1000
PLOT_INTERVAL=1000

GRAD_CLIP=1.0

# Base distribution / Scale config
SCALE_CAP=1.0
GLOWBASE_MIN_LOG=-5.0
GLOWBASE_MAX_LOG=5.0
SCALE_MAP="tanh"
GLOWBASE_LOGSCALE_FACTOR=1.0

# Augmentation schedule : Chute vers un palier de déquantification minimal à 120k

AUG_STOP_STEP=$(( ITERATIONS * 4 / 5 ))

AUG_PARAMS="noise_std:cos:0.05->0.002@${AUG_STOP_STEP},\
sd_affine:cos:0.05->0.005@${AUG_STOP_STEP},\
sd_deformation:linear:12.0->0.2@${AUG_STOP_STEP},\
sd_simulated_bias_field:cos:0.20->0.01@${AUG_STOP_STEP},\
sd_histogram_warping:cos:0.04->0.002@${AUG_STOP_STEP}"

DLBS_ROOT="/home/ntustison/Data/ds005752/BIDSAlignedToTemplate/"

mapfile -t T1 < <(ls -1 ${DLBS_ROOT}/sub-*/ses-*/anat/*T1w.nii.gz | sort)
mapfile -t T2 < <(ls -1 ${DLBS_ROOT}/sub-*/ses-*/anat/*T2w.nii.gz | sort)

echo "T1 Volumes trouvés: ${#T1[@]}"
echo "T2 Volumes trouvés: ${#T2[@]}"

python train_lamnr_glow_3d.py \
  --view "${T1[@]}" \
  --view "${T2[@]}" \
  --auto-resume \
  --H ${H} --W ${W} --D ${D} \
  --spatial-dims 3 \
  --L ${L} --K ${K} --hidden ${HIDDEN} \
  --batch ${BATCH} \
  --grad-accum ${GRAD_ACCUM} \
  --val-frac 0.1 \
  --max-iter "${ITERATIONS}" \
  --devices ${DEVICES} --precision ${PRECISION} --amp-dtype bf16 \
  --ema --ema-decay 0.9997 \
  --aug-schedules "${AUG_PARAMS}" \
  --lr ${LR} --warmup-iters ${WARMUP} \
  --lr-decay-gamma ${LR_DECAY_GAMMA} --lr-decay-steps ${LR_DECAY_STEPS} \
  --eval-interval ${EVAL_INTERVAL} --plot-interval ${PLOT_INTERVAL} \
  --grad-clip ${GRAD_CLIP} \
  --plateau-factor ${PLATEAU_FACTOR} --plateau-patience ${PLATEAU_PATIENCE} \
  --plateau-threshold ${PLATEAU_THRESHOLD} --plateau-cooldown ${PLATEAU_COOLDOWN} \
  --train-samples 3000 --val-samples ${VAL_SAMPLES} \
  --smooth-alpha 0.05 \
  --sample-mode model --sample-temp ${SAMPLE_TEMP} \
  --weighting fixed \
  --align "${ALIGN}" --align-weight "${ALIGN_WEIGHT}" \
  --vicreg-inv "${ALIGN_VICREG_INV}" \
  --vicreg-var "${ALIGN_VICREG_VAR}" \
  --vicreg-cov "${ALIGN_VICREG_COV}" \
  --vicreg-gamma "${ALIGN_VICREG_GAMMA}" \
  --scale-cap ${SCALE_CAP} \
  --glowbase-max-log ${GLOWBASE_MAX_LOG} --glowbase-min-log ${GLOWBASE_MIN_LOG} \
  --out-dir "${OUTDIR}" \
  --num-workers ${NUM_WORKERS} \
  --scale-map ${SCALE_MAP}
