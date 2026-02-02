#!/usr/bin/env bash
set -euo pipefail

# ---------- total steps ----------
iterations=120000              # phase 1
extra=0                       # phase 2
total=$((iterations + extra))  # horizon for phase-1 aug schedule

# ---------- 3D arch ----------
H=64
W=64
D=64               # depth for 3D Glow
L=3
K=24
hidden=320
BATCH=3             # start conservative in 3D; bump if VRAM allows

align=vicreg
align_weight=0.005
ALIGN_WARMUP=20000
OUTDIR="runs3d/hcp_t1_t2_fa_${H}x${W}x${D}_K${K}_L${L}_H${hidden}_${align}_6"

# ---------- screening configuration ----------
SCREEN_METHOD=cca           # none | cca | hsic
SCREEN_FRAC=0.25             # keep top 50% dims for alignment
SCREEN_WARMUP=10000          # start screening after N iters
SCREEN_REFRESH=0            # 0 = discover once; else refresh cadence
CCA_RIDGE=1e-3              # stability for CCA
PREFILTER_FRAC=0.5          # HSIC Pearson prefilter (ignored for CCA)

# Optimization
LR=2.5e-5      
WARMUP=5000  


# ---------- augmentation schedules ----------

# Phase 1: original decreasing schedule (strong -> weak)
aug_params_phase1="noise_std:cos:0.05->0.004@${total},\
sd_affine:cos:0.05->0.00@$((total)),\
sd_deformation:linear:12.0->0.6@$((total)),\
sd_simulated_bias_field:cos:0.20->0.03@${total},\
sd_histogram_warping:cos:0.04->0.008@${total}"


# Phase 2: template-focused fine-tune (no shape; mild intensity jitter)
aug_params_phase2="\
noise_std:cos:0.004->0.004@${extra},\
sd_affine:cos:0.00->0.00@${extra},\
sd_deformation:linear:0.0->0.0@${extra},\
sd_simulated_bias_field:cos:0.00->0.00@${extra},\
sd_histogram_warping:cos:0.004->0.004@${extra}"


python download_hcp_data.py

# ---------- Phase 1: strong -> weak aug ----------
python train_cohort_screened_3d.py \
  --view ~/.antstorch/hcp*T1Template.nii.gz \
  --view ~/.antstorch/hcp*T2Template.nii.gz \
  --view ~/.antstorch/hcp*FATemplate.nii.gz \
  --H ${H} --W ${W} --D ${D} \
  --spatial-dims 3 \
  --L ${L} --K ${K} --hidden ${hidden} \
  --batch ${BATCH} \
  --val-frac 0.0 \
  --max-iter "${iterations}" \
  --devices cuda:0 --precision mixed --amp-dtype bf16 \
  --ema --ema-decay 0.9997 \
  --auto-resume \
  --aug-schedules "${aug_params_phase1}" \
  --lr ${LR} --warmup-iters ${WARMUP} \
  --lr-decay-gamma 1.0 \
  --lr-decay-steps 0 \
  --eval-interval 1000 --plot-interval 1000 \
  --grad-clip 1.0 \
  --grad-accum 1 \
  --train-samples 3000 --val-samples 128 \
  --smooth-alpha 0.05 \
  --sample-mode model \
  --sample-temp 1.0 \
  --weighting fixed \
  --align "${align}" \
  --align-weight "${align_weight}" \
  --align-warmup "${ALIGN_WARMUP}" \
  --proj-hidden 320 --proj-dim 256 \
  --screen "${SCREEN_METHOD}" \
  --screen-warmup "${SCREEN_WARMUP}" \
  --screen-refresh "${SCREEN_REFRESH}" \
  --screen-frac "${SCREEN_FRAC}" \
  --cca-ridge "${CCA_RIDGE}" \
  --prefilter-frac "${PREFILTER_FRAC}" \
  --out-dir "${OUTDIR}"

