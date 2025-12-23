#!/usr/bin/env bash
set -eu pipefail

# total steps
iterations=120000          # phase 1
extra=0                    # phase 2
total=$((iterations + extra))   # horizon for phase-1 aug schedule

# 128×128 high-capacity arch
# H=128; W=128; L=5; K=12; hidden=192
# BATCH=64

# 256x256 high-capacity arch
H=256; W=256; L=6; K=10; hidden=192
BATCH=16
align=vicreg
align_weight=0.003
ALIGN_WARMUP=20000
OUTDIR="runs/hcp_t1_t2_fa_${H}x${W}_K${K}_L${L}_H${hidden}_${align}_5"

# Screening configuration
SCREEN_METHOD=none           # options: none | cca | hsic
SCREEN_FRAC=0.5             # keep top 50% dims for alignment
SCREEN_WARMUP=0             # start screening after N iters
SCREEN_REFRESH=0            # 0 = discover once; else refresh cadence
CCA_RIDGE=1e-3              # stability for CCA
PREFILTER_FRAC=0.5          # HSIC Pearson prefilter (ignored for CCA)

# Optimization
LR=5e-5          # half of the 128×128 LR
WARMUP=5000      # longer warmup than 128×128

# ------------------------------
# Augmentation schedules
# ------------------------------

# Phase 1: original decreasing schedule (strong -> weak)
aug_params_phase1="noise_std:cos:0.05->0.004@${total},\
sd_affine:cos:0.05->0.00@$((total*3/5)),\
sd_deformation:linear:12.0->0.6@$((total*7/10)),\
sd_simulated_bias_field:cos:0.20->0.03@${total},\
sd_histogram_warping:cos:0.04->0.008@${total}"

# Phase 2: template-focused fine-tune (no shape; mild intensity jitter)
aug_params_phase2="noise_std:cos:0.004->0.004@${extra},\
sd_affine:cos:0.00->0.00@${extra},\
sd_deformation:linear:0.0->0.0@${extra},\
sd_simulated_bias_field:cos:0.00->0.00@${extra},\
sd_histogram_warping:cos:0.008->0.008@${extra}"

SLICE_IDX=116


python download_hcp_data.py

# ---------- Phase 1: strong -> weak aug ----------
python train_2d.py \
  --view ~/.antstorch/hcp*T1Template.nii.gz \
  --view ~/.antstorch/hcp*T2Template.nii.gz \
  --view ~/.antstorch/hcp*FATemplate.nii.gz \
  --H ${H} --W ${W} --L ${L} --K ${K} --hidden ${hidden} \
  --batch ${BATCH} \
  --slice-idx ${SLICE_IDX} --val-frac 0.05 \
  --max-iter "${iterations}" \
  --devices cuda:1 --precision mixed --amp-dtype bf16 \
  --ema --ema-decay 0.9997 \
  --auto-resume \
  --aug-schedules "${aug_params_phase1}" \
  --lr ${LR} --warmup-iters ${WARMUP} \
  --eval-interval 5000 --plot-interval 5000 \
  --grad-clip 1.0 \
  --train-samples 3000 --val-samples 128 \
  --smooth-alpha 0.05 \
  --sample-mode model \
  --sample-temp 0.85 \
  --weighting fixed \
  --align "${align}" \
  --align-weight "${align_weight}" \
  --align-warmup "${ALIGN_WARMUP}" \
  --screen "${SCREEN_METHOD}" \
  --screen-warmup "${SCREEN_WARMUP}" \
  --screen-refresh "${SCREEN_REFRESH}" \
  --screen-frac "${SCREEN_FRAC}" \
  --cca-ridge "${CCA_RIDGE}" \
  --prefilter-frac "${PREFILTER_FRAC}" \
  --scale-cap 1.5 \
  --glowbase-max-log 2.0 \
  --glowbase-min-log -2.0 \
  --out-dir "${OUTDIR}"

CKPT="${OUTDIR}/training_state.pt"

for T in 0.7 0.85 1.0; do
  python train_2d.py \
    --resume "$CKPT" --use-ckpt-config \
    --view ~/.antstorch/hcp*T1Template.nii.gz \
    --view ~/.antstorch/hcp*T2Template.nii.gz \
    --view ~/.antstorch/hcp*FATemplate.nii.gz \
    --devices cuda:1 --precision mixed --amp-dtype bf16 \
    --sample-mode model --sample-temp "$T" \
    --lr 0 --warmup-iters 0 \
    --extra-iters 1 --eval-interval 1 \
    --out-dir "${OUTDIR}_tempsweep_T${T}"
done


# ---- Phase 2: resume, template-only fine-tune ----
# python train_2d.py \
#  --auto-resume --extra-iters ${extra} \
#  --view ~/Data/HCPTemplates/*/T_template0.nii.gz \
#  --view ~/Data/HCPTemplates/*/T_template1.nii.gz \
#  --view ~/Data/HCPTemplates/*/T_template2.nii.gz \
#  --H ${H} --W ${W} --L ${L} --K ${K} --hidden ${hidden} \
#  --batch ${BATCH} \
#  --slice-idx ${SLICE_IDX} --val-frac 0.0 \
#  --devices cuda:0 --precision mixed --amp-dtype bf16 \
#  --ema --ema-decay 0.9997 \
#  --aug-schedules "${aug_params_phase2}" \
#  --lr 3e-5 --warmup-iters 0 \
#  --grad-clip 1.0 \
#  --eval-interval 1000 --plot-interval 1000 \
#  --train-samples 3000 --val-samples 128 \
#  --smooth-alpha 0.05 \
#  --sample-mode model \
#  --weighting fixed \
#  --align "${align}" \
#  --align-weight "${align_weight}" \
#  --screen "${SCREEN_METHOD}" \
#  --screen-warmup "${SCREEN_WARMUP}" \
#  --screen-refresh "${SCREEN_REFRESH}" \
#  --screen-frac "${SCREEN_FRAC}" \
#  --cca-ridge "${CCA_RIDGE}" \
#  --prefilter-frac "${PREFILTER_FRAC}" \
#  --out-dir "${OUTDIR}"

