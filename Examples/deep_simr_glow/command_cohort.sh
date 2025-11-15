#!/usr/bin/env bash
set -eu pipefail

# total steps (single horizon across both phases)
iterations=80000          # phase 1
extra=40000               # phase 2
total=$((iterations + extra))

# 128×128 high-capacity arch
H=128; W=128; L=5; K=12; hidden=192
BATCH=64                  # drop to 48 if VRAM is tight
align=vicreg
align_weight=0.018
OUTDIR="runs/t1_t2_fa_${H}x${W}_${align}_K${K}_H${hidden}_${align}"

# one schedule over the FULL horizon (no discontinuities)
aug_params="noise_std:cos:0.05->0.004@${total},\
sd_affine:cos:0.05->0.00@$((total*3/5)),\
sd_deformation:linear:12.0->0.6@$((total*7/10)),\
sd_simulated_bias_field:cos:0.20->0.03@${total},\
sd_histogram_warping:cos:0.04->0.008@${total}"

# ---- Phase 1 ----
python train_cohort.py \
  --view ~/Data/NormalizingFlows/Nifti/*/T1.nii.gz \
  --view ~/Data/NormalizingFlows/Nifti/*/T2.nii.gz \
  --view ~/Data/NormalizingFlows/Nifti/*/FA.nii.gz \
  --H ${H} --W ${W} --L ${L} --K ${K} --hidden ${hidden} \
  --batch ${BATCH} \
  --slice-idx 60 --val-frac 0.10 \
  --max-iter "${iterations}" \
  --devices cuda:1 --precision mixed --amp-dtype bf16 \
  --ema --ema-decay 0.9997 \
  --auto-resume \
  --aug-schedules "${aug_params}" \
  --lr 1e-4 --warmup-iters 1000 \
  --eval-interval 1000 --plot-interval 1000 \
  --grad-clip 1.0 \
  --train-samples 3000 --val-samples 128 \
  --smooth-alpha 0.05 \
  --sample-mode model \
  --weighting fixed \
  --align "${align}" \
  --align-weight "${align_weight}" \
  --out-dir "${OUTDIR}"

# ---- Phase 2 (resume; SAME aug string) ----
#         - no warmup on resume → smoother
python train_cohort.py \
  --auto-resume --extra-iters ${extra} \
  --view ~/Data/NormalizingFlows/Nifti/*/T1.nii.gz \
  --view ~/Data/NormalizingFlows/Nifti/*/T2.nii.gz \
  --view ~/Data/NormalizingFlows/Nifti/*/FA.nii.gz \
  --H ${H} --W ${W} --L ${L} --K ${K} --hidden ${hidden} \
  --batch ${BATCH} \
  --slice-idx 60 --val-frac 0.10 \
  --devices cuda:1 --precision mixed --amp-dtype bf16 \
  --ema --ema-decay 0.9997 \
  --aug-schedules "${aug_params}" \
  --lr 3e-5 --warmup-iters 0
  --grad-clip 1.0 \
  --eval-interval 1000 --plot-interval 1000 \
  --train-samples 3000 --val-samples 128 \
  --smooth-alpha 0.05 \
  --sample-mode model \
  --weighting fixed \
  --align "${align}" \
  --align-weight "${align_weight}" \
  --out-dir "${OUTDIR}"
