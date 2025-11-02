#!/usr/bin/env bash

# Cohort trainer (glob-based views) — two-phase run with auto-resume
# Uses train_glob.py and repeated --view <glob> (files are paired by subject folder).
# Phase 1: 32x32 debug for 30k iters
# Phase 2: +30k extra iters (auto-resume), same architecture & run dir

iterations=200
extra=200
align=vicreg

# Augmentation schedule (same as your recent runs)
aug_params="noise_std:cos:0.08->0.01@${iterations},\
sd_affine:cos:0.04->0.00@$((iterations*3/5)),\
sd_deformation:linear:10.0->0.5@$((iterations*7/10)),\
sd_simulated_bias_field:cos:0.20->0.05@${iterations},\
sd_histogram_warping:cos:0.03->0.01@${iterations}"

# ----- Phase 1 -----
python train_cohort.py \
  --view ~/Data/NormalizingFlows/Nifti/*/T1.nii.gz \
  --view ~/Data/NormalizingFlows/Nifti/*/T2.nii.gz \
  --view ~/Data/NormalizingFlows/Nifti/*/FA.nii.gz \
  --H 64 --W 64 --L 4 --K 8 --hidden 96 \
  --batch 128 \
  --slice-idx 60 \
  --val-frac 0.10 \
  --max-iter "${iterations}" \
  --eval-interval 500 --plot-interval 500 \
  --devices cuda:1 --precision mixed --ema --ema-decay 0.9995 \
  --auto-resume \
  --aug-schedules "${aug_params}" \
  --lr 1e-4 --warmup-iters 400 \
  --train-samples 3000 --val-samples 128 \
  --smooth-alpha 0.05 \
  --sample-mode model --sample-temp 0.88 \
  --align "${align}" \
  --out-dir "runs/t1_t2_fa_64x64_test_${align}"

# ----- Phase 2 (extra iterations) -----
# Keep the SAME arch flags and out-dir so weights load cleanly.
python train_cohort.py \
  --auto-resume --extra-iters ${extra} \
  --view ~/Data/NormalizingFlows/Nifti/*/T1.nii.gz \
  --view ~/Data/NormalizingFlows/Nifti/*/T2.nii.gz \
  --view ~/Data/NormalizingFlows/Nifti/*/FA.nii.gz \
  --H 64 --W 64 --L 4 --K 8 --hidden 96 \
  --batch 128 \
  --slice-idx 60 --val-frac 0.10 \
  --eval-interval 100 --plot-interval 100 \
  --devices cuda:1 --precision mixed --ema --ema-decay 0.9995 \
  --aug-schedules "${aug_params}" \
  --lr 1e-4 --warmup-iters 400 \
  --train-samples 3000 --val-samples 128 \
  --smooth-alpha 0.05 \
  --sample-mode model --sample-temp 0.85 \
  --align "${align}" \
  --out-dir "runs/t1_t2_fa_64x64_${align}"

# echo "Done."
