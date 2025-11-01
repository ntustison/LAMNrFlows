
# iterations=30000
# align=vicreg

# aug_params="noise_std:cos:0.08->0.01@${iterations},\
# sd_affine:cos:0.04->0.00@$((iterations*4/5)),\
# sd_deformation:linear:8.0->0.5@$((iterations*7/10)),\
# sd_simulated_bias_field:cos:0.30->0.05@${iterations},\
# sd_histogram_warping:cos:0.04->0.01@${iterations}"

# python train_cohort.py \
#   --data cohort \
#   --data-root ~/Data/NormalizingFlows/Nifti \
#   --modalities T2 T1 FA \
#   --H 32 --W 32 --L 2 --K 4 --hidden 48 \
#   --batch 128 \
#   --slice-idx 60 \
#   --val-frac 0.10 \
#   --max-iter "${iterations}" \
#   --eval-interval 500 --plot-interval 500 \
#   --devices cuda:1 --precision mixed --ema --ema-decay 0.9995 \
#   --auto-resume \
#   --aug-schedules "${aug_params}" \
#   --lr 1e-4 --warmup-iters 400 \
#   --train-samples 3000 --val-samples 128 \
#   --smooth-alpha 0.05 \
#   --sample-mode model \
#   --sample-temp 0.85 \
#   --align "${align}" \
#   --out-dir "runs/t1_t2_fa_32x32_${align}"

extra=30000
align=vicreg
iterations=30000  # same as first phase, just to reuse aug_params endpoints
aug_params="noise_std:cos:0.08->0.01@${iterations},\
sd_affine:cos:0.04->0.00@$((iterations*4/5)),\
sd_deformation:linear:8.0->0.5@$((iterations*7/10)),\
sd_simulated_bias_field:cos:0.30->0.05@${iterations},\
sd_histogram_warping:cos:0.04->0.01@${iterations}"

python train_cohort.py \
  --auto-resume --extra-iters ${extra} \
  --data cohort \
  --data-root ~/Data/NormalizingFlows/Nifti \
  --modalities T2 T1 FA \
  --H 32 --W 32 --L 2 --K 4 --hidden 48 \
  --batch 128 \
  --slice-idx 60 --val-frac 0.10 \
  --devices cuda:1 --precision mixed --ema --ema-decay 0.9995 \
  --aug-schedules "${aug_params}" \
  --lr 1e-4 --warmup-iters 400 \
  --train-samples 3000 --val-samples 128 \
  --smooth-alpha 0.05 \
  --sample-mode model --sample-temp 0.85 \
  --align "${align}" \
  --out-dir "runs/t1_t2_fa_32x32_${align}"





