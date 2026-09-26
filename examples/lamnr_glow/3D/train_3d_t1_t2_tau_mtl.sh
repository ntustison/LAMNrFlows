#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/ntustison/Pkg/ANTsTorch:${PYTHONPATH:-}"

MANIFEST="oasis3_t1_t2_tau_mtl_manifest.csv"
CONFIG="oasis3_t1_t2_tau_mtl.json"
OUTDIR="runs3d/oasis3_t1_t2_tau_left_mtl_40x40x64_K8_12_16"

NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1

torchrun \
  --standalone \
  --nproc-per-node="${NUM_GPUS}" \
  -m antstorch.lamnr_flows.scripts.train_lamnr_flows_hybrid \
  --devices cuda \
  --manifest "${MANIFEST}" \
  --config "${CONFIG}" \
  --out-dir "${OUTDIR}" \
  --auto-resume \
  --precision float \
  --amp-dtype bf16 \
  --batch-size 16 \
  --accum-steps 1 \
  --train-samples 3000 \
  --val-samples 32 \
  --val-batches 8 \
  --val-fraction 0 \
  --max-iter 120000 \
  --lr 2.5e-5 \
  --warmup-iters 10000 \
  --lr-decay-gamma 0.5 \
  --lr-decay-steps 120000 \
  --weight-decay 1e-5 \
  --grad-clip 0.2 \
  --grad-checkpoint auto \
  --ema \
  --ema-decay 0.9997 \
  --plateau-factor 0.999999 \
  --plateau-patience 100000 \
  --plateau-threshold 1e-3 \
  --plateau-cooldown 5 \
  --eval-interval 1000 \
  --preview-interval 1000 \
  --preview-samples 100 \
  --preview-columns 10 \
  --sample-mode model \
  --sample-temp 0.75 \
  --alignment-latents all-pooled \
  --alignment-pool-size 2 \
  --weighting fixed \
  --tabular-base DiagGaussian \
  --align vicreg \
  --align-weight 3e-3 \
  --align-warmup 5000 \
  --screen none \
  --proj-dim 8 \
  --proj-hidden 32 \
  --vicreg-inv 10.0 \
  --vicreg-var 10.0 \
  --vicreg-cov 1.0 \
  --vicreg-gamma 1.0 \
  --screen-warmup 1000 \
  --screen-refresh 5000 \
  --screen-frac 0.5 \
  --cca-ridge 1e-3 \
  --prefilter-frac 0.5 \
  --num-workers 4 
