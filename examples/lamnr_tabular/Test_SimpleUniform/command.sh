#/usr/bin/zsh

python3 ../train_lamnr_flows_tabular.py \
  --views UniformSimulatedData/uniform_10000x4.csv \
  --output-prefix ./runs/uniform_singleview \
  --cuda-device "cuda:1" \
  \
  --base-distribution DiagGaussian \
  --base-sigma 1.0 \
  --base-min-log -5.0 \
  --base-max-log  5.0 \
  \
  --K 8 \
  --additive-first-n 6 \
  --scale-cap 1.5 \
  --spectral-norm-scales \
  --actnorm-every 0 \
  --mask-mode alternating \
  \
  --normalization 0mean \
  --add-noise-in normalized \
  --impute mean \
  --jitter-alpha 0.02 \
  --jitter-alpha-end 0.0 \
  --jitter-alpha-mode cosine \
  \
  --batch-size 256 \
  --lr 1e-4 \
  --weight-decay 5e-5 \
  \
  --penalty-type none \
  --tradeoff-mode uncertainty \
  --lambda-penalty 1.0 \
  --target-ratio 9.0 \
  --vicreg-w-inv 25.0 \
  --vicreg-w-var 25.0 \
  --vicreg-w-cov 1.0 \
  --vicreg-gamma 1.0 \
  --penalty-warmup-iters 400 \
  \
  --best-selection-metric val_bpd \
  --max-iter 1000 \
  --val-interval 10 \
  --early-stop-enabled \
  --early-stop-patience 30 \
  --early-stop-min-delta 1e-4 \
  --early-stop-min-iters 2000 \
  \
  --save-z \
  --verbose


python3 ../plot_csv_distributions.py runs/uniform_z_view0.csv --ncols 4 -o ./

Rscript sanity_check_gauss_decor.R  \
  --raw UniformSimulatedData/uniform_10000x4.csv \
  --z runs/uniform_z_view0.csv \
  --labelraw raw \
  --labelsz z \
  --outdir sanity_uniform
