#!/usr/bin/env python3
"""
Train SIMR normalizing-flow whiteners using ANTsTorch and (optionally) export latents/reconstructions.

Patched to work with dataset-owned normalization & the new apply() API.
- Adds CLI flags to control dataset normalization/jitter.
- Saves/loads per-view normalization stats ("dataset_normalizers").
- Backward compatible with older antstorch apply() that expects "use_training_standardization".

Exports (optional via flags):
  --save-z          : raw flow latents z (one CSV per view)
  --save-whitened   : PCA-projected / standardized latents (eps) (one CSV per view)
  --save-recon      : inverse-transformed reconstructions in observed scale (one CSV per view)

"""

import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import time
import math
import inspect
import warnings
import os
import json

from antstorch import lamnr_flows_whitener, apply_lamnr_flows_whitener



def _format_seconds(seconds: float) -> str:
    """
    Pretty-print seconds as Hh Mm Ss.
    """
    seconds = int(max(0, round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m or (h and s):
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def _estimate_seconds_per_iter_tabular(batch_size: int, total_dims: int, K: int) -> float:
    """
    Very rough heuristic for seconds/iteration for tabular flows.

    Scales with batch_size, total feature dimensionality, and flow depth (K).
    This is intentionally conservative and only meant to provide an order-of-magnitude ETA.
    """
    total_dims = max(1, int(total_dims))
    batch_size = max(1, int(batch_size))
    K = max(1, int(K))

    base = 0.002  # baseline overhead
    dim_term = 1e-5 * batch_size * total_dims
    depth_term = 5e-4 * (K / 32.0)
    return base + dim_term + depth_term


def _print_screen_dump(args, views):
    """
    Print a configuration screen dump similar to the Glow-based trainer.
    """
    n_views = len(views)
    n_samples = len(views[0]) if views else 0
    dims_per_view = [df.shape[1] for df in views]
    total_dims = sum(dims_per_view)

    print("\n=== Training configuration (tabular LAMNR flows) ===")
    print(f"Output prefix     : {args.output_prefix}")
    print(f"Number of views   : {n_views}")
    print(f"Samples per view  : {n_samples}")
    print(f"Dims per view     : {dims_per_view} (total={total_dims})")
    print(f"Base distribution : {args.base_distribution}")
    print(f"Flow depth K      : {args.K}")
    print(f"Scale cap         : {args.scale_cap}")
    print(f"Penalty type      : {args.penalty_type}")
    print(f"Tradeoff mode     : {args.tradeoff_mode}")
    print(f"Lambda penalty    : {args.lambda_penalty}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Max iterations    : {args.max_iter}")
    print(f"Val interval      : {args.val_interval}")
    print(f"CUDA device       : {args.cuda_device}")
    print(f"Seed              : {args.seed}")
    print("\nAll parsed arguments:")
    for k in sorted(vars(args).keys()):
        print(f"  {k}: {getattr(args, k)}")

    return n_samples, dims_per_view, total_dims


def _standardize_np(X: np.ndarray) -> np.ndarray:
    """
    Column-wise standardization with NaN handling: mean 0, std 1.
    Columns that are all-NaN are dropped.
    """
    X = np.asarray(X, dtype=float)
    # Impute NaNs with column means
    col_means = np.nanmean(X, axis=0)
    # Identify all-NaN columns (nanmean -> nan)
    valid = np.isfinite(col_means)
    if not np.all(valid):
        X = X[:, valid]
        col_means = col_means[valid]
    # Impute remaining NaNs
    inds = np.where(np.isnan(X))
    if inds[0].size > 0:
        X[inds] = np.take(col_means, inds[1])
    # Standardize
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    X = (X - mean) / std
    return X


def _rbf_gram(X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """
    RBF kernel Gram matrix with optional median heuristic for sigma.
    """
    X = np.asarray(X, dtype=float)
    sq_norms = np.sum(X * X, axis=1, keepdims=True)
    dist2 = sq_norms + sq_norms.T - 2.0 * (X @ X.T)
    # Numerical safety
    np.maximum(dist2, 0.0, out=dist2)

    if sigma is None or sigma <= 0.0:
        # Median heuristic on upper triangle
        triu = dist2[np.triu_indices_from(dist2, k=1)]
        triu = triu[np.isfinite(triu) & (triu > 0)]
        if triu.size == 0:
            sigma = 1.0
        else:
            med = np.median(triu)
            sigma = math.sqrt(0.5 * med) if med > 0 else 1.0

    K = np.exp(-dist2 / (2.0 * sigma * sigma))
    return K


def _center_gram(K: np.ndarray) -> np.ndarray:
    """
    Double-center a Gram matrix.
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / float(n)
    return H @ K @ H


def _hsic_value(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
    """
    Unnormalized HSIC estimator with RBF kernels.
    """
    n = X.shape[0]
    if n < 3:
        return 0.0
    K = _rbf_gram(X, sigma=sigma)
    L = _rbf_gram(Y, sigma=sigma)
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    # Biased estimator is fine for screening
    hsic = np.trace(Kc @ Lc) / ((n - 1.0) ** 2)
    return float(max(hsic, 0.0))


def _hsic_normalized(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
    """
    Normalized HSIC in [0, 1] via HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y)).
    """
    h_xy = _hsic_value(X, Y, sigma=sigma)
    if h_xy <= 0.0:
        return 0.0
    h_xx = _hsic_value(X, X, sigma=sigma)
    h_yy = _hsic_value(Y, Y, sigma=sigma)
    denom = math.sqrt(max(h_xx, 1e-12) * max(h_yy, 1e-12))
    val = h_xy / denom if denom > 0 else 0.0
    return float(max(min(val, 1.0), 0.0))


def _cca_max_corr(X: np.ndarray, Y: np.ndarray, reg: float = 1e-4) -> float:
    """
    Max canonical correlation between standardized X and Y via eigendecomposition.

    X, Y: shape [n, d1], [n, d2].
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = X.shape[0]
    if n < 3:
        return 0.0

    # Covariance matrices
    Cxx = (X.T @ X) / (n - 1.0)
    Cyy = (Y.T @ Y) / (n - 1.0)
    Cxy = (X.T @ Y) / (n - 1.0)

    # Regularization
    d1 = Cxx.shape[0]
    d2 = Cyy.shape[0]
    Cxx = Cxx + reg * np.eye(d1)
    Cyy = Cyy + reg * np.eye(d2)

    # Inverse square roots via eigen-decomposition
    ex, Ux = np.linalg.eigh(Cxx)
    ey, Uy = np.linalg.eigh(Cyy)
    ex[ex < 1e-12] = 1e-12
    ey[ey < 1e-12] = 1e-12
    Cxx_inv_sqrt = (Ux / np.sqrt(ex)) @ Ux.T
    Cyy_inv_sqrt = (Uy / np.sqrt(ey)) @ Uy.T

    T = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
    # Singular values of T are canonical correlations
    u, s, v = np.linalg.svd(T, full_matrices=False)
    if s.size == 0:
        return 0.0
    val = float(np.max(s))
    # Numerical clipping
    return float(max(min(val, 1.0), 0.0))


def _screen_views_for_alignment(args, views, verbose: bool = False):
    """
    Optional pre-training dependence screening using HSIC or CCA.

    If the average pairwise dependence across views is below args.screening_threshold,
    this function sets args.penalty_type = "none" (disabling latent alignment).
    """
    mode = getattr(args, "screening_mode", "none")
    if mode == "none":
        return
    if args.penalty_type == "none":
        return
    if len(views) < 2:
        return

    n = len(views[0])
    if n < 3:
        return

    frac = float(getattr(args, "screening_fraction", 0.2))
    frac = min(max(frac, 0.0), 1.0)
    max_samples = int(getattr(args, "screening_max_samples", 5000))
    thresh = float(getattr(args, "screening_threshold", 0.1))

    n_sample = max(10, min(n, max_samples, int(round(frac * n)) if frac > 0 else n))
    idx = np.random.choice(n, size=n_sample, replace=False)

    labels = [Path(p).stem for p in args.views]
    Xs = []
    for v in views:
        X = v.values
        X = X[idx, :]
        X = _standardize_np(X)
        Xs.append(X)

    scores = []
    pair_labels = []
    for i in range(len(Xs)):
        for j in range(i + 1, len(Xs)):
            Xi = Xs[i]
            Xj = Xs[j]
            if Xi.shape[0] < 3 or Xj.shape[0] < 3:
                s = 0.0
            else:
                if mode == "hsic":
                    s = _hsic_normalized(Xi, Xj, sigma=None)
                elif mode == "cca":
                    s = _cca_max_corr(Xi, Xj)
                else:
                    return  # unknown mode, silently skip
            scores.append(s)
            pair_labels.append(f"{labels[i]} vs {labels[j]}")

    if not scores:
        return

    scores = np.asarray(scores, dtype=float)
    mean_score = float(scores.mean())

    if verbose:
        print(f"=== Screening cross-view dependence (mode={mode}) ===")
        print(f"Using {n_sample} subjects (out of {n}) for screening.")
        for lbl, s in zip(pair_labels, scores):
            print(f"  {lbl}: {s:.4f}")
        print(f"  Average score: {mean_score:.4f}")
        print(f"  Threshold    : {thresh:.4f}")

    if mean_score < thresh:
        if verbose:
            print("  -> Average dependence below threshold; disabling alignment penalty (penalty_type='none').")
        args.penalty_type = "none"
    else:
        if verbose:
            print("  -> Dependence above threshold; keeping alignment penalty.")

def load_views(view_paths):
    views = []
    for p in view_paths:
        df = pd.read_csv(p)
        # Force numeric; non-numeric will become NaN → trainer/dataset handles imputation/normalization
        df = df.apply(pd.to_numeric, errors="coerce")
        views.append(df)
    # Basic shape check
    nset = {len(df) for df in views}
    if len(nset) != 1:
        raise ValueError(f"All views must have equal row counts; got {nset}")
    return views


def save_views(dfs, base_prefix: Path, tag: str):
    paths = []
    for i, df in enumerate(dfs):
        out = Path(f"{base_prefix}_{tag}_view{i}.csv")
        df.to_csv(out, index=False)
        paths.append(str(out))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", required=True, help="CSV paths for one or more views")
    ap.add_argument("--output-prefix", required=True, help="Prefix for saving models/metrics/config")

    # Model/base
    ap.add_argument("--base-distribution", default="GaussianPCA", choices=["GaussianPCA", "DiagGaussian"])
    ap.add_argument("--pca-latent-dimension", type=int, default=4)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--leaky-relu-negative-slope", type=float, default=0.2)

    # Flow/builder stability knobs
    ap.add_argument("--scale-cap", type=float, default=3.0, help="Bound for log-scale s via tanh; exp(s) in [e^-cap, e^cap]")
    ap.add_argument("--spectral-norm-scales", action="store_true", default=False, help="Apply spectral norm in scale MLP (if supported)")
    ap.add_argument("--additive-first-n", type=int, default=0, help="Use additive (no-scaling) couplings for first N layers")
    ap.add_argument("--actnorm-every", type=int, default=1, help="Insert ActNorm after every N couplings (1=after each)")
    ap.add_argument("--mask-mode", type=str, default="alternating", choices=["alternating", "rolling"], help="Mask alternation strategy")

    # Base distribution stability knobs
    ap.add_argument("--base-min-log", type=float, default=-5.0, help="Lower clamp for base log-scales (if supported)")
    ap.add_argument("--base-max-log", type=float, default=5.0, help="Upper clamp for base log-scales (if supported)")
    ap.add_argument("--base-sigma", type=float, default=0.1, help="Noise for GaussianPCA (if used)")

    # === Dataset normalization & jitter (new) ===
    ap.add_argument("--normalization", type=str, default="0mean", choices=["0mean","01","none"],
                    help="Per-view normalization mode: 0mean | 01 | none (None)")
    ap.add_argument("--add-noise-in", type=str, default="normalized", choices=["raw","normalized","none"],
                    help="Domain for alpha-jitter if enabled: raw/normalized/none")
    ap.add_argument("--impute", type=str, default="mean", choices=["none","mean","zero"],
                    help="Imputation applied after normalization")
    ap.add_argument("--dataset-normalizers-json", type=str, default=None,
                    help="If set, dump dataset normalization stats to this JSON file (one list item per view)")

    # Jitter schedule
    ap.add_argument("--jitter-alpha", type=float, default=0.0)
    ap.add_argument("--jitter-alpha-end", type=float, default=0.0)
    ap.add_argument("--jitter-alpha-mode", type=str, default="cosine", choices=["cosine", "linear", "exp"])
    ap.add_argument("--jitter-alpha-total-steps", type=int, default=None)

    # Optimization
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-iter", type=int, default=1200)
    ap.add_argument("--cuda-device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)

    # Penalty/tradeoff
    ap.add_argument("--tradeoff-mode", default="uncertainty", choices=["ema", "uncertainty", "fixed"])
    ap.add_argument("--target-ratio", type=float, default=9.0)
    ap.add_argument("--lambda-penalty", type=float, default=1.0)
    ap.add_argument("--ema-beta", type=float, default=0.98)

    # Latent alignment penalty / multi-view coupling
    ap.add_argument(
        "--penalty-type",
        default="barlow_twins_align",
        choices=[
            "decorrelate",
            "correlate",
            "barlow_twins_align",
            "pearson",
            "barlow_twins_multi",
            "vicreg",
            "info_nce",
            "hsic",
            "none",
        ],
    )

    # Barlow Twins (legacy + multi-view)
    ap.add_argument("--bt-lambda-diag", type=float, default=1.0)
    ap.add_argument("--bt-lambda-offdiag", type=float, default=5e-3)
    ap.add_argument("--bt-eps", type=float, default=1e-6)

    # InfoNCE temperature
    ap.add_argument(
        "--info-nce-T",
        type=float,
        default=0.2,
        help="Temperature for InfoNCE / NT-Xent when penalty_type='info_nce'.",
    )

    # VICReg weights
    ap.add_argument(
        "--vicreg-w-inv",
        type=float,
        default=25.0,
        help="Invariance (MSE) weight for VICReg when penalty_type='vicreg'.",
    )
    ap.add_argument(
        "--vicreg-w-var",
        type=float,
        default=25.0,
        help="Variance floor weight for VICReg when penalty_type='vicreg'.",
    )
    ap.add_argument(
        "--vicreg-w-cov",
        type=float,
        default=1.0,
        help="Covariance off-diagonal weight for VICReg when penalty_type='vicreg'.",
    )
    ap.add_argument(
        "--vicreg-gamma",
        type=float,
        default=1.0,
        help="Target std (gamma) for VICReg variance floor when penalty_type='vicreg'.",
    )

    # HSIC bandwidth
    ap.add_argument(
        "--hsic-sigma",
        type=float,
        default=0.0,
        help=(
            "RBF kernel bandwidth for HSIC when penalty_type='hsic'; "
            "0.0 = median heuristic per batch."
        ),
    )

    ap.add_argument("--penalty-warmup-iters", type=int, default=400)

    # Optional pre-training screening of cross-view dependence
    ap.add_argument(
        "--screening-mode",
        type=str,
        default="none",
        choices=["none", "hsic", "cca"],
        help="Optional pre-training dependence screening; may disable alignment if views are weakly related.",
    )
    ap.add_argument(
        "--screening-fraction",
        type=float,
        default=0.2,
        help="Fraction of subjects to sample for screening (0-1].",
    )
    ap.add_argument(
        "--screening-max-samples",
        type=int,
        default=5000,
        help="Maximum number of subjects to use for screening.",
    )
    ap.add_argument(
        "--screening-threshold",
        type=float,
        default=0.1,
        help="Minimum average dependence score required to keep alignment active. "
             "For 'cca' this is max canonical corr (0-1); for 'hsic' this is normalized HSIC (0-1).",
    )




# Scale regularizer
    ap.add_argument("--scale-penalty-weight", type=float, default=None,
                    help="Weight for mean|s| regularizer; passed only if whitener supports it")

    # Validation / early stopping
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--val-interval", type=int, default=200)
    ap.add_argument("--val-batch-size", type=int, default=2048)
    ap.add_argument("--early-stop-enabled", action="store_true", default=False)
    ap.add_argument("--early-stop-patience", type=int, default=300)
    ap.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    ap.add_argument("--early-stop-min-iters", type=int, default=600)
    ap.add_argument("--early-stop-beta", type=float, default=0.98)

    # Misc
    ap.add_argument("--best-selection-metric", default="val_bpd", choices=["val_bpd","smooth_total"])

    # Checkpointing / resume
    ap.add_argument("--resume-checkpoint", type=str, default=None)
    ap.add_argument("--save-checkpoint-dir", type=str, default=None)
    ap.add_argument("--checkpoint-interval", type=int, default=None, help="Defaults to --val-interval if omitted")
    ap.add_argument("--restore-best-for-final-eval", action="store_true", default=True)
    ap.add_argument("--verbose", action="store_true", default=False)

    # Optional exports
    ap.add_argument("--save-z", action="store_true", help="Export raw flow latents z_*_view{i}.csv")
    ap.add_argument("--save-whitened", default="pca", choices=["pca","full"])
    ap.add_argument("--save-recon", action="store_true", help="Export inverse reconstructions recon_view{i}.csv (observed scale)")

    args = ap.parse_args()
    verbose = args.verbose

    # Determinism
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load CSV views
    views = load_views(args.views)

    # Optional pre-training dependence screening (may disable alignment)
    _screen_views_for_alignment(args, views, verbose=verbose)

    # Screen dump of configuration & basic data stats
    if verbose:
        n_samples, dims_per_view, total_dims = _print_screen_dump(args, views)
    else:
        n_samples = len(views[0]) if views else 0
        dims_per_view = [df.shape[1] for df in views]
        total_dims = sum(dims_per_view)

    # Rough ETA based on a simple heuristic, similar in spirit to the Glow trainer
    if verbose and args.max_iter is not None and args.max_iter > 0:
        sec_per_iter = _estimate_seconds_per_iter_tabular(
            batch_size=args.batch_size,
            total_dims=total_dims,
            K=args.K,
        )
        eta_seconds = sec_per_iter * args.max_iter
        eta_str = _format_seconds(eta_seconds)
        finish_time = time.localtime(time.time() + eta_seconds)
        finish_str = time.strftime("%Y-%m-%d %H:%M:%S", finish_time)
        print(f"Estimated total training time (heuristic): ~{eta_str} (finish around {finish_str})")


    # === Prepare kwargs and filter by whitener signature for backward compatibility ===
    base_kwargs = dict(
        base_distribution=args.base_distribution,
        pca_latent_dimension=args.pca_latent_dimension,
        base_min_log=args.base_min_log,
        base_max_log=args.base_max_log,
        base_sigma=args.base_sigma,
    )

    flow_kwargs = dict(
        K=args.K,
        leaky_relu_negative_slope=args.leaky_relu_negative_slope,
        scale_cap=args.scale_cap,
        spectral_norm_scales=args.spectral_norm_scales,
        additive_first_n=args.additive_first_n,
        actnorm_every=args.actnorm_every,
        mask_mode=args.mask_mode,
    )

    # Normalization flags → trainer kwargs
    norm_mode = None if args.normalization == "none" else args.normalization
    train_kwargs = dict(
        jitter_alpha=args.jitter_alpha,
        jitter_alpha_end=args.jitter_alpha_end,
        jitter_alpha_mode=args.jitter_alpha_mode,
        jitter_alpha_total_steps=args.jitter_alpha_total_steps,

        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        max_iter=args.max_iter,
        cuda_device=args.cuda_device,
        seed=args.seed,

        tradeoff_mode=args.tradeoff_mode,
        target_ratio=args.target_ratio,
        lambda_penalty=args.lambda_penalty,
        ema_beta=args.ema_beta,

        # Latent alignment configuration
        penalty_type=args.penalty_type,
        bt_lambda_diag=args.bt_lambda_diag,
        bt_lambda_offdiag=args.bt_lambda_offdiag,
        bt_eps=args.bt_eps,
        info_nce_T=getattr(args, "info_nce_T", None),
        vicreg_w_inv=getattr(args, "vicreg_w_inv", None),
        vicreg_w_var=getattr(args, "vicreg_w_var", None),
        vicreg_w_cov=getattr(args, "vicreg_w_cov", None),
        vicreg_gamma=getattr(args, "vicreg_gamma", None),
        hsic_sigma=getattr(args, "hsic_sigma", None),
        penalty_warmup_iters=args.penalty_warmup_iters,

        val_fraction=args.val_fraction,
        val_interval=args.val_interval,
        val_batch_size=args.val_batch_size,

        early_stop_enabled=args.early_stop_enabled,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_min_iters=args.early_stop_min_iters,
        early_stop_beta=args.early_stop_beta,

        best_selection_metric=args.best_selection_metric,
        restore_best_for_final_eval=args.restore_best_for_final_eval,
        resume_checkpoint=args.resume_checkpoint,
        save_checkpoint_dir=args.save_checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        verbose=verbose,

        # New dataset-owned normalization/jitter knobs (only passed if supported)
        normalization=norm_mode,
        add_noise_in=args.add_noise_in,
        impute=args.impute,
        dataset_normalizers_dump_path=args.dataset_normalizers_json,
    )

    # Optional: scale penalty weight
    if args.scale_penalty_weight is not None:
        train_kwargs["scale_penalty_weight"] = args.scale_penalty_weight

    # Merge kwargs and filter by signature
    call_kwargs = dict(views=views)
    call_kwargs.update(base_kwargs)
    call_kwargs.update(flow_kwargs)
    call_kwargs.update(train_kwargs)

    sig = inspect.signature(lamnr_flows_whitener)
    filtered_kwargs = {k: v for k, v in call_kwargs.items() if k in sig.parameters}

    # Train
    if verbose:
        print("\n=== Train ===")
        missing = sorted(set(call_kwargs) - set(filtered_kwargs))
        if missing:
            print("Note: the following knobs are not supported by your installed whitener and were omitted:")
            for k in missing:
                print("  -", k)

    start_time = time.perf_counter()
    result = lamnr_flows_whitener(**filtered_kwargs)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"Time elapsed: {elapsed_time:.4f} seconds")

    # Print metrics
    if verbose:
        print("\n=== Metrics ===")
        for k, v in result.get("metrics", {}).items():
            print(f"{k}: {v}")

    # If trainer didn't dump JSON and user asked for it, try to write from returned stats
    if args.dataset_normalizers_json and os.path.dirname(args.dataset_normalizers_json):
        dn = result.get("dataset_normalizers", None)
        if dn is not None:
            os.makedirs(os.path.dirname(args.dataset_normalizers_json), exist_ok=True)
            with open(args.dataset_normalizers_json, "w", encoding="utf-8") as f:
                json.dump(dn, f, indent=2)

    # Optional exports using the apply helper
    base_prefix = Path(args.output_prefix)
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    if args.save_z or args.save_whitened or args.save_recon:

        if verbose:
            print("\n=== Save outputs ===")

        # Detect which apply() API we have
        apply_sig = inspect.signature(apply_lamnr_flows_whitener)
        supports_new = "normalization_mode" in apply_sig.parameters

        # Prepare normalization hints for apply()
        norm_stats = result.get("dataset_normalizers", None)
        apply_common = dict(
            batch_size=args.val_batch_size,
            device=args.cuda_device,
        )

        # Forward transforms
        if args.save_z:
            if supports_new:
                z_views = apply_lamnr_flows_whitener(
                    trainer_output=result,
                    data=views,
                    direction="forward",
                    output_space="z",
                    normalization_mode=norm_mode,
                    normalization_stats=norm_stats,
                    fit_stats_on_data_if_missing=(norm_stats is None and norm_mode is not None),
                    **apply_common,
                )
            else:
                z_views = apply_lamnr_flows_whitener(
                    trainer_output=result,
                    data=views,
                    direction="forward",
                    output_space="z",
                    use_training_standardization=True,
                    **apply_common,
                )
            z_paths = save_views(z_views, base_prefix, "z")
            if verbose:
                print("  z latents:")
                for p in z_paths:
                    print("  ", p)

        wh_views = None
        if args.save_whitened == "pca" and args.base_distribution == "GaussianPCA":
            if supports_new:
                wh_views = apply_lamnr_flows_whitener(
                    trainer_output=result,
                    data=views,
                    direction="forward",
                    output_space="whitened",
                    normalization_mode=norm_mode,
                    normalization_stats=norm_stats,
                    fit_stats_on_data_if_missing=(norm_stats is None and norm_mode is not None),
                    **apply_common,
                )
            else:
                wh_views = apply_lamnr_flows_whitener(
                    trainer_output=result,
                    data=views,
                    direction="forward",
                    output_space="whitened",
                    use_training_standardization=True,
                    **apply_common,
                )
            wh_paths = save_views(wh_views, base_prefix, "whitened")
            if verbose:
                print("  whitened pca latents:")
                for p in wh_paths:
                    print("  ", p)

        elif args.save_whitened == "full" and args.base_distribution == "GaussianPCA":
            if supports_new:
                wh_views = apply_lamnr_flows_whitener(
                    trainer_output=result,
                    data=views,
                    direction="forward",
                    output_space="whitened_full",
                    normalization_mode=norm_mode,
                    normalization_stats=norm_stats,
                    fit_stats_on_data_if_missing=(norm_stats is None and norm_mode is not None),
                    **apply_common,
                )
            else:
                wh_views = apply_lamnr_flows_whitener(
                    trainer_output=result,
                    data=views,
                    direction="forward",
                    output_space="whitened_full",
                    use_training_standardization=True,
                    **apply_common,
                )
            wh_paths = save_views(wh_views, base_prefix, "whitened_full")
            if verbose:
                print("  whitened full:")
                for p in wh_paths:
                    print("  ", p)

        # Reconstructions
        if args.save_recon:
            # Choose input space to match what we saved; default to whitened if requested & available, else z
            inv_input = "z"
            inv_data = None
            if args.base_distribution == "GaussianPCA" and wh_views is not None:
                inv_data = wh_views
                inv_input = "whitened" if args.save_whitened == "pca" else "whitened_full"
            else:
                if args.save_z and 'z_views' in locals():
                    inv_data = z_views
                else:
                    # compute z on the fly
                    if supports_new:
                        z_views = apply_lamnr_flows_whitener(
                            trainer_output=result,
                            data=views,
                            direction="forward",
                            output_space="z",
                            normalization_mode=norm_mode,
                            normalization_stats=norm_stats,
                            fit_stats_on_data_if_missing=(norm_stats is None and norm_mode is not None),
                            **apply_common,
                        )
                    else:
                        z_views = apply_lamnr_flows_whitener(
                            trainer_output=result,
                            data=views,
                            direction="forward",
                            output_space="z",
                            use_training_standardization=True,
                            **apply_common,
                        )
                    inv_data = z_views
                    inv_input = "z"

            if supports_new:
                recon_views = apply_lamnr_flows_whitener(
                    trainer_output=result["models"],   # list of models
                    data=inv_data,
                    direction="inverse",
                    input_space=inv_input,
                    normalization_mode=norm_mode,
                    normalization_stats=norm_stats,
                    fit_stats_on_data_if_missing=(norm_stats is None and norm_mode is not None),
                    **apply_common,
                )
            else:
                recon_views = apply_lamnr_flows_whitener(
                    trainer_output=result["models"],
                    data=inv_data,
                    direction="inverse",
                    input_space=inv_input,
                    use_training_standardization=True,
                    custom_standardizers=result.get("standardizers", None),
                    **apply_common,
                )
            recon_paths = save_views(recon_views, base_prefix, "recon")
            if verbose:
                print("  reconstructions:")
                for p in recon_paths:
                    print("  ", p)


if __name__ == "__main__":
    main()
