#!/usr/bin/env python3
"""
lamnr_glow_tool_3d.py — LAM-Flow (Glow 3D) Inference & Analysis Toolkit

A comprehensive suite for sampling, reconstruction, latent space analysis, and
conditional generation using trained 3D LAM-Flow models.

v0.5.0-3D (2026-02-20)

KEY FEATURES:
-------------
1. Sampling: Generate random 3D NIfTI volumes from the learned distribution.
2. Reconstruction: Sanity check (x -> z -> x_hat) and 3D NIfTI export.
3. Gaussian Fitting (gauss-fit): Fit a conditional Gaussian model to the latent space.
   - FORCED 'lowrank' (SVD) or 'diag' to avoid OOM errors on large 3D latents.
4. Imputation (gauss-impute): Predict missing 3D modalities (e.g., T1 -> FA).
5. Latent Manipulation:
   - recon-winsorize: Clamp latents to remove outliers (lesions/artifacts).
   - recon-interpolate: Interpolate 3D volumes between source and target.
   - calc-distance: Compute Euclidean distance to detect anomalies.

COMMANDS & EXAMPLES:
--------------------

1. FIT GAUSSIAN MODEL (Required for analysis)
   Fit a per-level Gaussian. Uses --cov-estimator lowrank by default for 3D volumes.
   
   python lamnr_glow_tool_3d.py gauss-fit \
     --ckpt runs/model_64x64x64/training_state.pt \
     --manifest data/manifest.csv \
     --views T1,FA \
     --volume-size 64x64x64 \
     --cov-estimator lowrank --rank 128 --sigma2 auto \
     --gauss-out output/model_lowrank.npz

2. CALCULATE LATENT DISTANCE (Anomaly Detection)
   Compute Euclidean distance between subject latents and the group mean.

   python lamnr_glow_tool_3d.py calc-distance \
     --ckpt runs/model.pt \
     --gauss output/model_lowrank.npz \
     --manifest data/patients.csv \
     --views T1 \
     --volume-size 64x64x64 \
     --out-csv output/distances.csv

3. INTERPOLATION (Style Transfer / Normalization)
   Interpolate between a source volume and a target volume (generates multiple NIfTI frames).
   
   python lamnr_glow_tool_3d.py recon-interpolate \
     --ckpt runs/model.pt \
     --source data/patient.nii.gz \
     --target data/atlas.nii.gz \
     --volume-size 64x64x64 \
     --steps 5 \
     --out-dir output/interp_frames/

4. WINSORIZATION (Lesion/Artifact Suppression)
   Clamp 3D latent vectors that exceed a quantile threshold (e.g., 99%).
   
   python lamnr_glow_tool_3d.py recon-winsorize \
     --ckpt runs/model.pt \
     --input data/lesion.nii.gz \
     --volume-size 64x64x64 \
     --quantile 0.99 \
     --out output/healed_lesion.nii.gz

5. SAMPLING
   Generate synthetic 3D NIfTI samples.

   python lamnr_glow_tool_3d.py sample \
     --ckpt runs/model.pt \
     --view-index 0 \
     --volume-size 64x64x64 \
     --n-samples 5 \
     --out-dir output/samples/

NOTE ON MEMORY:
---------------
Processing 3D volumes (e.g., 64x64x64 or 128x128x128) requires significant VRAM.
Keep --batch size low (1 or 2).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import csv
import sys
import time
import hashlib

import torch
import torch.nn.functional as F
import numpy as np
import ants

try:
    from tqdm import tqdm
except ImportError:
    print("[info] tqdm non trouvé. Exécutez `pip install tqdm` pour afficher les barres de progression.")
    tqdm = lambda x, **kwargs: x

# Ensure headless save works
import matplotlib
matplotlib.use("Agg")

__version__ = "0.5.0-3D"

# ---------------- antstorch / model factory -----------------
# Assumes antstorch has the 3D implementation available
try:
    from antstorch import create_glow_normalizing_flow_model_3d
except ImportError:
    # Fallback/Mock for environment testing without antstorch installed
    print("[warn] 'antstorch' not found. Ensure it is installed for 3D Glow models.")
    create_glow_normalizing_flow_model_3d = None

# ------------------------- utils ----------------------------

def parse_dhw(spec: str) -> Tuple[int, int, int]:
    try:
        parts = spec.lower().split("x")
        if len(parts) != 3: raise ValueError
        D, H, W = int(parts[0]), int(parts[1]), int(parts[2])
        assert D > 0 and H > 0 and W > 0
        return D, H, W
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid DxHxW spec '{spec}'. Expected like '64x64x64'.")

def set_deterministic(seed: int):
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalization to [0, 1] over spatial dims (D, H, W).
    Input: (B, C, D, H, W)
    """
    if not torch.is_floating_point(x):
        x = x.float()
    # Min/Max over spatial dims (2,3,4)
    x_min = x.amin(dim=(2, 3, 4), keepdim=True)
    x_max = x.amax(dim=(2, 3, 4), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def _coerce_5d(x, target_dhw: Tuple[int,int,int]=None):
    """Ensure tensor is (B, C, D, H, W)."""
    if not torch.is_tensor(x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        else:
            raise RuntimeError(f"Unexpected output type: {type(x)}")
    
    # If 4D (C, D, H, W) or (B, D, H, W), unsqueeze to 5D
    if x.ndim == 4:
        x = x.unsqueeze(0)
    
    x = x.float()

    if target_dhw is not None:
        dt, ht, wt = target_dhw
        d0, h0, w0 = x.shape[-3], x.shape[-2], x.shape[-1]
        if (d0, h0, w0) != (dt, ht, wt):
            x = F.interpolate(x, size=(dt, ht, wt), mode="trilinear", align_corners=False)
    
    return x

def save_nifti(x: torch.Tensor, out_path: Path, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
    """Save a single 3D/4D tensor to NIfTI using ANTsPy."""
    # x shape expected: (1, C, D, H, W) or (C, D, H, W)
    x = x.detach().cpu()
    if x.ndim == 5: x = x.squeeze(0) # Remove batch
    
    arr = x.numpy()
    
    # Handle channels for ANTs: 
    # ANTs expects (D, H, W) or (D, H, W, C)
    if arr.shape[0] == 1:
        arr = arr[0] # (D, H, W)
    else:
        # Transpose (C, D, H, W) -> (D, H, W, C)
        arr = np.transpose(arr, (1, 2, 3, 0))

    img = ants.from_numpy(arr)
    # Ensure spacing is float tuple
    sp = tuple(float(s) for s in spacing)
    try:
        img.set_spacing(sp)
    except:
        pass
    
    ants.image_write(img, str(out_path))

def _read_image_3d(path: Path, target_dhw: Optional[Tuple[int,int,int]] = None) -> torch.Tensor:
    """
    Read 3D NIfTI, normalize [0,1], return (1, C, D, H, W).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path}")
    
    img = ants.image_read(str(path))
    arr = img.numpy()
    
    # Standardize to (C, D, H, W)
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...] # (1, D, H, W)
    elif arr.ndim == 4:
        # ANTs reads as (D, H, W, C), we need (C, D, H, W)
        arr = np.transpose(arr, (3, 0, 1, 2))
        
    t = torch.from_numpy(arr).float()
    
    # Robust normalization
    q1, q99 = torch.quantile(t, 0.01), torch.quantile(t, 0.99)
    if q99 > q1:
        t = torch.clamp((t - q1) / (q99 - q1 + 1e-8), 0.0, 1.0)
    else:
        mn, mx = t.min(), t.max()
        t = (t - mn) / (mx - mn + 1e-8) if mx > mn else torch.zeros_like(t)

    # Resize if needed (trilinear)
    if target_dhw is not None:
        t = t.unsqueeze(0) # (1, C, D, H, W)
        t = F.interpolate(t, size=target_dhw, mode="trilinear", align_corners=False)
        return t
    
    return t.unsqueeze(0)


# ---------------------- Model Builders ----------------------

def build_model_from_config(cfg: dict, device: torch.device):
    D = int(cfg.get("D", 64))
    H = int(cfg.get("H", 64))
    W = int(cfg.get("W", 64))
    input_shape = (1, D, H, W)
    
    if create_glow_normalizing_flow_model_3d is None:
        raise ImportError("antstorch.create_glow_normalizing_flow_model_3d is required.")

    m = create_glow_normalizing_flow_model_3d(
        input_shape=input_shape,
        L=int(cfg.get("L", 3)),
        K=int(cfg.get("K", 16)),
        hidden_channels=int(cfg.get("hidden", 64)),
        base=str(cfg.get("base", "glow")),
        glowbase_logscale_factor=float(cfg.get("glowbase_logscale_factor", 3.0)),
        split_mode="channel",
        scale=True,
        scale_map=str(cfg.get("scale_map", "tanh")),
        net_actnorm=bool(cfg.get("net_actnorm", False))
    ).to(device).float().eval()
    
    # Attach shape for internal use
    m.input_shape = input_shape
    return m

def resolve_ckpt_path(p: Path) -> Path:
    if p.is_dir():
        for name in ("training_state.pt", "checkpoint.pt", "ckpt.pt", "model.pt"):
            cand = p / name
            if cand.exists(): return cand
    if not p.exists(): raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p

def load_weights_into_model(model, blob, view_idx: int, prefer_ema: bool = True):
    # Try EMA first
    if prefer_ema and isinstance(blob.get("ema"), list) and len(blob["ema"]) > 0:
        k = max(0, min(int(view_idx), len(blob["ema"]) - 1))
        sd = blob["ema"][k]
        if "state_dict" in sd: sd = sd["state_dict"]
        try:
            model.load_state_dict(sd, strict=False)
            return True, "ema"
        except: pass
        
    # Try models list
    if isinstance(blob.get("models"), list) and len(blob["models"]) > 0:
        k = max(0, min(int(view_idx), len(blob["models"]) - 1))
        sd = blob["models"][k]
        if "state_dict" in sd: sd = sd["state_dict"]
        model.load_state_dict(sd, strict=False)
        return True, "models"
        
    # Standard state_dict
    if "state_dict" in blob:
        model.load_state_dict(blob["state_dict"], strict=False)
        return True, "state_dict"
        
    return False, "failed"

def _prime_if_needed(model, D, H, W, device):
    """Run a dummy pass to initialize ActNorm if present."""
    dummy = torch.zeros(1, 1, D, H, W, device=device)
    try:
        model.inverse_and_log_det(dummy)
    except: pass

# ---------------------- Manifest Helpers ----------------------

def _read_manifest_csv(manifest_path: Path) -> Dict[str, List[str]]:
    with open(manifest_path, "r", newline="") as f:
        rdr = csv.reader(f)
        rows = list(rdr)
    if not rows: raise RuntimeError("Manifest empty")
    header = [h.strip() for h in rows[0]]
    cols = {h: [] for h in header}
    for r in rows[1:]:
        for h, v in zip(header, r):
            cols[h].append(v.strip())
    return cols

def _resolve_views(cols, root_dir, views_str):
    if not views_str: return list(cols.keys()), [cols[k] for k in cols.keys()]
    v_names = [v.strip() for v in views_str.split(",") if v.strip()]
    paths = []
    for v in v_names:
        if v not in cols: raise ValueError(f"View {v} not in manifest")
        # Resolve absolute paths
        abs_paths = []
        for p in cols[v]:
            pp = Path(p)
            if not pp.is_absolute(): pp = root_dir / pp
            abs_paths.append(pp)
        paths.append(abs_paths)
    return v_names, paths

# ---------------------- LowRank Math ----------------------

def _lowrank_from_Xc(Xc: np.ndarray, rank: int, sigma2: float | str, extra_ridge: float) -> dict:
    """Compute SVD-based low-rank covariance approximation.
       Xc: Centered data (N, D).
       Returns dict with U (D, r), eig (r), sigma2.
    """
    N, D = Xc.shape
    rmax = min(D, max(1, N - 1))
    r = int(max(1, min(rank, rmax)))
    
    # SVD on (N, D) -> U_svd(N,K), S(K), Vt(K,D)
    # We want eigenvectors of Cov = Xc.T @ Xc / (N-1)
    # Xc = U_svd * S * Vt
    # Cov = Vt.T * S^2 * Vt / (N-1)
    # So Cov eigenvectors are Vt.T (shape D, K)
    
    # Use float32 to save RAM during SVD if float64 is too big
    try:
        Ux, Svals, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        print("[warn] SVD failed on float64, trying float32")
        Ux, Svals, Vt = np.linalg.svd(Xc.astype(np.float32), full_matrices=False)
        
    eigs_all = (Svals ** 2) / max(1, (N - 1))
    
    # Top r components
    eig_r = eigs_all[:r].copy()
    U_cov = Vt[:r, :].T.copy() # (D, r)
    
    if isinstance(sigma2, str) and sigma2.lower() == "auto":
        # Average of remaining eigenvalues
        if eigs_all.shape[0] > r:
            sigma2_val = float(np.maximum(np.mean(eigs_all[r:]), 0.0))
        else:
            sigma2_val = 0.0
    else:
        sigma2_val = float(sigma2)
        
    sigma2_val += extra_ridge
    
    return {"type": "lowrank", "U": U_cov, "eig": eig_r, "sigma2": sigma2_val}

def _cond_mean_block_lowrank(U: np.ndarray, eig: np.ndarray, sigma2: float,
                             idx_U: list, idx_O: list,
                             mu: np.ndarray, ZO: np.ndarray,
                             base_ridge: float = 1e-6):
    """
    Conditional mean using LowRank inversion (Woodbury/Sherman-Morrison logic).
    Goal: E[z_U | z_O] = mu_U + Sig_UO * inv(Sig_OO) * (z_O - mu_O)
    
    Sig = U diag(eig) U.T + sigma2 I
    Sig_OO = U_O diag(eig) U_O.T + sigma2 I
    
    Inversion via Woodbury:
    inv(Sig_OO) = inv(sigma2 I + U_O Lam U_O.T)
                = (1/s2) I - (1/s2) U_O * inv(inv(Lam) + U_O.T U_O / s2) * U_O.T / s2
    """
    U = np.asarray(U, dtype=np.float64)
    eig = np.asarray(eig, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).ravel()
    
    U_O = U[idx_O, :] # (D_O, r)
    U_U = U[idx_U, :] # (D_U, r)
    
    diff_O = (ZO - mu[idx_O][None, :]).T # (D_O, N)
    
    r = eig.shape[0]
    
    # 1. Compute Matrix M = diag(1/eig) + (U_O.T @ U_O) / sigma2
    # If eig is close to 0, use large number
    inv_lam = np.diag(1.0 / (eig + 1e-12)) 
    M = inv_lam + (U_O.T @ U_O) / sigma2
    
    # 2. Solve M gamma = U_O.T @ diff_O
    rhs = U_O.T @ diff_O # (r, N)
    gamma = np.linalg.solve(M, rhs) # (r, N)
    
    # 3. Apply formula
    # inv(Sig_OO) * diff = (diff - U_O @ gamma) / sigma2
    w = (diff_O - U_O @ gamma) / sigma2 # (D_O, N)
    
    # 4. Result = mu_U + Sig_UO @ w
    # Sig_UO = U_U diag(eig) U_O.T
    # Result = mu_U + U_U diag(eig) (U_O.T @ w)
    
    # Optimized: U_O.T @ w is computed efficiently
    # Note: Woodbury simplification leads to:
    # E[z_U|z_O] = mu_U + U_U @ (inv(inv(Lam) + U_O.T U_O/s2) @ (U_O.T diff_O / s2))
    
    term1 = U_O.T @ diff_O / sigma2 # (r, N)
    term2 = np.linalg.solve(M, term1) # (r, N)
    delta = U_U @ term2 # (D_U, N)
    
    zU = mu[idx_U][:, None] + delta
    return zU.T # (N, D_U)

# ---------------------- Main Commands ----------------------

def main_sample(argv=None):
    ap = argparse.ArgumentParser("sample")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--volume-size", type=parse_dhw, default="64x64x64")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--devices", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    set_deterministic(args.seed)
    device = torch.device(args.devices)
    
    blob = torch.load(resolve_ckpt_path(Path(args.ckpt)), map_location=device, weights_only=False)
    cfg = blob.get("config", {"D": args.volume_size[0], "H": args.volume_size[1], "W": args.volume_size[2]})
    model = build_model_from_config(cfg, device)
    load_weights_into_model(model, blob, view_idx=0)
    _prime_if_needed(model, *args.volume_size, device)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[info] Sampling {args.n_samples} volumes...")
    with torch.no_grad():
        for i in range(args.n_samples):
            z_sample = model.sample(1, temperature=args.temperature)
            x_hat = _coerce_5d(z_sample, target_dhw=args.volume_size)
            x_hat = to01(x_hat)
            save_nifti(x_hat, Path(args.out_dir) / f"sample_{i:03d}.nii.gz")
    print("[ok] Done.")

def main_recon(argv=None):
    ap = argparse.ArgumentParser("recon")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--volume-size", type=parse_dhw, default="64x64x64")
    ap.add_argument("--devices", default="cuda:0")
    args = ap.parse_args(argv)

    device = torch.device(args.devices)
    blob = torch.load(resolve_ckpt_path(Path(args.ckpt)), map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    model = build_model_from_config(cfg, device)
    load_weights_into_model(model, blob, 0)
    _prime_if_needed(model, *args.volume_size, device)

    xi = _read_image_3d(Path(args.input), target_dhw=args.volume_size).to(device)
    
    with torch.no_grad():
        z, _ = model.inverse_and_log_det(xi)
        # z is list of tensors. 
        if isinstance(z, torch.Tensor): z = [z]
        xh, _ = model.forward_and_log_det(z)
        xh = to01(_coerce_5d(xh, target_dhw=args.volume_size))

    save_nifti(xh, Path(args.out))
    print(f"[ok] Saved {args.out}")


def main_gauss_fit(argv=None):
    ap = argparse.ArgumentParser("gauss-fit")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--gauss-out", required=True)
    ap.add_argument("--volume-size", type=parse_dhw, default="64x64x64")
    ap.add_argument("--batch", type=int, default=2, help="Keep low for 3D")
    ap.add_argument("--devices", default="cuda:0")
    
    ap.add_argument("--cov-estimator", choices=["lowrank", "diag"], default="lowrank")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--sigma2", default="auto")
    ap.add_argument("--cov-lam", type=float, default=1e-5)
    args = ap.parse_args(argv)

    device = torch.device(args.devices)
    blob = torch.load(resolve_ckpt_path(Path(args.ckpt)), map_location=device, weights_only=False)
    cfg = blob.get("config", {"D": args.volume_size[0], "H": args.volume_size[1], "W": args.volume_size[2]})
    model = build_model_from_config(cfg, device)

    # Resolve paths
    cols = _read_manifest_csv(Path(args.manifest))
    view_names, paths_per_view = _resolve_views(cols, Path(args.manifest).parent, args.views)
    N = len(paths_per_view[0])
    
    print(f"[info] Fitting 3D Gaussian on {N} subjects. Estimator: {args.cov_estimator}")

    # Latent collection
    # Structure: Z_levels[level_idx] -> Tensor (N, D_level)
    Z_levels = [] 
    
    for v_idx, vname in enumerate(view_names):
        print(f"  Encoding view: {vname}")
        load_weights_into_model(model, blob, v_idx)
        
        # Collect latents for this view
        latents_for_view = [] # list of levels, each is list of batches
        
        paths = paths_per_view[v_idx]
        for i in tqdm(range(0, N, args.batch), desc=f"Encoding {vname}", unit="batch"):
            batch_p = paths[i:i+args.batch]
            batch_x = []
            for p in batch_p:
                batch_x.append(_read_image_3d(p, args.volume_size).squeeze(0))
            
            xb = torch.stack(batch_x).to(device)
            
            with torch.no_grad():
                z_raw, _ = model.inverse_and_log_det(xb)
                if not isinstance(z_raw, list): z_raw = [z_raw]
                
                # Flatten spatial dims: (B, C, D, H, W) -> (B, Features)
                z_flat = [t.view(t.shape[0], -1).cpu() for t in z_raw]
                
                if not latents_for_view: 
                    latents_for_view = [[] for _ in z_flat]
                
                for l, t in enumerate(z_flat):
                    latents_for_view[l].append(t)

        # Concatenate batches -> (N, Feat) per level
        z_view_concat = [torch.cat(batches, dim=0) for batches in latents_for_view]
        
        # Merge with previous views
        if not Z_levels:
            Z_levels = z_view_concat
        else:
            # Concatenate features from different views at same level
            Z_levels = [torch.cat([Z_levels[l], z_view_concat[l]], dim=1) for l in range(len(Z_levels))]

    # Dimensions tracking
    dims_per_level_per_view = [] # Not fully tracked here for simplicity, reusing from 2D logic would be better if needed later
    # For now we assume strict order reconstruction in impute
    
    # Fit Stats
    mu_list = []
    Sigma_list = []
    
    for l, Z in enumerate(Z_levels):
        print(f"  Fitting Level {l}, Shape {Z.shape}")
        X = Z.numpy().astype(np.float64)
        mu = np.mean(X, axis=0)
        Xc = X - mu
        
        if args.cov_estimator == "lowrank":
            sig = _lowrank_from_Xc(Xc, args.rank, args.sigma2, args.cov_lam)
        else:
            sig = np.var(Xc, axis=0) + args.cov_lam # diag
            
        mu_list.append(mu)
        Sigma_list.append(sig)

    # Save
    pack = {
        "mode": "perlevel",
        "estimator": args.cov_estimator,
        "views": view_names,
        "N": N, "L": len(Z_levels),
        "D": args.volume_size[0], "H": args.volume_size[1], "W": args.volume_size[2],
        # Dims info helps reconstruction
        "dims_per_view_L0": [Z.shape[1] // len(view_names) for Z in Z_levels] # Approx
    }
    
    for i, (m, s) in enumerate(zip(mu_list, Sigma_list)):
        pack[f"mu_{i}"] = m
        if isinstance(s, dict):
            pack[f"Sigma_{i}_type"] = "lowrank"
            pack[f"Sigma_{i}_U"] = s["U"]
            pack[f"Sigma_{i}_eig"] = s["eig"]
            pack[f"Sigma_{i}_sigma2"] = s["sigma2"]
        else:
            pack[f"Sigma_{i}"] = s

    np.savez_compressed(args.gauss_out, **pack)
    print(f"[ok] Saved {args.gauss_out}")

def main_gauss_impute(argv=None):
    ap = argparse.ArgumentParser("gauss-impute")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gauss", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--views", required=True) # Full view list used in fit
    ap.add_argument("--observed", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--volume-size", type=parse_dhw, default="64x64x64")
    ap.add_argument("--devices", default="cuda:0")
    args = ap.parse_args(argv)

    device = torch.device(args.devices)
    blob = torch.load(resolve_ckpt_path(Path(args.ckpt)), map_location=device, weights_only=False)
    cfg = blob.get("config", {"D": args.volume_size[0], "H": args.volume_size[1], "W": args.volume_size[2]})
    model = build_model_from_config(cfg, device)

    # Load Gauss
    npz = np.load(args.gauss, allow_pickle=True)
    L = int(npz["L"])
    all_views = list(npz["views"])
    
    obs_views = [v.strip() for v in args.observed.split(",")]
    tgt_views = [v.strip() for v in args.target.split(",")]
    
    # Determine indices in flat vectors
    # This requires knowing the size of each view at each level.
    # Since we use Glow, splits are usually regular. 
    # We will infer sizes by encoding a dummy.
    dummy = torch.zeros(1, 1, args.volume_size[0], args.volume_size[1], args.volume_size[2], device=device)
    with torch.no_grad():
        z_dummy, _ = model.inverse_and_log_det(dummy)
        if not isinstance(z_dummy, list): z_dummy = [z_dummy]
        level_sizes = [t.numel() for t in z_dummy] # Size per ONE view per level

    # Build Slice Map
    # Flat vector at Level L is [View1_L ... ViewN_L]
    slice_map = [] # List of dict {view_name: (start, end)}
    for l in range(L):
        sz = level_sizes[l]
        d = {}
        curr = 0
        for v in all_views:
            d[v] = (curr, curr + sz)
            curr += sz
        slice_map.append(d)

    # Load Manifest
    cols = _read_manifest_csv(Path(args.manifest))
    _, paths_obs = _resolve_views(cols, Path(args.manifest).parent, args.observed)
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    N_sub = len(paths_obs[0])

    print(f"[info] Imputing {tgt_views} from {obs_views} for {N_sub} subjects.")

    for i in tqdm(range(N_sub), desc="Imputing volumes", unit="subj"):
        # 1. Encode Observed
        z_obs_levels = [[] for _ in range(L)] # Store vectors
        
        for v_idx, v_name in enumerate(obs_views):
            # Load specific weights for this view
            # Note: This is inefficient (reloading weights per subject/view), 
            # but safe for 3D memory.
            global_idx = all_views.index(v_name)
            load_weights_into_model(model, blob, global_idx)
            
            p = paths_obs[v_idx][i]
            x = _read_image_3d(p, args.volume_size).to(device)
            with torch.no_grad():
                z, _ = model.inverse_and_log_det(x)
                if not isinstance(z, list): z = [z]
                for l, t in enumerate(z):
                    z_obs_levels[l].append(t.view(-1).cpu().numpy())

        # 2. Condition & Predict per Level
        z_pred_levels = [] # List of tensors for reconstruction
        
        for l in range(L):
            # Construct Z_Observed_Concatenated
            # Note: The order must match the Gaussian fit order, filtering only obs
            # But here we have z_obs_levels filled by obs_views loop.
            # We need to construct the full vector Z_O corresponding to indices idx_O
            
            mu = npz[f"mu_{l}"]
            
            # Identify indices
            idx_O = []
            z_vals_O = []
            
            for v_name in obs_views:
                s, e = slice_map[l][v_name]
                idx_O.extend(range(s, e))
                # Find the value we encoded (match loop order above)
                z_vals_O.append(z_obs_levels[l][obs_views.index(v_name)])
            
            idx_O = np.array(idx_O)
            ZO = np.concatenate(z_vals_O) # Flat vector of all observed features
            
            # Identify Targets
            # We assume single target view for reconstruction simplicity, 
            # but math supports multiple. Let's predict the first target view.
            t_name = tgt_views[0]
            s_t, e_t = slice_map[l][t_name]
            idx_U = np.arange(s_t, e_t)
            
            # Load Sigma
            if f"Sigma_{l}_type" in npz and str(npz[f"Sigma_{l}_type"]) == "lowrank":
                U = npz[f"Sigma_{l}_U"]
                eig = npz[f"Sigma_{l}_eig"]
                s2 = npz[f"Sigma_{l}_sigma2"]
                
                # Conditional Mean
                z_target_flat = _cond_mean_block_lowrank(U, eig, s2, idx_U, idx_O, mu, ZO[:, None])
            else:
                # Diag fallback
                var = npz[f"Sigma_{l}"]
                # Independent means we just take the mean of the target
                z_target_flat = mu[idx_U][None, :]
            
            # Reshape back to tensor shape
            # We need to know original tensor shape.
            # z_dummy[l] has it.
            ref_shape = z_dummy[l].shape
            z_t_tensor = torch.from_numpy(z_target_flat).float().view(1, ref_shape[1], ref_shape[2], ref_shape[3], ref_shape[4])
            z_pred_levels.append(z_t_tensor.to(device))

        # 3. Decode
        tgt_global_idx = all_views.index(tgt_views[0])
        load_weights_into_model(model, blob, tgt_global_idx)
        
        with torch.no_grad():
            x_rec, _ = model.forward_and_log_det(z_pred_levels)
            x_rec = to01(_coerce_5d(x_rec, args.volume_size))
            
        out_name = Path(args.out_dir) / f"imputed_{i:04d}_{tgt_views[0]}.nii.gz"
        save_nifti(x_rec, out_name)
        print(f"  Saved {out_name}")

def main_recon_winsorize(argv=None):
    ap = argparse.ArgumentParser("recon-winsorize")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--volume-size", type=parse_dhw, default="64x64x64")
    ap.add_argument("--devices", default="cuda:0")
    args = ap.parse_args(argv)
    
    device = torch.device(args.devices)
    blob = torch.load(resolve_ckpt_path(Path(args.ckpt)), map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    model = build_model_from_config(cfg, device)
    load_weights_into_model(model, blob, 0)
    
    xi = _read_image_3d(Path(args.input), args.volume_size).to(device)
    
    with torch.no_grad():
        z, _ = model.inverse_and_log_det(xi)
        if not isinstance(z, list): z = [z]
        
        z_clamped = []
        for t in z:
            thresh = torch.quantile(torch.abs(t), args.quantile)
            z_clamped.append(torch.clamp(t, -thresh, thresh))
            
        xh, _ = model.forward_and_log_det(z_clamped)
        xh = to01(_coerce_5d(xh, args.volume_size))
        
    save_nifti(xh, Path(args.out))
    print(f"[ok] Saved {args.out}")

def main_calc_distance(argv=None):
    ap = argparse.ArgumentParser("calc-distance")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--volume-size", type=parse_dhw, default="64x64x64")
    ap.add_argument("--devices", default="cuda:0")
    args = ap.parse_args(argv)
    
    device = torch.device(args.devices)
    blob = torch.load(resolve_ckpt_path(Path(args.ckpt)), map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    model = build_model_from_config(cfg, device)
    load_weights_into_model(model, blob, 0)
    
    cols = _read_manifest_csv(Path(args.manifest))
    # Assume first column is what we want
    paths = [Path(p) for p in list(cols.values())[0]]
    
    results = []
    print(f"[info] Calculating latent L2 distance for {len(paths)} volumes")

    for p in tqdm(paths, desc="Distance calc", unit="vol"):
        try:
            xi = _read_image_3d(p, args.volume_size).to(device)
            with torch.no_grad():
                z, _ = model.inverse_and_log_det(xi)
                if not isinstance(z, list): z = [z]
                
                # Simple L2 norm of the latent vector
                dist = 0.0
                for t in z:
                    dist += torch.sum(t ** 2).item()
                results.append((p.name, math.sqrt(dist)))
        except Exception as e:
            print(f"[warn] Failed {p}: {e}")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "latent_l2_dist"])
        writer.writerows(results)
    print(f"[ok] Saved {args.out_csv}")

def main_recon_interpolate(argv=None):
    ap = argparse.ArgumentParser("recon-interpolate")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--volume-size", type=parse_dhw, default="64x64x64")
    ap.add_argument("--devices", default="cuda:0")
    args = ap.parse_args(argv)
    
    device = torch.device(args.devices)
    blob = torch.load(resolve_ckpt_path(Path(args.ckpt)), map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    model = build_model_from_config(cfg, device)
    load_weights_into_model(model, blob, 0)
    
    s = _read_image_3d(Path(args.source), args.volume_size).to(device)
    t = _read_image_3d(Path(args.target), args.volume_size).to(device)
    
    with torch.no_grad():
        zs, _ = model.inverse_and_log_det(s)
        zt, _ = model.inverse_and_log_det(t)
        if not isinstance(zs, list): zs = [zs]
        if not isinstance(zt, list): zt = [zt]
        
        Path(args.out_dir).mkdir(exist_ok=True, parents=True)
        
        alphas = np.linspace(0, 1, args.steps)
        for i, alpha in enumerate(alphas):
            zi = []
            for l in range(len(zs)):
                zi.append(zs[l] * (1 - alpha) + zt[l] * alpha)
            
            xi, _ = model.forward_and_log_det(zi)
            xi = to01(_coerce_5d(xi, args.volume_size))
            save_nifti(xi, Path(args.out_dir) / f"interp_{i:02d}_a{alpha:.2f}.nii.gz")
            
    print(f"[ok] Saved {args.steps} frames to {args.out_dir}")

# ---------------------- Entry Point ----------------------

if __name__ == "__main__":
    table = {
        "sample": main_sample,
        "recon": main_recon,
        "gauss-fit": main_gauss_fit,
        "gauss-impute": main_gauss_impute,
        "recon-winsorize": main_recon_winsorize,
        "calc-distance": main_calc_distance,
        "recon-interpolate": main_recon_interpolate
    }
    
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Available subcommands:", ", ".join(sorted(table.keys())))
        sys.exit(0)

    cmd = sys.argv.pop(1)
    if cmd not in table:
        print(f"Unknown command: {cmd}")
        print("Available:", ", ".join(sorted(table.keys())))
        sys.exit(1)

    sys.exit(table[cmd]())