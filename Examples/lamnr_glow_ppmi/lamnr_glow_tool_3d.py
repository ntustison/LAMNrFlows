#!/usr/bin/env python3
"""
lamnr_glow_tool_3d.py — LAM-Flow (Glow 3D) Inference & Analysis Toolkit

A comprehensive suite for sampling, reconstruction, latent space analysis, and
conditional generation using trained 3D LAM-Flow models.

v0.5.5-3D (STABLE ACTNORM & PUSH-THROUGH)

KEY FEATURES:
-------------
1. Sampling: Generate random 3D NIfTI volumes from the learned distribution.
2. Reconstruction: Sanity check (x -> z -> x_hat) and 3D NIfTI export.
3. Gaussian Fitting (gauss-fit): Fit a conditional Gaussian model to the latent space.
   - FORCED 'lowrank' (SVD) or 'diag' to avoid OOM errors on large 3D latents.
   - Includes latent circuit breakers (clamping) to prevent Out-Of-Distribution explosion.
4. Imputation (gauss-impute): Predict missing 3D modalities (e.g., T1 -> FA) using the stable Push-Through Woodbury identity.
5. Latent Manipulation & Analysis:
   - recon-template (NEW): Generate population average templates and Monte Carlo samples.
   - recon-winsorize: Clamp latents globally or per-level to remove outliers (lesions/artifacts).
   - recon-interpolate: Generate 3D NIfTI sequences interpolating between source and target/mean.
   - calc-distance: Compute Euclidean distance to detect anomalies against the group mean.

COMMANDS & EXAMPLES:
--------------------

1. FIT GAUSSIAN MODEL (Required for downstream analysis)
   Fit a per-level Gaussian. Uses --cov-estimator lowrank by default for 3D volumes.
   
   python lamnr_glow_tool_3d.py gauss-fit \
     --ckpt runs/model_64x64x64/training_state.pt \
     --manifest data/manifest.csv \
     --views T1,FA \
     --volume-size 64x64x64 \
     --cov-estimator lowrank --rank 256 \
     --gauss-out output/model_lowrank.npz

2. IMPUTATION (Conditional Generation)
   Impute a missing modality from an observed one.
   
   python lamnr_glow_tool_3d.py gauss-impute \
     --ckpt runs/model_64x64x64/training_state.pt \
     --gauss output/model_lowrank.npz \
     --manifest data/manifest_short.csv \
     --views T1,FA \
     --observed T1 --target FA \
     --volume-size 64x64x64 \
     --out-dir output/imputed_FA/

3. RECON-TEMPLATE (Population Average)
   Reconstruct the latent Gaussian mean and generate Monte Carlo variations.
   
   python lamnr_glow_tool_3d.py recon-template \
     --ckpt runs/model_64x64x64/training_state.pt \
     --gauss output/model_lowrank.npz \
     --manifest data/manifest.csv \
     --views T1 \
     --mc-samples 10 \
     --sharpen-image \
     --out-dir output/templates/

4. CALCULATE LATENT DISTANCE (Anomaly Detection)
   Compute Euclidean distance between subject latents and the Gaussian mean.

   python lamnr_glow_tool_3d.py calc-distance \
     --ckpt runs/model_64x64x64/training_state.pt \
     --gauss output/model_lowrank.npz \
     --manifest data/manifest.csv \
     --views T1 \
     --volume-size 64x64x64 \
     --out-csv output/distances.csv

5. INTERPOLATION (Style Transfer / Normalization)
   Interpolate a subject towards the population mean or a target image in sequential steps.
   
   python lamnr_glow_tool_3d.py recon-interpolate \
     --ckpt runs/model_64x64x64/training_state.pt \
     --gauss output/model_lowrank.npz \
     --manifest data/manifest_lesions.csv \
     --views T1 \
     --steps 5 \
     --out-dir output/interpolation/

6. WINSORIZATION (Lesion/Artifact Suppression)
   Clamp 3D latent vectors using hard thresholds or quantiles to "heal" pathologies.
   
   python lamnr_glow_tool_3d.py recon-winsorize \
     --ckpt runs/model_64x64x64/training_state.pt \
     --manifest data/manifest_lesions.csv \
     --views T1 \
     --hard-threshold 3.0 \
     --winsorize-level 0,0.95 \
     --out-dir output/winsorized/

7. SAMPLING
   Generate synthetic 3D NIfTI samples.

   python lamnr_glow_tool_3d.py sample \
     --ckpt runs/model_64x64x64/training_state.pt \
     --view-index 0 \
     --volume-size 64x64x64 \
     --n-samples 5 \
     --out-dir output/samples/

NOTE ON MEMORY & 3D GEOMETRY:
-----------------------------
Processing 3D volumes (e.g., 64x64x64) requires significant VRAM.
- Keep --batch size low (1 or 2).
- The script automatically processes Monte Carlo samples and matrix inversions sequentially 
  to prevent Out-Of-Memory (OOM) errors and tensor contiguity crashes.
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

import matplotlib
matplotlib.use("Agg")

__version__ = "0.5.5-3D"

try:
    from antstorch import create_glow_normalizing_flow_model_3d
except ImportError:
    print("[warn] 'antstorch' not found. Ensure it is installed for 3D Glow models.")
    create_glow_normalizing_flow_model_3d = None

# ------------------------- utils ----------------------------

def parse_dhw(spec: str) -> Tuple[int, int, int]:
    try:
        parts = spec.lower().split("x")
        D, H, W = int(parts[0]), int(parts[1]), int(parts[2])
        assert D > 0 and H > 0 and W > 0
        return D, H, W
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid DxHxW spec '{spec}'. Expected like '64x64x64'.")

def parse_dhw_float(spec: str) -> Tuple[float, float, float]:
    try:
        parts = spec.lower().split("x")
        D, H, W = float(parts[0]), float(parts[1]), float(parts[2])
        assert D > 0 and H > 0 and W > 0
        return D, H, W
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid spacing spec '{spec}'. Expected like '1.0x1.0x1.0'.")

def _gather_val_paths(val_list: Optional[list[str]], limit: int) -> list[Path]:
    from glob import glob
    paths: list[Path] = []
    tokens = val_list or []
    for tok in tokens:
        tok = os.path.expandvars(os.path.expanduser(tok))
        p = Path(tok)
        if p.exists() and p.is_file():
            if p.suffix.lower() in (".txt", ".lst", ".csv"):
                try:
                    with open(p, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line: paths.append(Path(os.path.expandvars(os.path.expanduser(line))))
                except Exception: pass
            else:
                paths.append(p)
        else:
            for g in sorted(glob(tok, recursive=True)):
                gp = Path(g)
                if gp.exists() and gp.is_file(): paths.append(gp)
    seen = set()
    uniq: list[Path] = []
    for p in paths:
        if p not in seen and p.exists() and p.is_file():
            uniq.append(p); seen.add(p)
        if len(uniq) >= int(limit): break
    return uniq

def set_deterministic(seed: int):
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to01(x: torch.Tensor, eps: float = 1e-8, winsorize: bool = True) -> torch.Tensor:
    """
    Normalise les volumes 3D (N, C, D, H, W) ou images 2D (N, C, H, W) entre 0 et 1.
    """
    if x.ndim < 4:
        return x
    
    # Pour un volume 3D, spatial_dims sera (2, 3, 4)
    spatial_dims = tuple(range(2, x.ndim))
    
    if winsorize:
        # torch.quantile ne supporte pas le float16, on force le calcul en float32
        x_calc = x.float()
        
        # Aplatit (N, C, D, H, W) en (N, C, D*H*W)
        x_flat = x_calc.flatten(start_dim=2)
        
        # Calcule les percentiles sur l'ensemble des voxels
        q_low = torch.quantile(x_flat, 0.01, dim=2, keepdim=True).to(x.dtype)
        q_high = torch.quantile(x_flat, 0.99, dim=2, keepdim=True).to(x.dtype)
        
        # Redimensionne en (N, C, 1, 1, 1) pour le broadcasting 3D
        view_shape = x.shape[:2] + (1,) * len(spatial_dims)
        x_min = q_low.view(view_shape)
        x_max = q_high.view(view_shape)
        
        # Écrête les voxels aberrants
        x = torch.clamp(x, min=x_min, max=x_max)
    else:
        # Comportement standard pour l'entraînement
        x_min = x.amin(dim=spatial_dims, keepdim=True)
        x_max = x.amax(dim=spatial_dims, keepdim=True)
        
    return (x - x_min) / (x_max - x_min + eps)

def _coerce_5d(x, target_dhw: Tuple[int,int,int]=None):
    if not torch.is_tensor(x):
        if isinstance(x, (list, tuple)): x = x[0]
        else: raise RuntimeError(f"Unexpected output type: {type(x)}")
    if x.ndim == 4: x = x.unsqueeze(0)
    x = x.float()
    if target_dhw is not None:
        dt, ht, wt = target_dhw
        d0, h0, w0 = x.shape[-3], x.shape[-2], x.shape[-1]
        if (d0, h0, w0) != (dt, ht, wt):
            x = F.interpolate(x, size=(dt, ht, wt), mode="trilinear", align_corners=False)
    return x

def save_nifti(x: torch.Tensor, out_path: Path, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
    x = x.detach().cpu()
    if x.ndim == 5: x = x.squeeze(0) 
    arr = x.numpy()
    if arr.shape[0] == 1: arr = arr[0] 
    else: arr = np.transpose(arr, (1, 2, 3, 0))
    img = ants.from_numpy(arr)
    sp = tuple(float(s) for s in spacing)
    try: img.set_spacing(sp)
    except: pass
    ants.image_write(img, str(out_path))

def _read_image_3d(path: Path, target_dhw: Optional[Tuple[int,int,int]] = None) -> torch.Tensor:
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"{path}")
    img = ants.image_read(str(path))
    arr = img.numpy()
    if arr.ndim == 3: arr = arr[np.newaxis, ...] 
    elif arr.ndim == 4: arr = np.transpose(arr, (3, 0, 1, 2))
    t = torch.from_numpy(arr).float()
    q1, q99 = torch.quantile(t, 0.01), torch.quantile(t, 0.99)
    if q99 > q1: t = torch.clamp((t - q1) / (q99 - q1 + 1e-8), 0.0, 1.0)
    else:
        mn, mx = t.min(), t.max()
        t = (t - mn) / (mx - mn + 1e-8) if mx > mn else torch.zeros_like(t)
    if target_dhw is not None:
        t = t.unsqueeze(0) 
        t = F.interpolate(t, size=target_dhw, mode="trilinear", align_corners=False)
        return t
    return t.unsqueeze(0)

def save_mid_slice_png(x: torch.Tensor, out_path: Path, slice_axis: int = 2):
    import matplotlib.pyplot as plt
    x_np = x.detach().cpu()
    if x_np.ndim == 5: x_np = x_np.squeeze(0) 
    D = x_np.shape[slice_axis+1] 
    mid_idx = D // 2
    if slice_axis == 0:
        slice_2d = x_np[0, mid_idx, :, :].numpy()
    elif slice_axis == 1:
        slice_2d = x_np[0, :, mid_idx, :].numpy()
    elif slice_axis == 2:
        slice_2d = x_np[0, :, :, mid_idx].numpy()
    else: 
        raise ValueError(f"Invalid slice_axis {slice_axis}. Must be 0, 1, or 2.")    
    plt.imsave(str(out_path), slice_2d, cmap="gray", vmin=0.0, vmax=1.0)

# ---------------------- Model Builders ----------------------

def build_model_from_config(cfg: dict, device: torch.device, target_dhw: Tuple[int, int, int] = None):
    if target_dhw is not None:
        D, H, W = int(target_dhw[0]), int(target_dhw[1]), int(target_dhw[2])
    else:
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
    def try_load(sd):
        try:
            model.load_state_dict(sd, strict=True)
            return True, None
        except Exception as e:
            try:
                model.load_state_dict(sd, strict=False)
                return True, f"non-strict: {e}"
            except Exception as e2:
                return False, f"failed: {e2}"

    def extract_sd(candidate):
        if isinstance(candidate, dict):
            if "state_dict" in candidate and isinstance(candidate["state_dict"], dict):
                return candidate["state_dict"]
            return candidate
        return None

    target_idx = int(view_idx)

    if prefer_ema and isinstance(blob.get("ema"), list) and len(blob["ema"]) > 0:
        max_idx = len(blob["ema"]) - 1
        k = max(0, min(target_idx, max_idx))
        sd = extract_sd(blob["ema"][k])
        if sd is not None:
            ok, note = try_load(sd)
            if ok:
                if target_idx > max_idx:
                    print(f"\n[AVERTISSEMENT CRITIQUE] L'index de vue {target_idx} n'existe pas dans le checkpoint (ema).")
                    print(f"-> Repli forcé sur les poids de l'index {k}.")
                    print(f"-> Attention : Les latents générés pour cette modalité seront hors distribution (Out-Of-Distribution) !\n")
                return True, f"ema slot={k} ({note})"
            
    if isinstance(blob.get("models"), list) and len(blob["models"]) > 0:
        max_idx = len(blob["models"]) - 1
        k = max(0, min(target_idx, max_idx))
        sd = extract_sd(blob["models"][k])
        if sd is not None:
            ok, note = try_load(sd)
            if ok:
                if target_idx > max_idx:
                    print(f"\n[AVERTISSEMENT CRITIQUE] L'index de vue {target_idx} n'existe pas dans le checkpoint (models).")
                    print(f"-> Repli forcé sur les poids de l'index {k}.")
                    print(f"-> Attention : Les latents générés pour cette modalité seront hors distribution (Out-Of-Distribution) !\n")
                return True, f"models slot={k} ({note})"
            
    if "state_dict" in blob:
        ok, note = try_load(blob["state_dict"])
        if ok:
            if target_idx > 0:
                 print(f"\n[AVERTISSEMENT CRITIQUE] Checkpoint unique (state_dict) détecté. Repli forcé de la vue {target_idx} sur l'unique vue disponible.\n")
            return True, f"state_dict ({note})"
            
    if isinstance(blob, dict) and all(isinstance(k, str) for k in blob.keys()) and any("." in k for k in blob.keys()):
        ok, note = try_load(blob)
        if ok: return True, f"raw ({note})"

    return False, "no valid weights found"


def _prime_if_needed(model, D, H, W, device):
    dummy = torch.randn(1, 1, D, H, W, device=device)
    try: model.inverse_and_log_det(dummy)
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
        abs_paths = []
        for p in cols[v]:
            pp = Path(p)
            if not pp.is_absolute(): pp = root_dir / pp
            abs_paths.append(pp)
        paths.append(abs_paths)
    return v_names, paths

# ---------------------- LowRank Math (STABLE PUSH-THROUGH) ----------------------

def _lowrank_from_Xc(Xc: np.ndarray, rank: int, sigma2: float | str, extra_ridge: float) -> dict:
    N, D = Xc.shape
    rmax = min(D, max(1, N - 1))
    r = int(max(1, min(rank, rmax)))
    try: Ux, Svals, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError: Ux, Svals, Vt = np.linalg.svd(Xc.astype(np.float32), full_matrices=False)
    eigs_all = (Svals ** 2) / max(1, (N - 1))
    eig_r = eigs_all[:r].copy()
    U_cov = Vt[:r, :].T.copy() 
    if isinstance(sigma2, str) and sigma2.lower() == "auto":
        sigma2_val = float(np.maximum(np.mean(eigs_all[r:]), 0.0)) if eigs_all.shape[0] > r else 0.0
    else:
        sigma2_val = float(sigma2)
    sigma2_val += extra_ridge
    return {"type": "lowrank", "U": U_cov, "eig": eig_r, "sigma2": sigma2_val}

def _cond_mean_block_lowrank(U: np.ndarray, eig: np.ndarray, sigma2: float,
                             idx_U: list, idx_O: list,
                             mu: np.ndarray, ZO: np.ndarray,
                             base_ridge: float = 1e-4):
    U = np.asarray(U, dtype=np.float64)
    eig = np.asarray(eig, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).ravel()
    
    if ZO.ndim == 1:
        ZO = ZO[None, :]
        
    U_O = U[idx_O, :] 
    U_U = U[idx_U, :] 
    
    s2 = max(float(sigma2) + float(base_ridge), 1e-6)
        
    dO = (ZO - mu[idx_O][None, :]).T 
    
    sqrt_eig = np.sqrt(np.clip(eig, 0.0, None))
    A_T = U_O.T * sqrt_eig[:, None]  
    
    K = A_T @ A_T.T + s2 * np.eye(len(eig), dtype=np.float64) 
    rhs = A_T @ dO 
    
    try:
        w = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(K, rhs, rcond=None)
        
    v_target = sqrt_eig[:, None] * w 
    projection = U_U @ v_target 
    zU = mu[idx_U][:, None] + projection
    
    return zU.T 

# ---------------------- Main Commands ----------------------

def main_recon(argv=None):
    ap = argparse.ArgumentParser("LAM‑Flow 3D reconstruction and latent editing tool (recon)")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--views", type=str, required=True)
    ap.add_argument("--view-index", type=int, default=0)
    ap.add_argument("--volume-size", type=parse_dhw, default="64x64x64")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--devices", type=str, default="cuda:0")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--gauss", type=str, default=None)
    ap.add_argument("--edit-levels", type=str, default="none")
    ap.add_argument("--edit-what", type=str, choices=["mean", "zero", "pc"], default="mean")
    ap.add_argument("--edit-pc-index", type=int, default=0)
    ap.add_argument("--edit-pc-scale", type=float, default=2.0)
    ap.add_argument("--edit-pc-center", type=str, choices=["sample", "mean"], default="sample")
    args = ap.parse_args(argv)

    device = torch.device(args.devices)
    blob = torch.load(resolve_ckpt_path(Path(args.ckpt)), map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    
    model = build_model_from_config(cfg, device, target_dhw=args.volume_size)
    all_views = [v.strip() for v in args.views.split(",") if v.strip()]
    vname = all_views[int(args.view_index)]
    
    Dc, Hc, Wc = args.volume_size
    _prime_if_needed(model, Dc, Hc, Wc, device)
    
    ok, note = load_weights_into_model(model, blob, int(args.view_index))
    if not ok: raise RuntimeError(f"Failed to load weights: {note}")

    cols = _read_manifest_csv(Path(args.manifest))
    vcol = cols[vname]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.gauss, allow_pickle=True) if args.gauss else None
    L_gauss = int(npz["L"]) if npz else 0

    bs = max(1, int(args.batch))
    for i, pth in enumerate(vcol[:bs]):
        xi = _read_image_3d(pth, target_dhw=args.volume_size).to(device)
        with torch.no_grad():
            z_raw, _ = model.inverse_and_log_det(xi)
            if not isinstance(z_raw, list): z_raw = [z_raw]
            xh_base, _ = model.forward_and_log_det(z_raw)
            xh_base = to01(_coerce_5d(xh_base, args.volume_size), winsorize=True)
            diff_base = to01(torch.abs(xi - xh_base), winsorize=True)
            
        base_name = Path(pth).name.split('.')[0]
        prefix = out_dir / f"recon_{i:03d}_{base_name}"
        save_nifti(xi, Path(f"{prefix}_orig.nii.gz"))
        save_nifti(xh_base, Path(f"{prefix}_recon.nii.gz"))
        save_nifti(diff_base, Path(f"{prefix}_diff.nii.gz"))
        save_mid_slice_png(xh_base, Path(f"{prefix}_recon_midslice.png"))

    print(f"[ok] Outputs in {out_dir}")
    return 0

def main_gauss_fit(argv=None):
    ap = argparse.ArgumentParser("gauss-fit")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--gauss-out", required=True)
    ap.add_argument("--volume-size", type=parse_dhw, default="64x64x64")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--devices", default="cuda:0")
    ap.add_argument("--cov-estimator", choices=["lowrank", "diag"], default="lowrank")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--sigma2", default="auto")
    ap.add_argument("--cov-lam", type=float, default=1e-5)
    args = ap.parse_args(argv)

    device = torch.device(args.devices)
    blob = torch.load(resolve_ckpt_path(Path(args.ckpt)), map_location=device, weights_only=False)
    cfg = blob.get("config", {})

    cols = _read_manifest_csv(Path(args.manifest))
    view_names, paths_per_view = _resolve_views(cols, Path(args.manifest).parent, args.views)
    N = len(paths_per_view[0])
    print(f"[info] Fitting 3D Gaussian on {N} subjects. Estimator: {args.cov_estimator}")

    Z_levels = [] 
    
    for v_idx, vname in enumerate(view_names):
        model_view = build_model_from_config(cfg, device, target_dhw=args.volume_size)
        
        _prime_if_needed(model_view, *args.volume_size, device)
        ok, note = load_weights_into_model(model_view, blob, v_idx)
        if not ok: raise RuntimeError(f"Weights failed for {vname}: {note}")
        
        latents_for_view = [] 
        paths = paths_per_view[v_idx]
        
        for i in tqdm(range(0, N, args.batch), desc=f"Encoding {vname}", unit="batch"):
            batch_p = paths[i:i+args.batch]
            batch_x = []
            for p in batch_p:
                batch_x.append(_read_image_3d(p, args.volume_size).squeeze(0))
            xb = torch.stack(batch_x).to(device)
            
            with torch.no_grad():
                z_raw, _ = model_view.inverse_and_log_det(xb)
                if not isinstance(z_raw, list): z_raw = [z_raw]
                
                # CORRECTION ICI: On construit la liste directement
                z_flat = []
                for t in z_raw:
                    t_safe = torch.clamp(t, min=-20.0, max=20.0)
                    z_flat.append(t_safe.view(t.shape[0], -1).cpu())

                if not latents_for_view: latents_for_view = [[] for _ in z_flat]
                for l, t in enumerate(z_flat): latents_for_view[l].append(t)

        z_view_concat = [torch.cat(batches, dim=0) for batches in latents_for_view]
        if not Z_levels: Z_levels = z_view_concat
        else: Z_levels = [torch.cat([Z_levels[l], z_view_concat[l]], dim=1) for l in range(len(Z_levels))]

    mu_list = []
    Sigma_list = []
    
    for l, Z in enumerate(Z_levels):
        X = Z.numpy().astype(np.float64)
        if not np.isfinite(X).all():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
        cap = np.percentile(np.abs(X), 99.9) + 1e-6
        X = np.clip(X, -cap, cap)
        
        mu = np.mean(X, axis=0)
        Xc = X - mu

        if args.cov_estimator == "lowrank":
            sig = _lowrank_from_Xc(Xc, args.rank, args.sigma2, args.cov_lam)
        else:
            sig = np.var(Xc, axis=0) + args.cov_lam 
            
        mu_list.append(mu)
        Sigma_list.append(sig)

    pack = {
        "mode": "perlevel",
        "estimator": args.cov_estimator,
        "views": view_names,
        "N": N, "L": len(Z_levels),
        "D": args.volume_size[0], "H": args.volume_size[1], "W": args.volume_size[2],
        "dims_per_view_L0": [Z.shape[1] // len(view_names) for Z in Z_levels] 
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

    out_path = Path(args.gauss_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **pack)
    print(f"[ok] Saved {out_path}")

def main_gauss_impute(argv=None):
    ap = argparse.ArgumentParser("gauss-impute")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gauss", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--views", required=True) 
    ap.add_argument("--observed", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--volume-size", type=parse_dhw, default="64x64x64")
    ap.add_argument("--devices", default="cuda:0")
    args = ap.parse_args(argv)

    device = torch.device(args.devices)
    blob = torch.load(resolve_ckpt_path(Path(args.ckpt)), map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    
    # 1. Lecture directe des tailles depuis le modèle Gaussien (plus de dummy_model)
    npz = np.load(args.gauss, allow_pickle=True)
    L = int(npz["L"])
    all_views = list(npz["views"])
    level_sizes = [int(sz) for sz in npz["dims_per_view_L0"]]
    
    obs_views = [v.strip() for v in args.observed.split(",")]
    tgt_views = [v.strip() for v in args.target.split(",")]
    
    slice_map = [] 
    for l in range(L):
        sz = level_sizes[l]
        d = {}
        curr = 0
        for v in all_views:
            d[v] = (curr, curr + sz)
            curr += sz
        slice_map.append(d)

    cols = _read_manifest_csv(Path(args.manifest))
    _, paths_obs = _resolve_views(cols, Path(args.manifest).parent, args.observed)
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    N_sub = len(paths_obs[0])

    print(f"[info] Imputing {tgt_views} from {obs_views} for {N_sub} subjects.")

    for i in tqdm(range(N_sub), desc="Imputing volumes", unit="subj"):
        z_obs_levels = [[] for _ in range(L)]
        obs_images = {}
        z_shapes = {} # Dictionnaire pour capturer dynamiquement les formes 5D réelles
        
        for v_idx, v_name in enumerate(obs_views):
            mdl_obs = build_model_from_config(cfg, device, target_dhw=args.volume_size)
            global_idx = all_views.index(v_name)
            
            ok, note = load_weights_into_model(mdl_obs, blob, global_idx)
            if not ok: raise RuntimeError(f"Weights failed for {v_name}: {note}")
            _prime_if_needed(mdl_obs, *args.volume_size, device)
            
            p = paths_obs[v_idx][i]
            x = _read_image_3d(p, args.volume_size).to(device)
            obs_images[v_name] = x
            with torch.no_grad():
                z, _ = mdl_obs.inverse_and_log_det(x)
                if isinstance(z, tuple): z = list(z)
                if not isinstance(z, list): z = [z]
                for l, t in enumerate(z):
                    z_obs_levels[l].append(t.view(-1).cpu().numpy())
                    # Capture de la forme exacte pour le décodage futur
                    if l not in z_shapes:
                        z_shapes[l] = t.shape

        z_pred_levels = [] 
        
        for l in range(L):
            mu = npz[f"mu_{l}"]
            idx_O = []
            z_vals_O = []
            
            for v_name in obs_views:
                s, e = slice_map[l][v_name]
                idx_O.extend(range(s, e))
                z_vals_O.append(z_obs_levels[l][obs_views.index(v_name)])
            
            idx_O = np.array(idx_O)
            ZO = np.concatenate(z_vals_O) 
            
            t_name = tgt_views[0]
            s_t, e_t = slice_map[l][t_name]
            idx_U = np.arange(s_t, e_t)
            
            if f"Sigma_{l}_type" in npz and str(npz[f"Sigma_{l}_type"]) == "lowrank":
                U = npz[f"Sigma_{l}_U"]
                eig = npz[f"Sigma_{l}_eig"]
                s2 = npz[f"Sigma_{l}_sigma2"]
                
                z_target_flat = _cond_mean_block_lowrank(U, eig, s2, idx_U, idx_O, mu, ZO)
            else:
                z_target_flat = mu[idx_U][None, :]
            
            # Application de la forme 5D capturée dynamiquement
            ref_shape = z_shapes[l]
            z_t_tensor = torch.from_numpy(z_target_flat).float().view(1, ref_shape[1], ref_shape[2], ref_shape[3], ref_shape[4])
            z_pred_levels.append(z_t_tensor.to(device))

        tgt_global_idx = all_views.index(tgt_views[0])
        mdl_tgt = build_model_from_config(cfg, device, target_dhw=args.volume_size)
        
        ok, note = load_weights_into_model(mdl_tgt, blob, tgt_global_idx)
        if not ok: raise RuntimeError(f"Weights failed for {tgt_views[0]}: {note}")
        _prime_if_needed(mdl_tgt, *args.volume_size, device)

        z_pred_clean = []
        for z in z_pred_levels:
            z_safe = torch.nan_to_num(z, nan=0.0, posinf=20.0, neginf=-20.0)
            z_safe = torch.clamp(z_safe, min=-20.0, max=20.0)
            z_pred_clean.append(z_safe)

        with torch.no_grad():
            x_rec, _ = mdl_tgt.forward_and_log_det(z_pred_clean)
            x_rec = to01(_coerce_5d(x_rec, args.volume_size), winsorize=True)

        out_path = Path(args.out_dir)
        out_name = out_path / f"imputed_{i:04d}_{tgt_views[0]}.nii.gz"
        save_nifti(x_rec, out_name)
        save_mid_slice_png(x_rec, out_path / f"imputed_{i:04d}_{tgt_views[0]}_midslice.png")
        
        for v_obs in obs_views:
            input_path = out_path / f"input_{i:04d}_{v_obs}.nii.gz"
            if not input_path.exists():
                save_nifti(obs_images[v_obs], input_path)
                save_mid_slice_png(obs_images[v_obs], out_path / f"input_{i:04d}_{v_obs}_midslice.png")

def main_recon_template(argv=None):
    """
    Reconstruit un template 3D (volume moyen) dans l'espace latent en utilisant le modèle Gaussien.
    Génère le volume décodé à partir de la moyenne (mu), et optionnellement, une moyenne de 
    plusieurs échantillons stochastiques de Monte Carlo.
    """
    ap = argparse.ArgumentParser("LAM-Flow 3D latent template reconstruction (recon-template)")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    ap.add_argument("--gauss", type=str, required=True, help="Gaussian model (.npz)")
    ap.add_argument("--manifest", type=str, required=True, help="Manifest CSV (for views validation)")
    ap.add_argument("--views", type=str, required=True, help="Views list (e.g. T1,FA)")
    ap.add_argument("--view-index", type=int, default=0, help="Which view to reconstruct")
    ap.add_argument("--volume-size", type=parse_dhw, default="64x64x64", help="DxHxW")
    ap.add_argument("--devices", type=str, default="cuda:0")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for templates")
    
    # Options de Monte Carlo
    ap.add_argument("--mc-samples", type=int, default=0,
                    help="Number of Monte Carlo samples to draw in latent space and average.")
    ap.add_argument("--mc-temp", type=float, default=1.0, help="Monte Carlo temperature.")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed for MC sampling.")
    ap.add_argument("--sharpen-image", action="store_true", 
                    help="Apply smoothing & Laplacian sharpening using ANTs before saving.")
    args = ap.parse_args(argv)

    device = torch.device(args.devices)
    mc_n = max(0, int(args.mc_samples))
    if mc_n > 0:
        set_deterministic(int(args.seed))

    # 1. Chargement Modèle
    blob = torch.load(resolve_ckpt_path(Path(args.ckpt)), map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    model = build_model_from_config(cfg, device, target_dhw=args.volume_size)
    _prime_if_needed(model, *args.volume_size, device)
    
    ok, note = load_weights_into_model(model, blob, int(args.view_index))
    if not ok: raise RuntimeError(f"Weights failed: {note}")

    views_list = [v.strip() for v in args.views.split(",") if v.strip()]
    vname = views_list[int(args.view_index)]

    # 2. Chargement Gaussien
    npz = np.load(args.gauss, allow_pickle=True)
    L = int(npz["L"])
    views_g = list(npz["views"])
    level_sizes = [int(sz) for sz in npz["dims_per_view_L0"]]
    
    if vname not in views_g: raise RuntimeError(f"View '{vname}' missing from Gaussian model.")
    v_idx = views_g.index(vname)
    
    slice_map = [] 
    for l in range(L):
        sz = level_sizes[l]
        d = {}
        curr = 0
        for v in views_g:
            d[v] = (curr, curr + sz)
            curr += sz
        slice_map.append(d)

    # 3. Sonde de la dimension latente 5D exacte
    dummy = torch.zeros(1, 1, *args.volume_size, device=device)
    with torch.no_grad():
        z_dummy, _ = model.inverse_and_log_det(dummy)
        if not isinstance(z_dummy, list): z_dummy = [z_dummy]
    z_shapes = [t.shape for t in z_dummy]

    # 4. Construction de la Moyenne (Mu)
    z_mu_list = []
    print(f"[info] Reconstructing base template for {vname} from Gaussian mean (mu)...")
    for l in range(L):
        mu_l = npz[f"mu_{l}"]
        s, e = slice_map[l][vname]
        mu_flat = mu_l[s:e]
        
        mu_tensor = torch.from_numpy(mu_flat).float().view(1, *z_shapes[l][1:]).to(device)
        z_mu_list.append(mu_tensor)

    with torch.no_grad():
        x_mu, _ = model.forward_and_log_det(z_mu_list)
        x_mu = to01(_coerce_5d(x_mu, args.volume_size), winsorize=True)

    # 5. Échantillonnage de Monte Carlo
    x_mc_mean = None
    if mc_n > 0:
        print(f"[info] Generating {mc_n} Monte Carlo samples at temp={args.mc_temp}...")
        z_mc_list = []
        for l in range(L):
            s, e = slice_map[l][vname]
            mu_flat = npz[f"mu_{l}"][s:e]
            Dv = e - s
            
            if f"Sigma_{l}_type" in npz and str(npz[f"Sigma_{l}_type"]) == "lowrank":
                U_full = npz[f"Sigma_{l}_U"]
                eig = npz[f"Sigma_{l}_eig"] * (args.mc_temp ** 2)
                sigma2 = float(npz[f"Sigma_{l}_sigma2"]) * (args.mc_temp ** 2)
                
                U_v = U_full[s:e, :]
                
                # Formule : z = mu + U * sqrt(eig) * xi + sqrt(sigma2) * eps
                xi = np.random.randn(U_v.shape[1], mc_n)
                A = U_v * np.sqrt(np.clip(eig, a_min=0.0, a_max=None))[np.newaxis, :]
                z_samp = mu_flat[:, None] + A @ xi
                
                if sigma2 > 0.0:
                    eps = np.random.randn(Dv, mc_n)
                    z_samp = z_samp + math.sqrt(max(sigma2, 0.0)) * eps
                
                z_samp = z_samp.T # Format (mc_n, D)
            else:
                raise RuntimeError("Only 'lowrank' covariance is currently supported for 3D MC sampling.")

            z_samp_tensor = torch.from_numpy(z_samp).float().view(mc_n, *z_shapes[l][1:]).to(device)
            z_mc_list.append(z_samp_tensor)

        # Décodage Séquentiel (Crucial pour éviter un OOM en 3D)
        print(f"[info] Decoding MC samples sequentially to preserve VRAM...")
        x_mc_sum = torch.zeros_like(x_mu)
        for i in tqdm(range(mc_n), desc="Decoding MC", unit="vol"):
            z_mc_single = [z[i:i+1] for z in z_mc_list]
            with torch.no_grad():
                xi, _ = model.forward_and_log_det(z_mc_single)
                xi = to01(_coerce_5d(xi, args.volume_size), winsorize=True)
            x_mc_sum += xi
            
        x_mc_mean = x_mc_sum / float(mc_n)

    # 6. Sharpening (Optionnel)
    if args.sharpen_image:
        print("[info] Applying Laplacian sharpening...")
        import ants
        
        x_mu_cpu = x_mu.cpu().squeeze().numpy()
        ants_mu = ants.iMath_sharpen(ants.smooth_image(ants.from_numpy(x_mu_cpu), 1.0))
        x_mu = torch.from_numpy(ants_mu.numpy()).view(1, 1, *args.volume_size).to(device)
        
        if x_mc_mean is not None:
            x_mc_cpu = x_mc_mean.cpu().squeeze().numpy()
            ants_mc = ants.iMath_sharpen(ants.smooth_image(ants.from_numpy(x_mc_cpu), 1.0))
            x_mc_mean = torch.from_numpy(ants_mc.numpy()).view(1, 1, *args.volume_size).to(device)

    # 7. Sauvegarde
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_nifti(x_mu, out_dir / f"template_{vname}_mu.nii.gz")
    save_mid_slice_png(x_mu, out_dir / f"template_{vname}_mu_midslice.png")
    
    if x_mc_mean is not None:
        save_nifti(x_mc_mean, out_dir / f"template_{vname}_mc_mean.nii.gz")
        save_mid_slice_png(x_mc_mean, out_dir / f"template_{vname}_mc_mean_midslice.png")
        
        diff = to01(torch.abs(x_mu - x_mc_mean), winsorize=True)
        save_nifti(diff, out_dir / f"template_{vname}_diff.nii.gz")
        save_mid_slice_png(diff, out_dir / f"template_{vname}_diff_midslice.png")

    print(f"[ok] Saved templates to {out_dir}")
    return 0

def main_recon_temperature(argv=None):
    """
    Encode an image, scales the latents by a temperature factor (tau), and reconstructs. 
    Supports global temperatures and per-level overrides.
    """
    ap = argparse.ArgumentParser("LAM-Flow Recon Temperature Scaling")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    ap.add_argument("--manifest", type=str, required=True, help="Manifest CSV")
    ap.add_argument("--views", type=str, required=True, help="Views list (e.g. T1,FA)")
    ap.add_argument("--view-index", type=int, default=0, help="View to process")
    ap.add_argument("--slice-axis", type=int, required=True)
    ap.add_argument("--slice-index", type=int, required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--devices", type=str, default="cuda:0")
    ap.add_argument("--out", type=str, required=True, help="Output PNG panel")
    
    # Option de Température Globale
    ap.add_argument("--tau", type=float, default=0.99, 
                    help="Global temperature scaling factor (default: 0.99). Values < 1.0 pull towards the mean.")
    
    # Option par niveau (Granularité)
    ap.add_argument("--tau-level", action="append", type=str,
                    help="Override temperature for a specific level. Format 'level,value'. "
                         "Example: '--tau-level 0,0.95 --tau-level 4,0.8'. "
                         "Can be repeated.")

    args = ap.parse_args(argv)
    device = torch.device(args.devices)

    # --- Parsing des overrides par niveau ---
    level_overrides = {}
    if args.tau_level:
        for item in args.tau_level:
            try:
                parts = item.split(',')
                if len(parts) != 2: raise ValueError
                lvl = int(parts[0])
                val = float(parts[1])
                level_overrides[lvl] = val
            except ValueError:
                raise RuntimeError(f"Invalid format for --tau-level: '{item}'. Expected 'level,value'.")
        print(f"[info] Per-level overrides: {level_overrides}")

    # 1. Chargement Modèle & Données
    ckpt_path = resolve_ckpt_path(Path(args.ckpt))
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)
    cfg = blob.get("config", {})
    Hc, Wc = int(cfg.get("H", 128)), int(cfg.get("W", 128))
    model = build_model_from_config(cfg if cfg else {"H": Hc, "W": Wc}, device=device)
    model.eval()
    _prime_if_needed(model, Hc, Wc, device=device)

    manifest_path = Path(args.manifest)
    cols = _read_manifest_csv(manifest_path)
    view_names, per_view_paths = _resolve_views(cols, manifest_path.parent, args.views)
    
    views_list = [v.strip() for v in args.views.split(",")]
    vname = views_list[int(args.view_index)]
    
    ok, note = load_weights_into_model(model, blob, view_idx=int(args.view_index), prefer_ema=True, view_name=vname, cfg_views=views_list)
    if not ok: raise RuntimeError(f"Weights failed: {note}")

    paths = per_view_paths[int(args.view_index)]
    xs = []
    limit = min(int(args.batch), len(paths))
    for i in range(limit):
        xi = _read_image_any(paths[i], args.slice_axis, args.slice_index)
        xi = torch.nn.functional.interpolate(xi.unsqueeze(0), size=(Hc, Wc), mode="bilinear").squeeze(0)
        xi = to01(xi.unsqueeze(0)).squeeze(0)
        xs.append(xi)
        
    # Correction PyTorch : Transfert du tenseur sur le périphérique
    xb = torch.stack(xs, dim=0).to(device)
    
    # 3. Encodage x -> z
    z_list = _encode_latents(model, xb)
    
    # 4. Temperature Scaling Granulaire
    z_scaled_list = []
    print(f"[info] Scaling latents with temperature...")

    for l, z in enumerate(z_list):
        # Déterminer la valeur cible de la température pour ce niveau
        if l in level_overrides:
            tau = float(level_overrides[l])
            mode_str = "Override"
        else:
            tau = float(args.tau)
            mode_str = "Global"

        # Multiplication scalaire (Conserve la distribution Gaussienne)
        z_scaled = z * tau
        
        print(f"  Level {l} ({mode_str}): tau={tau}")
        z_scaled_list.append(z_scaled)

    # 5. Décodage & Sauvegarde
    xh = _decode_latents(model, z_scaled_list, target_hw=(Hc, Wc))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if os.path.splitext(args.out)[1].lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        panel = make_recon_panel(xb, xh)
        save_grid(panel, Path(args.out), nrow=3, target_hw=(Hc, Wc))
        print(f"[ok] Saved temperature scaled recon panel to {args.out}")
    elif os.path.splitext(args.out)[1].lower() in [".nii", ".gz"]:
        import ants
        xh = _coerce_nchw_4d(xh, target_hw=(Hc, Wc))
        ants_img = ants.from_numpy(np.flip(xh.squeeze().cpu().numpy(), axis=(1,0))) 
        ants.image_write(ants_img, args.out)
        print(f"[ok] Saved temperature scaled recon to {args.out}")

def main_calc_distance(argv=None):
    """
    Calcule la distance Euclidienne (L2) entre les latents d'une image 3D et une référence.
    Référence par défaut : Moyenne Gaussienne (Mu) extraite du modèle .npz.
    Référence optionnelle : Une image cible (--target-image).
    """
    ap = argparse.ArgumentParser("LAM-Flow 3D Latent Distance Calculator")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint")
    ap.add_argument("--gauss", required=True, help="Gaussian model (.npz)")
    ap.add_argument("--manifest", required=True, help="Input manifest CSV")
    ap.add_argument("--views", required=True, help="Views header (e.g. T1,FA)")
    ap.add_argument("--view-index", type=int, default=0, help="View index to analyze")
    ap.add_argument("--volume-size", type=parse_dhw, default="64x64x64", help="DxHxW")
    ap.add_argument("--batch", type=int, default=1, help="Batch size (keep low for 3D)")
    ap.add_argument("--devices", default="cuda:0")
    ap.add_argument("--out-csv", required=True, help="Output CSV file path")
    ap.add_argument("--save-levels", action=argparse.BooleanOptionalAction, default=True,
                    help="Include separate columns for distance at each level.")
    ap.add_argument("--target-image", type=str, default=None,
                    help="Optional target 3D image path. If set, calculates distance to this image instead of the Gaussian mean.")
    args = ap.parse_args(argv)
    
    device = torch.device(args.devices)
    
    # 1. Chargement du modèle
    blob = torch.load(resolve_ckpt_path(Path(args.ckpt)), map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    model = build_model_from_config(cfg, device, target_dhw=args.volume_size)
    
    _prime_if_needed(model, *args.volume_size, device)
    ok, note = load_weights_into_model(model, blob, int(args.view_index))
    if not ok: raise RuntimeError(f"Weights failed: {note}")
    
    views_list = [v.strip() for v in args.views.split(",")]
    vname = views_list[int(args.view_index)]

    # 2. Chargement du modèle Gaussien (pour obtenir Mu et la structure des niveaux)
    npz = np.load(args.gauss, allow_pickle=True)
    L = int(npz["L"])
    views_g = list(npz["views"])
    level_sizes = [int(sz) for sz in npz["dims_per_view_L0"]]
    
    if vname not in views_g: 
        raise RuntimeError(f"View '{vname}' missing from Gaussian model.")
    v_idx_g = views_g.index(vname)
    
    slice_map = [] 
    for l in range(L):
        sz = level_sizes[l]
        d = {}
        curr = 0
        for v in views_g:
            d[v] = (curr, curr + sz)
            curr += sz
        slice_map.append(d)

    # 3. Parsing du Manifest
    cols = _read_manifest_csv(Path(args.manifest))
    _, per_view_paths = _resolve_views(cols, Path(args.manifest).parent, args.views)
    paths = per_view_paths[int(args.view_index)]
    total_imgs = len(paths)

# ---------------------------------------------------------
    # 4. PRÉPARATION DE LA RÉFÉRENCE ET DE LA VARIANCE
    # ---------------------------------------------------------
    reference_latents = [] 
    variance_latents = [] 

    # A. Extraction systématique de la variance (depuis le modèle Gaussien)
    print(f"[info] Extracting latent variances for standardized distance...")
    for l in range(L):
        s, e = slice_map[l][vname]
        
        if f"Sigma_{l}_type" in npz and str(npz[f"Sigma_{l}_type"]) == "lowrank":
            U_full = npz[f"Sigma_{l}_U"]
            eig = npz[f"Sigma_{l}_eig"]
            sigma2 = float(npz[f"Sigma_{l}_sigma2"])
            
            U_v = U_full[s:e, :]
            # Calcul rapide de la diagonale sans instancier la matrice géante
            var_flat = np.sum((U_v ** 2) * eig, axis=1) + sigma2
        else:
            Sig_full = npz[f"Sigma_{l}"]
            if Sig_full.ndim == 1:
                var_flat = Sig_full[s:e]
            else:
                var_flat = np.diag(Sig_full)[s:e]
                
        var_tensor = torch.from_numpy(var_flat).float().to(device).view(1, -1)
        variance_latents.append(var_tensor)

    # B. Détermination de la référence (Cible ou Moyenne)
    if args.target_image:
        tgt_path = Path(args.target_image)
        if not tgt_path.exists(): raise FileNotFoundError(f"Target image not found: {tgt_path}")
        print(f"[info] Reference: Target Image ({tgt_path.name})")
        
        xt = _read_image_3d(tgt_path, target_dhw=args.volume_size).to(device)
        with torch.no_grad():
            z_tgt_list, _ = model.inverse_and_log_det(xt)
            if not isinstance(z_tgt_list, list): z_tgt_list = [z_tgt_list]
            
        for z in z_tgt_list:
            reference_latents.append(z.view(1, -1).detach()) 
    else:
        print(f"[info] Reference: Gaussian Mean (Mu)")
        for l in range(L):
            mu_l = npz[f"mu_{l}"]
            s, e = slice_map[l][vname]
            mu_flat = mu_l[s:e]
            mu_tensor = torch.from_numpy(mu_flat).float().to(device).view(1, -1)
            reference_latents.append(mu_tensor)

    # ---------------------------------------------------------
    # 5. BOUCLE DE CALCUL
    # ---------------------------------------------------------
    bs = max(1, int(args.batch))
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] Calculating standardized L2 distances for {total_imgs} volumes...")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        
        header = ["path", "total_distance"]
        if args.save_levels:
            header += [f"dist_L{l}" for l in range(L)]
        writer.writerow(header)

        for i in tqdm(range(0, total_imgs, bs), desc="Distance calc", unit="batch"):
            batch_paths = paths[i : i + bs]
            xs = []
            valid_paths = []
            
            for p in batch_paths:
                try:
                    xi = _read_image_3d(p, target_dhw=args.volume_size)
                    xs.append(xi.squeeze(0))
                    valid_paths.append(str(p))
                except Exception as e:
                    print(f"[warn] Failed to read {p}: {e}")

            if not xs: continue

            xb = torch.stack(xs, dim=0).to(device)
            B = xb.shape[0]

            with torch.no_grad():
                z_list, _ = model.inverse_and_log_det(xb)
                if not isinstance(z_list, list): z_list = [z_list]
                
            dists_per_level = np.zeros((B, L), dtype=np.float64)
            
            for l, z in enumerate(z_list):
                ref = reference_latents[l] 
                var = variance_latents[l] # La variance de la population
                z_flat = z.view(B, -1)     
                
                # Distance Euclidienne Standardisée (Mahalanobis diagonale)
                # On divise le carré de la différence par la variance avant de faire la somme
                dist_sq = torch.sum(((z_flat - ref) ** 2) / (var + 1e-6), dim=1).cpu().numpy()
                dists_per_level[:, l] = np.sqrt(dist_sq)

            for b_idx in range(B):
                total_dist = np.sqrt(np.sum(dists_per_level[b_idx] ** 2))
                row = [valid_paths[b_idx], f"{total_dist:.6f}"]
                if args.save_levels:
                    row.extend([f"{d:.6f}" for d in dists_per_level[b_idx]])
                writer.writerow(row)

    print(f"[ok] Distances written to {out_csv}")


def main_recon_interpolate(argv=None):
    """
    Interpole entre une cible (Moyenne Gaussienne ou Image Cible) et un sujet 3D.
    Utilise l'interpolation sphérique (Slerp) centrée sur \mu pour éviter les artéfacts hors-distribution.
    Génère une séquence de volumes NIfTI représentant la transition.
    """
    
    def _slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, DOT_THRESHOLD: float = 0.9995):
        v0_norm = torch.norm(v0, dim=1, keepdim=True)
        v1_norm = torch.norm(v1, dim=1, keepdim=True)
        
        v0_norm_safe = torch.clamp(v0_norm, min=1e-8)
        v1_norm_safe = torch.clamp(v1_norm, min=1e-8)
        
        v0_dir = v0 / v0_norm_safe
        v1_dir = v1 / v1_norm_safe
        
        dot = torch.sum(v0_dir * v1_dir, dim=1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0)
        
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)
        
        lerp_mask = (torch.abs(dot) > DOT_THRESHOLD)
        
        mag = (1.0 - t) * v0_norm + t * v1_norm
        
        slerp_dir = (torch.sin((1.0 - t) * omega) / sin_omega) * v0_dir + (torch.sin(t * omega) / sin_omega) * v1_dir
        res = slerp_dir * mag
        
        lerp_res = (1.0 - t) * v0 + t * v1
        
        return torch.where(lerp_mask, lerp_res, res)

    ap = argparse.ArgumentParser("LAM-Flow 3D Latent Interpolation")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    ap.add_argument("--gauss", type=str, required=True, help="Gaussian model (.npz) - Required for Mu-centered Slerp.")
    ap.add_argument("--manifest", type=str, required=True, help="Manifest CSV (Source images)")
    ap.add_argument("--views", type=str, required=True, help="Views list (e.g. T1,FA)")
    ap.add_argument("--view-index", type=int, default=0, help="View to process")
    ap.add_argument("--volume-size", type=str, default="64x64x64", help="DxHxW") 
    ap.add_argument("--batch", type=int, default=1, help="Number of source subjects to process")
    ap.add_argument("--devices", type=str, default="cuda:0")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for NIfTI frames")
    
    # Options d'interpolation
    ap.add_argument("--target-image", type=str, default=None,
                    help="Optional target image path. If not set, interpolates towards Gaussian mean.")
    ap.add_argument("--steps", type=int, default=5, 
                    help="Number of interpolation steps (frames) to generate between Target (t=0) and Source (t=1).")
    args = ap.parse_args(argv)
    
    device = torch.device(args.devices)
    
    if isinstance(args.volume_size, str):
        v_parts = args.volume_size.lower().split("x")
        args.volume_size = (int(v_parts[0]), int(v_parts[1]), int(v_parts[2]))

    # 1. Chargement Modèle
    blob = torch.load(resolve_ckpt_path(Path(args.ckpt)), map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    model = build_model_from_config(cfg, device, target_dhw=args.volume_size)
    _prime_if_needed(model, *args.volume_size, device)
    
    ok, note = load_weights_into_model(model, blob, int(args.view_index))
    if not ok: raise RuntimeError(f"Weights failed: {note}")

    views_list = [v.strip() for v in args.views.split(",")]
    vname = views_list[int(args.view_index)]

    # 2. Chargement Manifest Source
    cols = _read_manifest_csv(Path(args.manifest))
    _, per_view_paths = _resolve_views(cols, Path(args.manifest).parent, args.views)
    paths = per_view_paths[int(args.view_index)]

    # 3. Chargement Inconditionnel de la Moyenne Gaussienne (\mu)
    print(f"[info] Loading Gaussian Mean (Mu) for centering...")
    npz = np.load(args.gauss, allow_pickle=True)
    L = int(npz["L"])
    views_g = list(npz["views"])
    level_sizes = [int(sz) for sz in npz["dims_per_view_L0"]]
    
    if vname not in views_g: raise RuntimeError(f"View '{vname}' missing from Gaussian model.")
    
    # Extraction des tenseurs \mu par niveau
    mu_tensors = []
    for l in range(L):
        curr = 0
        for v in views_g:
            if v == vname:
                s, e = curr, curr + level_sizes[l]
                break
            curr += level_sizes[l]
            
        mu_flat = npz[f"mu_{l}"][s:e]
        mu_tensors.append(torch.from_numpy(mu_flat).float().to(device))

    # 4. Détermination de la Cible (z_target)
    z_target_list = []
    is_slerp = False
    
    if args.target_image:
        tgt_path = Path(args.target_image)
        if not tgt_path.exists(): raise FileNotFoundError(f"Target image not found: {tgt_path}")
        print(f"[info] Target: Specific Image ({tgt_path.name}) -> Using Mu-centered Slerp")
        
        xt = _read_image_3d(tgt_path, target_dhw=args.volume_size).to(device)
        with torch.no_grad():
            z_tgt_raw, _ = model.inverse_and_log_det(xt)
            if not isinstance(z_tgt_raw, list): z_tgt_raw = [z_tgt_raw]
            for z in z_tgt_raw:
                z_target_list.append(z.clone()) # (1, C, D, H, W)
        is_slerp = True
    else:
        print(f"[info] Target: Gaussian Mean (Mu) -> Using standard Lerp")
        # On utilise un encodage factice pour obtenir la forme 5D exacte de la cible
        dummy = torch.zeros(1, 1, args.volume_size[0], args.volume_size[1], args.volume_size[2], device=device)
        with torch.no_grad():
            z_dummy, _ = model.inverse_and_log_det(dummy)
            if not isinstance(z_dummy, list): z_dummy = [z_dummy]

        for l in range(L):
            ref_shape = z_dummy[l].shape
            mu_t = mu_tensors[l].view(1, ref_shape[1], ref_shape[2], ref_shape[3], ref_shape[4])
            z_target_list.append(mu_t)

    # 5. Boucle d'Interpolation (Sujet par Sujet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bs = max(1, int(args.batch))
    limit = min(bs, len(paths))

    alphas = np.linspace(0.0, 1.0, args.steps)
    
    for i in range(limit):
        pth = paths[i]
        print(f"\n[info] Interpolating Source: {pth.name}")
        xs = _read_image_3d(pth, target_dhw=args.volume_size).to(device)
        
        with torch.no_grad():
            z_source_list, _ = model.inverse_and_log_det(xs)
            if not isinstance(z_source_list, list): z_source_list = [z_source_list]
            
            base_name = Path(pth).name.split('.')[0]
            subj_dir = out_dir / f"interp_{i:03d}_{base_name}"
            subj_dir.mkdir(exist_ok=True)
            
            for step_idx, alpha in enumerate(alphas):
                z_interp_list = []
                for l, (z_src, z_tgt, mu_t) in enumerate(zip(z_source_list, z_target_list, mu_tensors)):
                    
                    B, C, D_dim, H, W = z_src.shape
                    
                    z_src_flat = z_src.view(B, -1)
                    z_tgt_flat = z_tgt.view(B, -1)
                    mu_flat_b = mu_t.view(1, -1)
                    
                    if is_slerp:
                        # Slerp centré sur \mu
                        z_src_c = z_src_flat - mu_flat_b
                        z_tgt_c = z_tgt_flat - mu_flat_b
                        z_new_c = _slerp(float(alpha), z_tgt_c, z_src_c)
                        z_new_flat = z_new_c + mu_flat_b
                    else:
                        # Lerp vers la moyenne
                        z_new_flat = z_tgt_flat + float(alpha) * (z_src_flat - z_tgt_flat)
                    
                    z_new = z_new_flat.view(B, C, D_dim, H, W)
                    z_interp_list.append(z_new)
                
                xh, _ = model.forward_and_log_det(z_interp_list)
                xh = to01(_coerce_5d(xh, args.volume_size), winsorize=True)
                
                out_name = subj_dir / f"frame_{step_idx:02d}_alpha{alpha:.2f}.nii.gz"
                save_nifti(xh, out_name)
                save_mid_slice_png(xh, subj_dir / f"frame_{step_idx:02d}_alpha{alpha:.2f}_midslice.png")
                
        print(f"[ok] Generated {args.steps} frames in {subj_dir}")

    return 0

def main_sample(argv=None):
    ap = argparse.ArgumentParser("LAM‑Flow 3D sample tool")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--view-index", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--volume-size", type=parse_dhw, default="64x64x64")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--devices", type=str, default="cuda:0")
    ap.add_argument("--out-dir", type=str, default="samples_3d")
    args = ap.parse_args(argv)

    ckpt_path = resolve_ckpt_path(Path(args.ckpt))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu") if args.devices.lower() == "cpu" else torch.device(args.devices.split(",")[0])
    set_deterministic(args.seed)

    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    
    model = build_model_from_config(cfg, device, target_dhw=args.volume_size)
    
    Dc, Hc, Wc = args.volume_size
    _prime_if_needed(model, Dc, Hc, Wc, device)
    
    ok, note = load_weights_into_model(model, blob, view_idx=int(args.view_index), prefer_ema=bool(args.ema))
    if not ok: raise RuntimeError(f"Weights failed: {note}")

    if args.n_samples > 0:
        print(f"[info] sampling {args.n_samples} volumes @ temp={args.temperature}")
        with torch.no_grad():
            for i in range(args.n_samples):
                try: z_sample = model.sample(1, temperature=float(args.temperature))
                except TypeError: z_sample = model.sample(1) 
                    
                x = _coerce_5d(z_sample, target_dhw=(Dc, Hc, Wc))
                x = to01(x, winsorize=True)
                
                out_file = out_dir / f"sample_{i:04d}.nii.gz"
                save_nifti(x, out_file)
                save_mid_slice_png(x, out_dir / f"sample_{i:04d}_midslice.png")

        print(f"[ok] wrote {args.n_samples} samples to {out_dir}")

if __name__ == "__main__":
    table = {
        "sample": main_sample,
        "recon": main_recon,
        "gauss-fit": main_gauss_fit,
        "gauss-impute": main_gauss_impute,
        "recon-temperature": main_recon_temperature,
        "calc-distance": main_calc_distance,
        "recon-interpolate": main_recon_interpolate,
        "recon-template": main_recon_template
    }
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Available subcommands:", ", ".join(sorted(table.keys())))
        sys.exit(0)
    cmd = sys.argv.pop(1)
    if cmd not in table:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
    sys.exit(table[cmd]())