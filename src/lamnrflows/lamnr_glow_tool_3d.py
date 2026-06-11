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
   - recon-template: Generate population average templates and Monte Carlo samples.
   - recon-temperature: Modulate latents globally or per-level to remove outliers (lesions/artifacts).
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

6. Modulation (temperature)
   Modulate 3D latent vectors.
   
   python lamnr_glow_tool_3d.py recon-temperature \
     --ckpt runs/model_64x64x64/training_state.pt \
     --manifest data/manifest_lesions.csv \
     --views T1 \
     --tau 0.95 \
     --out-dir output/scaled_tau095/

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
import warnings
from xml.parsers.expat import model

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

_orig_double = torch.Tensor.double

def _mps_safe_double(self, *args, **kwargs):
    # Si le tenseur est sur puce Apple, on force float32 car float64 fait planter
    if self.device.type == 'mps':
        return self.float(*args, **kwargs)
    return _orig_double(self, *args, **kwargs)

torch.Tensor.double = _mps_safe_double
# -----------------------------------------

def parse_hw(spec: str) -> Tuple[int, int]:
    try:
        a, b = spec.lower().split("x")
        H, W = int(a), int(b)
        assert H > 0 and W > 0
        return H, W
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid HxW spec '{spec}'. Expected like '128x128'.")

def parse_hw_float(spec: str) -> Tuple[float, float]:
    try:
        a, b = spec.lower().split("x")
        H, W = float(a), float(b)
        assert H > 0 and W > 0
        return H, W
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid spacing spec '{spec}'. Expected like '0.8x0.8'.")

def parse_hwd(spec: str) -> Tuple[int, int, int]:
    try:
        parts = spec.lower().split("x")
        H, W, D = int(parts[0]), int(parts[1]), int(parts[2])
        assert D > 0 and H > 0 and W > 0
        return H, W, D
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid HxWxD spec '{spec}'. Expected like '64x64x64'.")

def parse_hwd_float(spec: str) -> Tuple[float, float, float]:
    try:
        parts = spec.lower().split("x")
        H, W, D = float(parts[0]), float(parts[1]), float(parts[2])
        assert D > 0 and H > 0 and W > 0
        return H, W, D
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid spacing spec '{spec}'. Expected like '1.0x1.0x1.0'.")

def parse_mn(spec: str) -> Tuple[int, int]:
    try:
        m, n = spec.lower().split("x")
        M, N = int(m), int(n)
        assert M > 0 and N > 0
        return M, N
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid --grid-size '{spec}'. Expected like '6x8' (rows×cols).")

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
    Normalise les volumes 3D (N, C, H, W, D) ou images 2D (N, C, H, W) entre 0 et 1.
    """
    # 1. Élimination pure et simple des NaNs/Infs issus de l'augmentation ANTs
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

    if x.ndim < 4:
        return x
    
    # Pour un volume 3D, spatial_dims sera (2, 3, 4)
    spatial_dims = tuple(range(2, x.ndim))

    if winsorize:
        # torch.quantile ne supporte pas le float16, on force le calcul en float32
        x_calc = x.float()
        
        # Aplatit (N, C, H, W, D) en (N, C, H*W*D)
        x_flat = x_calc.flatten(start_dim=2)
        
        # Calcule les percentiles sur l'ensemble des voxels
        q_low = torch.quantile(x_flat, 0.01, dim=2, keepdim=True).to(x.dtype)
        q_high = torch.quantile(x_flat, 0.99, dim=2, keepdim=True).to(x.dtype)
        
        # Redimensionne en (N, C, 1, 1, 1) pour le broadcasting 3D
        view_shape = x.shape[:2] + (1,) * len(spatial_dims)
        x_min = q_low.view(view_shape)
        x_max = q_high.view(view_shape)
        
    else:
        # Comportement standard pour l'entraînement
        x_min = x.amin(dim=spatial_dims, keepdim=True)
        x_max = x.amax(dim=spatial_dims, keepdim=True)

    norm = (x - x_min) / (x_max - x_min + eps)
    return torch.clamp(norm, eps, 1.0 - eps)

def _coerce_5d(x, target_hwd=None):
    """Garantit que la sortie est un tenseur 5D (B, C, H, W, D) float32."""
    if isinstance(x, (list, tuple)):
        x = x[-1] # Prendre la sortie finale du flux
        
    if x.ndim == 4:
        x = x.unsqueeze(0) # Ajouter batch si manquant
        
    x = x.float()
    
    if target_hwd is not None:
        Hc, Wc, Dc = target_hwd
        # Vérifie les 3 dernières dimensions (spatiales)
        if x.shape[-3:] != (Hc, Wc, Dc):
            x = F.interpolate(x, size=(Hc, Wc, Dc), mode="trilinear", align_corners=False)

    return x

from typing import Tuple, Optional
import torch
import numpy as np
import ants

@torch.no_grad()
def resample_with_ants_spacing_3d(x: torch.Tensor,
                                  native_spacing: Tuple[float, float, float],
                                  target_spacing: Tuple[float, float, float]) -> torch.Tensor:
    """
    Resample 5D tensor (N, C, D, H, W) to a target physical spacing using ANTsPy (use_voxels=False).
    If C>1, channels are resampled independently and stacked back.
    """
    device, dtype = x.device, x.dtype
    N, C = x.shape[0], x.shape[1]
    outs = []
    
    for c in range(C):
        xs = []
        for i in range(N):
            arr = x[i, c].detach().cpu().numpy()
            img = ants.from_numpy(arr)
            
            # Application de l'espacement 3D natif
            try:
                img.set_spacing((float(native_spacing[0]), float(native_spacing[1]), float(native_spacing[2])))
            except Exception:
                img.spacing = (float(native_spacing[0]), float(native_spacing[1]), float(native_spacing[2]))
                
            # Rééchantillonnage physique 3D
            img_r = ants.resample_image(img, 
                                        (float(target_spacing[0]), float(target_spacing[1]), float(target_spacing[2])),
                                        use_voxels=False, 
                                        interp_type=0)
            
            xs.append(torch.from_numpy(img_r.numpy()).to(device=device, dtype=dtype))
        outs.append(torch.stack(xs, dim=0))
        
    y = torch.stack(outs, dim=1)  # (N, C, h, w, d)
    return y

@torch.no_grad()
def resample_with_ants_size_3d(x: torch.Tensor,
                               target_size: Tuple[int, int, int],
                               native_spacing: Optional[Tuple[float, float, float]] = None) -> torch.Tensor:
    """
    Resample 5D tensor (N, C, H, W, D) to a target voxel size (H, W, D) using ANTsPy (use_voxels=True).
    """
    device, dtype = x.device, x.dtype
    N, C = x.shape[0], x.shape[1]
    outs = []
    
    for c in range(C):
        xs = []
        for i in range(N):
            arr = x[i, c].detach().cpu().numpy()
            img = ants.from_numpy(arr)
            
            if native_spacing is not None:
                try:
                    img.set_spacing((float(native_spacing[0]), float(native_spacing[1]), float(native_spacing[2])))
                except Exception:
                    img.spacing = (float(native_spacing[0]), float(native_spacing[1]), float(native_spacing[2]))
                    
            # Rééchantillonnage par nombre de voxels 3D
            img_r = ants.resample_image(img, 
                                        (int(target_size[0]), int(target_size[1]), int(target_size[2])),
                                        use_voxels=True, 
                                        interp_type=0)
            
            xs.append(torch.from_numpy(img_r.numpy()).to(device=device, dtype=dtype))
        outs.append(torch.stack(xs, dim=0))
        
    y = torch.stack(outs, dim=1)  # (N, C, d, h, w)
    return y

import numpy as np
import torch
import ants
from pathlib import Path

def save_nifti(x: torch.Tensor, out_path: Path | str, reference_image: ants.ANTsImage | None = None):
    """
    Sauvegarde un tenseur 3D au format NIfTI.
    Utilise from_numpy_like pour hériter de la géométrie exacte si une référence est fournie.
    """
    x = x.detach().cpu()
    
    # Nettoyage des dimensions (retrait du batch si présent)
    if x.ndim == 5: 
        x = x.squeeze(0) 
        
    arr = x.numpy()
    
    # Gestion des canaux (C, D, H, W) -> (D, H, W, C) pour ANTs multi-canaux
    if arr.shape[0] == 1: 
        arr = arr[0] 
    else: 
        arr = np.transpose(arr, (1, 2, 3, 0))
        
    # Création de l'image ANTs
    if reference_image is not None:
        img = ants.from_numpy_like(arr, reference_image)
    else:
        img = ants.from_numpy(arr)
        # Fallback de sécurité (1 mm isotropique)
        try:
            img.set_spacing((1.0, 1.0, 1.0))
        except Exception:
            pass
            
    ants.image_write(img, str(out_path))

import torchvision as tv
import torch
from pathlib import Path

def save_grid(x: torch.Tensor, out_path: Path | str, nrow: int, slice_axis: int = 2, winsorize: bool = True):
    """
    Extrait une coupe 2D du milieu de chaque volume 3D dans un lot (batch) 
    et sauvegarde le tout sous forme de grille d'images (PNG/JPG).
    
    Input attendu : Tenseur 5D de forme (B, C, Dim1, Dim2, Dim3)
    """
    # 1. Préparation du tenseur (CPU et float32)
    x = x.detach().cpu().float()
    
    # Sécurité : S'assurer d'avoir 5 dimensions
    if x.ndim == 4:
        x = x.unsqueeze(0)
        
    # 2. Normalisation globale sur le volume 3D
    # (Utilise la fonction to01 déjà présente dans votre script)
    x = to01(x, winsorize=winsorize)
    
    # 3. Détermination de la dimension à couper 
    # (+2 pour ignorer les dimensions Batch et Channel)
    dim_to_slice = slice_axis + 2
    
    # Trouver l'indice de la coupe centrale
    mid_idx = x.shape[dim_to_slice] // 2
    
    # 4. Extraction de la coupe pour tout le lot simultanément
    if slice_axis == 0:
        x_2d = x[:, :, mid_idx, :, :]
    elif slice_axis == 1:
        x_2d = x[:, :, :, mid_idx, :]
    elif slice_axis == 2:
        x_2d = x[:, :, :, :, mid_idx]
    else:
        raise ValueError(f"slice_axis invalide ({slice_axis}). Doit être 0, 1 ou 2.")
        
    # x_2d est maintenant de forme (B, C, H, W)
    
    # 5. Création du dossier cible et sauvegarde
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    tv.utils.save_image(x_2d, str(out_path), nrow=int(nrow))

def _read_image_3d(path: Path, target_hwd: tuple[int, int, int]) -> torch.Tensor:
    import ants
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"{path}")
    
    img = ants.image_read(str(path))
    H, W, D = target_hwd
    
    resize_factor = min(float(H)/float(img.shape[0]), 
                        float(W)/float(img.shape[1]),
                        float(D)/float(img.shape[2]))
    
    spacing = (img.spacing[0] / resize_factor, 
               img.spacing[1] / resize_factor,
               img.spacing[2] / resize_factor)   
    
    img = ants.resample_image(img, spacing, use_voxels=False, interp_type=0)
    img = ants.pad_or_crop_image_to_size(img, (H, W, D))
    
    img = img * ants.get_mask(img) 
    
    arr = img.numpy()
    if arr.ndim == 3: 
        arr = arr[np.newaxis, ...] # (1, H, W, D)
        
    t = torch.from_numpy(arr).float()
    
    # Normalisation Min-Max robuste pour l'inférence
    x_min = t.amin(dim=(1, 2, 3), keepdim=True)
    x_max = t.amax(dim=(1, 2, 3), keepdim=True)
    t = (t - x_min) / (x_max - x_min + 1e-8)
    
    return t # Retourne (1, H, W, D)
    
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

def _encode_latents(model, xb: torch.Tensor) -> List[torch.Tensor]:
    """
    Pousse un lot de volumes (5D) vers les latents multi-échelles z_list.
    Input: xb (B, 1, H, W, D)
    Output: Liste de tenseurs per-level
    """
    # S'assurer que le tenseur est au bon format pour la 3D
    if xb.ndim != 5:
        raise RuntimeError(f"Encodage 3D attend 5 dimensions, reçu {xb.ndim}")
        
    device_type = xb.device.type
    # Désactiver l'autocast pour garantir la précision numérique des flows
    with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=False):
        if hasattr(model, "inverse_and_log_det"):
            z, _ = model.inverse_and_log_det(xb)
        elif hasattr(model, "inverse"):
            z, _ = model.inverse(xb)
        else:
            raise RuntimeError("Le modèle manque de mapping inverse (inverse_and_log_det).")
            
    return z if isinstance(z, (list, tuple)) else [z]

def _decode_latents(model, z_list: List[torch.Tensor], target_hwd: Tuple[int, int, int]) -> torch.Tensor:
    """Décode une liste de latents multi-échelles vers l'espace image 3D."""
    if not isinstance(z_list, (list, tuple)):
        z_list = [z_list]
        
    device = z_list[0].device
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
        if hasattr(model, "forward_and_log_det"):
            xh, _ = model.forward_and_log_det(z_list)
        else:
            raise RuntimeError("Le modèle n'expose pas forward_and_log_det; décodage impossible.")
            
    return _coerce_5d(xh, target_hwd=target_hwd)

def _edit_latents_to_mean_for_view_3d(
    z_list: List[torch.Tensor],
    gauss_blob: Dict[str, Any],
    view_name: str,
    levels_to_edit: List[int],
    mode: str = "mean",
    pc_index: int = 0,
    pc_scale: float = 2.0,
    pc_center: str = "sample",
    pc_k: int = 64,
    pc_beta: float = 0.0
) -> List[torch.Tensor]:
    import numpy as np

    if not levels_to_edit:
        return z_list

    views, dims_tbl, shapes_by_view, L = _validate_gauss_blob(gauss_blob)

    try:
        v_idx = views.index(view_name)
    except ValueError:
        raise RuntimeError(f"[recon] View '{view_name}' not found in Gaussian header {views}.")

    mu_list = gauss_blob["mu"]
    Sigma_list = gauss_blob.get("Sigma", None)

    raw_slices = gauss_blob.get("level_view_slices", None)
    level_view_slices: List[Dict[int, Tuple[int, int]]] = []
    V = len(views)
    if raw_slices is not None:
        for l in range(L):
            row = raw_slices[l]
            if isinstance(row, dict):
                row_int = {int(k): tuple(v) for k, v in row.items()}
            else:
                row_int = {vi: tuple(row[vi]) for vi in range(V)}
            level_view_slices.append(row_int)
    else:
        for l in range(L):
            off = 0
            row_int = {}
            for vi in range(V):
                d = int(np.asarray(dims_tbl[vi][l]).item())
                row_int[vi] = (off, off + d)
                off += d
            level_view_slices.append(row_int)

    levels_set = {int(l) for l in levels_to_edit}
    z_out: List[torch.Tensor] = []

    for l, z_l in enumerate(z_list):
        if l not in levels_set:
            z_out.append(z_l)
            continue

        if z_l.ndim != 5:
            raise RuntimeError(f"[recon] Expected 5D latent at level {l}, got shape {tuple(z_l.shape)}.")

        # L'ordre spatial du modèle est H, W, D
        B, C, H_z, W_z, D_z = z_l.shape 
        # La forme sauvegardée par gauss-fit respecte l'ordre du modèle
        Cg, Hg, Wg, Dg = shapes_by_view[v_idx][l]

        a, b = level_view_slices[l][v_idx]

        mu_level = np.asarray(mu_list[l], dtype=np.float64).ravel()
        mu_view_flat = mu_level[a:b]
        
        # Adaptation 3D : (1, C, H, W, D)
        mu_view = torch.as_tensor(mu_view_flat, dtype=z_l.dtype, device=z_l.device).view(1, C, H_z, W_z, D_z)

        if mode == "mean":
            z_l_edit = mu_view.expand(B, C, H_z, W_z, D_z)

        elif mode == "zero":
            z_l_edit = torch.zeros_like(z_l)

        elif mode == "pc":
            Sigma_l = Sigma_list[l] if isinstance(Sigma_list, (list, tuple)) else Sigma_list
            Dv = C * H_z * W_z * D_z
            
            if isinstance(Sigma_l, dict) and Sigma_l.get("type") == "lowrank":
                U = np.asarray(Sigma_l["U"], dtype=np.float64)
                eig = np.asarray(Sigma_l["eig"], dtype=np.float64)
                sigma2 = float(Sigma_l.get("sigma2", 0.0))
                U_v = U[a:b, :]
                Sv = (U_v * eig[np.newaxis, :]) @ U_v.T
                if sigma2 > 0.0:
                    Sv = Sv + sigma2 * np.eye(Dv, dtype=np.float64)
            else:
                S = np.asarray(Sigma_l, dtype=np.float64)
                if S.ndim == 1:
                    Sv = np.diag(S[a:b])
                else:
                    Sv = S[a:b, a:b]

            Sv = 0.5 * (Sv + Sv.T)
            w, V_mat = np.linalg.eigh(Sv)

            k = int(pc_index)
            col = -1 - k
            direction_np = V_mat[:, col]
            lam = float(max(w[col], 0.0))
            step = float(pc_scale) * (lam ** 0.5 if lam > 0.0 else 0.0)

            # Adaptation 3D
            direction_t = torch.from_numpy(direction_np.astype(np.float32)).view(1, C, H_z, W_z, D_z).to(z_l.device, z_l.dtype)

            if pc_center.lower() == "mean":
                base = mu_view.expand(B, C, H_z, W_z, D_z)
            else:
                base = z_l

            z_l_edit = base + step * direction_t
            print(f"[recon] level {l}, view '{view_name}': PC{pc_index} lambda={lam:.3e}, step={step:.3e}, center={pc_center}")

        elif mode == "pc_denoise":
            Sigma_l = Sigma_list[l] if isinstance(Sigma_list, (list, tuple)) else Sigma_list
            Dv = C * H_z * W_z * D_z
            
            if isinstance(Sigma_l, dict) and Sigma_l.get("type") == "lowrank":
                U = np.asarray(Sigma_l["U"], dtype=np.float64)
                eig = np.asarray(Sigma_l["eig"], dtype=np.float64)
                sigma2 = float(Sigma_l.get("sigma2", 0.0))
                U_v = U[a:b, :]
                Sv = (U_v * eig[np.newaxis, :]) @ U_v.T
                if sigma2 > 0.0:
                    Sv = Sv + sigma2 * np.eye(Dv, dtype=np.float64)
            else:
                S = np.asarray(Sigma_l, dtype=np.float64)
                if S.ndim == 1:
                    Sv = np.diag(S[a:b])
                else:
                    Sv = S[a:b, a:b]

            Sv = 0.5 * (Sv + Sv.T)
            w, V_mat = np.linalg.eigh(Sv)

            V_desc = V_mat[:, ::-1]
            k_keep = min(max(int(pc_k), 0), V_desc.shape[1])
            V_t = torch.from_numpy(V_desc.astype(np.float32)).to(z_l.device, z_l.dtype)

            z_flat = z_l.view(B, -1)
            mu_flat = mu_view.view(1, -1)
            y = torch.matmul(z_flat - mu_flat, V_t)

            if k_keep < V_t.shape[1]:
                tail = y[:, k_keep:]
                if float(pc_beta) == 0.0:
                    y[:, k_keep:] = 0.0
                else:
                    y[:, k_keep:] = float(pc_beta) * tail

            z_flat_edit = mu_flat + torch.matmul(y, V_t.T)
            z_l_edit = z_flat_edit.view(B, C, H_z, W_z, D_z) # Adaptation 3D

            print(f"[recon] level {l}, view '{view_name}': pc_denoise k_keep={k_keep}, tail_beta={pc_beta:.3f}")

        else:
            raise ValueError(f"[recon] Unknown edit mode '{mode}'.")

        z_out.append(z_l_edit)

    return z_out

def _validate_gauss_blob(g: dict):
    """
    Validate required fields in the serialized Gaussian blob from gauss-fit.
    Returns (views, dims_tbl, shapes_by_view, L) if valid, else raises RuntimeError
    with a detailed, actionable message.
    """
    import numpy as _np

    def _shape_of(x):
        try:
            return f"{len(x)}" if hasattr(x, "__len__") else "n/a"
        except Exception:
            return "n/a"

    def _prod_all(t):
        try:
            # Fonction générique pour calculer le produit de toutes les dimensions
            import math
            return math.prod(int(v) for v in t)
        except Exception:
            return None

    errors = []
    views = g.get("views", None)
    dims_tbl = g.get("dims_per_level_per_view", None)  # V × L
    shapes_by_view = g.get("shapes_by_view", None)     # V × L × (C,D,H,W) ou (C,H,W)
    L_raw = g.get("L", None)

    # 1) Presence / types
    if not isinstance(views, (list, tuple)) or len(views) == 0 or not all(isinstance(v, str) for v in views):
        errors.append(f"- 'views' missing or invalid; expected non-empty list[str], got: {type(views).__name__} with len={_shape_of(views)}")

    if dims_tbl is None or not isinstance(dims_tbl, (list, tuple)):
        errors.append(f"- 'dims_per_level_per_view' missing or invalid; expected list[list[int]], got: {type(dims_tbl).__name__}")
    if shapes_by_view is None or not isinstance(shapes_by_view, (list, tuple)):
        errors.append(f"- 'shapes_by_view' missing or invalid; expected list[list[tuple]], got: {type(shapes_by_view).__name__}")

    # L must be a positive integer
    try:
        L = int(L_raw)
        if L <= 0:
            errors.append(f"- 'L' present but non-positive; expected integer > 0, got: {L_raw!r}")
    except Exception:
        errors.append(f"- 'L' missing or not an int; got: {L_raw!r}")

    # If any structural errors so far, raise early with context
    if errors:
        raise RuntimeError(
            "[gauss] Invalid gaussian file structure:\n"
            + "\n".join(errors)
        )

    # 2) Dimensions across views
    V = len(views)
    if len(dims_tbl) != V:
        errors.append(f"- dims_per_level_per_view has V={len(dims_tbl)} rows but views has V={V}")
    if len(shapes_by_view) != V:
        errors.append(f"- shapes_by_view has V={len(shapes_by_view)} rows but views has V={V}")

    # Ensure each view has L entries
    bad_dims_rows = [vi for vi in range(V) if not isinstance(dims_tbl[vi], (list, tuple)) or len(dims_tbl[vi]) != L]
    bad_shapes_rows = [vi for vi in range(V) if not isinstance(shapes_by_view[vi], (list, tuple)) or len(shapes_by_view[vi]) != L]
    if bad_dims_rows:
        errors.append(f"- dims_per_level_per_view rows with wrong length L={L}: {bad_dims_rows[:10]} (showing first 10)")
    if bad_shapes_rows:
        errors.append(f"- shapes_by_view rows with wrong length L={L}: {bad_shapes_rows[:10]} (showing first 10)")

    # 3) Per-level consistency: dims_tbl[v][ℓ] == prod(shapes_by_view[v][ℓ])
    mismatches = []
    for vi in range(V):
        if vi in bad_dims_rows or vi in bad_shapes_rows:
            continue
        for l in range(L):
            try:
                d_tbl = int(_np.asarray(dims_tbl[vi][l]).item() if hasattr(dims_tbl[vi][l], "item") else dims_tbl[vi][l])
            except Exception:
                d_tbl = None
            d_shp = _prod_all(shapes_by_view[vi][l])
            if d_tbl is None or d_shp is None or d_tbl != d_shp:
                mismatches.append((vi, l, d_tbl, d_shp))
                if len(mismatches) >= 20:
                    break
        if len(mismatches) >= 20:
            break
    if mismatches:
        msg = "\n".join([f"  - view[{vi}]='{views[vi]}', level {l}: dims_tbl={dt} vs Prod(shape)={ds}"
                        for (vi, l, dt, ds) in mismatches])
        errors.append(f"- dims_per_level_per_view does not match shapes_by_view for some entries (showing up to 20):\n{msg}")

    if errors:
        # Helpful footer with quick hints
        footer = (
            "\nHints:\n"
            "  • Re-run gauss-fit to regenerate the file if you changed model config (H/W/D, K, levels).\n"
            "  • Ensure --views in gauss-fit matches the manifest header order you expect to use in imputation.\n"
            "  • Verify that your serialized file includes the new fields written by the updated gauss-fit."
        )
        raise RuntimeError("[gauss] Inconsistent gaussian metadata:\n" + "\n".join(errors) + footer)

    # Normalize dims_tbl to pure Python ints
    dims_tbl_py = [[int(_np.asarray(d).item() if hasattr(d, "item") else d) for d in row] for row in dims_tbl]

    return views, dims_tbl_py, shapes_by_view, L


def _load_gaussian_model(gauss_path: Path) -> Dict[str, Any]:
    """
    Load Gaussian model saved by gauss-fit (.pt or .npz).
    Returns a dict with keys:
    - mode: "perlevel" or "merged"
    - estimator: "full"|"diag"|"lw"|"oas"|"lowrank"
    - views: list[str]
    - N, H, W, L: ints (and potentially D for 3D)
    - dims_per_level_per_view: V x L list of ints
    - shapes_by_view: optional V x L list of tuples
    - level_view_slices: optional L x V list of (start,end) in level-flat space
    - mu: list[(D_l,)] if perlevel else (D_total,)
    - Sigma: list[np.ndarray or dict] if perlevel else np.ndarray or dict
    """
    gauss_path = Path(gauss_path)
    if not gauss_path.exists():
        raise FileNotFoundError(f"Gaussian file not found: {gauss_path}")

    if str(gauss_path).endswith(".pt"):
        try:
            blob = torch.load(gauss_path, map_location="cpu", weights_only=True)
        except Exception as e:
            print(f"[warn] weights_only load failed ({e.__class__.__name__}: {e}); retrying without weights_only")
            blob = torch.load(gauss_path, map_location="cpu")
        return blob

    npz = np.load(str(gauss_path), allow_pickle=True)
    keys = set(npz.files)
    blob: Dict[str, Any] = {}

    def _scalar(k, cast=int, default=None):
        if k in keys:
            try:
                return cast(np.array(npz[k]).ravel()[0])
            except Exception:
                try:
                    return cast(npz[k].tolist())
                except Exception:
                    return cast(npz[k])
        return default

    blob["mode"] = (np.array(npz["mode"]).tolist() if "mode" in keys else "perlevel")
    blob["estimator"] = (np.array(npz["estimator"]).tolist() if "estimator" in keys else "full")
    blob["N"] = _scalar("N", int, None)
    blob["H"] = _scalar("H", int, None)
    blob["W"] = _scalar("W", int, None)
    # Ligne ajoutée pour supporter la 3D sans casser la rétrocompatibilité 2D
    blob["D"] = _scalar("D", int, None) 
    blob["L"] = _scalar("L", int, None)

    if "views" in keys:
        vv = np.array(npz["views"]).tolist()
        blob["views"] = [str(x) for x in (vv if isinstance(vv, list) else [vv])]

    # dims and stats may be JSON strings inside NPZ
    if "dims_json" in keys:
        blob["dims_per_level_per_view"] = json.loads(str(np.array(npz["dims_json"]).tolist()))
    if "stats_json" in keys:
        blob["stats"] = json.loads(str(np.array(npz["stats_json"]).tolist()))

    # shapes and slices (optional)
    if "shapes_json" in keys:
        blob["shapes_by_view"] = json.loads(str(np.array(npz["shapes_json"]).tolist()))
    if "slices_json" in keys:
        blob["level_view_slices"] = json.loads(str(np.array(npz["slices_json"]).tolist()))

    # per-level preferred path
    L = int(blob.get("L", 0) or 0)
    if any(f.startswith("mu_") for f in keys):
        mu_list, Sig_list = [], []
        for i in range(L):
            mu_list.append(np.array(npz[f"mu_{i}"]))
            if f"Sigma_{i}_type" in keys and str(np.array(npz[f"Sigma_{i}_type"]).tolist()) == "lowrank":
                Sig_list.append({
                    "type": "lowrank",
                    "U": np.array(npz[f"Sigma_{i}_U"]),
                    "eig": np.array(npz[f"Sigma_{i}_eig"]),
                    "sigma2": float(np.array(npz[f"Sigma_{i}_sigma2"]).ravel()[0]),
                })
            else:
                Sig_list.append(np.array(npz.get(f"Sigma_{i}")))
        blob["mu"] = mu_list
        blob["Sigma"] = Sig_list
        blob["mode"] = "perlevel"
        return blob

    # merged fallback
    if "mu" in keys:
        blob["mu"] = np.array(npz["mu"])
        if "Sigma_type" in keys and str(np.array(npz["Sigma_type"]).tolist()) == "lowrank":
            blob["Sigma"] = {
                "type": "lowrank",
                "U": np.array(npz["Sigma_U"]),
                "eig": np.array(npz["Sigma_eig"]),
                "sigma2": float(np.array(npz["Sigma_sigma2"]).ravel()[0]),
            }
        elif "Sigma" in keys:
            blob["Sigma"] = np.array(npz["Sigma"])
        return blob

    # legacy object arrays
    if "mu" in keys and np.array(npz["mu"]).dtype == object:
        blob["mu"] = np.array(npz["mu"]).tolist()
        if "Sigma" in keys:
            blob["Sigma"] = np.array(npz["Sigma"]).tolist()
        return blob

    raise RuntimeError(f"Unrecognized NPZ contents in {gauss_path}; keys={sorted(keys)}")

# ---------------------- Model Builders ----------------------

def _gather_val_paths(val_list: Optional[list[str]], limit: int) -> list[Path]:
    """
    Unified input method: accept one or more tokens that may be
      - a glob pattern (quote in shell to avoid pre-expansion OR pass many expanded files),
      - a text file listing one path per line,
      - a direct image path.
    Returns up to `limit` unique, existing Paths.
    """
    from glob import glob
    import os
    paths: list[Path] = []
    tokens = val_list or []
    for tok in tokens:
        tok = os.path.expandvars(os.path.expanduser(tok))
        p = Path(tok)
        if p.exists() and p.is_file():
            # If it's a text file, read lines; else treat as a direct image path
            if p.suffix.lower() in (".txt", ".lst", ".csv"):
                try:
                    with open(p, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                paths.append(Path(os.path.expandvars(os.path.expanduser(line))))
                except Exception:
                    pass
            else:
                paths.append(p)
        else:
            # Treat as glob pattern
            for g in sorted(glob(tok, recursive=True)):
                gp = Path(g)
                if gp.exists() and gp.is_file():
                    paths.append(gp)
    # unique / preserve order
    seen = set()
    uniq: list[Path] = []
    for p in paths:
        if p not in seen and p.exists() and p.is_file():
            uniq.append(p); seen.add(p)
        if len(uniq) >= int(limit):
            break
    return uniq


def build_model_from_config(cfg: dict, device: torch.device, target_hwd: Tuple[int, int, int] = None):
    if target_hwd is not None:
        H, W, D = int(target_hwd[0]), int(target_hwd[1]), int(target_hwd[2])
    else:
        H = int(cfg.get("H", 64))
        W = int(cfg.get("W", 64))
        D = int(cfg.get("D", 64))
        
    input_shape = (1, H, W, D)
    if create_glow_normalizing_flow_model_3d is None:
        raise ImportError("antstorch.create_glow_normalizing_flow_model_3d is required.")

    raw_k = cfg.get("K", 32)
    parsed_k = [int(x) for x in raw_k] if isinstance(raw_k, (list, tuple)) else int(raw_k)
    raw_hidden_channels = cfg.get("hidden", 128)
    parsed_hidden_channels = [int(x) for x in raw_hidden_channels] if isinstance(raw_hidden_channels, (list, tuple)) else int(raw_hidden_channels)
    
    m = create_glow_normalizing_flow_model_3d(
        input_shape=input_shape,
        L=int(cfg.get("L", 3)),
        K=parsed_k,
        hidden_channels=parsed_hidden_channels,
        base=str(cfg.get("base", "glow")),
        glowbase_logscale_factor=float(cfg.get("glowbase_logscale_factor", 1.0)),
        glowbase_min_log=float(cfg.get("glowbase_min_log", -5.0)),
        glowbase_max_log=float(cfg.get("glowbase_max_log", 5.0)),
        split_mode="channel",
        scale=True,
        scale_map=str(cfg.get("scale_map", "tanh")),
        leaky=0.0,
        net_actnorm=bool(cfg.get("net_actnorm", False)),
        scale_cap=float(cfg.get("scale_cap", 0.5)),
    ).to(device).float().eval()
    
    if not hasattr(m, "input_shape"):
        m.input_shape = input_shape
    return m

def resolve_ckpt_path(p: Path) -> Path:
    if p.is_dir():
        for name in ("training_state.pt", "checkpoint.pt", "ckpt.pt", "model.pt"):
            cand = p / name
            if cand.exists(): return cand
    if not p.exists(): raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p

def load_weights_into_model(model, blob, view_idx: int, prefer_ema: bool = True,
                            view_name: str | None = None, cfg_views: list[str] | None = None):
    """
    Robustly load weights for a given view into `model`.

    Selection order:
      (a) if prefer_ema: blob["ema"][slot]
      (b) blob["models"][slot]
      (c) blob["state_dict"]
      (d) raw dict of param tensors
    The `slot` comes from `cfg_views` name mapping if available, else `view_idx`.
    """
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

    # map manifest name -> checkpoint slot if we know the training order
    vidx_eff = int(view_idx)
    if cfg_views and view_name in cfg_views:
        vidx_eff = cfg_views.index(view_name)

    # (a) EMA list
    if prefer_ema and isinstance(blob.get("ema"), (list, tuple)) and len(blob["ema"]) > 0:
        k = max(0, min(vidx_eff, len(blob["ema"]) - 1))
        sd = extract_sd(blob["ema"][k])
        if sd is not None:
            ok, note = try_load(sd)
            if ok:
                return True, ("ema", f"slot={k}")

    # (b) models list
    if isinstance(blob.get("models"), (list, tuple)) and len(blob["models"]) > 0:
        k = max(0, min(vidx_eff, len(blob["models"]) - 1))
        sd = extract_sd(blob["models"][k])
        if sd is not None:
            ok, note = try_load(sd)
            if ok:
                return True, ("models", f"slot={k}")

    # (c) single state_dict
    if isinstance(blob.get("state_dict"), dict):
        ok, note = try_load(blob["state_dict"])
        if ok:
            return True, ("state_dict", None)

    # (d) raw dict with param keys
    if isinstance(blob, dict) and all(isinstance(k, str) for k in blob.keys()) and any("." in k for k in blob.keys()):
        ok, note = try_load(blob)
        if ok:
            return True, ("raw", None)

    return False, ("none", "no recognizable weights in blob")


def _prime_if_needed(model, H, W, D, device):
    dummy = torch.randn(1, 1, H, W, D, device=device)
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
    
    # 1. SVD Randomisée via PyTorch (Extrêmement rapide)
    # Convertir en tenseur PyTorch
    Xc_tensor = torch.tensor(Xc, dtype=torch.float32)
    
    # q=r force l'algorithme à ne chercher que les 'r' premières composantes (ici 256)
    _, S_tensor, V_tensor = torch.svd_lowrank(Xc_tensor, q=r)
    
    Svals = S_tensor.numpy()
    # torch.svd_lowrank retourne V avec la dimension (D, r)
    # Cela correspond directement à la forme transposée dont nous avons besoin pour U_cov
    U_cov = V_tensor.numpy().copy() 
    
    # 2. Calcul des valeurs propres pour les composantes principales
    eig_r = (Svals ** 2) / max(1, (N - 1))
    
    # 3. Estimation du bruit résiduel (sigma2)
    if isinstance(sigma2, str) and sigma2.lower() == "auto":
        # Variance totale = somme des carrés des éléments de Xc divisée par (N-1)
        total_variance = np.sum(Xc ** 2) / max(1, (N - 1))
        explained_variance = np.sum(eig_r)
        
        # Le reste de la variance est attribué au bruit
        residual_variance = max(0.0, total_variance - explained_variance)
        num_remaining_eigs = min(N, D) - r
        
        sigma2_val = float(residual_variance / num_remaining_eigs) if num_remaining_eigs > 0 else 0.0
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

def _flatten_latents_by_level(z_list) -> List:
    """
    Input: list of tensors or a single tensor; each tensor shape (B, C, H, W, D), (B, C, H, W) or (B, D).
    Output: list of (B, D_l) 2-D tensors per level.
    """
    if not isinstance(z_list, (list, tuple)):
        z_list = [z_list]
    outs = []
    for z in z_list:
        if z.ndim == 5:  # <-- Prise en charge de la 3D
            B, C, H, W, D = z.shape
            outs.append(z.reshape(B, C * H * W * D))
        elif z.ndim == 4: # Gardé par rétrocompatibilité ou architectures mixtes
            B, C, H, W = z.shape
            outs.append(z.reshape(B, C * H * W))
        elif z.ndim == 2:
            outs.append(z)
        else:
            raise RuntimeError(f"Unexpected latent shape: {tuple(z.shape)}")
    return outs

def _concat_views_per_level(z_per_view_per_level: List[List]) -> List:
    """
    z_per_view_per_level: list over views (V) of list over levels (L) of (N, D_lv)
    Returns: list over levels (L) of (N, sum_v D_lv)
    """
    V = len(z_per_view_per_level)
    L = len(z_per_view_per_level[0])
    outs = []
    for l in range(L):
        cols = [z_per_view_per_level[v][l] for v in range(V)]
        outs.append(torch.cat(cols, dim=1))
    return outs

def _np_stats(mat) -> Dict[str, float]:
    # mat is 2D or 1D vector of eigenvalues
    if isinstance(mat, dict) and mat.get('type') == 'lowrank':
        return _lowrank_stats(mat)
    if mat.ndim == 2:
        vals = np.linalg.eigvalsh(mat)
    else:
        vals = np.asarray(mat).ravel()
    vmin = float(vals.min(initial=np.inf))
    vmax = float(vals.max(initial=0.0))
    cond = float(vmax / (vmin + 1e-12)) if vmax > 0 else float("inf")
    return {"lambda_min": vmin, "lambda_max": vmax, "cond": cond}

def _cov_full(X: np.ndarray, ridge: float) -> np.ndarray:
    # X: (N, D), zero-mean assumed? We'll subtract mean outside.
    N = X.shape[0]
    S = (X.T @ X) / max(1, (N - 1))
    if ridge and ridge > 0.0:
        S = S + float(ridge) * np.eye(S.shape[0], dtype=S.dtype)
    return S

def _cov_diag(X: np.ndarray, ridge: float) -> np.ndarray:
    var = X.var(axis=0, ddof=1)
    if ridge and ridge > 0.0:
        var = var + float(ridge)
    return var  # 1D

def _cov_oas(X: np.ndarray, extra_ridge: float) -> np.ndarray:
    """
    Oracle Approximating Shrinkage toward scaled identity: (1-a)S + a*(tr(S)/p)I
    Uses Chen et al. 2010 closed form.
    """
    N, p = X.shape
    S = (X.T @ X) / max(1, (N - 1))
    mu = np.trace(S) / p
    # Frobenius norm of S
    trS2 = float(np.sum(S * S))
    trS = float(np.trace(S))
    # OAS shrinkage factor
    # guard for tiny denominators
    denom = (N + 1 - 2.0 / p) * (trS2 - (trS * trS) / p)
    if denom <= 0:
        a = 1.0
    else:
        a = ((1.0 - 2.0 / p) * trS2 + (trS * trS)) / denom
        a = max(0.0, min(1.0, a))
    S_shrunk = (1.0 - a) * S + a * mu * np.eye(p, dtype=S.dtype)
    if extra_ridge and extra_ridge > 0.0:
        S_shrunk = S_shrunk + float(extra_ridge) * np.eye(p, dtype=S.dtype)
    return S_shrunk

def _lowrank_stats(sig: dict) -> dict:
    eig = np.asarray(sig.get("eig", []), dtype=float)
    sigma2 = float(sig.get("sigma2", 0.0))
    lam_min = float(sigma2)
    lam_max = float((eig.max() if eig.size > 0 else 0.0) + sigma2)
    cond = float(lam_max / (lam_min + 1e-12)) if lam_max > 0 else float("inf")
    return {"lambda_min": lam_min, "lambda_max": lam_max, "cond": cond}

def _fit_gaussian_blocks(X_blocks: List[np.ndarray], estimator: str, shrinkage: float, cov_lam: float) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Fit Gaussian to concatenated blocks (per level). Returns (mu, Sigma, meta).
    For 'diag', Sigma is 1D vector of variances.
    """
    X = np.concatenate(X_blocks, axis=1) if len(X_blocks) > 1 else X_blocks[0]
    mu = X.mean(axis=0)
    Xc = X - mu
    est = estimator.lower()
    if est == "full":
        Sigma = _cov_full(Xc, ridge=float(shrinkage) + float(cov_lam))
    elif est == "diag":
        Sigma = _cov_diag(Xc, ridge=float(shrinkage) + float(cov_lam))
    elif est in ("oas", "lw", "ledoitwolf"):
        # Treat lw as oas for now; both shrink toward scaled identity; OAS has closed form
        Sigma = _cov_oas(Xc, extra_ridge=float(cov_lam))
    else:
        raise RuntimeError(f"Unknown --cov-estimator: {estimator}")
    stats = _np_stats(Sigma if Sigma.ndim == 2 else Sigma)
    return mu, Sigma, stats

def _ckpt_fingerprint(ckpt_path: Path) -> str:
    try:
        h = hashlib.sha1()
        h.update(ckpt_path.read_bytes()[:1024*1024])  # first 1MB
        h.update(str(ckpt_path.stat().st_size).encode())
        h.update(str(int(ckpt_path.stat().st_mtime)).encode())
        return h.hexdigest()
    except Exception:
        return "unknown"


def _save_gauss_npz(blob: Dict[str, Any], out_path: Path):
    import json
    pack = {
        "mode": blob["mode"],
        "estimator": blob["estimator"],
        "N": np.int64(blob["N"]),
        "H": np.int64(blob["H"]),
        "W": np.int64(blob["W"]),
        "L": np.int64(blob["L"]),
        "views": np.array(blob["views"], dtype=object),
        "dims_json": json.dumps(blob["dims_per_level_per_view"]),
        "shapes_json": json.dumps(blob["shapes_by_view"]),
        "slices_json": json.dumps(blob["level_view_slices"]),
        "stats_json": json.dumps(blob["stats"]),
    }
    if blob["mode"] == "perlevel":
        for i, mu in enumerate(blob["mu"]):
            pack[f"mu_{i}"] = np.asarray(mu)
            S = blob["Sigma"][i]
            if isinstance(S, dict) and S.get("type") == "lowrank":
                pack[f"Sigma_{i}_type"] = "lowrank"
                pack[f"Sigma_{i}_U"] = np.asarray(S["U"])
                pack[f"Sigma_{i}_eig"] = np.asarray(S["eig"])
                pack[f"Sigma_{i}_sigma2"] = np.asarray([S["sigma2"]], dtype=np.float64)
            else:
                pack[f"Sigma_{i}"] = np.asarray(S)
    else:
        pack["mu"] = np.asarray(blob["mu"])
        S = blob["Sigma"]
        if isinstance(S, dict) and S.get("type") == "lowrank":
            pack["Sigma_type"] = "lowrank"
            pack["Sigma_U"] = np.asarray(S["U"])
            pack["Sigma_eig"] = np.asarray(S["eig"])
            pack["Sigma_sigma2"] = np.asarray([S["sigma2"]], dtype=np.float64)
        else:
            pack["Sigma"] = np.asarray(S)

    np.savez_compressed(out_path, **pack)


# ---------------------- Main Commands ----------------------

def main_recon(argv=None):
    ap = argparse.ArgumentParser("LAM‑Flow 3D reconstruction tool (recon)")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    ap.add_argument("--manifest", type=str, required=True, help="CSV with per-view file paths")
    ap.add_argument("--views", type=str, required=True, help="Comma list of views (e.g., T1,FA)")
    ap.add_argument("--view-index", type=int, default=0, help="Which view to load (0-based)")
    ap.add_argument("--volume-size", type=str, required=True, help="Size format HxWxD (e.g., 64x80x64)")
    ap.add_argument("--reference-image", type=str, default=None, help="Optional 3D image to define the header.")
    ap.add_argument("--batch", type=int, default=1, help="Batch size (keep low for 3D)")
    ap.add_argument("--devices", type=str, default="cuda:0")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for NIfTI volumes")
    
    # Options d'Édition Gaussienne
    ap.add_argument("--gauss", type=str, default=None, help="Gaussian model for latent editing.")
    ap.add_argument("--edit-levels", type=str, default="none", help="Levels to project (e.g. '0,1,2', 'all')")
    ap.add_argument("--edit-what", type=str, choices=["mean", "zero", "pc", "pc_denoise"], default="mean")
    ap.add_argument("--edit-pc-index", type=int, default=0)
    ap.add_argument("--edit-pc-scale", type=float, default=2.0)
    ap.add_argument("--edit-pc-center", type=str, choices=["sample", "mean"], default="sample")
    ap.add_argument("--edit-pc-k", type=int, default=64)
    ap.add_argument("--edit-pc-beta", type=float, default=0.0)
    args = ap.parse_args(argv)

    device = torch.device(args.devices)
    ckpt_path = resolve_ckpt_path(Path(args.ckpt))
    
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)

    cfg = blob.get("config", {})
    try:
        Hc, Wc, Dc = [int(x) for x in args.volume_size.split("x")]
    except:
        raise ValueError(f"Invalid --volume-size {args.volume_size}. Expected HxWxD.")

    model = build_model_from_config(cfg if cfg else {"H": Hc, "W": Wc, "D": Dc}, device=device, target_hwd=(Hc, Wc, Dc))
    model.eval()

    manifest_path = Path(args.manifest)
    with open(manifest_path, "r") as f:
        header = [h.strip() for h in f.readline().strip().split(",")]
        all_views = [v.strip() for v in args.views.split(",") if v.strip()]
        v_idx_map = {v: header.index(v) for v in all_views}
        rows = []
        for line in f:
            parts = [s.strip() for s in line.strip().split(",")]
            if not parts or all(p == "" for p in parts): continue
            rows.append([Path(parts[v_idx_map[v]]) for v in all_views])

    vname = all_views[args.view_index]
    vcol = [r[all_views.index(vname)] for r in rows]

    ok, note = load_weights_into_model(model, blob, view_idx=all_views.index(vname), prefer_ema=True, view_name=vname, cfg_views=all_views)
    if not ok: raise RuntimeError(f"Failed to load weights: {note}")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bs = max(1, int(args.batch))
    
    # Traitement par lots pour la 3D
    for batch_idx in range(0, min(bs, len(vcol))):
        pth = vcol[batch_idx]
        xi = _read_image_3d(pth, target_hwd=(Hc, Wc, Dc)) # Doit retourner (1, H, W)
        xb = xi.unsqueeze(0).to(device=device, dtype=torch.float32) # (1, 1, H, W, D)

        # Base reconstruction
        z_list = _encode_latents(model, xb)
        xh = _decode_latents(model, z_list, target_hwd=(Hc, Wc, Dc))

        xh_edit = None
        if args.gauss and args.edit_levels.lower() != "none":
            gauss_blob = _load_gaussian_model(Path(args.gauss))
            _, _, _, L = _validate_gauss_blob(gauss_blob)

            spec = args.edit_levels.strip().lower()
            levels = list(range(L)) if spec == "all" else [int(p) for p in spec.split(",") if p.strip()]
            levels = sorted({l for l in levels if 0 <= l < L})

            if levels:
                z_list_edit = _edit_latents_to_mean_for_view_3d(
                    z_list, gauss_blob, vname, levels_to_edit=levels,
                    mode=args.edit_what, pc_index=args.edit_pc_index,
                    pc_scale=args.edit_pc_scale, pc_center=args.edit_pc_center,
                    pc_k=args.edit_pc_k, pc_beta=args.edit_pc_beta
                )
                xh_edit = _decode_latents(model, z_list_edit, target_hwd=(Hc, Wc, Dc))

        # Sauvegarde 3D
        diff = torch.abs(xb - xh)
        base_name = f"subj_{batch_idx:04d}_{vname}"
        
        if args.reference_image is not None:
            reference_image = ants.image_read(args.reference_image)
            save_nifti(xb, out_dir / f"{base_name}_orig.nii.gz", reference_image)
            save_nifti(xh, out_dir / f"{base_name}_recon.nii.gz", reference_image)
            save_nifti(diff, out_dir / f"{base_name}_diff.nii.gz", reference_image)
        else:
            save_nifti(xb, out_dir / f"{base_name}_orig.nii.gz")
            save_nifti(xh, out_dir / f"{base_name}_recon.nii.gz")
            save_nifti(diff, out_dir / f"{base_name}_diff.nii.gz")
        
        print(f"[{batch_idx}] max |x - x_hat| = {float(diff.max().item()):.6f}")

        if xh_edit is not None:
            diff_edit = torch.abs(xb - xh_edit)
            if args.reference_image is not None:
                save_nifti(xh_edit, out_dir / f"{base_name}_edit.nii.gz", reference_image)
                save_nifti(diff_edit, out_dir / f"{base_name}_diff_edit.nii.gz", reference_image)
            else:
                save_nifti(xh_edit, out_dir / f"{base_name}_edit.nii.gz")
                save_nifti(diff_edit, out_dir / f"{base_name}_diff_edit.nii.gz")
            print(f"[{batch_idx}] max |x - x_hat_edit| = {float(diff_edit.max().item()):.6f}")

    print(f"[recon] Volumes written to {out_dir}")
    return 0

def main_gauss_fit(argv: List[str] | None = None):

    try:
        from tqdm import tqdm
    except ImportError:
        print("[info] tqdm not found. Install with `pip install tqdm` for progress bars.")
        tqdm = lambda x, **kwargs: x

    def _sanitize_latents_array(X, cap_quantile=99.9, hard_cap=None):
        X = np.asarray(X, dtype=np.float64)
        stats = {}
        nf = ~np.isfinite(X)
        nf_count = int(nf.sum())
        if nf_count:
            X[nf] = 0.0
        stats["nonfinite"] = nf_count

        if hard_cap is None:
            q = np.percentile(np.abs(X), [50, 90, 99, cap_quantile])
            cap = float(q[-1] + 1e-12)
            stats["abs_quantiles"] = {"p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]), f"p{cap_quantile}": float(q[3])}
        else:
            cap = float(hard_cap)
            stats["abs_quantiles"] = None

        pre = X.copy()
        np.clip(X, -cap, cap, out=X)
        clipped = int(np.sum(pre != X))
        stats["cap"] = cap
        stats["clipped"] = clipped
        return X, stats

    def _cov_stats(Sd):
        Sd = 0.5 * (Sd + Sd.T)
        w = np.linalg.eigvalsh(Sd)
        lam_min = float(np.min(w))
        lam_max = float(np.max(w))
        cond = float(lam_max / max(lam_min, 1e-300))
        tr = float(np.trace(Sd))
        diag_mean = float(np.mean(np.diag(Sd)))
        return lam_min, lam_max, cond, tr, diag_mean

    def _dense_from_cov(Sigma, D):
        if isinstance(Sigma, dict) and Sigma.get("type") == "lowrank":
            U = np.asarray(Sigma["U"], dtype=np.float64)
            eig = np.asarray(Sigma["eig"], dtype=np.float64)
            sigma2 = float(Sigma.get("sigma2", 0.0))
            return U @ (np.diag(eig) @ U.T) + sigma2 * np.eye(U.shape[0])
        S = np.asarray(Sigma, dtype=np.float64)
        if S.ndim == 1:
            return np.diag(S)
        return S

    def _scrub_row_outliers(z_per_view_per_level, per_view_paths, view_names, thresh=1e6):
        N = z_per_view_per_level[0][0].shape[0]
        bad = set()
        bad_paths = []
        for v_idx, vname in enumerate(view_names):
            for li, Z in enumerate(z_per_view_per_level[v_idx]):
                row_max = Z.detach().abs().amax(dim=1)
                bad_idx = torch.nonzero(row_max > thresh, as_tuple=False).view(-1).cpu().tolist()
                if bad_idx:
                    print(f"[scrub] {vname} L{li}: {len(bad_idx)} outliers > {thresh:g}: {bad_idx[:8]}")
                    bad.update(bad_idx)
            for bi in sorted(set(bad)):
                if bi < len(per_view_paths[v_idx]):
                    bad_paths.append({"view": vname, "idx": int(bi), "path": str(per_view_paths[v_idx][bi])})
        if not bad:
            return z_per_view_per_level, per_view_paths, list(range(N)), []
        keep = sorted(set(range(N)) - bad)
        kt = torch.tensor(keep, dtype=torch.long)
        z_clean = [[Z.index_select(0, kt) for Z in z_per_view_per_level[v]] for v in range(len(view_names))]
        per_paths_clean = [[plist[i] for i in keep] for plist in per_view_paths]
        print(f"[scrub] dropped {len(bad)} subjects; new N={len(keep)}")
        return z_clean, per_paths_clean, keep, bad_paths

    ap = argparse.ArgumentParser("LAM-Flow conditional Gaussian fitter (gauss-fit) 3D")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint.")
    ap.add_argument("--manifest", type=str, required=True, help="CSV manifest.")
    ap.add_argument("--views", type=str, default=None, help="Views to include (e.g., T1,FA).")
    ap.add_argument("--volume-size", type=str, required=True, help="Size format HxWxD (e.g., 64x80x64)")
    ap.add_argument("--batch", type=int, default=1, help="Batch size (keep low for 3D).")
    ap.add_argument("--devices", type=str, default="cuda:0")

    # Gaussian options
    ap.add_argument("--cov-mode", type=str, choices=["perlevel","merged"], default="perlevel")
    ap.add_argument("--cov-estimator", type=str, choices=["full","diag","oas","lw","lowrank"], default="lowrank")
    ap.add_argument("--rank", type=int, default=128, help="Matrix rank for 'lowrank'.")
    ap.add_argument("--sigma2", type=str, default="auto")
    ap.add_argument("--shrinkage", type=str, default="1e-6")
    ap.add_argument("--cov-lam", type=float, default=1e-6)
    ap.add_argument("--jitter", type=float, default=1e-4)

    # Outputs
    ap.add_argument("--gauss-out", type=str, required=True, help="Output path (.npz).")
    ap.add_argument("--gauss-summary", type=str, default="", help="JSON summary path.")
    ap.add_argument("--save-fp", type=int, default=64)
    args = ap.parse_args(argv)

    @torch.no_grad()
    def _probe_latent_shapes_for_view(model, state_blob, view_idx, Hc, Wc, Dc, device):
        ok, note = load_weights_into_model(model, state_blob, view_idx=view_idx, prefer_ema=True)
        if not ok:
            raise RuntimeError(f"load_weights_into_model failed for view {view_idx}: {note}")
        # Adaptation 3D
        x0 = torch.zeros(1, 1, int(Hc), int(Wc), int(Dc), device=device, dtype=torch.float32)
        if hasattr(model, "inverse_and_log_det"):
            z, _ = model.inverse_and_log_det(x0)
        elif hasattr(model, "inverse"):
            z, _ = model.inverse(x0)
        else:
            raise RuntimeError("Model lacks inverse mapping")
        z_list = z if isinstance(z, (list, tuple)) else [z]
        # Adaptation 3D : retourne (C, H, W, D)
        return [(int(t.shape[1]), int(t.shape[2]), int(t.shape[3]), int(t.shape[4])) for t in z_list]

    device = torch.device("cpu") if args.devices.lower() == "cpu" else torch.device(args.devices.split(",")[0])
    ckpt_path = resolve_ckpt_path(Path(args.ckpt))
    manifest_path = Path(args.manifest).resolve()
    manifest_dir = manifest_path.parent
    
    try:
        Hc, Wc, Dc = [int(x) for x in args.volume_size.split("x")]
    except:
        raise ValueError(f"Invalid --volume-size {args.volume_size}. Expected HxWxD.")

    try:
        state_blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state_blob = torch.load(ckpt_path, map_location=device)
        
    cfg = state_blob.get("config", {})
    cfg_views = list(cfg.get("views", [])) if isinstance(cfg.get("views"), (list, tuple)) else None

    # model instantiation needs H, W, D
    model = build_model_from_config(cfg if cfg else {"H": Hc, "W": Wc, "D": Dc}, device=device)
    model.eval()

    cols = _read_manifest_csv(manifest_path)
    view_names, per_view_paths = _resolve_views(cols, manifest_dir, args.views)
    V = len(view_names)
    N = len(per_view_paths[0])
    N_original = int(N)
    print(f"[info] views: {view_names} (V={V}); subjects: N={N}")

    z_per_view_per_level: List[List[torch.Tensor]] = [None] * V

    for v_idx, vname in enumerate(view_names):
        model = build_model_from_config(cfg if cfg else {"H": Hc, "W": Wc, "D": Dc}, device=device)
        model.eval()

        ok, note = load_weights_into_model(
            model, state_blob, view_idx=v_idx, prefer_ema=True,
            view_name=vname, cfg_views=cfg_views
        )
        if not ok:
            raise RuntimeError(f"Failed to load weights for view {v_idx} ({vname}): {note}")

        paths = per_view_paths[v_idx]
        bs = max(1, int(args.batch))

        latents_per_level_list: List[List[torch.Tensor]] | None = None

        def _flush_batch(xlist: List[torch.Tensor]):
            nonlocal latents_per_level_list
            xb = torch.stack(xlist, dim=0).to(device=device, dtype=torch.float32)
            with torch.no_grad(), torch.amp.autocast(device.type, enabled=False):
                if hasattr(model, "inverse_and_log_det"):
                    z, _ = model.inverse_and_log_det(xb) 
                elif hasattr(model, "inverse"):
                    z, _ = model.inverse(xb)
                else:
                    raise RuntimeError("Model lacks inverse mapping")
            zl = _flatten_latents_by_level(z) # Fonction inchangée (aplatit le (B,C,H,W,D) en (B,D_l))
            if latents_per_level_list is None:
                latents_per_level_list = [[] for _ in range(len(zl))]
            for li, arr in enumerate(zl):
                latents_per_level_list[li].append(arr.detach().cpu())

        batch = []
        for pth in tqdm(paths, desc=f"Encoding {vname}", unit="vol"):
            # Adaptation 3D : Utilisation de _read_image_3d
            xi = _read_image_3d(pth, target_hwd=(Hc, Wc, Dc)) 
            batch.append(xi)
            if len(batch) >= bs:
                _flush_batch(batch); batch = []
        if batch:
            _flush_batch(batch)

        if latents_per_level_list is None:
            raise RuntimeError(f"No latents collected for view {v_idx} ({vname}).")
        z_per_view_per_level[v_idx] = [torch.cat(chunks, dim=0) for chunks in latents_per_level_list]

    z_per_view_per_level, per_view_paths, keep_idx, bad_paths = _scrub_row_outliers(
        z_per_view_per_level, per_view_paths, view_names, thresh=1e6
    )
    N = len(keep_idx)

    Z_levels = _concat_views_per_level(z_per_view_per_level)
    L = len(Z_levels)
    dims_per_level_per_view = [[int(t.shape[1]) for t in vlist] for vlist in z_per_view_per_level]

    # Adaptation 3D pour la forme (C, H, W, D)
    shapes_by_view: List[List[Tuple[int,int,int,int]]] = []
    for v_idx in range(V):
        shp = _probe_latent_shapes_for_view(model, state_blob, v_idx, Hc, Wc, Dc, device)
        shapes_by_view.append(shp)

    level_view_slices: List[Dict[int, Tuple[int,int]]] = []
    for l in range(L):
        off = 0
        row_dict = {}
        for v_idx in range(V):
            d = int(dims_per_level_per_view[v_idx][l])
            row_dict[v_idx] = (off, off + d)
            off += d
        level_view_slices.append(row_dict)

    estimator = args.cov_estimator.lower()
    shrink = args.shrinkage
    try:
        shrink_val = float(shrink)
    except Exception:
        shrink_val = 0.0

    out_blob: Dict[str, Any] = {
        "mode": args.cov_mode,
        "estimator": estimator,
        "shrinkage": shrink,
        "cov_lam": float(args.cov_lam),
        "jitter": float(args.jitter),
        "views": view_names,
        "N": int(N),             
        "H": int(Hc),
        "W": int(Wc),
        "D": int(Dc), # Ajout 3D
        "L": int(L),
        "dims_per_level_per_view": dims_per_level_per_view,
        "shapes_by_view": shapes_by_view,
        "level_view_slices": level_view_slices,
        "config": cfg,
        "ckpt_path": str(ckpt_path),
        "ckpt_fingerprint": _ckpt_fingerprint(ckpt_path),
        "created_utc": int(time.time()),
    }

    stats_list = []
    if args.cov_mode == "perlevel":
        mu_list, Sigma_list = [], []
        cap_quant = 99.9         
        hard_cap = None          

        for l, Zl in enumerate(Z_levels):
            X = Zl.detach().cpu().numpy().astype("float64")
            mu = X.mean(axis=0)
            Xc = X - mu

            Xc_clean, sstats = _sanitize_latents_array(Xc, cap_quantile=cap_quant, hard_cap=hard_cap)
            D_l = Xc_clean.shape[1]
            
            if estimator == "lowrank":
                Sigma = _lowrank_from_Xc(
                    Xc_clean, rank=int(args.rank),
                    sigma2=args.sigma2, extra_ridge=float(args.cov_lam)
                )
                if isinstance(Sigma, dict) and Sigma.get("type") == "lowrank":
                    print(f"[lowrank L{l}] eff_rank={Sigma['U'].shape[1]}  sigma2={Sigma['sigma2']:.3g}")
            else:
                _mu_unused, Sigma, _stats_unused = _fit_gaussian_blocks(
                    [Xc_clean], estimator=estimator, shrinkage=shrink_val, cov_lam=float(args.cov_lam)
                )

            row_dims = [int(dims_per_level_per_view[v][l]) for v in range(V)]
            offs = [0]
            for d in row_dims: offs.append(offs[-1] + d)

            if estimator == "lowrank":
                U = Sigma["U"]
                eig = Sigma["eig"]
                sigma2 = float(Sigma["sigma2"])
                
                lam_min = sigma2
                lam_max = float((np.max(eig) if eig.size > 0 else 0.0) + sigma2)
                cond = lam_max / max(lam_min, 1e-300)
                tr = float(np.sum(eig) + D_l * sigma2)
                diag_mean = tr / D_l
                
                # Calcul de la trace par bloc sans instancier la matrice dense
                blk_tr = []
                for a, b in zip(offs[:-1], offs[1:]):
                    U_v = U[a:b, :]
                    blk_var = np.sum((U_v ** 2) * eig, axis=1) + sigma2
                    blk_tr.append(float(np.sum(blk_var)))
            else:
                Sd = _dense_from_cov(Sigma, D_l)
                lam_min, lam_max, cond, tr, diag_mean = _cov_stats(Sd)
                blk_tr = [float(np.trace(Sd[a:b, a:b])) for a, b in zip(offs[:-1], offs[1:])]
                del Sd # Libération explicite de la mémoire

            stats = {
                "lambda_min": lam_min, "lambda_max": lam_max, "cond": cond, 
                "trace": tr, "diag_mean": diag_mean,
                "winsor_cap": float(sstats.get("cap", 0.0)),
                "winsor_clipped": int(sstats.get("clipped", 0)),
                "winsor_nonfinite": int(sstats.get("nonfinite", 0)),
            }

            print("[fit Σ L{} by view] ".format(l) + " ".join("v{}:{:.3e}".format(vi, t) for vi, t in enumerate(blk_tr)))

            mu_list.append(mu); 
            Sigma_list.append(Sigma); 
            stats_list.append(stats)
            import gc
            del X, Xc, Xc_clean
            gc.collect()

        out_blob["mu"] = mu_list
        out_blob["Sigma"] = Sigma_list
        out_blob["stats"] = stats_list

    if args.save_fp == 32:
        def _cast_fp32_inplace(ob):
            def cast(x): return x.astype(np.float32) if isinstance(x, np.ndarray) and x.dtype == np.float64 else x
            if isinstance(ob.get("mu"), list):
                ob["mu"] = [cast(m) for m in ob["mu"]]
                newS = []
                for S in ob["Sigma"]:
                    if isinstance(S, dict) and S.get("type") == "lowrank":
                        U = S.get("U"); eig = S.get("eig"); sigma2 = float(S.get("sigma2", 0.0))
                        S = {"type":"lowrank", "U": (cast(U)), "eig": (cast(eig)), "sigma2": float(sigma2)}
                    else:
                        S = cast(S)
                    newS.append(S)
                ob["Sigma"] = newS
        _cast_fp32_inplace(out_blob)

    out_path = Path(args.gauss_out); out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if str(out_path).endswith(".pt"):
            torch.save(out_blob, out_path, pickle_protocol=5)
        elif str(out_path).endswith(".npz"):
            _save_gauss_npz(out_blob, out_path) 
        print(f"[ok] wrote Gaussian model: {out_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save --gauss-out: {e}")

    if args.gauss_summary:
        js = {
            "mode": out_blob["mode"], "estimator": out_blob["estimator"],
            "views": out_blob["views"],
            "N": out_blob["N"], "H": out_blob["H"], "W": out_blob["W"], "D": out_blob["D"], "L": out_blob["L"],
            "dims_per_level_per_view": out_blob["dims_per_level_per_view"],
            "stats": out_blob["stats"],
        }
        js["dropped_subjects"] = {
           "count": int(N_original - N), "original_N": int(N_original), "kept_N": int(N), "details": bad_paths
        }
        js_path = Path(args.gauss_summary); js_path.parent.mkdir(parents=True, exist_ok=True)
        with open(js_path, "w") as f:
            json.dump(js, f, indent=2)
        print(f"[ok] wrote summary JSON: {js_path}")

def main_gauss_impute(argv=None):
    ap = argparse.ArgumentParser("gauss-impute")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gauss", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--views", required=True) 
    ap.add_argument("--observed", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--tau", type=float, default=1.0,
                    help="Facteur de température pour projeter l'imputation sur l'ensemble typique (défaut: 1.0).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--volume-size", type=parse_hwd, default="64x64x64")
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
            mdl_obs = build_model_from_config(cfg, device, target_hwd=args.volume_size)
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
        mdl_tgt = build_model_from_config(cfg, device, target_hwd=args.volume_size)
        
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
    ap.add_argument("--volume-size", type=parse_hwd, default="64x64x64", help="DxHxW")
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
    model = build_model_from_config(cfg, device, target_hwd=args.volume_size)
    _prime_if_needed(model, *args.volume_size, device)
    
    ok, note = load_weights_into_model(model, blob, int(args.view_index))
    if not ok: raise RuntimeError(f"Weights failed: {note}")

    model = model.to(device)

    views_list = [v.strip() for v in args.views.split(",") if v.strip()]
    vname = views_list[int(args.view_index)]

    # 2. Chargement Gaussien
    npz = np.load(args.gauss, allow_pickle=True)
    L = int(npz["L"])
    views_g = list(npz["views"])
    raw_dims = npz["dims_json"]
    dims_str = raw_dims.item() if hasattr(raw_dims, "item") and raw_dims.ndim == 0 else raw_dims[0]
    dims_list = json.loads(dims_str) if isinstance(dims_str, str) else dims_str

    # Récupère [43008, 43008, 86016]
    level_sizes = dims_list[args.view_index]

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

def main_recon_cohort_template(argv=None):
    """
    Génère un template 2D spécifique à une cohorte (Cohort Average).
    Lit un manifest, encode toutes les images dans l'espace latent, 
    calcule le barycentre (moyenne Frechèt) et le décode.
    """

    def frechet_mean_spherical(z_points, max_iter=50, tol=1e-5, verbose=True):
        norms = torch.norm(z_points, p=2, dim=1, keepdim=True)
        z_spheres = z_points / norms.clamp(min=1e-8)
        
        mu = torch.mean(z_spheres, dim=0)
        mu = mu / torch.norm(mu).clamp(min=1e-8)
        
        for i in range(max_iter):
            dot_prods = torch.matmul(z_spheres, mu).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            thetas = torch.acos(dot_prods).unsqueeze(1)
            
            diff = z_spheres - (dot_prods.unsqueeze(1) * mu.unsqueeze(0))
            diff_norms = torch.norm(diff, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            
            tangent_vectors = (diff / diff_norms) * thetas
            tangent_mean = torch.mean(tangent_vectors, dim=0)
            tangent_mean_norm = torch.norm(tangent_mean)
            
            if verbose:
                print(f"    Iter {i+1}/{max_iter} | Erreur (Norme Tangente) : {tangent_mean_norm.item():.6f}")
        
            if tangent_mean_norm < tol:
                if verbose:
                    print(f"    -> Convergence atteinte à l'itération {i+1} !")
                break
                
            mu = mu * torch.cos(tangent_mean_norm) + (tangent_mean / tangent_mean_norm) * torch.sin(tangent_mean_norm)
            mu = mu / torch.norm(mu).clamp(min=1e-8)
            
        avg_norm = torch.mean(norms)
        return mu * avg_norm

    ap = argparse.ArgumentParser("LAM-Flow 3D Cohort Template (recon-cohort-template)")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    ap.add_argument("--manifest", type=str, required=True, help="Manifest CSV with cohort images")
    ap.add_argument("--views", type=str, required=True, help="Comma list of views (e.g. T1,FA)")
    ap.add_argument("--view-index", type=int, default=0, help="Which view to use")
    ap.add_argument("--volume-size", type=parse_hwd, default="64x64x64", help="Target spatial dims")
    ap.add_argument("--devices", type=str, default="cuda:0")
    ap.add_argument("--out", type=str, required=True, help="Output NIfTI filename")
    ap.add_argument("--sharpen-image", action="store_true", help="Apply Laplacian sharpening")
    args = ap.parse_args(argv)

    device = torch.device(args.devices)

    # 1. Chargement Modèle
    ckpt_path = resolve_ckpt_path(Path(args.ckpt))
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    model = build_model_from_config(cfg, device, target_hwd=args.volume_size)
    model.eval()

    # 2. Chargement Manifeste
    cols = _read_manifest_csv(Path(args.manifest))
    views_list = [v.strip() for v in args.views.split(",")]
    vname = views_list[args.view_index]
    _, per_view_paths = _resolve_views(cols, Path(args.manifest).parent, vname)

    global_idx = views_list.index(vname) if vname in views_list else args.view_index
    ok, note = load_weights_into_model(model, blob, global_idx)
    if not ok: raise RuntimeError(f"Weights failed for {vname}: {note}")
    _prime_if_needed(model, *args.volume_size, device)

    paths = per_view_paths[0]
    N = len(paths)
    if N == 0: raise RuntimeError("Aucune image trouvée dans le manifest.")
    print(f"[info] Calcul de la moyenne latente 3D pour {N} sujets (Vue: {vname})...")

    # 3. Encodage et Stockage Latent (sur CPU pour économiser la VRAM)
    z_all_subjects = None
    
    for p in tqdm(paths, desc="Encoding Cohort 3D", unit="vol"):
        x = _read_image_3d(p, args.volume_size).to(device)

        with torch.no_grad():
            if x.dim() == 4:
                x = x.unsqueeze(1)            
            z_list, _ = model.inverse_and_log_det(x)
            if isinstance(z_list, tuple): z_list = list(z_list)
            if not isinstance(z_list, list): z_list = [z_list]

        if z_all_subjects is None:
            z_all_subjects = [ [] for _ in range(len(z_list)) ]

        for l in range(len(z_list)):
            # CRITIQUE 3D : .cpu() libère la VRAM à chaque itération
            z_all_subjects[l].append(z_list[l].cpu()) 

    # 4. Calcul de la Moyenne Tangentielle et Décodage
    z_mean = []
    print(f"[info] Calcul de la moyenne de Fréchet sur la variété sphérique (3D)...")
    
    for l in range(len(z_all_subjects)):
        # Empilement pour le niveau l (toujours sur CPU): forme (N, C, D, H, W)
        z_stack = torch.cat(z_all_subjects[l], dim=0)
        N_subj, C, D_z, H_z, W_z = z_stack.shape
        
        # Aplatissement en 2D (N, D) et transfert sur GPU pour le calcul rapide
        z_flat = z_stack.view(N_subj, -1).to(device)
        
        # Calcul de la moyenne riemannienne
        mu_flat = frechet_mean_spherical(z_flat)
        
        # Remodelage vers les dimensions spatiales 3D (1, C, D, H, W)
        z_mean.append(mu_flat.view(1, C, D_z, H_z, W_z))

    with torch.no_grad():
        # z_mean est déjà sur le GPU grâce à l'étape précédente
        x_recon, _ = model.forward_and_log_det(z_mean)
        x_recon = to01(_coerce_5d(x_recon, args.volume_size))

    # 5. Sauvegarde
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(x_recon, outp)
    print(f"[ok] Template de cohorte 3D sauvegardé : {outp}")

def main_recon_temperature(argv=None):
    """
    Encode un volume 3D, met à l'échelle les latents par un facteur de température (tau), 
    et reconstruit le volume. Supporte les températures globales et par niveau.
    """
    ap = argparse.ArgumentParser("LAM-Flow 3D Recon Temperature Scaling")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    ap.add_argument("--manifest", type=str, required=True, help="Manifest CSV")
    ap.add_argument("--views", type=str, required=True, help="Views list (e.g. T1,FA)")
    ap.add_argument("--view-index", type=int, default=0, help="View to process")
    ap.add_argument("--volume-size", type=parse_hwd, default="64x64x64", help="Target spatial dimensions")
    ap.add_argument("--devices", type=str, default="cuda:0")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for scaled NIfTI volumes")
    
    # Option de Température Globale
    ap.add_argument("--tau", type=float, default=0.99, 
                    help="Global temperature scaling factor (default: 0.99). Values < 1.0 pull towards the mean.")
    
    # Option par niveau (Granularité)
    ap.add_argument("--tau-level", action="append", type=str,
                    help="Override temperature for a specific level. Format 'level,value'.")

    args = ap.parse_args(argv)
    device = torch.device(args.devices)

    # --- Parsing des overrides par niveau ---
    level_overrides = {}
    if args.tau_level:
        for item in args.tau_level:
            try:
                parts = item.split(',')
                if len(parts) != 2: raise ValueError
                lvl, val = int(parts[0]), float(parts[1])
                level_overrides[lvl] = val
            except ValueError:
                raise RuntimeError(f"Invalid format for --tau-level: '{item}'. Expected 'level,value'.")
        print(f"[info] Per-level overrides: {level_overrides}")

    # 1. Chargement Modèle
    ckpt_path = resolve_ckpt_path(Path(args.ckpt))
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    
    model = build_model_from_config(cfg, device, target_hwd=args.volume_size)
    model.eval()

    # 2. Chargement Manifeste & Poids
    cols = _read_manifest_csv(Path(args.manifest))
    views_list = [v.strip() for v in args.views.split(",")]
    vname = views_list[args.view_index]
    _, per_view_paths = _resolve_views(cols, Path(args.manifest).parent, vname)
    
    # Trouver l'index global de la vue si le checkpoint contient plusieurs vues
    global_idx = views_list.index(vname) if vname in views_list else args.view_index
    
    ok, note = load_weights_into_model(model, blob, global_idx)
    if not ok: raise RuntimeError(f"Weights failed for {vname}: {note}")
    _prime_if_needed(model, *args.volume_size, device)

    paths = per_view_paths[0]
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[info] Processing {len(paths)} volumes with base tau={args.tau}...")

    # 3. Traitement Itératif (Volume par Volume pour économiser la VRAM)
    for i, p in enumerate(tqdm(paths, desc="Scaling Volumes", unit="vol")):
        # Lecture 3D
        x = _read_image_3d(p, args.volume_size).to(device)
        
        # Encodage
        with torch.no_grad():
            z_list, _ = model.inverse_and_log_det(x)
            if isinstance(z_list, tuple): z_list = list(z_list)
            if not isinstance(z_list, list): z_list = [z_list]

        # 4. Temperature Scaling Granulaire
        z_scaled_list = []
        for l, z in enumerate(z_list):
            tau = float(level_overrides.get(l, args.tau))
            # La multiplication scalaire s'applique à toutes les dimensions (B, C, H, W, D)
            z_scaled_list.append(z * tau)

        # 5. Décodage
        with torch.no_grad():
            x_rec, _ = model.forward_and_log_det(z_scaled_list)
            x_rec = to01(_coerce_5d(x_rec, args.volume_size), winsorize=True)

        # 6. Sauvegarde NIfTI
        out_name = out_path / f"scaled_tau{args.tau}_{i:04d}_{vname}.nii.gz"
        save_nifti(x_rec, out_name)

    print(f"[ok] Saved {len(paths)} temperature scaled 3D volumes to {out_path}")


def main_calc_distance(argv=None):
    """
    Calcule la distance Euclidienne, geodesique, ou Mahalanobis.
    Référence par défaut : Moyenne Gaussienne (Mu) extraite du modèle .npz.
    Référence optionnelle : Une image cible (--target-image).
    Nouveau mode : Distances croisées (NxN) via --pairwise.
    """
    ap = argparse.ArgumentParser("LAM-Flow 3D Latent Distance Calculator")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint")
    ap.add_argument("--gauss", required=True, help="Gaussian model (.npz)")
    ap.add_argument("--views", required=True, help="Views header (e.g. T1,FA)")
    ap.add_argument("--view-index", type=int, default=0, help="View index to analyze")
    ap.add_argument("--volume-size", default="64x64x64", help="HxWxD")
    ap.add_argument("--batch", type=int, default=1, help="Batch size (keep low for 3D)")
    ap.add_argument("--workers", type=int, default=4, 
                    help="Number of parallel I/O workers for loading images.")    
    ap.add_argument("--devices", default="cuda:0")
    ap.add_argument("--out-csv", required=True, help="Output CSV file path")
    ap.add_argument("--save-levels", action=argparse.BooleanOptionalAction, default=True,
                    help="Include separate columns for distance at each level.")
    ap.add_argument("--distance-metric", type=str, default="geodesic", choices=["euclidean", "mahalanobis", "geodesic"],
                    help="Which distance metric to use.")
    ap.add_argument("--variance-epsilon", type=float, default=1e-6, required=False, 
                    help="Regularization parameter for numerical stability.")
    
    # Options Source / Target
    ap.add_argument("--manifest", type=str, default=None, 
                    help="Input manifest CSV (optional if --source-image is provided)")
    ap.add_argument("--source-image", type=str, default=None,
                    help="Optional single source 3D image. If set, ignores the manifest.")
    ap.add_argument("--target-image", type=str, default=None,
                    help="Optional target 3D image path.")
    ap.add_argument("--pairwise", action=argparse.BooleanOptionalAction, default=False,
                    help="Calculate N x N pairwise distances between all subjects in the manifest.")
    
    args = ap.parse_args(argv)
    
    if not args.manifest and not args.source_image:
        raise ValueError("You must provide either --manifest or --source-image.")

    # Avertissement contourné si mode pairwise actif
    if args.distance_metric == "geodesic" and not args.target_image and not getattr(args, 'pairwise', False): 
        warnings.warn(
           "Geodesic distance requires a target image. Falling back to Mahalanobis distance for Gaussian Mean.", 
           UserWarning )
        args.distance_metric = "mahalanobis"

    device = torch.device(args.devices)
    
    if isinstance(args.volume_size, str):
        v_parts = args.volume_size.lower().split("x")
        args.volume_size = (int(v_parts[0]), int(v_parts[1]), int(v_parts[2]))
    
    # 1. Chargement du modèle
    blob = torch.load(resolve_ckpt_path(Path(args.ckpt)), map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    model = build_model_from_config(cfg, device, target_hwd=args.volume_size)
    
    _prime_if_needed(model, *args.volume_size, device)
    ok, note = load_weights_into_model(model, blob, int(args.view_index))
    if not ok: raise RuntimeError(f"Weights failed: {note}")
    
    views_list = [v.strip() for v in args.views.split(",")]
    vname = views_list[int(args.view_index)]

    # 2. Chargement du modèle Gaussien
    npz = np.load(args.gauss, allow_pickle=True)
    L = int(npz["L"])
    views_g = [str(v) for v in np.array(npz["views"]).tolist()]
    
    if vname not in views_g: 
        raise RuntimeError(f"View '{vname}' missing from Gaussian model.")
    
    import json
    raw_slices = json.loads(str(np.array(npz["slices_json"]).tolist()))
    slice_map = [] 
    for l in range(L):
        d = {}
        for v_idx, v in enumerate(views_g):
            s, e = raw_slices[l][str(v_idx)]
            d[v] = (int(s), int(e))
        slice_map.append(d)

    # 3. Parsing Source
    paths = []
    if args.source_image:
        src_path = Path(args.source_image)
        if not src_path.exists(): raise FileNotFoundError(f"Source image not found: {src_path}")
        paths = [src_path]
        print(f"[info] Using single source image: {src_path.name}")
    else:
        cols = _read_manifest_csv(Path(args.manifest))
        _, per_view_paths = _resolve_views(cols, Path(args.manifest).parent, args.views)
        paths = per_view_paths[int(args.view_index)]
    
    total_imgs = len(paths)

    # 4. PRÉPARATION DE LA VARIANCE (Et Référence si non Pairwise)
    reference_latents = [] 
    variance_latents = [] 

    print(f"[info] Extracting latent variances for standardized distance...")
    for l in range(L):
        s, e = slice_map[l][vname]
        if f"Sigma_{l}_type" in npz and str(npz[f"Sigma_{l}_type"]) == "lowrank":
            U_full = npz[f"Sigma_{l}_U"]
            eig = npz[f"Sigma_{l}_eig"]
            sigma2 = float(npz[f"Sigma_{l}_sigma2"])
            U_v = U_full[s:e, :]
            var_flat = np.sum((U_v ** 2) * eig, axis=1) + sigma2
        else:
            Sig_full = npz[f"Sigma_{l}"]
            if Sig_full.ndim == 1:
                var_flat = Sig_full[s:e]
            else:
                var_flat = np.diag(Sig_full)[s:e]
                
        var_tensor = torch.from_numpy(var_flat).float().to(device).view(1, -1)
        variance_latents.append(var_tensor)

    if not getattr(args, 'pairwise', False):
        if args.target_image:
            tgt_path = Path(args.target_image)
            if not tgt_path.exists(): raise FileNotFoundError(f"Target image not found: {tgt_path}")
            print(f"[info] Reference: Target Image ({tgt_path.name})")
            xt = _read_image_3d(tgt_path, target_hwd=args.volume_size).unsqueeze(0).to(device)
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
                mu_tensor = torch.from_numpy(mu_l[s:e]).float().to(device).view(1, -1)
                reference_latents.append(mu_tensor)

    # 5. BOUCLE DE CALCUL
    bs = max(1, int(args.batch))
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor
    def _load_single(p):
        try:
            return _read_image_3d(p, target_hwd=args.volume_size), str(p)
        except Exception:
            return None, None

    # -- NOUVELLE LOGIQUE : PAIRWISE (NxN) --
    if getattr(args, 'pairwise', False):
        print(f"[info] Calculating Pairwise (NxN) distances for {total_imgs} volumes...")
        all_z_levels = [[] for _ in range(L)]
        all_paths = []

        try:
            iterable = tqdm(range(0, total_imgs, bs), desc="Distance calc", unit="batch")
        except ImportError:
            iterable = range(0, total_imgs, bs)

        for i in iterable:
            batch_paths = paths[i : i + bs]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(_load_single, batch_paths))
            
            xs = [res[0] for res in results if res[0] is not None]
            valid_paths = [res[1] for res in results if res[1] is not None]
            if not xs: continue

            xb = torch.stack(xs, dim=0).to(device)
            with torch.no_grad():
                z_list, _ = model.inverse_and_log_det(xb)
                if not isinstance(z_list, list): z_list = [z_list]
                
            # Mouvement crucial vers CPU pour éviter l'OOM
            for l, z in enumerate(z_list):
                all_z_levels[l].append(z.cpu())
            all_paths.extend(valid_paths)

        N_valid = len(all_paths)
        total_dist_sq = torch.zeros((N_valid, N_valid), dtype=torch.float64)

        print(f"[info] Executing matrix multiplications on CPU for {N_valid}x{N_valid} combinations...")
        for l in range(L):
            Z_l = torch.cat(all_z_levels[l], dim=0).view(N_valid, -1)
            
            if args.distance_metric == "mahalanobis":
                var = variance_latents[l].cpu() + args.variance_epsilon
                Z_scaled = Z_l / torch.sqrt(var)
                dist_l = torch.cdist(Z_scaled, Z_scaled, p=2)
            elif args.distance_metric == "geodesic":
                Z_norm = F.normalize(Z_l, p=2, dim=1)
                cos_sim = torch.mm(Z_norm, Z_norm.t())
                cos_sim = torch.clamp(cos_sim, min=-1.0 + args.variance_epsilon, max=1.0 - args.variance_epsilon)
                dist_l = torch.acos(cos_sim)
            else:
                dist_l = torch.cdist(Z_l, Z_l, p=2)

            total_dist_sq += (dist_l.double() ** 2)

        total_dist = torch.sqrt(total_dist_sq).numpy()

        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            # En-tête : "subject", "sub-01_T1w.nii.gz", "sub-02_T1w.nii.gz"...
            header = ["subject"] + [Path(p).name for p in all_paths]
            writer.writerow(header)
            
            for i in range(N_valid):
                row = [Path(all_paths[i]).name] + [f"{d:.6f}" for d in total_dist[i]]
                writer.writerow(row)

        print(f"[ok] Pairwise distances written to {out_csv}")
        return 0

    # -- ANCIENNE LOGIQUE : 1 TARGET (ou MU) --
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["path", "total_distance"]
        if args.save_levels:
            header += [f"dist_L{l}" for l in range(L)]
        writer.writerow(header)

        try:
            iterable = tqdm(range(0, total_imgs, bs), desc="Distance calc", unit="batch")
        except ImportError:
            iterable = range(0, total_imgs, bs)

        for i in iterable:
            batch_paths = paths[i : i + bs]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(_load_single, batch_paths))
                
            xs = [res[0] for res in results if res[0] is not None]
            valid_paths = [res[1] for res in results if res[1] is not None]
            if not xs: continue

            xb = torch.stack(xs, dim=0).to(device)
            B = xb.shape[0]

            with torch.no_grad():
                z_list, _ = model.inverse_and_log_det(xb)
                if not isinstance(z_list, list): z_list = [z_list]
                
            dists_per_level = np.zeros((B, L), dtype=np.float64)
            
            for l, z in enumerate(z_list):
                ref = reference_latents[l] 
                z_flat = z.view(B, -1)     

                if args.distance_metric == "mahalanobis":
                    var = variance_latents[l] + args.variance_epsilon
                    dist_sq = torch.sum(((z_flat - ref) ** 2) / var, dim=1)
                    dists_per_level[:, l] = np.sqrt(dist_sq.cpu().numpy())
                elif args.distance_metric == "geodesic":
                    cos_sim = F.cosine_similarity(z_flat, ref, dim=1)
                    cos_sim = torch.clamp(cos_sim, min=-1.0 + args.variance_epsilon, max=1.0 - args.variance_epsilon)
                    geodesic_dist = torch.acos(cos_sim)
                    dists_per_level[:, l] = geodesic_dist.cpu().numpy()                        
                else: 
                    dist_sq = torch.sum((z_flat - ref) ** 2, dim=1)
                    dists_per_level[:, l] = np.sqrt(dist_sq.cpu().numpy())

            for b_idx in range(B):
                total_dist = np.sqrt(np.sum(dists_per_level[b_idx] ** 2))
                row = [valid_paths[b_idx], f"{total_dist:.6f}"]
                if args.save_levels:
                    row.extend([f"{d:.6f}" for d in dists_per_level[b_idx]])
                writer.writerow(row)
                
                if args.source_image:
                    print(f"\n[Result] Standardized Distance: {total_dist:.4f}")

    print(f"[ok] Distances written to {out_csv}")
    return 0
    
def main_recon_interpolate(argv=None):
    """
    Interpole entre une cible (Moyenne Gaussienne ou Image Cible) et un sujet 3D.
    """
    def _slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, DOT_THRESHOLD: float = 0.9990):
        v0_norm = torch.norm(v0, dim=1, keepdim=True)
        v1_norm = torch.norm(v1, dim=1, keepdim=True)
        
        v0_norm_safe = torch.clamp(v0_norm, min=1e-8)
        v1_norm_safe = torch.clamp(v1_norm, min=1e-8)
        
        v0_dir = v0 / v0_norm_safe
        v1_dir = v1 / v1_norm_safe
        
        dot = torch.sum(v0_dir * v1_dir, dim=1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0)
        
        # --- CORRECTION ANTI-NAN ---
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)
        
        # 1. Empêcher la division par zéro mathématique (0.0 / 0.0 -> NaN)
        sin_omega_safe = torch.where(sin_omega == 0.0, torch.ones_like(sin_omega) * 1e-6, sin_omega)
        
        lerp_mask = (torch.abs(dot) > DOT_THRESHOLD)
        mag = (1.0 - t) * v0_norm + t * v1_norm
        
        slerp_dir = (torch.sin((1.0 - t) * omega) / sin_omega_safe) * v0_dir + (torch.sin(t * omega) / sin_omega_safe) * v1_dir
        res = slerp_dir * mag
        
        lerp_res = (1.0 - t) * v0 + t * v1
        
        # 2. Sécurité supplémentaire : nettoyer tout NaN résiduel avant le torch.where
        res = torch.nan_to_num(res, nan=0.0)
        
        return torch.where(lerp_mask, lerp_res, res)

    def _nlerp(t: float, v0: torch.Tensor, v1: torch.Tensor):
        """ Normalized Lerp : 100% stable numériquement, sans trigonométrie """
        # 1. Calcul des magnitudes cibles
        v0_norm = torch.norm(v0, dim=1, keepdim=True)
        v1_norm = torch.norm(v1, dim=1, keepdim=True)
        target_mag = (1.0 - t) * v0_norm + t * v1_norm
        
        # 2. Interpolation linéaire standard (Lerp)
        lerp_val = (1.0 - t) * v0 + t * v1
        lerp_norm = torch.norm(lerp_val, dim=1, keepdim=True)
        
        # 3. Projection sur la sphère latente (Inflation)
        lerp_dir = lerp_val / torch.clamp(lerp_norm, min=1e-8)
        
        return lerp_dir * target_mag


    ap = argparse.ArgumentParser("LAM-Flow 3D Latent Interpolation")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    ap.add_argument("--gauss", type=str, required=True, help="Gaussian model (.npz)")
    ap.add_argument("--views", type=str, required=True, help="Views list (e.g. T1,FA)")
    ap.add_argument("--view-index", type=int, default=0, help="View to process")
    ap.add_argument("--volume-size", type=str, default="64x64x64", help="HxWxD") 
    ap.add_argument("--batch", type=int, default=1, help="Number of source subjects")
    ap.add_argument("--devices", type=str, default="cuda:0")
    ap.add_argument("--out", type=str, required=True, help="Output NIfTI file path")
    
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--source-image", type=str, default=None)
    ap.add_argument("--target-image", type=str, default=None)
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument('--interp-type', choices=["nlerp", "slerp"], default="slerp")
    ap.add_argument("--interp-level", action="append", type=str)
    
    args = ap.parse_args(argv)
    
    if not args.manifest and not args.source_image:
        raise ValueError("You must provide either --manifest or --source-image.")
        
    device = torch.device(args.devices)
    
    level_overrides = {}
    if args.interp_level:
        for item in args.interp_level:
            try:
                parts = item.split(',')
                lvl = int(parts[0]); val = float(parts[1])
                level_overrides[lvl] = val
            except: pass

    if isinstance(args.volume_size, str):
        v_parts = args.volume_size.lower().split("x")
        args.volume_size = (int(v_parts[0]), int(v_parts[1]), int(v_parts[2]))

    # 1. Chargement Modèle (EMA désactivé par sécurité d'inférence)
    blob = torch.load(resolve_ckpt_path(Path(args.ckpt)), map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    model = build_model_from_config(cfg, device, target_hwd=args.volume_size)
    _prime_if_needed(model, *args.volume_size, device)
    
    ok, note = load_weights_into_model(model, blob, int(args.view_index), prefer_ema=False)
    if not ok: raise RuntimeError(f"Weights failed: {note}")

    views_list = [v.strip() for v in args.views.split(",")]
    vname = views_list[int(args.view_index)]

    # 2. Chargement Source
    paths = []
    if args.source_image:
        src_path = Path(args.source_image)
        paths = [src_path]
    else:
        cols = _read_manifest_csv(Path(args.manifest))
        _, per_view_paths = _resolve_views(cols, Path(args.manifest).parent, args.views)
        paths = per_view_paths[int(args.view_index)]

    # 3. Chargement de la Moyenne Gaussienne (\mu)
    print(f"[info] Loading Gaussian Mean (Mu)...")
    gauss_blob = _load_gaussian_model(Path(args.gauss))
    views_g, dims_tbl, shapes_by_view, L = _validate_gauss_blob(gauss_blob)
    
    if vname not in views_g: 
        raise RuntimeError(f"View '{vname}' missing from Gaussian model.")
    v_idx_g = views_g.index(vname)
    mu_list_raw = gauss_blob["mu"]

    # Extraction des tranches (slices) pour la vue cible
    level_view_slices = []
    raw_slices = gauss_blob.get("level_view_slices", None)
    if raw_slices:
        for l in range(L):
            row = raw_slices[l]
            if isinstance(row, dict): row = {int(k): tuple(v) for k, v in row.items()}
            else: row = {vi: tuple(row[vi]) for vi in range(len(views_g))}
            level_view_slices.append(row)
    else:
        for l in range(L):
            off = 0; row = {}
            for vi in range(len(views_g)):
                d = int(np.asarray(dims_tbl[vi][l]).item()); row[vi] = (off, off+d); off += d
            level_view_slices.append(row)

    # Préparation des tenseurs \mu par niveau
    mu_tensors = []
    for l in range(L):
        a, b = level_view_slices[l][v_idx_g]
        mu_flat = np.asarray(mu_list_raw[l], dtype=np.float64).ravel()[a:b]
        mu_tensors.append(torch.from_numpy(mu_flat).float().to(device))

    # 4. Détermination de la Cible
    z_target_list = []
    is_slerp = args.interp_type == "slerp"
    
    if args.target_image:
        tgt_path = Path(args.target_image)
        xt = _read_image_3d(tgt_path, target_hwd=args.volume_size).unsqueeze(0).to(device)
        with torch.no_grad():
            z_tgt_raw, _ = model.inverse_and_log_det(xt)
            if not isinstance(z_tgt_raw, list): z_tgt_raw = [z_tgt_raw]
            for z in z_tgt_raw:
                z_target_list.append(z.clone().contiguous())
        is_slerp = True
    else:
        dummy = torch.zeros(1, 1, args.volume_size[0], args.volume_size[1], args.volume_size[2], device=device)
        with torch.no_grad():
            z_dummy, _ = model.inverse_and_log_det(dummy)
            if not isinstance(z_dummy, list): z_dummy = [z_dummy]

        for l in range(L):
            ref_shape = z_dummy[l].shape 
            mu_t = mu_tensors[l].view(1, ref_shape[1], ref_shape[2], ref_shape[3], ref_shape[4])
            z_target_list.append(mu_t.to(device).contiguous())
    
    # 5. Boucle d'Interpolation
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    limit = min(max(1, int(args.batch)), len(paths))
    
    for i in range(limit):
        pth = paths[i]
        xs = _read_image_3d(pth, target_hwd=args.volume_size).unsqueeze(0).to(device)
        
        with torch.no_grad():
            z_source_list, _ = model.inverse_and_log_det(xs)
            if not isinstance(z_source_list, list): z_source_list = [z_source_list]
            
            z_interp_list = []
            for l, (z_src, z_tgt, mu_t) in enumerate(zip(z_source_list, z_target_list, mu_tensors)):
                t_level = level_overrides.get(l, float(args.t))

                B, C, H_lvl, W_lvl, D_lvl = z_src.shape

                z_src_flat = z_src.contiguous().view(B, -1)
                z_tgt_flat = z_tgt.contiguous().view(B, -1)
                mu_flat_b = mu_t.view(1, -1)

                if is_slerp:
                    z_src_c = z_src_flat - mu_flat_b
                    z_tgt_c = z_tgt_flat - mu_flat_b
                    z_new_c = _slerp(t_level, z_tgt_c, z_src_c) 
                    z_new_flat = z_new_c + mu_flat_b
                else:
                    z_new_flat = _nlerp(t_level, mu_flat_b, z_src_flat)

                # FORÇAGE DE LA CONTIGUITÉ MÉMOIRE (Crucial pour Glow 3D)
                z_new = z_new_flat.view(B, C, H_lvl, W_lvl, D_lvl).contiguous()
                
                # SONDE DE DÉBOGAGE
                if torch.isnan(z_new).any():
                    print(f"  [CRITIQUE] NaNs détectés au niveau {l} pendant l'interpolation ! t={t_level}")

                z_interp_list.append(z_new)
            
            xh, _ = model.forward_and_log_det(z_interp_list)
            xh = to01(_coerce_5d(xh, target_hwd=args.volume_size), winsorize=True)
            
            if limit > 1:
                base_name = Path(pth).name.split('.')[0]
                if out_path.name.endswith('.nii.gz'):
                    out_name = out_path.parent / f"{out_path.name[:-7]}_{i:03d}_{base_name}.nii.gz"
                else:
                    out_name = out_path.parent / f"{out_path.stem}_{i:03d}_{base_name}{out_path.suffix}"
            else:
                out_name = out_path

            save_nifti(xh, out_name)
                
        print(f"[ok] Generated interpolated 3D volume at {out_name}")

    return 0

# --------------------------- main ---------------------------
import argparse
from pathlib import Path
import json
import torch
import torch.nn.functional as F

@torch.no_grad()
def reconstruct_batch(model, xb: torch.Tensor):
    """
    Round-trip x -> z -> x_hat using MultiscaleFlow APIs for 3D volumes.
    Uses model.inverse_and_log_det(x) to obtain latents, then decodes with
    model.forward_and_log_det(z_list) for a noise-free reconstruction.
    Returns x_hat tensor typically (N, C, D, H, W) or (N, C, H, W, D).
    """
    # Push x to latents
    if hasattr(model, "inverse_and_log_det"):
        z, _ = model.inverse_and_log_det(xb)
    elif hasattr(model, "inverse"):
        z, _ = model.inverse(xb)
    else:
        raise RuntimeError("Model lacks inverse mapping (inverse_and_log_det / inverse).")

    # Ensure list of tensors for multiscale
    z_list = z if isinstance(z, (list, tuple)) else [z]

    # Decode using the flow graph
    if hasattr(model, "forward_and_log_det"):
        xh, _ = model.forward_and_log_det(z_list)
        # Extraction des 3 dimensions spatiales de l'entrée d'origine
        target_shape = (xb.shape[-3], xb.shape[-2], xb.shape[-1])
        return _coerce_5d(xh, target_hwd=target_shape)
    else:
        raise RuntimeError(
            "Model does not expose forward_and_log_det(z_list); cannot decode latents. "
            "Please update your flow wrapper to include forward_and_log_det."
        )

def make_recon_panel(x: torch.Tensor, xh: torch.Tensor) -> torch.Tensor:
    """
    Create a 3-column panel per sample: [x | x_hat | abs(x-x_hat)], stacked into a grid batch.
    Input: (N,1,H,W) in [0,1]
    Output: (3N,1,H,W) suitable for save_grid with nrow=3
    """
    x = _coerce_nchw_4d(x, target_hw=(x.shape[-2], x.shape[-1]))
    xh = _coerce_nchw_4d(xh, target_hw=(x.shape[-2], x.shape[-1]))
    diff = torch.abs(x - xh)
    diff_max = float(diff.max().item())

    print(f"[info] recon: max |x - x_hat| = {diff_max:.6f}")

    diff = to01(diff)
    panels = []
    for i in range(x.shape[0]):
        panels.append(x[i:i+1])
        panels.append(xh[i:i+1])
        panels.append(diff[i:i+1])

    return torch.cat(panels, dim=0)

def make_recon_panel_3d(x: torch.Tensor, xh: torch.Tensor) -> torch.Tensor:
    """
    Create a sequence of 3 volumes per sample: [x | x_hat | abs(x-x_hat)], stacked into a batch.
    Input: (N, C, H, W, D) in [0,1]
    Output: (3N, C, H, W, D) suitable for 4D NIfTI export or 2D slice grids.
    """
    target_shape = (x.shape[-3], x.shape[-2], x.shape[-1])
    
    x = _coerce_5d(x, target_hwd=target_shape)
    xh = _coerce_5d(xh, target_hwd=target_shape)
    
    diff = torch.abs(x - xh)
    diff_max = float(diff.max().item())

    print(f"[info] recon: max |x - x_hat| = {diff_max:.6f}")

    # Normalisation de la différence pour mettre en évidence les erreurs
    diff = to01(diff)
    
    panels = []
    for i in range(x.shape[0]):
        panels.append(x[i:i+1])
        panels.append(xh[i:i+1])
        panels.append(diff[i:i+1])

    return torch.cat(panels, dim=0)
@torch.no_grad()
def sample_with_temperature(model, n: int, temp: float):
    """
    Robust sampler that tries to honor temperature:
      1) model.sample(n, temperature=temp)
      2) model.sample(n, T=temp)
      3) Temporarily set q0 temperatures if available, then model.sample(n)
      4) Fallback: model.sample(n) with a warning (temperature likely ignored)
    Returns a tensor (N,C,H,W) or a (list/tuple, we coerce later).
    """

    def _try_set_temperature_on_q0(model, temp: float) -> Optional[Tuple[object, list]]:
        """
        Best-effort: if model has q0 distributions, try to set their temperature then return a handle to restore later.
        Returns (container, prev_values) if successful, otherwise None.
        """
        q0 = getattr(model, "q0", None)
        if q0 is None:
            return None
        # q0 may be a list/tuple of base dists, or a single object with sub-d dists
        bases = []
        if isinstance(q0, (list, tuple)):
            bases = list(q0)
        else:
            # try common patterns
            for attr in ("q0s", "bases", "base"):
                cand = getattr(q0, attr, None)
                if cand is not None:
                    if isinstance(cand, (list, tuple)):
                        bases = list(cand)
                    else:
                        bases = [cand]
                    break
            if not bases:
                bases = [q0]

        prev = []
        did_any = False
        for b in bases:
            # Try common APIs
            if hasattr(b, "T"):
                try:
                    prev.append(("T", float(getattr(b, "T"))))
                    setattr(b, "T", float(temp))
                    did_any = True
                    continue
                except Exception:
                    pass
            if hasattr(b, "temperature"):
                try:
                    prev.append(("temperature", float(getattr(b, "temperature"))))
                    setattr(b, "temperature", float(temp))
                    did_any = True
                    continue
                except Exception:
                    pass
            if hasattr(b, "set_temperature"):
                try:
                    prev.append(("call", None))
                    b.set_temperature(float(temp))
                    did_any = True
                    continue
                except Exception:
                    pass
        if did_any:
            return (bases, prev)
        return None

    def _restore_q0_temperature(handle: Tuple[object, list] | None):
        if handle is None:
            return
        bases, prev = handle
        # prev aligns with bases by index if we stored in order
        if not isinstance(bases, (list, tuple)):
            return
        j = 0
        for i, b in enumerate(bases):
            if j >= len(prev):
                break
            tag, val = prev[j]
            j += 1
            try:
                if tag == "T" and hasattr(b, "T"):
                    setattr(b, "T", float(val))
                elif tag == "temperature" and hasattr(b, "temperature"):
                    setattr(b, "temperature", float(val))
                # if tag == "call", there may be no easy restore
            except Exception:
                pass

    # 1) Try explicit kw "temperature"
    try:
        return model.sample(n, temperature=float(temp))
    except TypeError:
        pass
    except Exception as e:
        # some models accept the kw but error elsewhere—propagate later if all fails
        err1 = e
    # 2) Try "T"
    try:
        return model.sample(n, T=float(temp))
    except TypeError:
        pass
    except Exception as e2:
        err2 = e2
    # 3) Try setting q0 temperature temporarily
    handle = None
    try:
        handle = _try_set_temperature_on_q0(model, float(temp))
        if handle is not None:
            out = model.sample(n)
            return out
    finally:
        _restore_q0_temperature(handle)
    # 4) Fallback: plain sample, warn
    print(f"[warn] temperature={temp} may have been ignored by model.sample; "
          f"no compatible API found. Consider implementing explicit prior sampling for this model.")
    return model.sample(n)
    
def main_sample(argv=None):
    ap = argparse.ArgumentParser("LAM‑Flow 3D sample grid tool")
    ap.add_argument("--version", action="store_true", help="Print version and exit")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file or directory")
    ap.add_argument("--view-index", type=int, default=0, help="Which view to sample (0-based)")
    ap.add_argument("--sample-grid-size", type=parse_mn, default=None, help="Grid as MxN (rows×cols), e.g., 4x4")
    ap.add_argument("--image-size", type=parse_hw, default="128x128", help="Per-tile 2D size HxW for the saved snapshot PNG")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (prior noise scale)")
    ap.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True, help="Prefer EMA weights if present")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed for reproducible sampling")
    ap.add_argument("--devices", type=str, default="cuda:0", help='Device like "cuda:0" or "cpu"')
    ap.add_argument("--sample-grid-out", type=str, default="", help="Output PNG filename (default auto next to ckpt)")
    ap.add_argument("--save-volumes-dir", type=str, default="", 
                    help="If specified, saves each generated 3D sample as an individual NIfTI volume in this directory.")
    
    # Paramètres de coupe pour la visualisation 2D des volumes 3D
    ap.add_argument("--slice-axis", type=int, default=0, help="Axis for NIfTI slicing (0: Sagittal/Axial depending on orient, etc.)")
    ap.add_argument("--slice-index", type=int, default=32, help="Slice index for extracting the 2D snapshot from 3D volume")
    
    # Options pour l'échantillonnage de l'enveloppe convexe typique 3D
    ap.add_argument("--sampling-strategy", type=str, default="pure", choices=["pure", "projected-hull", "geodesic-hull"],
                    help="Sampling strategy: standard Gaussian (pure), normalized linear convex hull (projected-hull), or Riemannian Frechet mean hull (geodesic-hull).")
    ap.add_argument("--hull-k", type=int, default=3, help="Number of empirical support samples to mix per generated point.")

    # Reconstruction sanity check options
    ap.add_argument("--recon", type=int, default=0, help="If >0, run reconstruction sanity check on N validation volumes.")
    ap.add_argument("--val-list", type=str, nargs="+", default=None, help="One or more inputs: globs (quoted) and/or files (txt lists or volume files).")
    ap.add_argument("--recon-out", type=str, default="", help="Output PNG for recon panel (default auto next to ckpt).")

    # ANTs 3D resampling options
    ap.add_argument("--resample-spacing", type=parse_hwd_float, default=None,
                    help="Physical spacing DxSxT (e.g., 1.0x0.8x0.8 mm). Uses ANTs resample_image.")
    ap.add_argument("--resample-size", type=parse_hwd, default=None,
                    help="Voxel size DxHxW (e.g., 48x64x56). Uses ANTs resample_image.")
    ap.add_argument("--native-spacing", type=parse_hwd_float, default=None,
                    help="Override native spacing DxSxT (mm) if not present in checkpoint config.")

    args = ap.parse_args(argv)
    if args.version:
        print(__version__)
        return

    if args.resample_spacing is not None and args.resample_size is not None:
        raise SystemExit("Specify only one of --resample-spacing or --resample-size (not both).")

    ckpt_path = resolve_ckpt_path(Path(args.ckpt))
    out_dir = ckpt_path.parent

    device = torch.device("cpu") if args.devices.lower() == "cpu" else torch.device(args.devices.split(",")[0])
    set_deterministic(args.seed)

    print(f"[info] lamnr_glow_tool_3d {__version__}")
    print(f"[info] loading checkpoint: {ckpt_path}")

    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)

    cfg = blob.get("config", {})
    
    # Construction du modèle à partir des configurations 3D natives
    model = build_model_from_config(cfg, device=device)

    # Extraction du spacing 3D natif
    native_spacing = None
    if args.native_spacing is not None:
        native_spacing = (float(args.native_spacing[0]), float(args.native_spacing[1]), float(args.native_spacing[2]))
    else:
        for key in ("spacing", "pixdim", "voxel_spacing", "voxel_size"):
            if key in cfg:
                val = cfg[key]
                try:
                    if isinstance(val, (list, tuple)) and len(val) >= 3:
                        native_spacing = (float(val[0]), float(val[1]), float(val[2]))
                        break
                except Exception:
                    pass
    if native_spacing is None:
        native_spacing = (1.0, 1.0, 1.0)

    ok, src_note = load_weights_into_model(model, blob, view_idx=int(args.view_index), prefer_ema=bool(args.ema))
    if not ok:
        raise RuntimeError(f"Could not load weights from checkpoint ({src_note})")
    which_src, note = src_note
    print(f"[info] weights loaded from: {which_src} (view {args.view_index})")

    # Amorce (Prime) du réseau avec la taille spatiale 3D native
    Hc, Wc, Dc = model.input_shape[-3], model.input_shape[-2], model.input_shape[-1]
    _prime_if_needed(model, Hc, Wc, Dc, device)

    # Helper interne pour extraire une coupe 2D d'un tenseur 3D (B, C, D, H, W)
    def _extract_2d_slice(tensor_5d, axis, index):
        idx = min(max(0, int(index)), tensor_5d.shape[axis + 2] - 1)
        if axis == 0:
            return tensor_5d[:, :, idx, :, :]
        elif axis == 1:
            return tensor_5d[:, :, :, idx, :]
        else:
            return tensor_5d[:, :, :, :, idx]

    # Reconstruction sanity check (volumes 3D complets)
    if int(args.recon) > 0:
        val_paths = _gather_val_paths(getattr(args, "val_list", None), limit=int(args.recon))
        if not val_paths:
            raise SystemExit("Recon requested but no validation images found.")
        print(f"[info] recon: loading {len(val_paths)} volume(s) for 3D round-trip test")
        xs = []
        for pth in val_paths:
            try:
                # Lecture du volume 3D complet (pas d'extraction de coupe ici pour respecter la SVD)
                xi = _read_image_3d(pth)  # Doit retourner un tenseur (1, H, W, D) ou similaire
            except Exception as e:
                print(f"[warn] skipping {pth}: {e}")
                continue
            # Redimensionnement volumétrique Trilinéraire pour correspondre au domaine du flux
            xi = F.interpolate(xi.unsqueeze(0), size=(Hc, Wc, Dc), mode="trilinear", align_corners=False).squeeze(0)
            xs.append(xi)
        if not xs:
            raise SystemExit("Recon: no readable volumes after parsing inputs.")
        xb = torch.stack(xs, dim=0).to(device=device, dtype=torch.float32)  # [N, C, Dc, Hc, Wc]
        
        try:
            xh = reconstruct_batch(model, xb)
        except Exception as e:
            raise SystemExit(f"Recon failed: {e}")
            
        # Extraction de la coupe 2D sur le batch réel et reconstruit pour le panneau PNG de diagnostic
        xb_slice = _extract_2d_slice(xb, args.slice_axis, args.slice_index)
        xh_slice = _extract_2d_slice(xh, args.slice_axis, args.slice_index)
        panel = make_recon_panel(xb_slice, xh_slice)
        
        if args.recon_out:
            recon_out = Path(args.recon_out)
            if not recon_out.is_absolute():
                recon_out = out_dir / recon_out
        else:
            recon_out = out_dir / f"recon_view{int(args.view_index)}_N{panel.shape[0]//3}_{Hc}x{Wc}x{Dc}_slice{args.slice_index}.png"
        panel = F.interpolate(panel, size=(Hc, Wc), mode="bilinear", align_corners=False)
        save_grid(panel, recon_out, nrow=3)
        print(f"[ok] 3D recon slice panel saved: {recon_out}")

    # Section Échantillonnage 3D

    if args.sample_grid_size is not None:
        M, N = args.sample_grid_size
        total = int(M) * int(N)

        if args.sampling_strategy == "pure":
            print(f"[info] sampling {total} volumes @ temp={args.temperature} as {M}x{N}")
            try:
                s = sample_with_temperature(model, total, float(args.temperature))
                x = s[0] if isinstance(s, (list, tuple)) else s
            except Exception as e:
                raise RuntimeError(f"sampling failed: {e}")
        else:
            print(f"[info] sampling {total} volumes via 3D Typical Convex Hull '{args.sampling_strategy}' (k={args.hull_k})")
            support_paths = _gather_val_paths(getattr(args, "val_list", None), limit=9999)
            if not support_paths:
                raise SystemExit("Hull sampling requested but no 3D support cohort found in --val-list.")

            print(f"[info] Projecting {len(support_paths)} volumetric scans into latent space...")
            
            # --- ENCODAGE CORRECT (Image -> Latent via 'inverse') ---
            z_list = []
            saved_latent_shape = None
            saved_is_tuple = False
            
            for pth in tqdm(support_paths, desc="Encoding Volumes", unit="scan"):
                try:
                    # Lecture et redimensionnement à la volée (comme dans votre bloc gauss-fit)
                    xi = _read_image_3d(pth, target_hwd=(Hc, Wc, Dc)) 
                    xb = xi.unsqueeze(0).to(device=device, dtype=torch.float32) # (1, C, H, W, D)
                except Exception as e:
                    print(f"\n[warn] Failed to read/interpolate {pth}: {e}")
                    continue
                
                try:
                    with torch.no_grad():
                        # L'encodage se fait via 'inverse' dans ce framework
                        if hasattr(model, "inverse_and_log_det"):
                            z_raw, _ = model.inverse_and_log_det(xb) 
                        elif hasattr(model, "inverse"):
                            z_raw, _ = model.inverse(xb)
                        else:
                            raise RuntimeError("Model lacks inverse mapping for encoding")
                        
                        # Sauvegarde et aplatissement (similaire à _flatten_latents_by_level)
                        if isinstance(z_raw, (list, tuple)):
                            latent_shape = [zl.shape for zl in z_raw]
                            # Aplatit chaque niveau à partir de la dimension 1, puis concatène
                            z_flat_subject = torch.cat([zl.flatten(start_dim=1) for zl in z_raw], dim=1)
                            if saved_latent_shape is None:
                                saved_latent_shape = latent_shape
                                saved_is_tuple = isinstance(z_raw, tuple)
                        else:
                            latent_shape = z_raw.shape
                            z_flat_subject = z_raw.flatten(start_dim=1)
                            if saved_latent_shape is None:
                                saved_latent_shape = latent_shape
                        
                        z_list.append(z_flat_subject.cpu()) # Ajoute un (1, D_total)
                except Exception as e:
                    print(f"\n[warn] Failed to encode {pth}: {e}")
                    continue

            if not z_list:
                raise SystemExit("Could not encode any 3D support vectors.")

            # Construction de la matrice d'enveloppe (N_support, D_total)
            Z_flat = torch.cat(z_list, dim=0).to(device)
            N_support, D_latent = Z_flat.shape
            print(f"[info] Encoded matrix shape: {Z_flat.shape}")

            # --- MANIPULATION LATENTE (Enveloppe Convexe) ---
            dirichlet = torch.distributions.Dirichlet(torch.ones(args.hull_k, device=device))
            alpha = dirichlet.sample((total,))
            idx = torch.randint(0, N_support, (total, args.hull_k), device=device)
            Z_picked = Z_flat[idx]

            empirical_radius = torch.mean(torch.norm(Z_flat, p=2, dim=-1))

            if args.sampling_strategy == "projected-hull":
                z_linear = torch.sum(alpha.unsqueeze(-1) * Z_picked, dim=1)
                z_sample_flat = empirical_radius * (z_linear / torch.norm(z_linear, p=2, dim=-1, keepdim=True))
                
            elif args.sampling_strategy == "geodesic-hull":
                Z_picked_unit = Z_picked / torch.norm(Z_picked, p=2, dim=-1, keepdim=True)
                z_init = torch.sum(alpha.unsqueeze(-1) * Z_picked_unit, dim=1)
                z_geo = z_init / torch.norm(z_init, p=2, dim=-1, keepdim=True)
                
                for _ in range(8):
                    cos_theta = torch.bmm(Z_picked_unit, z_geo.unsqueeze(-1)).squeeze(-1)
                    cos_theta = torch.clamp(cos_theta, -0.9999, 0.9999)
                    theta = torch.acos(cos_theta)
                    sin_theta = torch.sin(theta)
                    scale = torch.where(sin_theta > 1e-5, theta / sin_theta, torch.ones_like(theta))
                    tangent_vectors = (Z_picked_unit - cos_theta.unsqueeze(-1) * z_geo.unsqueeze(1)) * scale.unsqueeze(-1)
                    mean_tangent = torch.sum(alpha.unsqueeze(-1) * tangent_vectors, dim=1)
                    norm_tangent = torch.norm(mean_tangent, p=2, dim=-1, keepdim=True)
                    norm_tangent = torch.clamp(norm_tangent, min=1e-5)
                    z_geo = z_geo * torch.cos(0.5 * norm_tangent) + (mean_tangent / norm_tangent) * torch.sin(0.5 * norm_tangent)
                    z_geo = z_geo / torch.norm(z_geo, p=2, dim=-1, keepdim=True)
                
                z_sample_flat = z_geo * empirical_radius

            # --- RECONSTRUCTION TOPOLOGIQUE ---
            if isinstance(saved_latent_shape, list):
                z_sample_list = []
                current_idx = 0
                for shape in saved_latent_shape:
                    # shape est (1, C, H, W, D). On remplace le 1 par 'total'
                    target_shape = (total,) + tuple(shape)[1:] 
                    num_elements = torch.prod(torch.tensor(target_shape[1:])).item()
                    zl_flat = z_sample_flat[:, current_idx : current_idx + num_elements]
                    z_sample_list.append(zl_flat.view(*target_shape).to(device))
                    current_idx += num_elements
                
                if saved_is_tuple:
                    z_sample = tuple(z_sample_list)
                else:
                    z_sample = z_sample_list
            else:
                target_shape = (total,) + tuple(saved_latent_shape)[1:]
                z_sample = z_sample_flat.view(*target_shape).to(device)

            print_shape = z_sample[0].shape if isinstance(z_sample, (list, tuple)) else z_sample.shape
            print(f"[info] Generated {total} latent samples with structural shape {print_shape} using strategy '{args.sampling_strategy}'.")

            # --- DÉCODAGE CORRECT (Latent -> Image via 'forward') ---
            try:
                with torch.no_grad():
                    # Dans ce framework, forward génère l'image à partir du latent
                    if hasattr(model, "forward_and_log_det"):
                        x_out, _ = model.forward_and_log_det(z_sample)
                    else:
                        x_out = model(z_sample) # Appelle model.forward()
                        
                    if isinstance(x_out, (list, tuple)):
                        x = x_out[0] # Extrait l'image si c'est un tuple (x, log_det)
                    else:
                        x = x_out
            except Exception as e:
                raise RuntimeError(f"Decoding (Latent -> Image) failed: {e}")

        # Recalage optionnel ANTs (Versions 3D des fonctions de rééchantillonnage)
        if args.resample_spacing is not None:
            target_spacing = (float(args.resample_spacing[0]), float(args.resample_spacing[1]), float(args.resample_spacing[2]))
            print(f"[info] ANTs 3D resample by spacing: {native_spacing} -> {target_spacing}")
            x = resample_with_ants_spacing_3d(x, native_spacing=native_spacing, target_spacing=target_spacing)
        elif args.resample_size is not None:
            target_size = (int(args.resample_size[0]), int(args.resample_size[1]), int(args.resample_size[2]))
            print(f"[info] ANTs 3D resample by voxel size -> {target_size}")
            x = resample_with_ants_size_3d(x, target_size=target_size, native_spacing=native_spacing)

        
        # --- NOUVEAU BLOC : Sauvegarde des Volumes 3D (NIfTI) ---
        if args.save_volumes_dir:
            import ants
            import os
            vol_out_dir = Path(args.save_volumes_dir)
            if not vol_out_dir.is_absolute():
                vol_out_dir = out_dir / vol_out_dir
            vol_out_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"[info] Saving {total} generated 3D volumes to {vol_out_dir} ...")
            # x a la forme (Batch, Channels, H, W, D)
            for i in range(total):
                # Extraction du volume individuel. On prend le premier canal [i, 0]
                vol_tensor = x[i, 0].cpu().numpy()
                
                # Création de l'image ANTs
                ants_img = ants.from_numpy(vol_tensor)
                
                # Application de l'espacement physique correct (natif ou redimensionné)
                if args.resample_spacing is not None:
                    ants_img.set_spacing((float(args.resample_spacing[0]), float(args.resample_spacing[1]), float(args.resample_spacing[2])))
                else:
                    ants_img.set_spacing(native_spacing)
                
                # Nommage du fichier
                it = blob.get("iter", None)
                it_str = (f"_it{int(it)-1:06d}" if isinstance(it, int) and it > 0 else "")
                vol_name = f"sample{it_str}_view{int(args.view_index)}_{args.sampling_strategy}_idx{i:03d}.nii.gz"
                
                ants.image_write(ants_img, str(vol_out_dir / vol_name))
            print(f"[ok] Volumes saved successfully.")
        # --------------------------------------------------------

        # Extraction de la coupe 2D cible
        from torchvision.utils import save_image  # <-- Import direct de la fonction officielle

        # Extraction de la coupe 2D cible
        print(f"[info] Slicing generated 3D volumes (axis={args.slice_axis}, index={args.slice_index}) for grid visualization.")
        x_slice = _extract_2d_slice(x, args.slice_axis, args.slice_index)

        # 1. Sécurisation du parsing de la taille
        if isinstance(args.image_size, str):
            th, tw = args.image_size.lower().split('x')
            target_h, target_w = int(th), int(tw)
        else:
            target_h, target_w = int(args.image_size[0]), int(args.image_size[1])

        # 2. Redimensionnement
        if (target_h, target_w) != (int(x_slice.shape[-2]), int(x_slice.shape[-1])):
            x_slice = F.interpolate(x_slice, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # 3. Forcer le format strict (Batch, Canal, Hauteur, Largeur)
        if x_slice.dim() == 3:
            x_slice = x_slice.unsqueeze(1) # Restaure le canal (1) si une fonction l'a écrasé
            
        # Détermination du chemin de sortie
        if args.sample_grid_out:
            out_path = Path(args.sample_grid_out)
            if not out_path.is_absolute():
                out_path = out_dir / out_path
        else:
            it = blob.get("iter", None)
            it_str = (f"_it{int(it)-1:06d}" if isinstance(it, int) and it > 0 else "")
            out_name = (f"samples3d{it_str}_view{int(args.view_index)}_{args.sampling_strategy}_"
                        f"slice{args.slice_index}_{int(M)}x{int(N)}.png")
            out_path = out_dir / out_name

        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 4. SAUVEGARDE DIRECTE : On bypasse 'save_grid'
        # x_slice est garanti d'être de forme [10, 1, 128, 128]
        save_image(x_slice.float().cpu(), out_path, nrow=int(N))
        print(f"[ok] wrote summary grid PNG: {out_path}")

        # Métadonnées JSON enrichies pour la traçabilité 3D
        meta = {
            "version": __version__,
            "ckpt": str(ckpt_path),
            "view_index": int(args.view_index),
            "sampling_strategy": args.sampling_strategy,
            "hull_k": int(args.hull_k) if args.sampling_strategy != "pure" else None,
            "slice_axis": int(args.slice_axis),
            "slice_index": int(args.slice_index),
            "sample_grid_size": [int(M), int(N)],
            "ckpt_native_shape_3d": [int(Hc), int(Wc), int(Dc)],
            "seed": int(args.seed),
            "out": str(out_path),
            "native_spacing_3d": list(native_spacing),
        }
        try:
            with open(out_path.with_suffix(".json"), "w") as f:
                json.dump(meta, f, indent=2)
            print(f"[ok] wrote: {out_path.with_suffix('.json')}")
        except Exception as e:
            print(f"[warn] could not write metadata json: {e}")

if __name__ == "__main__":
    table = {
        "sample": main_sample,
        "recon": main_recon,
        "gauss-fit": main_gauss_fit,
        "gauss-impute": main_gauss_impute,
        "recon-temperature": main_recon_temperature,
        "calc-distance": main_calc_distance,
        "recon-interpolate": main_recon_interpolate,
        "recon-template": main_recon_template,
        "recon-cohort-template": main_recon_cohort_template,
        
    }
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Available subcommands:", ", ".join(sorted(table.keys())))
        sys.exit(0)
    cmd = sys.argv.pop(1)
    if cmd not in table:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
    sys.exit(table[cmd]())