#!/usr/bin/env python3
"""
lamnr_glow_tool.py — Sample M×N image grids from trained LAM-Flow (Glow 2D) checkpoints.

v0.3.1 (2025-11-09)
- v0.3.8: Add `--cov-estimator lowrank` with `--rank` and `--sigma2`,
  storing per-level factors U (D×r), eig (r), sigma2.

- Add `gauss-fit` subcommand to fit a Conditional Gaussian over multiview latents
  (perlevel|merged) from a strict, row-ordered manifest; missing files cause a hard error.
  Produces a serialized model (`--gauss-out`) and optional summary JSON (`--gauss-summary`).
  Quick usage:
  
  ```bash
  python lamnr_glow_tool.py gauss-fit \
    --ckpt runs/t1_t2_fa_128x128_vicreg/training_state.pt \
    --manifest /data/lam/manifest.csv \
    --views T1,T2,FA \
    --slice-axis 2 --slice-index 64 \
    --batch 64 --devices cuda:0 \
    --cov-mode perlevel \
    --cov-estimator full --shrinkage 1e-6 --cov-lam 0.00 \
    --gauss-out /data/lam/models/t1t2fa_gauss_perlevel.pt \
    --gauss-summary /data/lam/models/t1t2fa_gauss_perlevel.json
  ```

- Add ANTs resampling by physical spacing (--resample-spacing SxT) and by voxel size (--resample-size HxW).
- Add --native-spacing override.
- Ensure priming happens at checkpoint-native size to avoid internal shape mismatches.
- Prefer EMA weights if available; robust checkpoint loader.
- Save a metadata JSON next to the PNG output for reproducibility.

Usage examples
--------------
# 6×8 grid, 192×192 tiles, sample at ckpt-native size, then resample tiles to 0.8×0.8 mm (ANTs), then save:
python lamnr_glow_tool.py \
  --ckpt runs/t1_t2_fa_256x256_vicreg/training_state.pt \
  --view-index 1 \
  --grid-size 6x8 \
  --image-size 192x192 \
  --resample-spacing 0.8x0.8
  --out samples_view1.png

# If native spacing is not in the checkpoint, provide it:
python lamnr_glow_tool.py \
  --ckpt runs/t1_t2_fa_256x256_vicreg/training_state.pt \
  --view-index 1 \
  --grid-size 6x8 \
  --image-size 192x192 \
  --native-spacing 1.0x1.0 \
  --resample-spacing 0.7x0.7

# Resample by voxel count (use_voxels=True) to 192×192 before final grid save:
python lamnr_glow_tool.py \
  --ckpt runs/t1_t2_fa_256x256_vicreg/training_state.pt \
  --view-index 0 \
  --grid-size 5x10 \
  --resample-size 192x192 \
  --out samples_rsz.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import csv
import sys
import time
import hashlib

import torch
import torch.nn.functional as F
import torchvision as tv
from PIL import Image
import numpy as np

import ants

# Ensure headless save works
import matplotlib
matplotlib.use("Agg")

__version__ = "0.3.9"

# ---------------- antstorch / model factory -----------------
from antstorch import create_glow_normalizing_flow_model_2d

# ------------------------- utils ----------------------------
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


@torch.no_grad()
def warmup_actnorm_with_real_batch(model, x_real: torch.Tensor):
    """
    Run a single real-batch pass to stabilize ActNorm / data-dependent inits.
    Best-effort; safe to skip on failure.
    """
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        return
    x1 = x_real[:1].to(dev, torch.float32)
    for fn in ("log_prob", "inverse_and_log_det", "__call__"):
        if hasattr(model, fn):
            try:
                getattr(model, fn)(x1)
                break
            except Exception:
                continue

def to01(
    x: torch.Tensor,
    eps: float = 1e-8,
    winsorize: bool = False,
    upper_q: float = 0.99,
) -> torch.Tensor:
    """
    Per-image/channel normalization to [0, 1] over spatial dims (H,W).

    If winsorize=True, values above the per-image/channel upper_q quantile
    are clipped before min-max normalization, so a few spikes do not
    dominate the scaling.

    Expects x shaped (N, C, H, W) or anything that can be viewed that way.
    """
    if not torch.is_floating_point(x):
        x = x.float()

    if winsorize:
        # Flatten spatial dims, compute quantile per (N,C), then reshape
        N, C, H, W = x.shape
        flat = x.view(N, C, -1)                              # (N,C,HW)
        hi = torch.quantile(flat, upper_q, dim=-1, keepdim=True)  # (N,C,1)
        hi = hi.view(N, C, 1, 1)                             # (N,C,1,1)
        x = torch.minimum(x, hi)

    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


@torch.no_grad()
def resample_with_ants_spacing(x: torch.Tensor,
                               native_spacing: Tuple[float, float],
                               target_spacing: Tuple[float, float]) -> torch.Tensor:
    """
    Resample (N,C,H,W) to a target physical spacing using ANTsPy (use_voxels=False).
    If C>1, channels are resampled independently and stacked back.
    """
    device, dtype = x.device, x.dtype
    N, C, H, W = x.shape
    outs = []
    for c in range(C):
        xs = []
        for i in range(N):
            arr = x[i, c].detach().cpu().numpy()
            img = ants.from_numpy(arr)
            try:
                img.set_spacing((float(native_spacing[0]), float(native_spacing[1])))
            except Exception:
                img.spacing = (float(native_spacing[0]), float(native_spacing[1]))
            img_r = ants.resample_image(img, (float(target_spacing[0]), float(target_spacing[1])),
                                        use_voxels=False, interp_type=0)
            xs.append(torch.from_numpy(img_r.numpy()).to(device=device, dtype=dtype))
        outs.append(torch.stack(xs, dim=0))
    y = torch.stack(outs, dim=1)  # (N,C,h,w)
    return y

@torch.no_grad()
def resample_with_ants_size(x: torch.Tensor,
                            target_size: Tuple[int, int],
                            native_spacing: Optional[Tuple[float, float]] = None) -> torch.Tensor:
    """
    Resample (N,C,H,W) to a target voxel size (H,W) using ANTsPy (use_voxels=True).
    If native_spacing is provided, attaches it before resampling (helpful for certain ANTs backends).
    """
    device, dtype = x.device, x.dtype
    N, C, H, W = x.shape
    outs = []
    for c in range(C):
        xs = []
        for i in range(N):
            arr = x[i, c].detach().cpu().numpy()
            img = ants.from_numpy(arr)
            if native_spacing is not None:
                try:
                    img.set_spacing((float(native_spacing[0]), float(native_spacing[1])))
                except Exception:
                    img.spacing = (float(native_spacing[0]), float(native_spacing[1]))
            img_r = ants.resample_image(img, (int(target_size[0]), int(target_size[1])),
                                        use_voxels=True, interp_type=0)
            xs.append(torch.from_numpy(img_r.numpy()).to(device=device, dtype=dtype))
        outs.append(torch.stack(xs, dim=0))
    y = torch.stack(outs, dim=1)  # (N,C,h,w)
    return y


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

@torch.no_grad()
def _prime_if_needed(model, H: int, W: int, device: torch.device):
    x = torch.zeros(1, 1, int(H), int(W), device=device, dtype=torch.float32)
    try:
        _ = model.inverse_and_log_det(x)
    except Exception:
        try:
            _ = model.log_prob(x)
        except Exception:
            pass

def _coerce_nchw_4d(x, target_hw=None):
    if isinstance(x, (list, tuple)):
        cands = [t for t in x if torch.is_tensor(t) and t.dim() in (3, 4)]
        if not cands:
            raise RuntimeError("Sample output is not a tensor.")
        areas, fixed = [], []
        for t in cands:
            if t.dim() == 3:
                if t.shape[-1] in (1, 3) and (t.shape[0] not in (1, 3)):
                    t = t.permute(2, 0, 1).contiguous()
                t = t.unsqueeze(0)
            elif t.dim() == 4 and t.shape[-1] in (1,3) and t.shape[1] not in (1,3):
                t = t.permute(0, 3, 1, 2).contiguous()
            fixed.append(t)
            areas.append(int(t.shape[-1]) * int(t.shape[-2]))
        x = fixed[int(torch.tensor(areas).argmax().item())]

    if not torch.is_tensor(x):
        raise RuntimeError(f"Unexpected sample output type: {type(x)}")

    if x.dim() == 3:
        if x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
            x = x.permute(2, 0, 1).contiguous()
        x = x.unsqueeze(0)
    if x.dim() == 4 and x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
        x = x.permute(0, 3, 1, 2).contiguous()

    if x.size(1) not in (1, 3):
        x = x.mean(dim=1, keepdim=True)
    x = x.float()
    try:
        if (x.amin() < 0.0) or (x.amax() > 1.0):
            x = to01(x)
    except Exception:
        pass

    if target_hw is not None:
        Ht, Wt = int(target_hw[0]), int(target_hw[1])
        H0, W0 = int(x.shape[-2]), int(x.shape[-1])
        if (H0, W0) != (Ht, Wt):
            x = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)
    return x

def save_grid(x: torch.Tensor, out_path: Path, nrow: int, target_hw: Tuple[int, int] | None):

    x = _coerce_nchw_4d(x, target_hw=target_hw)
    x = torch.clamp(x, 0.0, 1.0)
    
    out_path = Path(out_path)
    ext = "".join(out_path.suffixes).lower()

    if ".nii" in ext:
        import ants
        import numpy as np

        arr = x.detach().cpu().numpy()
        
        if arr.shape[0] > 1:
            arr_ants = np.transpose(arr.squeeze(1), (1, 2, 0))
        else:
            arr_ants = arr[0, 0]
            
        ants_img = ants.from_numpy(np.flip(arr_ants, axis=1)) 
        ants.image_write(ants_img, str(out_path))

    else:
        import torchvision as tv
        tv.utils.save_image(x, str(out_path), nrow=int(nrow))
      
# ---------------------- data loading helpers ----------------------

def _read_image_any(path: Path, slice_axis: int, slice_index: int) -> torch.Tensor:
    """
    Read a 2D image from disk and return a tensor (1,H,W) in float32 [0,1].
    - PNG/JPG/TIFF: read via PIL and scale to [0,1].
    - NIfTI (.nii/.nii.gz): read via ANTsPy, slice with ants.slice_image(image, axis, idx).
      If idx is out of bounds, it is clamped to the valid range. collapse_strategy=0.
    """
    path = Path(path)
    ext = path.suffix.lower()
    # Simple 2D images via PIL
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        img = Image.open(path).convert("L")
        arr = torch.from_numpy(np.array(img)).float() / 255.0
        return arr.unsqueeze(0)

    # Handle NIfTI via ANTs
    is_nii = (ext == ".nii") or (ext == ".gz" and path.name.endswith(".nii.gz"))
    if is_nii:
        img = ants.image_read(str(path))

        # Clamp slice index to valid range for the chosen axis
        try:
            shp = tuple(int(v) for v in img.shape)
        except Exception:
            # Fallback if shape is unavailable
            shp = None

        ax = int(slice_axis)
        idx = int(slice_index)
        if shp is not None and 0 <= ax < len(shp):
            if idx < 0: idx = 0
            if idx >= shp[ax]: idx = shp[ax] // 2  # middle if out of range
        try:
            img2d = ants.slice_image(img, axis=ax, idx=idx, collapse_strategy=0)
        except Exception:
            # Fallback: try middle slice along axis 0
            try:
                mid0 = shp[0] // 2 if (shp is not None and len(shp) > 0) else 0
                img2d = ants.slice_image(img, axis=0, idx=mid0, collapse_strategy=0)
            except Exception as e:
                raise RuntimeError(f"ANTs slice_image failed for axis={ax}, idx={idx}: {e}")

        arr = img2d.numpy()
        # Ensure 2D
        if arr.ndim > 2:
            arr = np.squeeze(arr)
            if arr.ndim > 2:
                arr = arr[..., 0]

        arr = torch.from_numpy(arr).float()
        # Robust normalize to [0,1]
        try:
            a_min = torch.quantile(arr, 0.01)
            a_max = torch.quantile(arr, 0.99)
            if a_max <= a_min:
                raise ValueError("degenerate quantiles")
            arr = torch.clamp((arr - a_min) / (a_max - a_min + 1e-8), 0.0, 1.0)
        except Exception:
            mn, mx = torch.min(arr), torch.max(arr)
            arr = (arr - mn) / (mx - mn + 1e-8) if (mx > mn) else torch.zeros_like(arr)
        return arr.unsqueeze(0)

    # Fallback: try PIL
    img = Image.open(path).convert("L")
    arr = torch.from_numpy(np.array(img)).float() / 255.0
    return arr.unsqueeze(0)

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


def build_model_from_config(cfg: dict, device: torch.device):
    H = int(cfg.get("H", 128))
    W = int(cfg.get("W", 128))
    input_shape = (1, H, W)
    m = create_glow_normalizing_flow_model_2d(
        input_shape=input_shape,
        L=int(cfg.get("L", 4)),
        K=int(cfg.get("K", 3)),
        hidden_channels=int(cfg.get("hidden", 96)),
        base=str(cfg.get("base", "glow")),
        glowbase_logscale_factor=float(cfg.get("glowbase_logscale_factor", 3.0)),
        glowbase_min_log=float(cfg.get("glowbase_min_log", -5.0)),
        glowbase_max_log=float(cfg.get("glowbase_max_log", 5.0)),
        split_mode="channel",
        scale=True,
        scale_map=str(cfg.get("scale_map", "tanh")),
        leaky=0.0,
        net_actnorm=bool(cfg.get("net_actnorm", False)),
        scale_cap=float(cfg.get("scale_cap", 2.0)),
    ).to(device).float().eval()
    if not hasattr(m, "input_shape"):
        m.input_shape = input_shape
    return m


# ---------------------- reconstruction sanity check ----------------------

# ---------------------- reconstruction sanity check ----------------------
@torch.no_grad()
def reconstruct_batch(model, xb: torch.Tensor):
    """
    Round-trip x -> z -> x_hat using MultiscaleFlow APIs.
    Uses model.inverse_and_log_det(x) to obtain latents, then decodes with
    model.forward_and_log_det(z_list) for a noise-free reconstruction.
    Returns x_hat tensor (N,1,H,W).
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
        return _coerce_nchw_4d(xh, target_hw=(xb.shape[-2], xb.shape[-1]))
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



def _encode_latents(model, xb: torch.Tensor) -> List[torch.Tensor]:
    """
    Push batch x -> multiscale latents z_list using the flow inverse.
    Returns a list of per-level tensors.
    """
    xb = _coerce_nchw_4d(xb, target_hw=(xb.shape[-2], xb.shape[-1]))
    device_type = xb.device.type
    with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=False):
        if hasattr(model, "inverse_and_log_det"):
            z, _ = model.inverse_and_log_det(xb)
        elif hasattr(model, "inverse"):
            z, _ = model.inverse(xb)
        else:
            raise RuntimeError("Model lacks inverse mapping (inverse_and_log_det / inverse).")
    return z if isinstance(z, (list, tuple)) else [z]


def _decode_latents(model, z_list: List[torch.Tensor], target_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Decode list of multiscale latents back to image space using the flow forward.
    """
    if not isinstance(z_list, (list, tuple)):
        z_list = [z_list]
    device = z_list[0].device
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
        if hasattr(model, "forward_and_log_det"):
            xh, _ = model.forward_and_log_det(z_list)
        else:
            raise RuntimeError("Model does not expose forward_and_log_det(z_list); cannot decode latents.")
    return _coerce_nchw_4d(xh, target_hw=target_hw)


def _edit_latents_to_mean_for_view(
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
    """
    For a given view, modify specified levels' latents.

    mode:
      - "mean":  replace with per-level Gaussian mean for that view.
      - "zero":  replace with zeros.
      - "pc":    add a shift along a principal component of the per-level, per-view covariance.

    PC editing:
      • Extract Σ_{ℓ,v} block for this view at each level ℓ.
      • Eigendecompose Σ_{ℓ,v} and take PC index `pc_index` (0 = largest variance).
      • Step size is pc_scale * sqrt(lambda_k).
      • Center:
          - "sample": z_edit = z + step * q_k
          - "mean":   z_edit = μ + step * q_k
    """
    import numpy as np

    if not levels_to_edit:
        return z_list

    # Validate Gaussian metadata and get canonical fields
    views, dims_tbl, shapes_by_view, L = _validate_gauss_blob(gauss_blob)

    try:
        v_idx = views.index(view_name)
    except ValueError:
        raise RuntimeError(
            f"[recon] View '{view_name}' not found in Gaussian 'views' header {views}."
        )

    if str(gauss_blob.get("mode", "perlevel")).lower() != "perlevel":
        raise RuntimeError(
            "[recon] Latent editing currently requires a perlevel Gaussian ('--cov-mode perlevel')."
        )

    mu_list = gauss_blob["mu"]  # list over levels
    Sigma_list = gauss_blob.get("Sigma", None)

    # Build per-level slice offsets for each view.
    raw_slices = gauss_blob.get("level_view_slices", None)
    level_view_slices: List[Dict[int, Tuple[int, int]]] = []

    V = len(views)
    if raw_slices is not None:
        for l in range(L):
            row = raw_slices[l]
            if isinstance(row, dict):
                # JSON round-trip typically stringifies keys
                row_int = {int(k): tuple(v) for k, v in row.items()}
            else:
                # Fallback: list-of-tuples in header order
                row_int = {vi: tuple(row[vi]) for vi in range(V)}
            level_view_slices.append(row_int)
    else:
        # Rebuild from dims_tbl (same logic as in gauss-impute)
        for l in range(L):
            off = 0
            row_int = {}
            for vi in range(V):
                d_raw = dims_tbl[vi][l]
                d = int(np.asarray(d_raw).item() if hasattr(d_raw, "item") else d_raw)
                row_int[vi] = (off, off + d)
                off += d
            level_view_slices.append(row_int)

    if len(z_list) != L:
        raise RuntimeError(
            f"[recon] Model has {len(z_list)} latent levels but Gaussian reports L={L}."
        )

    levels_set = {int(l) for l in levels_to_edit}
    z_out: List[torch.Tensor] = []

    for l, z_l in enumerate(z_list):
        if l not in levels_set:
            z_out.append(z_l)
            continue

        if z_l.ndim != 4:
            raise RuntimeError(
                f"[recon] Expected 4D latent at level {l}, got shape {tuple(z_l.shape)}."
            )

        B, C, H, W = z_l.shape
        Cg, Hg, Wg = shapes_by_view[v_idx][l]
        if (C, H, W) != (Cg, Hg, Wg):
            raise RuntimeError(
                f"[recon] Latent shape mismatch for view '{view_name}', level {l}: "
                f"model (C,H,W)=({C},{H},{W}) vs Gaussian ({Cg},{Hg},{Wg})."
            )

        a, b = level_view_slices[l][v_idx]

        # Per-level, per-view mean in flattened order
        mu_level = np.asarray(mu_list[l], dtype=np.float64).ravel()
        if b > mu_level.shape[0]:
            raise RuntimeError(
                f"[recon] Gaussian mean for level {l} is too short (len={mu_level.shape[0]}), "
                f"expected at least {b}."
            )

        mu_view_flat = mu_level[a:b]
        mu_view = torch.as_tensor(
            mu_view_flat, dtype=z_l.dtype, device=z_l.device
        ).view(1, C, H, W)

        if mode == "mean":
            # Replace with Gaussian mean
            z_l_edit = mu_view.expand(B, C, H, W)

        elif mode == "zero":
            # Replace with zeros
            z_l_edit = torch.zeros_like(z_l)

        elif mode == "pc":
            if Sigma_list is None:
                raise RuntimeError(
                    "[recon] Gaussian blob has no 'Sigma' field; cannot perform PC editing."
                )

            # Select Σ_l (per-level)
            Sigma_l = Sigma_list[l] if isinstance(Sigma_list, (list, tuple)) else Sigma_list

            # Build per-view covariance block Σ_{ℓ,v}
            Dv = C * H * W
            if isinstance(Sigma_l, dict) and Sigma_l.get("type") == "lowrank":
                U = np.asarray(Sigma_l["U"], dtype=np.float64)    # (D_total, r)
                eig = np.asarray(Sigma_l["eig"], dtype=np.float64)  # (r,)
                sigma2 = float(Sigma_l.get("sigma2", 0.0))

                U_v = U[a:b, :]  # (D_v, r)
                if U_v.shape[0] != Dv:
                    raise RuntimeError(
                        f"[recon] lowrank U slice has wrong length at level {l}, view '{view_name}'. "
                        f"expected {Dv}, got {U_v.shape[0]}"
                    )
                # Σ_{ℓ,v} = U_v diag(eig) U_v^T + σ² I  (computed without forming full Σ_l)
                Sv = (U_v * eig[np.newaxis, :]) @ U_v.T
                if sigma2 > 0.0:
                    Sv = Sv + sigma2 * np.eye(Dv, dtype=np.float64)
            else:
                S = np.asarray(Sigma_l, dtype=np.float64)
                if S.ndim == 1:
                    # Diagonal covariance: block is simply diag of the relevant entries
                    diag_v = S[a:b]
                    Sv = np.diag(diag_v)
                else:
                    Sv = S[a:b, a:b]

            # Symmetrize for numerical safety
            Sv = 0.5 * (Sv + Sv.T)

            # Eigendecomposition of per-view block
            w, V = np.linalg.eigh(Sv)  # ascending eigenvalues
            if w.size == 0:
                raise RuntimeError(f"[recon] Empty covariance block at level {l}, view '{view_name}'.")

            # pc_index = 0 => largest eigenvalue
            k = int(pc_index)
            if k < 0 or k >= w.size:
                raise RuntimeError(
                    f"[recon] pc_index={pc_index} out of range for level {l}, "
                    f"view '{view_name}' (dim={w.size})."
                )
            col = -1 - k  # 0 -> last, 1 -> second last, etc.
            direction_np = V[:, col]
            lam = float(max(w[col], 0.0))
            step = float(pc_scale) * (lam ** 0.5 if lam > 0.0 else 0.0)

            direction_t = torch.from_numpy(direction_np.astype(np.float32)).view(
                1, C, H, W
            ).to(z_l.device, z_l.dtype)

            if pc_center.lower() == "mean":
                base = mu_view.expand(B, C, H, W)
            else:  # "sample"
                base = z_l

            z_l_edit = base + step * direction_t

            print(
                f"[recon] level {l}, view '{view_name}': PC{pc_index} "
                f"lambda={lam:.3e}, step={step:.3e}, center={pc_center}"
            )

        elif mode == "pc_denoise":
            if Sigma_list is None:
                raise RuntimeError(
                    "[recon] Gaussian blob has no 'Sigma' field; cannot perform PC denoising."
                )

            # Select Σ_l (per-level)
            Sigma_l = Sigma_list[l] if isinstance(Sigma_list, (list, tuple)) else Sigma_list

            # Build per-view covariance block Σ_{ℓ,v}
            Dv = C * H * W
            if isinstance(Sigma_l, dict) and Sigma_l.get("type") == "lowrank":
                U = np.asarray(Sigma_l["U"], dtype=np.float64)    # (D_total, r)
                eig = np.asarray(Sigma_l["eig"], dtype=np.float64)  # (r,)
                sigma2 = float(Sigma_l.get("sigma2", 0.0))

                U_v = U[a:b, :]  # (D_v, r)
                if U_v.shape[0] != Dv:
                    raise RuntimeError(
                        f"[recon] lowrank U slice has wrong length at level {l}, view '{view_name}'. "
                        f"expected {Dv}, got {U_v.shape[0]}"
                    )
                # Σ_{ℓ,v} = U_v diag(eig) U_v^T + σ² I  (computed without forming full Σ_l)
                Sv = (U_v * eig[np.newaxis, :]) @ U_v.T
                if sigma2 > 0.0:
                    Sv = Sv + sigma2 * np.eye(Dv, dtype=np.float64)
            else:
                S = np.asarray(Sigma_l, dtype=np.float64)
                if S.ndim == 1:
                    # Diagonal covariance: block is simply diag of the relevant entries
                    diag_v = S[a:b]
                    Sv = np.diag(diag_v)
                else:
                    Sv = S[a:b, a:b]

            # Symmetrize for numerical safety
            Sv = 0.5 * (Sv + Sv.T)

            # Eigendecomposition of per-view block
            w, V = np.linalg.eigh(Sv)  # ascending eigenvalues
            if w.size == 0:
                raise RuntimeError(
                    f"[recon] Empty covariance block at level {l}, view '{view_name}'."
                )

            # Work in descending-variance order
            V_desc = V[:, ::-1]
            w_desc = w[::-1]

            # Number of PCs to preserve
            k_keep = int(pc_k)
            if k_keep < 0:
                k_keep = 0
            if k_keep > V_desc.shape[1]:
                k_keep = V_desc.shape[1]

            V_t = torch.from_numpy(V_desc.astype(np.float32)).to(z_l.device, z_l.dtype)  # (Dv, Dv)

            z_flat = z_l.view(B, -1)
            mu_flat = mu_view.view(1, -1)
            y = torch.matmul(z_flat - mu_flat, V_t)  # (B, Dv) in PC coordinates

            if k_keep < V_t.shape[1]:
                tail = y[:, k_keep:]
                if float(pc_beta) == 0.0:
                    y[:, k_keep:] = 0.0
                else:
                    y[:, k_keep:] = float(pc_beta) * tail

            z_flat_edit = mu_flat + torch.matmul(y, V_t.T)
            z_l_edit = z_flat_edit.view(B, C, H, W)

            print(
                f"[recon] level {l}, view '{view_name}': pc_denoise k_keep={k_keep}, tail_beta={pc_beta:.3f}"
            )

        else:
            raise ValueError(
                f"[recon] Unknown edit mode '{mode}', expected 'mean', 'zero', 'pc', or 'pc_denoise'."
            )

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

    def _prod3(t):
        try:
            c, h, w = int(t[0]), int(t[1]), int(t[2])
            return c * h * w
        except Exception:
            return None

    errors = []
    views = g.get("views", None)
    dims_tbl = g.get("dims_per_level_per_view", None)  # V × L
    shapes_by_view = g.get("shapes_by_view", None)     # V × L × (C,H,W)
    L_raw = g.get("L", None)

    # 1) Presence / types
    if not isinstance(views, (list, tuple)) or len(views) == 0 or not all(isinstance(v, str) for v in views):
        errors.append(f"- 'views' missing or invalid; expected non-empty list[str], got: {type(views).__name__} with len={_shape_of(views)}")

    if dims_tbl is None or not isinstance(dims_tbl, (list, tuple)):
        errors.append(f"- 'dims_per_level_per_view' missing or invalid; expected list[list[int]], got: {type(dims_tbl).__name__}")
    if shapes_by_view is None or not isinstance(shapes_by_view, (list, tuple)):
        errors.append(f"- 'shapes_by_view' missing or invalid; expected list[list[tuple(C,H,W)]], got: {type(shapes_by_view).__name__}")

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

    # 3) Per-level consistency: dims_tbl[v][ℓ] == C*H*W from shapes_by_view[v][ℓ]
    mismatches = []
    for vi in range(V):
        if vi in bad_dims_rows or vi in bad_shapes_rows:
            continue
        for l in range(L):
            try:
                d_tbl = int(_np.asarray(dims_tbl[vi][l]).item() if hasattr(dims_tbl[vi][l], "item") else dims_tbl[vi][l])
            except Exception:
                d_tbl = None
            d_shp = _prod3(shapes_by_view[vi][l])
            if d_tbl is None or d_shp is None or d_tbl != d_shp:
                mismatches.append((vi, l, d_tbl, d_shp))
                if len(mismatches) >= 20:
                    break
        if len(mismatches) >= 20:
            break
    if mismatches:
        msg = "\n".join([f"  - view[{vi}]='{views[vi]}', level {l}: dims_tbl={dt} vs C*H*W={ds}"
                        for (vi, l, dt, ds) in mismatches])
        errors.append(f"- dims_per_level_per_view does not match shapes_by_view for some entries (showing up to 20):\n{msg}")

    if errors:
        # Helpful footer with quick hints
        footer = (
            "\nHints:\n"
            "  • Re-run gauss-fit to regenerate the file if you changed model config (H/W, K, levels).\n"
            "  • Ensure --views in gauss-fit matches the manifest header order you expect to use in imputation.\n"
            "  • Verify that your serialized file includes the new fields written by the updated gauss-fit."
        )
        raise RuntimeError("[gauss] Inconsistent gaussian metadata:\n" + "\n".join(errors) + footer)

    # Normalize dims_tbl to pure Python ints (in case np types snuck in)
    dims_tbl_py = [[int(_np.asarray(d).item() if hasattr(d, "item") else d) for d in row] for row in dims_tbl]

    return views, dims_tbl_py, shapes_by_view, L


def make_recon_panel_with_edit(x: torch.Tensor, xh: torch.Tensor, xh_edit: torch.Tensor) -> torch.Tensor:
    """
    Create a 5-column panel per sample:
      [x | x_hat | x_hat_edit | abs(x-x_hat) | abs(x-x_hat_edit)].
    Input: (N,1,H,W) in [0,1]
    Output: (5N,1,H,W) suitable for save_grid with nrow=5
    """
    x = _coerce_nchw_4d(x, target_hw=(x.shape[-2], x.shape[-1]))
    xh = _coerce_nchw_4d(xh, target_hw=(x.shape[-2], x.shape[-1]))
    xh_edit = _coerce_nchw_4d(xh_edit, target_hw=(x.shape[-2], x.shape[-1]))

    diff_orig = torch.abs(x - xh)
    diff_edit = torch.abs(x - xh_edit)

    diff_orig_max = float(diff_orig.max().item())
    diff_edit_max = float(diff_edit.max().item())
    print(f"[info] recon: max |x - x_hat| = {diff_orig_max:.6f} (orig)")
    print(f"[info] recon: max |x - x_hat_edit| = {diff_edit_max:.6f} (edit)")

    diff_orig = to01(diff_orig)
    diff_edit = to01(diff_edit)

    panels = []
    for i in range(x.shape[0]):
        panels.append(x[i:i+1])
        panels.append(xh[i:i+1])
        panels.append(xh_edit[i:i+1])
        panels.append(diff_orig[i:i+1])
        panels.append(diff_edit[i:i+1])

    return torch.cat(panels, dim=0)

def resolve_ckpt_path(p: Path) -> Path:
    if p.is_dir():
        for name in ("training_state.pt", "checkpoint.pt", "ckpt.pt", "model.pt"):
            cand = p / name
            if cand.exists():
                return cand
        raise FileNotFoundError(f"No checkpoint found under directory: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p



# --------------------------- main ---------------------------
def main_sample():
    ap = argparse.ArgumentParser("LAM‑Flow sample grid tool")
    ap.add_argument("--version", action="store_true", help="Print version and exit")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file or directory")
    ap.add_argument("--view-index", type=int, default=0, help="Which view to sample (0-based)")
    ap.add_argument("--sample-grid-size", type=parse_mn, default=None, help="Grid as MxN (rows×cols), e.g., 6x8")
    ap.add_argument("--image-size", type=parse_hw, default="128x128", help="Per-tile size HxW for saved PNG")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (prior noise scale)")
    ap.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True, help="Prefer EMA weights if present")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed for reproducible sampling")
    ap.add_argument("--devices", type=str, default="cuda:0", help='Device like "cuda:0" or "cpu"')
    ap.add_argument("--sample-grid-out", type=str, default="", help="Output PNG filename (default auto next to ckpt)")
    ap.add_argument("--slice-axis", type=int, default=0, help="Axis for NIfTI slicing with ANTs (default: 0)")
    ap.add_argument("--slice-index", type=int, default=120, help="Slice index for NIfTI slicing with ANTs (default: 120)")
    # Reconstruction sanity check options
    ap.add_argument("--recon", type=int, default=0, help="If >0, run reconstruction sanity check on N validation images.")
    ap.add_argument("--val-list", type=str, nargs="+", default=None, help="One or more inputs: globs (quoted) and/or files (txt lists or image files). Examples: '/data/*/T1.nii.gz' or list.txt")
    
    
    ap.add_argument("--recon-out", type=str, default="", help="Output PNG for recon panel (default auto next to ckpt).")

    # ANTs resampling options
    ap.add_argument("--resample-spacing", type=parse_hw_float, default=None,
                    help="Physical spacing SxT (e.g., 0.8x0.8 mm). Uses ANTs resample_image(use_voxels=False).")
    ap.add_argument("--resample-size", type=parse_hw, default=None,
                    help="Voxel size HxW (e.g., 192x192). Uses ANTs resample_image(use_voxels=True).")
    ap.add_argument("--native-spacing", type=parse_hw_float, default=None,
                    help="Override native spacing SxT (mm) if not present in checkpoint config.")

    args = ap.parse_args()
    if args.version:
        print(__version__)
        return

    # sanity: disallow specifying both spacing & size simultaneously
    if args.resample_spacing is not None and args.resample_size is not None:
        raise SystemExit("Specify only one of --resample-spacing or --resample-size (not both).")

    ckpt_path = resolve_ckpt_path(Path(args.ckpt))
    out_dir = ckpt_path.parent

    device = torch.device("cpu") if args.devices.lower() == "cpu" else torch.device(args.devices.split(",")[0])
    set_deterministic(args.seed)

    print(f"[info] lamnr_glow_tool {__version__}")
    print(f"[info] loading checkpoint: {ckpt_path}")

    # torch.load with safe default when possible
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=True)  # PyTorch >=2.5
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)

    cfg = blob.get("config", {})
    model = build_model_from_config(cfg if cfg else {"H": args.image_size[0], "W": args.image_size[1]}, device=device)

    # Determine native spacing from ckpt if available or CLI override
    native_spacing = None
    if args.native_spacing is not None:
        native_spacing = (float(args.native_spacing[0]), float(args.native_spacing[1]))
    else:
        for key in ("spacing", "pixdim", "voxel_spacing", "voxel_size", "voxel_spacing_xy"):
            if key in cfg:
                val = cfg[key]
                try:
                    if isinstance(val, (list, tuple)) and len(val) >= 2:
                        native_spacing = (float(val[0]), float(val[1]))
                        break
                    if isinstance(val, dict) and "x" in val and "y" in val:
                        native_spacing = (float(val["x"]), float(val["y"]))
                        break
                except Exception:
                    pass
    if native_spacing is None:
        native_spacing = (1.0, 1.0)

    ok, src_note = load_weights_into_model(model, blob, view_idx=int(args.view_index), prefer_ema=bool(args.ema))
    if not ok:
        raise RuntimeError(f"Could not load weights from checkpoint ({src_note})")
    which_src, note = src_note
    if note:
        print(f"[warn] weight load note: {note}")
    print(f"[info] weights loaded from: {which_src} (view {args.view_index})")

    # Prime using ckpt-native spatial size
    Hc, Wc = model.input_shape[-2], model.input_shape[-1]
    _prime_if_needed(model, Hc, Wc, device)

    if (args.image_size[0], args.image_size[1]) != (Hc, Wc):
        msg = (f"[warn] Requested --image-size {args.image_size[0]}x{args.image_size[1]} differs from ckpt-native "
               f"{Hc}x{Wc}. Sampling at ckpt size; any ANTs resampling will be applied after sampling, "
               f"then the grid tiles will be resized to --image-size for saving.")
        print(msg)


    # Reconstruction sanity check (optional)
    if int(args.recon) > 0:
        val_paths = _gather_val_paths(getattr(args, "val_list", None), limit=int(args.recon))
        if not val_paths:
            raise SystemExit("Recon requested but no validation images found. Use --val-list or --val-dir/--val-glob.")
        print(f"[info] recon: loading {len(val_paths)} image(s) for round-trip test")
        xs = []
        for pth in val_paths:
            try:
                xi = _read_image_any(pth, args.slice_axis, args.slice_index)  # (1,H,W) in [0,1]
            except Exception as e:
                print(f"[warn] skipping {pth}: {e}")
                continue
            # Resize to ckpt-native size for a valid bijective mapping
            xi = F.interpolate(xi.unsqueeze(0), size=(Hc, Wc), mode="bilinear", align_corners=False).squeeze(0)
            xs.append(xi)
        if not xs:
            raise SystemExit("Recon: no readable images after parsing inputs.")
        xb = torch.stack(xs, dim=0).to(device=device, dtype=torch.float32)  # (N,1,Hc,Wc)
        # Forward -> latents -> reconstruct
        try:
            xh = reconstruct_batch(model, xb)
        except Exception as e:
            raise SystemExit(f"Recon failed: {e}")
        # Build 3-column panel
        panel = make_recon_panel(xb, xh)  # (3N,1,Hc,Wc)
        # Determine output path
        if args.recon_out:
            recon_out = Path(args.recon_out)
            if not recon_out.is_absolute():
                recon_out = out_dir / recon_out
        else:
            recon_out = out_dir / f"recon_view{int(args.view_index)}_N{panel.shape[0]//3}_{Hc}x{Wc}.png"
        save_grid(panel, recon_out, nrow=3, target_hw=(Hc, Wc))
        print(f"[ok] recon panel saved: {recon_out}")

    # Sample
    if args.sample_grid_size is not None:

        M, N = args.sample_grid_size
        total = int(M) * int(N)
        print(f"[info] sampling {total} images @ temp={args.temperature} as {M}x{N}")

        try:
            s = sample_with_temperature(model, total, float(args.temperature))
            x = s[0] if isinstance(s, (list, tuple)) else s
        except Exception as e:
            raise RuntimeError(f"sampling failed: {e}")

        # Optional ANTs resampling
        if args.resample_spacing is not None:
            target_spacing = (float(args.resample_spacing[0]), float(args.resample_spacing[1]))
            if tuple(round(s, 6) for s in target_spacing) != tuple(round(s, 6) for s in native_spacing):
                print(f"[info] ANTs resample by spacing: {native_spacing} -> {target_spacing}")
                x = resample_with_ants_spacing(x, native_spacing=native_spacing,
                                            target_spacing=target_spacing)
            else:
                print("[info] Requested spacing equals native spacing; skipping ANTs resampling.")
        elif args.resample_size is not None:
            target_size = (int(args.resample_size[0]), int(args.resample_size[1]))
            if tuple(target_size) != (int(x.shape[-2]), int(x.shape[-1])):
                print(f"[info] ANTs resample by voxel size: {(int(x.shape[-2]), int(x.shape[-1]))} -> {target_size}")
                x = resample_with_ants_size(x, target_size=target_size, native_spacing=native_spacing)
            else:
                print("[info] Requested voxel size equals current; skipping ANTs resampling.")

        # Compose and save grid
        if args.sample_grid_out:
            out_path = Path(args.sample_grid_out)
            if not out_path.is_absolute():
                out_path = out_dir / out_path
        else:
            it = blob.get("iter", None)
            it_str = (f"_it{int(it)-1:06d}" if isinstance(it, int) and it > 0 else "")
            out_name = (f"samples{it_str}_view{int(args.view_index)}_temp{float(args.temperature):.2f}_"
                        f"{int(M)}x{int(N)}_{int(args.image_size[0])}x{int(args.image_size[1])}.png")
            out_path = out_dir / out_name

        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_grid(x, out_path, nrow=int(N), target_hw=(int(args.image_size[0]), int(args.image_size[1])))
        print(f"[ok] wrote: {out_path}")

        # Metadata JSON
        meta = {
            "version": __version__,
            "ckpt": str(ckpt_path),
            "weights_source": which_src,
            "view_index": int(args.view_index),
            "sample_grid_size": [int(M), int(N)],
            "image_size_saved": [int(args.image_size[0]), int(args.image_size[1])],
            "ckpt_native_size": [int(Hc), int(Wc)],
            "temperature": float(args.temperature),
            "seed": int(args.seed),
            "devices": args.devices,
            "out": str(out_path),
            "iter": int(blob.get("iter", -1)) if isinstance(blob.get("iter", None), int) else None,
            "config": cfg if isinstance(cfg, dict) else None,
            "native_spacing": list(native_spacing) if native_spacing is not None else None,
            "resample_spacing": list(args.resample_spacing) if args.resample_spacing is not None else None,
            "resample_size": list(args.resample_size) if args.resample_size is not None else None
        }
        try:
            with open(out_path.with_suffix(".json"), "w") as f:
                json.dump(meta, f, indent=2)
            print(f"[ok] wrote: {out_path.with_suffix('.json')}")
        except Exception as e:
            print(f"[warn] could not write metadata json: {e}")



def main_recon(argv=None):
    ap = argparse.ArgumentParser("LAM‑Flow reconstruction tool (recon)")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file or directory")
    ap.add_argument("--manifest", type=str, required=True, help="CSV with per-view file paths")
    ap.add_argument("--views", type=str, required=True, help="Comma list of views to reconstruct (e.g., T1,T2,FA)")
    ap.add_argument("--view-index", type=int, default=0, help="Which single view's weights to load (0-based)")
    ap.add_argument("--slice-axis", type=int, required=True)
    ap.add_argument("--slice-index", type=int, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--devices", type=str, default="cuda:0")
    ap.add_argument("--out", type=str, required=True, help="Output PNG panel")
    ap.add_argument("--gauss", type=str, default=None,
                    help="Optional Gaussian model (.npz or .pt) from gauss-fit for latent editing.")
    ap.add_argument("--edit-levels", type=str, default="none",
                    help="Levels to project to the Gaussian mean, e.g. '0', '0,1,2', '0-2,4', or 'all'. "
                         "'none' (default) disables latent editing.")
    ap.add_argument("--edit-what", type=str, choices=["mean", "zero", "pc", "pc_denoise"], default="mean",
                    help=("What to insert at selected levels: Gaussian mean ('mean'), zeros ('zero'), "
                          "or a principal-component shift ('pc')."))
    ap.add_argument("--edit-pc-index", type=int, default=0, 
                    help="Principal component index (0 = largest variance) within the selected view/level (for --edit-what pc).")
    ap.add_argument("--edit-pc-scale", type=float, default=2.0,
                    help="Scale k in ±k·σ along the chosen PC (for --edit-what pc). Use a negative value for the opposite direction.")
    ap.add_argument("--edit-pc-center", type=str, choices=["sample", "mean"], default="sample",
                    help=("Center for PC editing: 'sample' adds the PC shift to each sample's latent; "
                          "'mean' starts from the Gaussian mean."))
    ap.add_argument("--edit-pc-k", type=int, default=64,
                    help=("For 'pc_denoise': number of top principal components to preserve; "
                          "the remaining PCs will be shrunk."))
    ap.add_argument("--edit-pc-beta", type=float, default=0.0,
                    help=("For 'pc_denoise': shrink factor for tail PCs (0 = full projection, "
                          "1 = no change in tail)."))
    args = ap.parse_args(argv)

    device = torch.device(args.devices)
    ckpt_path = resolve_ckpt_path(Path(args.ckpt))
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)

    cfg = blob.get("config", {})
    Hc = int(cfg.get("H", 128))
    Wc = int(cfg.get("W", 128))
    model = build_model_from_config(cfg if cfg else {"H": Hc, "W": Wc}, device=device)
    model.eval()

    manifest_path = Path(args.manifest)
    with open(manifest_path, "r") as f:
        header = [h.strip() for h in f.readline().strip().split(",")]
        all_views = [v.strip() for v in args.views.split(",") if v.strip()]
        if not set(all_views).issubset(set(header)):
            missing = [v for v in all_views if v not in header]
            raise RuntimeError(f"Views {missing} not found in manifest header: {header}")
        v_idx_map = {v: header.index(v) for v in all_views}
        rows = []
        for line in f:
            parts = [s.strip() for s in line.strip().split(",")]
            if not parts or all(p == "" for p in parts):
                continue
            paths = [Path(parts[v_idx_map[v]]) for v in all_views]
            rows.append(paths)

    if not rows:
        raise RuntimeError(f"No valid rows found in manifest: {manifest_path}")

    vname = all_views[int(args.view_index)]
    vcol = [r[all_views.index(vname)] for r in rows]

    ok, note = load_weights_into_model(
        model,
        blob,
        view_idx=all_views.index(vname),
        prefer_ema=True,
        view_name=vname,
        cfg_views=all_views,
    )
    if not ok:
        raise RuntimeError(f"Failed to load weights for view '{vname}': {note}")

    xs = []
    bs = max(1, int(args.batch))
    for pth in vcol[:bs]:
        xi = _read_image_any(pth, int(args.slice_axis), int(args.slice_index))
        xi = torch.nn.functional.interpolate(
            xi.unsqueeze(0),
            size=(Hc, Wc),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        xi = to01(xi.unsqueeze(0)).squeeze(0)
        xs.append(xi)

    xb = torch.stack(xs, dim=0).to(device=device, dtype=torch.float32)

    # Base reconstruction (x -> z -> x_hat) for this view
    z_list = _encode_latents(model, xb)
    xh = _decode_latents(model, z_list, target_hw=(Hc, Wc))

    # Optional Gaussian-based latent editing
    xh_edit = None
    edit_levels = None
    edit_mode = None
    gauss_path = None

    if args.gauss:
        gauss_path = Path(args.gauss)
        gauss_blob = _load_gaussian_model(gauss_path)

        # Parse level specification
        spec = (args.edit_levels or "none").strip().lower()
        if spec not in ("none", ""):
            # Determine how many levels are available from the Gaussian blob
            try:
                _, _, _, L = _validate_gauss_blob(gauss_blob)
            except Exception as e:
                raise RuntimeError(f"[recon] Invalid Gaussian blob for editing: {e}")

            if spec == "all":
                levels = list(range(L))
            else:
                levels = []
                for part in spec.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if "-" in part:
                        a_str, b_str = part.split("-", 1)
                        a = int(a_str)
                        b = int(b_str)
                        if a > b:
                            a, b = b, a
                        levels.extend(list(range(a, b + 1)))
                    else:
                        levels.append(int(part))
                # Clamp to valid level range
                if max(levels) >= L:
                    raise ValueError(f"edit-level exceeded max number of levels L={L}")
                levels = sorted({l for l in levels if 0 <= l < L})

            if levels:
                z_list_edit = _edit_latents_to_mean_for_view(
                    z_list,
                    gauss_blob,
                    vname,
                    levels_to_edit=levels,
                    mode=args.edit_what,
                    pc_index=getattr(args, "edit_pc_index", 0),
                    pc_scale=getattr(args, "edit_pc_scale", 2.0),
                    pc_center=getattr(args, "edit_pc_center", "sample"),
                    pc_k=getattr(args, "edit_pc_k", 64),
                    pc_beta=getattr(args, "edit_pc_beta", 0.0),
                )
                xh_edit = _decode_latents(model, z_list_edit, target_hw=(Hc, Wc))
                edit_levels = levels
                edit_mode = args.edit_what

    outp = Path(args.out)
    if xh_edit is None:
        panel = make_recon_panel(xb, xh)
        ncol = 3
        diff = torch.abs(xb - xh)
        max_abs_diff = float(diff.max().item())
        meta_edit = None
    else:
        panel = make_recon_panel_with_edit(xb, xh, xh_edit)
        ncol = 5
        diff_orig = torch.abs(xb - xh)
        diff_edit = torch.abs(xb - xh_edit)
        max_abs_diff = float(diff_orig.max().item())
        max_abs_diff_edit = float(diff_edit.max().item())
        meta_edit = {
            "gauss": str(gauss_path) if gauss_path is not None else None,
            "edit_levels": edit_levels,
            "edit_mode": edit_mode,
            "max_abs_diff_edit": max_abs_diff_edit,
        }
        if edit_mode == "pc":
            meta_edit["edit_pc_index"] = getattr(args, "edit_pc_index", 0)
            meta_edit["edit_pc_scale"] = getattr(args, "edit_pc_scale", 2.0)
            meta_edit["edit_pc_center"] = getattr(args, "edit_pc_center", "sample")

    save_grid(panel, outp, nrow=ncol, target_hw=(Hc, Wc))
    print(f"[recon] wrote {outp}")

    # Metadata sidecar JSON
    try:
        meta = {
            "tool": "lamnr_glow_tool",
            "mode": "recon",
            "version": __version__,
            "ckpt": str(ckpt_path),
            "manifest": str(manifest_path),
            "view": str(vname),
            "view_index": int(args.view_index),
            "slice_axis": int(args.slice_axis),
            "slice_index": int(args.slice_index),
            "batch": int(len(xs)),
            "Hc": int(Hc),
            "Wc": int(Wc),
            "ncol": int(ncol),
            "max_abs_diff": max_abs_diff,
        }
        if meta_edit is not None:
            meta["latent_edit"] = meta_edit
        meta_path = outp.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[recon] wrote metadata {meta_path}")
    except Exception as e:
        print(f"[warn] could not write recon metadata json: {e}")

    return 0
# ---------------------- gauss-fit (conditional Gaussian) ----------------------
def _read_manifest_csv(manifest_path: Path) -> Dict[str, List[str]]:
    """
    Read a CSV manifest where each row is one subject and each column is a view.
    Returns a dict: {header_name: [paths...]} with equal-length lists.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", newline="") as f:
        rdr = csv.reader(f)
        rows = list(rdr)
    if not rows:
        raise RuntimeError("Manifest is empty.")
    header = rows[0]
    if any(h is None or h.strip() == "" for h in header):
        raise RuntimeError("Manifest header has empty column name(s).")
    cols = {h: [] for h in header}
    for i, r in enumerate(rows[1:], start=2):
        if len(r) != len(header):
            raise RuntimeError(f"Row {i} has {len(r)} cells but header has {len(header)} columns.")
        for h, v in zip(header, r):
            cols[h].append(v.strip())
    return cols

def _resolve_views(cols: Dict[str, List[str]], manifest_dir: Path, views_cli: str | None) -> Tuple[List[str], List[List[Path]]]:
    # Decide which columns are views and resolve to absolute Paths
    if views_cli is None or views_cli.strip() == "":
        view_names = list(cols.keys())
    else:
        view_names = [v.strip() for v in views_cli.split(",") if v.strip()]
        for v in view_names:
            if v not in cols:
                raise RuntimeError(f"--views specified '{v}', but it's not in manifest header.")
    # Build per-view list of Paths; enforce non-empty and existence
    per_view_paths: List[List[Path]] = []
    for v in view_names:
        paths_v: List[Path] = []
        for s in cols[v]:
            if s == "":
                raise RuntimeError(f"Manifest has a blank cell under view '{v}'.")
            pth = Path(s)
            if not pth.is_absolute():
                pth = (manifest_dir / pth).resolve()
            if not pth.exists() or not pth.is_file():
                raise FileNotFoundError(f"Missing/unreadable file for view '{v}': {pth}")
            paths_v.append(pth)
        per_view_paths.append(paths_v)
    # Ensure equal subject count
    n_set = {len(x) for x in per_view_paths}
    if len(n_set) != 1:
        raise RuntimeError(f"Views have inconsistent row counts: {[len(x) for x in per_view_paths]}")
    return view_names, per_view_paths

def _flatten_latents_by_level(z_list) -> List:
    """
    Input: list of tensors or a single tensor; each tensor shape (B, C, H, W) or (B, D).
    Output: list of (B, D_l) 2-D tensors per level.
    """
    if not isinstance(z_list, (list, tuple)):
        z_list = [z_list]
    outs = []
    for z in z_list:
        if z.ndim == 4:
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

def _lowrank_from_Xc(Xc: np.ndarray, rank: int, sigma2: float | str, extra_ridge: float) -> dict:
    N, D = Xc.shape
    rmax = min(D, max(1, N - 1))
    r = int(max(1, min(rank, rmax)))
    Ux, Svals, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigs_all = (Svals ** 2) / max(1, (N - 1))
    eig_r = eigs_all[:r].copy()
    U = Vt[:r, :].T.copy()
    if isinstance(sigma2, str) and sigma2.lower() == "auto":
        if eigs_all.shape[0] > r:
            sigma2_val = float(np.maximum(np.mean(eigs_all[r:]), 0.0))
        else:
            sigma2_val = 0.0
    else:
        sigma2_val = float(sigma2)
    sigma2_val = float(max(0.0, sigma2_val)) + float(max(0.0, extra_ridge))
    return {"type":"lowrank", "U": U, "eig": eig_r, "sigma2": sigma2_val}

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


def main_recon_template(argv=None):
    """
    Reconstruct a latent-space template for a single view using a Gaussian model
    from `gauss-fit`. The base template is decode(mu) for that view. Optionally,
    draw Monte Carlo samples in latent space and average their reconstructions.
    """
    ap = argparse.ArgumentParser("LAM-Flow latent template reconstruction (recon-template)")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file or directory")
    ap.add_argument("--gauss", type=str, required=True, help="Gaussian model (.npz or .pt) from gauss-fit")
    ap.add_argument("--views", type=str, required=True,
                    help="Comma list of views matching training (e.g., T1,T2,FA)")
    ap.add_argument("--view-index", type=int, default=0,
                    help="Which view to use (0-based index into --views)")
    ap.add_argument("--devices", type=str, default="cuda:0",
                    help='Device like "cuda:0" or "cpu"')
    ap.add_argument("--out", type=str, required=True, help="Output PNG filename")
    ap.add_argument("--mc-samples", type=int, default=0,
                    help="If >0, draw this many Monte Carlo samples in latent space and "
                         "average their reconstructions in image space.")
    ap.add_argument("--mc-temp", type=float, default=1.0, help="Monte Carlo temperature.")
    ap.add_argument("--seed", type=int, default=12345,
                    help="Random seed used when --mc-samples > 0")
    args = ap.parse_args(argv)

    device = torch.device(args.devices)

    # Optionally fix randomness for Monte Carlo sampling
    mc_n = max(0, int(args.mc_samples))
    if mc_n > 0:
        set_deterministic(int(args.seed))

    # ------------------------------- checkpoint --------------------------------
    ckpt_path = resolve_ckpt_path(Path(args.ckpt))
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)

    cfg = blob.get("config", {}) or {}
    Hc = int(cfg.get("H", 128))
    Wc = int(cfg.get("W", 128))

    model = build_model_from_config(cfg if cfg else {"H": Hc, "W": Wc}, device=device)
    model.eval()
    _prime_if_needed(model, Hc, Wc, device=device)

    cfg_views = list(cfg.get("views", [])) if isinstance(cfg.get("views"), (list, tuple)) else None

    views_cli = [v.strip() for v in str(args.views).split(",") if v.strip()]
    if not views_cli:
        raise RuntimeError("[recon-template] --views must list at least one view name.")

    # Prefer config views if present and consistent in length
    if cfg_views and len(cfg_views) == len(views_cli):
        all_views = cfg_views
    else:
        all_views = views_cli

    if not (0 <= int(args.view_index) < len(all_views)):
        raise RuntimeError(
            f"[recon-template] --view-index {args.view_index} is out of range for views {all_views}."
        )
    vname = all_views[int(args.view_index)]

    ok, note = load_weights_into_model(
        model,
        blob,
        view_idx=all_views.index(vname),
        prefer_ema=True,
        view_name=vname,
        cfg_views=all_views,
    )
    if not ok:
        raise RuntimeError(f"[recon-template] Failed to load weights for view '{vname}': {note}")

    print(f"[recon-template] using view '{vname}' on device {device}")

    # ------------------------------- Gaussian ----------------------------------
    gauss_blob = _load_gaussian_model(Path(args.gauss))
    views_g, dims_tbl, shapes_by_view, L = _validate_gauss_blob(gauss_blob)

    if vname not in views_g:
        raise RuntimeError(
            f"[recon-template] View '{vname}' not present in Gaussian header views={views_g}."
        )
    v_idx = views_g.index(vname)

    if str(gauss_blob.get("mode", "perlevel")).lower() != "perlevel":
        raise RuntimeError(
            "[recon-template] currently requires a per-level Gaussian ('--cov-mode perlevel')."
        )

    mu_list = gauss_blob.get("mu", None)
    if mu_list is None:
        raise RuntimeError("[recon-template] Gaussian blob has no 'mu' field.")

    # Build per-level slice offsets for each view, matching gauss-fit logic
    raw_slices = gauss_blob.get("level_view_slices", None)
    level_view_slices: List[Dict[int, Tuple[int, int]]] = []

    V = len(views_g)
    if raw_slices is not None:
        for l in range(L):
            row = raw_slices[l]
            if isinstance(row, dict):
                # JSON round-trip typically stringifies keys
                row_int = {int(k): tuple(v) for k, v in row.items()}
            else:
                # Fallback: list-of-tuples in header order
                row_int = {vi: tuple(row[vi]) for vi in range(V)}
            level_view_slices.append(row_int)
    else:
        # Rebuild from dims_tbl
        for l in range(L):
            off = 0
            row_int = {}
            for vi in range(V):
                d_raw = dims_tbl[vi][l]
                d = int(np.asarray(d_raw).item() if hasattr(d_raw, "item") else d_raw)
                row_int[vi] = (off, off + d)
                off += d
            level_view_slices.append(row_int)

    # ------------------------- construct mean latents --------------------------
    z_mu_list: List[torch.Tensor] = []
    for l in range(L):
        Cg, Hg, Wg = shapes_by_view[v_idx][l]
        a, b = level_view_slices[l][v_idx]

        mu_level = np.asarray(mu_list[l], dtype=np.float64).ravel()
        if b > mu_level.shape[0]:
            raise RuntimeError(
                f"[recon-template] Gaussian mean for level {l} is too short (len={mu_level.shape[0]}), "
                f"expected at least {b}."
            )
        mu_view_flat = mu_level[a:b]
        if mu_view_flat.shape[0] != Cg * Hg * Wg:
            raise RuntimeError(
                f"[recon-template] Level {l} mean slice for view '{vname}' has length {mu_view_flat.shape[0]}, "
                f"expected {Cg * Hg * Wg} from shapes_by_view."
            )

        mu_view = torch.from_numpy(mu_view_flat.astype(np.float32)).to(device=device)
        z_mu_list.append(mu_view.view(1, Cg, Hg, Wg))

    # Decode mean template
    x_mu = _decode_latents(model, z_mu_list, target_hw=(Hc, Wc))
    x_mu = to01(x_mu)

    # ----------------------- Monte Carlo latent sampling -----------------------
    x_mc_mean = None
    if mc_n > 0:
        Sigma_list = gauss_blob.get("Sigma", None)
        if Sigma_list is None:
            raise RuntimeError(
                "[recon-template] Gaussian blob has no 'Sigma' field; cannot run Monte Carlo sampling."
            )

        def _sample_gaussian_block(mu_flat: np.ndarray, Sigma_block, n: int, jitter: float = 1e-6, temperature: float=1.0) -> np.ndarray:
            """
            Sample z ~ N(mu_flat, Sigma_block) for a single view/level block.
            Sigma_block may be:
              - dict(type='lowrank', U (D×r), eig (r), sigma2)
              - 1D diag vector
              - 2D full covariance matrix
            Returns array of shape (n, D).
            """
            mu = np.asarray(mu_flat, dtype=np.float64).reshape(-1)
            D = mu.shape[0]
            if D == 0:
                return np.zeros((n, 0), dtype=np.float64)

            # Low-rank parameterization
            if isinstance(Sigma_block, dict) and Sigma_block.get("type") == "lowrank":
                U = np.asarray(Sigma_block["U"], dtype=np.float64)   # (D, r)
                eig = np.asarray(Sigma_block["eig"], dtype=np.float64).reshape(-1)  # (r,)
                eig = eig * (temperature ** 2)
                sigma2 = float(Sigma_block.get("sigma2", 0.0))
                sigma2 = sigma2 * (temperature ** 2)
                if U.shape[0] != D:
                    raise RuntimeError(
                        f"[recon-template] lowrank U has wrong number of rows (got {U.shape[0]}, expected {D})."
                    )
                r = U.shape[1]
                if eig.shape[0] != r:
                    raise RuntimeError(
                        f"[recon-template] lowrank eig has length {eig.shape[0]}, expected {r}."
                    )

                # mu + U diag(sqrt(eig)) xi + sqrt(sigma2) eps
                xi = np.random.randn(r, n)
                A = U * np.sqrt(np.clip(eig, a_min=0.0, a_max=None))[np.newaxis, :]
                z = mu[:, None] + A @ xi
                if sigma2 > 0.0:
                    eps = np.random.randn(D, n)
                    z = z + math.sqrt(max(sigma2, 0.0)) * eps
                return z.T

            # Diagonal covariance
            S = np.asarray(Sigma_block, dtype=np.float64)
            if S.ndim == 1:
                var = np.clip(S, a_min=0.0, a_max=None)
                var = var * (temperature ** 2)
                std = np.sqrt(var + float(jitter))
                eps = np.random.randn(n, D)
                return (mu[None, :] + eps * std[None, :])

            # Full covariance
            if S.ndim != 2:
                raise RuntimeError(f"[recon-template] Sigma_block has unexpected ndim={S.ndim} (expected 1 or 2).")
            if S.shape[0] != S.shape[1] or S.shape[0] != D:
                raise RuntimeError(
                    f"[recon-template] Sigma_block shape {S.shape} incompatible with D={D}."
                )
            S = 0.5 * (S + S.T)
            S = S * (temperature ** 2)
            I = np.eye(D, dtype=np.float64)
            jj = float(jitter)
            L = None
            for _ in range(7):
                try:
                    L = np.linalg.cholesky(S + jj * I)
                    break
                except np.linalg.LinAlgError:
                    jj *= 10.0
            if L is None:
                # Eigen fallback
                w, V = np.linalg.eigh(S)
                w_clamped = np.clip(w, a_min=1e-12, a_max=None)
                L = (V * np.sqrt(w_clamped)[np.newaxis, :]) @ V.T
            eps = np.random.randn(D, n)
            z = mu[:, None] + L @ eps
            return z.T

        z_mc_list: List[torch.Tensor] = []
        Sigma_mode = Sigma_list
        for l in range(L):
            Cg, Hg, Wg = shapes_by_view[v_idx][l]
            Dv = Cg * Hg * Wg
            a, b = level_view_slices[l][v_idx]

            mu_level = np.asarray(mu_list[l], dtype=np.float64).ravel()
            if b > mu_level.shape[0]:
                raise RuntimeError(
                    f"[recon-template] Gaussian mean for level {l} is too short (len={mu_level.shape[0]}), "
                    f"expected at least {b}."
                )
            mu_view_flat = mu_level[a:b]
            if mu_view_flat.shape[0] != Dv:
                raise RuntimeError(
                    f"[recon-template] Level {l} mean slice for view '{vname}' has length {mu_view_flat.shape[0]}, "
                    f"expected {Dv}."
                )

            # Select per-level Sigma and restrict to this view block
            Sigma_l = Sigma_mode[l] if isinstance(Sigma_mode, (list, tuple)) else Sigma_mode
            if isinstance(Sigma_l, dict) and Sigma_l.get("type") == "lowrank":
                U_full = np.asarray(Sigma_l["U"], dtype=np.float64)
                eig = np.asarray(Sigma_l["eig"], dtype=np.float64)
                sigma2 = float(Sigma_l.get("sigma2", 0.0))
                U_v = U_full[a:b, :]
                if U_v.shape[0] != Dv:
                    raise RuntimeError(
                        f"[recon-template] lowrank U slice has wrong length at level {l}, view '{vname}'. "
                        f"expected {Dv}, got {U_v.shape[0]}"
                    )
                Sigma_block = {"type": "lowrank", "U": U_v, "eig": eig, "sigma2": sigma2}
            else:
                S_full = np.asarray(Sigma_l, dtype=np.float64)
                if S_full.ndim == 1:
                    Sigma_block = S_full[a:b]
                else:
                    Sigma_block = S_full[a:b, a:b]

            z_samples_flat = _sample_gaussian_block(mu_view_flat, Sigma_block, mc_n, temperature=args.mc_temp)
            if z_samples_flat.shape != (mc_n, Dv):
                raise RuntimeError(
                    f"[recon-template] sampled block has shape {z_samples_flat.shape}, expected ({mc_n}, {Dv})."
                )

            z_samples = torch.from_numpy(z_samples_flat.astype(np.float32)).to(device=device)
            z_mc_list.append(z_samples.view(mc_n, Cg, Hg, Wg))

        # Decode all Monte Carlo latents, then average in image space
        x_mc_stack = _decode_latents(model, z_mc_list, target_hw=(Hc, Wc))  # (mc_n,1,H,W)
        x_mc_stack = to01(x_mc_stack)
        x_mc_mean = x_mc_stack.mean(dim=0, keepdim=True)  # (1,1,H,W)

        print(f"[recon-template] Monte Carlo mean computed from {mc_n} samples.")

    # ----------------------------- save panel -----------------------------
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if x_mc_mean is not None:
        # Panel: [decode(mu) | MC mean | abs difference]
        panel = make_recon_panel(x_mu, x_mc_mean)
        nrow = 3
    else:
        panel = x_mu
        nrow = 1

    save_grid(panel, outp, nrow=nrow, target_hw=(Hc, Wc))
    print(f"[recon-template] wrote {outp}")


def main_gauss_fit(argv: List[str] | None = None):

    def _sanitize_latents_array(X, cap_quantile=99.9, hard_cap=None):
        """
        X: (N, D) numpy float64
        - Replaces non-finites with 0
        - Clips to a symmetric cap derived from a high quantile (or a fixed hard_cap)
        Returns X_sanitized, stats dict
        """
        X = np.asarray(X, dtype=np.float64)
        stats = {}
        # non-finite -> 0
        nf = ~np.isfinite(X)
        nf_count = int(nf.sum())
        if nf_count:
            X[nf] = 0.0
        stats["nonfinite"] = nf_count

        # derive cap from quantile if not given
        if hard_cap is None:
            q = np.percentile(np.abs(X), [50, 90, 99, cap_quantile])
            cap = float(q[-1] + 1e-12)
            stats["abs_quantiles"] = {"p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]), f"p{cap_quantile}": float(q[3])}
        else:
            cap = float(hard_cap)
            stats["abs_quantiles"] = None

        # clip
        pre = X.copy()
        np.clip(X, -cap, cap, out=X)
        clipped = int(np.sum(pre != X))
        stats["cap"] = cap
        stats["clipped"] = clipped
        return X, stats

    def _cov_stats(Sd):
        # Sd: dense (D,D) numpy array
        Sd = 0.5 * (Sd + Sd.T)
        w = np.linalg.eigvalsh(Sd)
        lam_min = float(np.min(w))
        lam_max = float(np.max(w))
        cond = float(lam_max / max(lam_min, 1e-300))
        tr = float(np.trace(Sd))
        diag_mean = float(np.mean(np.diag(Sd)))
        return lam_min, lam_max, cond, tr, diag_mean


    def _dense_from_cov(Sigma, D):
        # Accepts dense, diag, or lowrank dict; returns dense (D,D)
        if isinstance(Sigma, dict) and Sigma.get("type") == "lowrank":
            U = np.asarray(Sigma["U"], dtype=np.float64)      # (D,r)
            eig = np.asarray(Sigma["eig"], dtype=np.float64)  # (r,)
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
                row_max = Z.detach().abs().amax(dim=1)  # (N,)
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

    ap = argparse.ArgumentParser("LAM-Flow conditional Gaussian fitter (gauss-fit)")
    ap.add_argument("--ckpt", type=str, required=True, 
                    help="Path to the trained model checkpoint (.pt).")
    ap.add_argument("--manifest", type=str, required=True, 
                    help="CSV file listing image paths for each view.")
    ap.add_argument("--views", type=str, default=None, 
                    help="Comma-separated list of views to include (e.g., T1,FA). Defaults to all columns in the manifest.")
    ap.add_argument("--slice-axis", type=int, required=True, 
                    help="Axis for slice extraction (0, 1, or 2) using ANTs.")
    ap.add_argument("--slice-index", type=int, required=True, 
                    help="Index of the slice to extract along the specified axis.")
    ap.add_argument("--batch", type=int, default=64, 
                    help="Batch size for encoding images into latent vectors.")
    ap.add_argument("--devices", type=str, default="cuda:0", 
                    help="Computing device (e.g., 'cuda:0', 'cpu', 'mps').")

    # Gaussian options
    ap.add_argument("--cov-mode", type=str, choices=["perlevel","merged"], default="perlevel",
                    help="Covariance mode: per resolution level (perlevel) or concatenated (merged).")
    ap.add_argument("--cov-estimator", type=str, choices=["full","diag","oas","lw","lowrank"], default="full",
                    help="Method for estimating the covariance matrix.")
    ap.add_argument("--rank", type=int, default=64, 
                    help="Matrix rank for the 'lowrank' estimator.")
    ap.add_argument("--sigma2", type=str, default="auto", 
                    help="Residual variance for 'lowrank' (numeric value or 'auto' for the mean of the remaining eigenvalues).")
    ap.add_argument("--shrinkage", type=str, default="1e-6", 
                    help="Shrinkage factor for regularizing the covariance.")
    ap.add_argument("--cov-lam", type=float, default=1e-6, 
                    help="Tikhonov (ridge) regularization added to the diagonal.")
    ap.add_argument("--jitter", type=float, default=1e-4, 
                    help="Numerical stability noise added during conditional imputation.")

    # Outputs
    ap.add_argument("--gauss-out", type=str, required=True, 
                    help="Output path for the serialized Gaussian model (.npz or .pt).")
    ap.add_argument("--gauss-summary", type=str, default="", 
                    help="Path to save a JSON summary of the covariance statistics.")
    ap.add_argument("--save-fp", type=int, default=64, 
                    help="Numerical precision for saving matrices (32 or 64-bit).")
    args = ap.parse_args(argv)

    @torch.no_grad()
    def _probe_latent_shapes_for_view(model, state_blob, view_idx, Hc, Wc, device):
        ok, note = load_weights_into_model(model, state_blob, view_idx=view_idx, prefer_ema=True)
        if not ok:
            raise RuntimeError(f"load_weights_into_model failed for view {view_idx}: {note}")
        x0 = torch.zeros(1, 1, int(Hc), int(Wc), device=device, dtype=torch.float32)
        if hasattr(model, "inverse_and_log_det"):
            z, _ = model.inverse_and_log_det(x0)
        elif hasattr(model, "inverse"):
            z, _ = model.inverse(x0)
        else:
            raise RuntimeError("Model lacks inverse mapping")
        z_list = z if isinstance(z, (list, tuple)) else [z]
        return [(int(t.shape[1]), int(t.shape[2]), int(t.shape[3])) for t in z_list]

    device = torch.device("cpu") if args.devices.lower() == "cpu" else torch.device(args.devices.split(",")[0])
    ckpt_path = resolve_ckpt_path(Path(args.ckpt))
    manifest_path = Path(args.manifest).resolve()
    manifest_dir = manifest_path.parent
    print(f"[info] gauss-fit: checkpoint: {ckpt_path}")
    print(f"[info] gauss-fit: manifest:   {manifest_path}")

    # Load checkpoint
    try:
        state_blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state_blob = torch.load(ckpt_path, map_location=device)
    cfg = state_blob.get("config", {})
    cfg_views = list(cfg.get("views", [])) if isinstance(cfg.get("views"), (list, tuple)) else None

    Hc = int(cfg.get("H", 128)); Wc = int(cfg.get("W", 128))
    model = build_model_from_config(cfg if cfg else {"H": Hc, "W": Wc}, device=device)
    model.eval()
    _prime_if_needed(model, Hc, Wc, device=device)

    # Parse manifest / views (preserve header or --views subset order)
    cols = _read_manifest_csv(manifest_path)
    view_names, per_view_paths = _resolve_views(cols, manifest_dir, args.views)  # header order
    V = len(view_names)
    N = len(per_view_paths[0])
    assert all(len(pp) == N for pp in per_view_paths), "All views must have the same number of subjects"
    N_original = int(N)
    print(f"[info] views: {view_names} (V={V}); subjects: N={N}")


    # Extract latents per view/level (x->z)
    z_per_view_per_level: List[List[torch.Tensor]] = [None] * V
    cfg_views = list(cfg.get("views", [])) if isinstance(cfg.get("views"), (list, tuple)) else None

    for v_idx, vname in enumerate(view_names):
        # fresh model instance for THIS view (mirrors trainer behavior)
        model = build_model_from_config(cfg if cfg else {"H": Hc, "W": Wc}, device=device)
        model.eval()
        _prime_if_needed(model, Hc, Wc, device=device)

        # robust loader (name-mapped → EMA → models → state_dict → raw)
        ok, note = load_weights_into_model(
            model, state_blob, view_idx=v_idx, prefer_ema=True,
            view_name=vname, cfg_views=cfg_views
        )
        if not ok:
            raise RuntimeError(f"Failed to load weights for view {v_idx} ({vname}): {note}")
        else:
            try:
                src, msg = note
            except Exception:
                src, msg = str(note), None
            print(f"[info] view {v_idx} ({vname}) weights source: {src}{(' | ' + str(msg)) if msg else ''}")

        paths = per_view_paths[v_idx]
        bs = max(1, int(args.batch))

        # ---- real-batch ActNorm warmup with REAL data ----
        try:
            warm_xs = []
            warm_n = min(bs, len(paths))
            for pth in paths[:warm_n]:
                xi = _read_image_any(pth, int(args.slice_axis), int(args.slice_index))  # (1,h,w) in [0,1]
                xi = torch.nn.functional.interpolate(
                    xi.unsqueeze(0), size=(Hc, Wc), mode="bilinear", align_corners=False
                ).squeeze(0)
                xi = to01(xi.unsqueeze(0)).squeeze(0)  # match trainer preprocessing
                warm_xs.append(xi)
            if warm_xs:
                xb_warm = torch.stack(warm_xs, dim=0).to(device=device, dtype=torch.float32)
                warmup_actnorm_with_real_batch(model, xb_warm)
        except Exception as _e:
            print(f"[warn] warmup failed for view {v_idx}: {_e}")

        latents_per_level_list: List[List[torch.Tensor]] | None = None

        def _flush_batch(xlist: List[torch.Tensor]):
            nonlocal latents_per_level_list
            xb = torch.stack(xlist, dim=0).to(device=device, dtype=torch.float32)
            with torch.no_grad(), torch.amp.autocast(device.type, enabled=False):
                if hasattr(model, "inverse_and_log_det"):
                    z, _ = model.inverse_and_log_det(xb)  # x->z
                elif hasattr(model, "inverse"):
                    z, _ = model.inverse(xb)
                else:
                    raise RuntimeError("Model lacks inverse mapping")
            zl = _flatten_latents_by_level(z)  # list of (B, D_l)
            if latents_per_level_list is None:
                latents_per_level_list = [[] for _ in range(len(zl))]
            for li, arr in enumerate(zl):
                latents_per_level_list[li].append(arr.detach().cpu())

        # main batching (preprocess exactly like warmup)
        batch = []
        for pth in paths:
            xi = _read_image_any(pth, int(args.slice_axis), int(args.slice_index))  # (1,h,w) in [0,1]
            xi = torch.nn.functional.interpolate(
                xi.unsqueeze(0), size=(Hc, Wc), mode="bilinear", align_corners=False
            ).squeeze(0)
            xi = to01(xi.unsqueeze(0)).squeeze(0)
            batch.append(xi)
            if len(batch) >= bs:
                _flush_batch(batch); batch = []
        if batch:
            _flush_batch(batch)

        if latents_per_level_list is None:
            raise RuntimeError(f"No latents collected for view {v_idx} ({vname}). Check manifest paths and preprocessing.")
        z_per_view_per_level[v_idx] = [torch.cat(chunks, dim=0) for chunks in latents_per_level_list]

    z_per_view_per_level, per_view_paths, keep_idx, bad_paths = _scrub_row_outliers(
        z_per_view_per_level, per_view_paths, view_names, thresh=1e6
    )
    N = len(keep_idx)

    # Concatenate views per level (header order)
    Z_levels = _concat_views_per_level(z_per_view_per_level)  # list of (N, D_level)
    L = len(Z_levels)
    dims_per_level_per_view = [[int(t.shape[1]) for t in vlist] for vlist in z_per_view_per_level]  # V × L

    # Shapes and slices (store both forms)
    shapes_by_view: List[List[Tuple[int,int,int]]] = []
    for v_idx in range(V):
        shp = _probe_latent_shapes_for_view(model, state_blob, v_idx, Hc, Wc, device)
        if len(shp) != L:
            raise RuntimeError(f"Expected {L} levels, got {len(shp)} for view '{view_names[v_idx]}'")
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

    # Fit Gaussians (float64)
    estimator = args.cov_estimator.lower()
    shrink = args.shrinkage
    if estimator == "full" and shrink == "auto":
        shrink_val = 0.0
    else:
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
        "N": int(N),             # <-- now reflects the kept cohort
        "H": int(Hc),
        "W": int(Wc),
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

        # Optional CLI control (add these to argparse if you want):
        cap_quant = 99.9         # high-quantile cap (used if hard_cap is None)
        hard_cap = None          # e.g., set to 50.0 to force absolute cap

        for l, Zl in enumerate(Z_levels):

            X = Zl.detach().cpu().numpy().astype("float64")  # (N,D)
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
                    U = Sigma["U"]; eig = Sigma["eig"]; sigma2 = float(Sigma.get("sigma2", 0.0))
                    print(f"[lowrank L{l}] eff_rank={U.shape[1]}  sum_eig={np.sum(eig):.3g}  sigma2={sigma2:.3g}")

            else:
                _mu_unused, Sigma, _stats_unused = _fit_gaussian_blocks(
                    [Xc_clean],
                    estimator=estimator,
                    shrinkage=shrink_val,
                    cov_lam=float(args.cov_lam)
                )

            # Stats from the Sigma we actually save
            Sd = _dense_from_cov(Sigma, D_l)
            lam_min, lam_max, cond, tr, diag_mean = _cov_stats(Sd)
            stats = {"lambda_min": lam_min, "lambda_max": lam_max,
                    "cond": cond, "trace": tr, "diag_mean": diag_mean,
                    "winsor_cap": float(sstats.get("cap", 0.0)),
                    "winsor_clipped": int(sstats.get("clipped", 0)),
                    "winsor_nonfinite": int(sstats.get("nonfinite", 0)),
                    "winsor_abs_quantiles": sstats.get("abs_quantiles")
                    }
            # print(f"[fit Σ L{l}] λmin={lam_min:.3e} λmax={lam_max:.3e} cond={cond:.3e} tr={tr:.3e} diag={diag_mean:.3e} D={D_l}")
            # print(f"[debug] dims_per_level_per_view L{l}:",
            #     [int(dims_per_level_per_view[v][l]) for v in range(V)])

            # Per-view block traces at level l
            row_dims = [int(dims_per_level_per_view[v][l]) for v in range(V)]
            offs = [0]
            for d in row_dims:
                offs.append(offs[-1] + d)
            blk_tr = [float(np.trace(Sd[a:b, a:b])) for a, b in zip(offs[:-1], offs[1:])]
            print("[fit Σ L{} by view] ".format(l) + " ".join("v{}:{:.3e}".format(vi, t) for vi, t in enumerate(blk_tr)))

            mu_list.append(mu); Sigma_list.append(Sigma); stats_list.append(stats)

        out_blob["mu"] = mu_list
        out_blob["Sigma"] = Sigma_list
        out_blob["stats"] = stats_list

    # Optional FP cast
    if args.save_fp == 32:
        def _cast_fp32_inplace(ob):
            def cast(x):
                return x.astype(np.float32) if isinstance(x, np.ndarray) and x.dtype == np.float64 else x
            if isinstance(ob.get("mu"), list):
                ob["mu"] = [cast(m) for m in ob["mu"]]
                newS = []
                for S in ob["Sigma"]:
                    if isinstance(S, dict) and S.get("type") == "lowrank":
                        U = S.get("U"); eig = S.get("eig"); sigma2 = float(S.get("sigma2", 0.0))
                        S = {"type":"lowrank",
                             "U": (U.astype(np.float32) if isinstance(U, np.ndarray) else U),
                             "eig": (eig.astype(np.float32) if isinstance(eig, np.ndarray) else eig),
                             "sigma2": float(sigma2)}
                    else:
                        S = cast(S)
                    newS.append(S)
                ob["Sigma"] = newS
            else:
                ob["mu"] = cast(ob["mu"])
                if isinstance(ob["Sigma"], dict) and ob["Sigma"].get("type") == "lowrank":
                    U = ob["Sigma"].get("U"); eig = ob["Sigma"].get("eig"); sigma2 = float(ob["Sigma"].get("sigma2", 0.0))
                    ob["Sigma"] = {"type":"lowrank",
                                   "U": (U.astype(np.float32) if isinstance(U, np.ndarray) else U),
                                   "eig": (eig.astype(np.float32) if isinstance(eig, np.ndarray) else eig),
                                   "sigma2": float(sigma2)}
                else:
                    ob["Sigma"] = cast(ob["Sigma"])
        _cast_fp32_inplace(out_blob)

    # Save
    out_path = Path(args.gauss_out); out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if str(out_path).endswith(".pt"):
            torch.save(out_blob, out_path, pickle_protocol=5, _use_new_zipfile_serialization=True)
        elif str(out_path).endswith(".npz"):
            _save_gauss_npz(out_blob, out_path)  # ensure dims/shapes/slices are serialized
        else:
            raise ValueError("Unsupported save format. Use .npz or .pt.")
        print(f"[ok] wrote Gaussian model: {out_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save --gauss-out: {e}")

    # Summary JSON
    if args.gauss_summary:
        js = {
            "mode": out_blob["mode"],
            "estimator": out_blob["estimator"],
            "shrinkage": out_blob["shrinkage"],
            "cov_lam": out_blob["cov_lam"],
            "jitter": out_blob["jitter"],
            "views": out_blob["views"],
            "N": out_blob["N"], "H": out_blob["H"], "W": out_blob["W"], "L": out_blob["L"],
            "dims_per_level_per_view": out_blob["dims_per_level_per_view"],
            "stats": out_blob["stats"],
            "ckpt_fingerprint": out_blob["ckpt_fingerprint"],
        }
        dropped_count = int(N_original - N)
        js["dropped_subjects"] = {
           "count": dropped_count,
           "original_N": int(N_original),
           "kept_N": int(N),
           "details": bad_paths,
           "thresh_max_abs_z": 1e6,
        }
        js_path = Path(args.gauss_summary); js_path.parent.mkdir(parents=True, exist_ok=True)
        with open(js_path, "w") as f:
            json.dump(js, f, indent=2)
        print(f"[ok] wrote summary JSON: {js_path}")


def _load_gaussian_model(gauss_path: Path) -> Dict[str, Any]:
    """
    Load Gaussian model saved by gauss-fit (.pt or .npz).
    Returns a dict with keys:
    - mode: "perlevel" or "merged"
    - estimator: "full"|"diag"|"lw"|"oas"|"lowrank"
    - views: list[str]
    - N, H, W, L: ints
    - dims_per_level_per_view: V x L list of ints
    - shapes_by_view: optional V x L list of (C,H,W)
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
        except Exception as e:  # UnpicklingError from weights_only lands here
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


def main_gauss_impute(argv=None):
    """
    Impute one or more target modalities given observed modalities, using a Gaussian
    model produced by `gauss-fit` and a trained LAM-Flow checkpoint.
    """

    def _load_gaussian_model(gauss_path: Path) -> Dict[str, Any]:
        """
        Load Gaussian model saved by gauss-fit (.pt or .npz).
        Returns a dict with keys:
        - mode: "perlevel" or "merged"
        - estimator: "full"|"diag"|"lw"|"oas"|"lowrank"
        - views: list[str]
        - N, H, W, L: ints
        - dims_per_level_per_view: V x L list of ints
        - shapes_by_view: optional V x L list of (C,H,W)
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
            except Exception as e:  # UnpicklingError from weights_only lands here
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

    # ---------- helpers (lowrank + robust SPD solve + torch cov-space solver) ----------

    def _cond_mean_block_lowrank(U: np.ndarray, eig: np.ndarray, sigma2: float,
                                 idx_U: list[int], idx_O: list[int],
                                 mu: np.ndarray, ZO: np.ndarray,
                                 base_ridge: float = 0.0, max_tries: int = 10):

        def _spd_solve_cholesky(SOO: np.ndarray, B: np.ndarray, base_ridge: float = 0.0, max_tries: int = 8):
            SOO = np.asarray(SOO, dtype=np.float64)
            B   = np.asarray(B,   dtype=np.float64)
            SOO = 0.5 * (SOO + SOO.T)
            D = SOO.shape[0]
            I = np.eye(D, dtype=np.float64)
            lam = max(float(base_ridge), 1e-8 * (np.trace(SOO) / max(D, 1)))
            for _ in range(max_tries):
                try:
                    L = np.linalg.cholesky(SOO + lam * I)
                    Y = np.linalg.solve(L, B)
                    X = np.linalg.solve(L.T, Y)
                    if np.all(np.isfinite(X)):
                        return X, lam
                except np.linalg.LinAlgError:
                    pass
                lam *= 10.0
            X, *_ = np.linalg.lstsq(SOO + lam * I, B, rcond=None)
            return X, lam

        U   = np.asarray(U,   dtype=np.float64)
        eig = np.asarray(eig, dtype=np.float64)
        mu  = np.asarray(mu,  dtype=np.float64).ravel()
        ZO  = np.asarray(ZO,  dtype=np.float64)
        U_O = U[idx_O, :]    # (D_O, r)
        U_U = U[idx_U, :]    # (D_U, r)
        Lam = np.diag(eig) if eig.ndim == 1 else eig
        SOO = U_O @ Lam @ U_O.T
        if sigma2 > 0.0:
            SOO = SOO + float(sigma2) * np.eye(SOO.shape[0], dtype=np.float64)
        dO = (ZO - mu[idx_O][None, :]).T  # (D_O, N)
        X, lam = _spd_solve_cholesky(SOO, dO, base_ridge=base_ridge, max_tries=max_tries)
        Y  = U_O.T @ X       # (r,N)
        TY = Lam @ Y         # (r,N)
        add = U_U @ TY       # (D_U,N)
        zU = mu[idx_U][:, None] + add  # (D_U,N)
        return zU.T, lam, SOO  # (N,D_U), lam, SOO

    @torch.no_grad()
    def _torch_conditional_gaussian_impute(
        z_obs_np, idx_obs, idx_mis, mu_np, Sigma_np,
        jitter: float = 1e-4, sample: bool = False, tau: float = 1.0, max_tries: int = 7,
    ):
        device = torch.device("cpu")
        z_obs = torch.as_tensor(z_obs_np, dtype=torch.double, device=device)      # (B, d_O)
        mu    = torch.as_tensor(mu_np,    dtype=torch.double, device=device).view(-1)
        S     = torch.as_tensor(Sigma_np, dtype=torch.double, device=device)
        idx_O = torch.as_tensor(idx_obs, dtype=torch.long, device=device)
        idx_M = torch.as_tensor(idx_mis, dtype=torch.long, device=device)

        mu_O, mu_M = mu[idx_O], mu[idx_M]
        S_OO = 0.5*(S.index_select(0, idx_O).index_select(1, idx_O) +
                    S.index_select(0, idx_O).index_select(1, idx_O).T)
        S_MO = S.index_select(0, idx_M).index_select(1, idx_O)
        S_MM = 0.5*(S.index_select(0, idx_M).index_select(1, idx_M) +
                    S.index_select(0, idx_M).index_select(1, idx_M).T)

        I = torch.eye(S_OO.shape[0], dtype=S_OO.dtype, device=device)
        jj = float(jitter)
        for _ in range(max_tries):
            try:
                L = torch.linalg.cholesky(S_OO + jj*I)
                break
            except RuntimeError:
                jj *= 10.0

        d = (z_obs - mu_O.unsqueeze(0)).T
        y = torch.linalg.solve_triangular(L, d, upper=False)
        alpha = torch.linalg.solve_triangular(L.T, y, upper=True).T  # (B, d_O)
        mean_cond = mu_M.unsqueeze(0) + alpha @ S_MO.T               # (B, d_M)

        if not sample:
            return mean_cond.to(torch.float32).cpu().numpy()

        S_OM = S_MO.T
        yK = torch.linalg.solve_triangular(L, S_OM, upper=False)
        K  = torch.linalg.solve_triangular(L.T, yK, upper=True)
        S_cond = 0.5*((S_MM - S_MO @ K) + (S_MM - S_MO @ K).T)

        Iu = torch.eye(S_cond.shape[0], dtype=S_cond.dtype, device=device)
        jj2 = float(jitter)
        for _ in range(max_tries):
            try:
                Lc = torch.linalg.cholesky(S_cond + jj2*Iu)
                break
            except RuntimeError:
                jj2 *= 10.0

        B = z_obs.shape[0]
        eps = torch.randn((B, Lc.shape[0]), dtype=S_cond.dtype, device=device)
        samples = mean_cond + (eps @ Lc.T)*float(tau)
        return samples.to(torch.float32).cpu().numpy()

    # ---------------------------------- args ----------------------------------
    import argparse
    ap = argparse.ArgumentParser("LAM-Flow Gaussian imputation (gauss-impute)")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to the trained Glow model checkpoint (.pt).")
    ap.add_argument("--gauss", type=str, required=True,
                    help="Path to the serialized Gaussian model (.npz or .pt) generated by gauss-fit.")
    ap.add_argument("--manifest", type=str, required=True,
                    help="CSV manifest containing paths to the observed images.")
    ap.add_argument("--views", type=str, required=True,
                    help="Comma-separated list of all views in the Gaussian model (must match fit order).")
    ap.add_argument("--observed", type=str, required=True,
                    help="Comma-separated list of views used as predictors (e.g., 'T1').")
    ap.add_argument("--target", type=str, required=True,
                    help="Comma-separated list of views to be imputed (e.g., 'FA').")
    ap.add_argument("--slice-axis", type=int, required=True,
                    help="Axis index for slicing NIfTI images (0, 1, or 2).")
    ap.add_argument("--slice-index", type=int, required=True,
                    help="Slice index to extract from the input volumes.")
    ap.add_argument("--devices", type=str, default="cuda:0",
                    help="Computing device to use (e.g., 'cuda:0' or 'cpu').")
    ap.add_argument("--batch", type=int, default=64,
                    help="Batch size for processing. Reduce to 1-4 if experiencing memory issues.")
    ap.add_argument("--strategy", type=str, choices=["mean", "sample"], default="mean",
                    help="Imputation strategy: 'mean' for conditional mean (sharp/clean) or 'sample' for stochastic sampling (textured).")
    ap.add_argument("--samples", type=int, default=1,
                    help="Number of stochastic samples to generate per subject if strategy is 'sample'.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Scaling factor for the latent variance during stochastic sampling.")
    ap.add_argument("--outdir", type=str, required=True,
                    help="Directory where the imputed images will be saved.")
    ap.add_argument("--output-format", type=str, choices=["nii","nii.gz","png"], default="nii.gz",
                    help="File format for saving the results. NIfTI (.nii.gz) is recommended for medical data.")
    ap.add_argument("--pairs-csv", type=str, default=None,
                    help="Optional path to save a CSV mapping observed files to their imputed counterparts.")
    ap.add_argument("--seed", type=int, default=1234,
                    help="Random seed for reproducible stochastic sampling.")
    ap.add_argument("--safe-latent", type=str, choices=["none","clamp"], default="none",
                    help="Optional safety mechanism to prevent extreme values in the imputed latent space.")
    ap.add_argument("--safe-k", type=float, default=2.0,
                    help="The k-sigma threshold used if safe-latent is set to 'clamp'.")
    args = ap.parse_args(argv)

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    obs   = [v.strip() for v in args.observed.split(",") if v.strip()]
    tgt   = [v.strip() for v in args.target.split(",") if v.strip()]
    if not views or not obs or not tgt:
        raise ValueError("views/observed/target must be non-empty")
    if any(v not in views for v in (obs + tgt)):
        bad = [v for v in (obs + tgt) if v not in views]
        raise ValueError(f"observed/target not all in --views: {bad}")
    if set(obs) & set(tgt):
        both = sorted(set(obs) & set(tgt))
        raise ValueError(f"--observed and --target overlap: {both}")

    # --------------------------------- model ----------------------------------
    device = torch.device(args.devices)
    ckpt_path = resolve_ckpt_path(Path(args.ckpt))
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)
    cfg = blob.get("config", {})
    Hc = int(cfg.get("H", 128)); Wc = int(cfg.get("W", 128))
    model = build_model_from_config(cfg if cfg else {"H": Hc, "W": Wc}, device=device)
    _prime_if_needed(model, Hc, Wc, device=device)

    cfg_views = list(cfg.get("views", [])) if isinstance(cfg.get("views"), (list, tuple)) else None

    # ------------------------------- gaussian ---------------------------------
    gauss_path = Path(args.gauss)
    g = _load_gaussian_model(gauss_path)                   # <-- ACTUALLY LOAD IT

    # Strong validation; returns canonical fields we use below
    views_header, dims_tbl, shapes_by_view, L = _validate_gauss_blob(g)

    # Bind metadata AFTER load/validate
    mode      = str(g.get("mode", "perlevel")).lower()
    estimator = str(g.get("estimator", "full")).lower()
    stats     = g.get("stats", None)

    # Σ stats (optional)
    if isinstance(stats, list):
        for i, st in enumerate(stats):
            if isinstance(st, dict) and "lambda_min" in st:
                print(f"[Σ L{i}] λmin={st.get('lambda_min'):.3e}, λmax={st.get('lambda_max'):.3e}, cond={st.get('cond'):.3e}")

    # ---------- enforce header view order (exact as fit) and rebuild slices from dims ----------
    views = views_header[:]             # force the exact order used at fit time
    obs   = [v for v in views if v in obs]
    tgt   = [v for v in views if v in tgt]
    view_index = {v: i for i, v in enumerate(views)}  # header index mapping

    # Rebuild per-level slice offsets directly from dims_tbl (the authority that Σ used)
    level_view_slices = []
    V = len(views)
    for l in range(L):
        off = 0
        row = {}
        for vi in range(V):
            d = int(dims_tbl[vi][l])
            row[vi] = (off, off + d)
            off += d
        level_view_slices.append(row)

    # Quick internal consistency between slices and dims_tbl
    for l in range(L):
        for vi, vname in enumerate(views):
            a, b = level_view_slices[l][vi]
            exp = int(dims_tbl[vi][l])
            if (b - a) != exp:
                raise RuntimeError(
                    f"[gauss] slice mismatch at level {l}, view '{vname}': slice={b-a}, dims_tbl={exp}. "
                    "Re-run gauss-fit; your stored dims and shapes are inconsistent."
                )


    # ------------------------------- manifest ---------------------------------
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    with open(manifest_path, "r") as f:
        header_row = f.readline().strip().split(",")
        col_idx = []
        for v in views:
            try:
                col_idx.append(header_row.index(v))
            except ValueError:
                raise ValueError(f"View '{v}' not found in manifest header: {header_row}")
        rows = []
        for line in f:
            parts = [s.strip() for s in line.strip().split(",")]
            if not parts or all(p == "" for p in parts):
                continue
            if len(parts) < max(col_idx)+1:
                raise ValueError("Manifest row has too few columns")
            paths = [Path(parts[j]) for j in col_idx]
            if any(str(p)=="" for p in paths):
                raise ValueError("Manifest has empty cell; missing files are not allowed")
            if any(not p.exists() for p in paths):
                missing = [str(p) for p in paths if not p.exists()]
                raise FileNotFoundError(f"Missing files in manifest row: {missing}")
            rows.append(paths)
    N = len(rows)
    per_view_paths = [[rows[i][v] for i in range(N)] for v in range(len(views))]

    # ----------------------- encode observed latents --------------------------
    @torch.no_grad()
    def encode_view(vname: str):
        v_idx = view_index[vname]

        # Fresh model instance for THIS view (mirrors gauss-fit)
        mdl = build_model_from_config(cfg if cfg else {"H": Hc, "W": Wc}, device=device)
        mdl.eval()
        _prime_if_needed(mdl, Hc, Wc, device=device)

        ok, note = load_weights_into_model(
            mdl, blob, view_idx=v_idx, prefer_ema=True, view_name=vname, cfg_views=cfg_views
        )
        if not ok:
            raise RuntimeError(f"Failed to load weights for view {v_idx} ({vname}): {note}")

        bs = max(1, int(args.batch))

        # Real-batch ActNorm warmup (safe no-op if helper missing)
        try:
            warm_n = min(bs, len(per_view_paths[v_idx]))
            warm_xs = []
            for pth in per_view_paths[v_idx][:warm_n]:
                xi = _read_image_any(pth, int(args.slice_axis), int(args.slice_index))  # (1,h,w) in [0,1]
                xi = torch.nn.functional.interpolate(
                    xi.unsqueeze(0), size=(Hc, Wc), mode="bilinear", align_corners=False
                ).squeeze(0)
                xi = to01(xi.unsqueeze(0)).squeeze(0)  # match gauss-fit/train preprocessing
                warm_xs.append(xi)
            if warm_xs and "warmup_actnorm_with_real_batch" in globals():
                xb_warm = torch.stack(warm_xs, dim=0).to(device=device, dtype=torch.float32)
                warmup_actnorm_with_real_batch(mdl, xb_warm)
        except Exception as _e:
            print(f"[warn] warmup failed for view {v_idx} ({vname}): {_e}")

        acc = None
        batch = []

        def flush():
            nonlocal acc, batch
            xb = torch.stack(batch, dim=0).to(device=device, dtype=torch.float32)
            batch = []
            with torch.no_grad(), torch.amp.autocast(
                device_type=("cuda" if device.type == "cuda" else "cpu"), enabled=False
            ):
                if hasattr(mdl, "inverse_and_log_det"):
                    z, _ = mdl.inverse_and_log_det(xb)  # x->z
                elif hasattr(mdl, "inverse"):
                    z, _ = mdl.inverse(xb)
                else:
                    raise RuntimeError("Model lacks inverse mapping")
            zl = _flatten_latents_by_level(z)  # list of (B, D_l)
            if acc is None:
                acc = [[] for _ in range(len(zl))]
            for li, arr in enumerate(zl):
                acc[li].append(arr.detach().cpu())

        for pth in per_view_paths[v_idx]:
            xi = _read_image_any(pth, int(args.slice_axis), int(args.slice_index))  # (1,h,w) in [0,1]
            xi = torch.nn.functional.interpolate(
                xi.unsqueeze(0), size=(Hc, Wc), mode="bilinear", align_corners=False
            ).squeeze(0)
            xi = to01(xi.unsqueeze(0)).squeeze(0)  # explicit, like gauss-fit
            batch.append(xi)
            if len(batch) >= bs:
                flush()
        if batch:
            flush()
        return [torch.cat(chunks, dim=0) for chunks in acc]


    obs_latents = {v: encode_view(v) for v in obs}

    # Assert concatenation dims match the Gaussian layout per level
    for l in range(L):
        D_expected = sum(level_view_slices[l][view_index[v]][1] -
                         level_view_slices[l][view_index[v]][0] for v in obs)
        D_got = sum(obs_latents[v][l].shape[1] for v in obs)
        if D_expected != D_got:
            raise RuntimeError(f"[gauss-impute] Level {l}: observed latent width mismatch; "
                               f"got {D_got}, expected {D_expected}. Check dims_per_level_per_view and view order.")

    # ----------------------------- conditioning ------------------------------
    mu_list  = g["mu"]    if mode == "perlevel" else [g["mu"]]
    Sig_list = g["Sigma"] if mode == "perlevel" else [g["Sigma"]]
    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)

    for tname in tgt:
        t_vidx = view_index[tname]

        per_level_U = []
        for l in range(L):
            mu  = mu_list[l]  if mode == "perlevel" else mu_list[0]
            Sig = Sig_list[l] if mode == "perlevel" else Sig_list[0]

            slices_l = level_view_slices[l]  # dict: header view idx -> (a,b)
            idx_O = []
            for v in obs:
                a, b = slices_l[view_index[v]]
                idx_O.extend(range(a, b))
            aU, bU = slices_l[t_vidx]
            idx_U = list(range(aU, bU))

            # Build Z_O first (concatenate observed views at this level)
            ZO_parts = [obs_latents[v][l].numpy().astype("float64") for v in views if v in obs]
            ZO = np.concatenate(ZO_parts, axis=1)

            N  = ZO.shape[0]

            # Fast dimension guard (informative)
            D_expected = sum(slices_l[view_index[v]][1] - slices_l[view_index[v]][0] for v in obs)
            if ZO.shape[1] != D_expected:
                raise RuntimeError(f"[gauss-impute] Level {l}: ZO width={ZO.shape[1]} doesn't match "
                                   f"sum of observed blocks={D_expected}.")

            # Now safe to form μ-partitions (used for diagnostics)
            mu_O = np.asarray(mu, dtype=np.float64)[idx_O]
            mu_U = np.asarray(mu, dtype=np.float64)[idx_U]

            # Σ vs empirical check (cheap sanity signal; only for full/OAS/LW where Σ is 2-D)
            if not isinstance(Sig, dict):
                S = np.asarray(Sig, dtype=np.float64) if Sig is not None else None
                if S is not None and S.ndim == 2:
                    SOO_g   = S[np.ix_(idx_O, idx_O)]
                    ZOc     = ZO - ZO.mean(axis=0, keepdims=True)  # unbiased sample covariance of Z_O
                    SOO_emp = (ZOc.T @ ZOc) / max(ZO.shape[0]-1, 1)
                    tr_g    = float(np.trace(SOO_g))
                    tr_emp  = float(np.trace(SOO_emp))
                    diag_g  = float(np.mean(np.diag(SOO_g)))
                    diag_emp= float(np.mean(np.diag(SOO_emp)))
                    print(f"[Σ check L{l}] tr_g/tr_emp={tr_g/(tr_emp+1e-12):.3g}  "
                          f"diag_g/diag_emp={diag_g/(diag_emp+1e-12):.3g}  "
                          f"D_O={len(idx_O)}  D_U={len(idx_U)}")

            if isinstance(Sig, dict) and Sig.get("type") == "lowrank":
                U      = np.asarray(Sig["U"],   dtype=np.float64)
                eig    = np.asarray(Sig["eig"], dtype=np.float64)
                sigma2 = float(Sig.get("sigma2", 0.0))

                zU, lam_used, SOO = _cond_mean_block_lowrank(
                    U, eig, sigma2, idx_U, idx_O,
                    (mu_list[l] if mode == "perlevel" else mu_list[0]), ZO,
                    base_ridge=float(g.get("cov_lam", 0.0)), max_tries=10
                )
                dev = float(np.mean(np.linalg.norm(zU - mu_U[None, :], axis=1)))
                tgt_dev = float(np.mean(np.linalg.norm(ZO - mu_O[None, :], axis=1)))
                print(f"[cond L{l}] lowrank Σ: ||zU-μ||_mean={dev:.4g} (ridge={lam_used:.3e}, target≈{tgt_dev:.3g})")

                if args.strategy == "sample":
                    U_U = U[idx_U, :]; U_O = U[idx_O, :]
                    Lam = np.diag(eig)
                    S_UU = U_U @ Lam @ U_U.T + sigma2*np.eye(len(idx_U))
                    S_UO = U_U @ Lam @ U_O.T
                    S_OU = S_UO.T
                    Y = np.linalg.solve(SOO, S_OU)
                    Sig_cond = S_UU - S_UO @ Y
                    Sig_cond = 0.5*(Sig_cond + Sig_cond.T) + 1e-6*np.eye(Sig_cond.shape[0])
                    Lc = np.linalg.cholesky(Sig_cond + 1e-12*np.eye(Sig_cond.shape[0]))
                    rng = np.random.default_rng(int(args.seed))
                    eps = rng.standard_normal(size=(N, zU.shape[1]))
                    zU = zU + (eps @ Lc.T) * float(args.temperature)
                    if args.safe_latent == "clamp":
                        stdU = np.sqrt(np.clip(np.diag(Sig_cond), 1e-12, None))
                        lower = mu_U - float(args.safe_k)*stdU
                        upper = mu_U + float(args.safe_k)*stdU
                        zU = np.clip(zU, lower[None,:], upper[None,:])

            else:
                S = np.asarray(Sig, dtype=np.float64)
                if S.ndim == 1:  # diag Σ
                    zU = np.tile(mu_U[None, :], (N, 1))
                    print(f"[cond L{l}] diag Σ: using μ only; ||zU-μ||_mean=0.0")
                    if args.strategy == "sample":
                        rng = np.random.default_rng(int(args.seed))
                        std = np.sqrt(np.clip(S[idx_U], 1e-12, None))
                        eps = rng.standard_normal(size=(N, zU.shape[1]))
                        zU = zU + eps * std[None,:] * float(args.temperature)
                        if args.safe_latent == "clamp":
                            lower = mu_U - float(args.safe_k)*std
                            upper = mu_U + float(args.safe_k)*std
                            zU = np.clip(zU, lower[None,:], upper[None,:])
                else:
                    # covariance-space torch Cholesky (matches working evaluator)
                    zU = _torch_conditional_gaussian_impute(
                        z_obs_np=ZO, idx_obs=idx_O, idx_mis=idx_U,
                        mu_np=(mu_list[l] if mode=="perlevel" else mu_list[0]),
                        Sigma_np=S, jitter=float(g.get("jitter", 1e-4)),
                        sample=(args.strategy=="sample"), tau=float(args.temperature)
                    )
                    dev = float(np.mean(np.linalg.norm(zU - mu_U[None,:], axis=1)))
                    tgt_dev = float(np.mean(np.linalg.norm(ZO - mu_O[None,:], axis=1)))
                    print(f"[cond L{l}] torch-Chol: ||zU-μ||_mean={dev:.4g} (jitter={float(g.get('jitter',1e-4)):.1e}, target≈{tgt_dev:.3g})")

            per_level_U.append(torch.from_numpy(zU).float())

        # ----------------------------- decode (z->x) ---------------------------
        # Build a fresh target model to avoid ActNorm contamination from prior loads
        t_vidx = view_index[tname]
        mdl_t = build_model_from_config(cfg if cfg else {"H": Hc, "W": Wc}, device=device)
        mdl_t.eval()
        _prime_if_needed(mdl_t, Hc, Wc, device=device)

        # name-aware loading aligned with gauss-fit
        ok, note = load_weights_into_model(
            mdl_t, blob, view_idx=t_vidx, prefer_ema=True, view_name=tname, cfg_views=cfg_views
        )
        if not ok:
            raise RuntimeError(f"Failed to load weights for target view {t_vidx} ({tname}): {note}")

        mdl_dev = next(mdl_t.parameters()).device
        zU_tensors = []
        for l, flat in enumerate(per_level_U):
            C, H, W = shapes_by_view[view_index[tname]][l]
            zL = flat.view(N, C, H, W).to(device=mdl_dev, dtype=torch.float32)
            zL = torch.nan_to_num(zL, nan=0.0, posinf=20.0, neginf=-20.0)
            zL = torch.clamp(zL, -20.0, 20.0)
            zU_tensors.append(zL)
        z_pack = zU_tensors if len(zU_tensors) > 1 else zU_tensors[0]

        if hasattr(mdl_t, "forward_and_log_det"):
            xh, _ = mdl_t.forward_and_log_det(z_pack)
        elif hasattr(mdl_t, "forward"):
            xh, _ = mdl_t.forward(z_pack)
        else:
            raise RuntimeError("Model lacks forward mapping")

        xh = _coerce_nchw_4d(xh, target_hw=(Hc, Wc))
        print(f"[decode] pre-to01 min/max: {float(xh.min()):.3f}/{float(xh.max()):.3f}")
        xh = to01(xh).detach().cpu()

        out_dir = Path(args.outdir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(N):
            file_name = f"{i:06d}_{tname}.{args.output_format}" 
            save_grid(xh[i:i+1], out_dir / file_name, nrow=1, target_hw=(Hc, Wc))

        print(f"[gauss-impute] wrote {N} NIfTI images for target={tname} -> {out_dir}")


if __name__ == "__main__":
    import sys, inspect

    # discover subcommands from functions named main_*
    table = {
        name[len("main_"):].replace("_", "-"): obj
        for name, obj in globals().items()
        if name.startswith("main_") and callable(obj)
    }

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Subcommands:", ", ".join(sorted(table)))
        sys.exit(0)

    sub = sys.argv.pop(1)
    fn = table.get(sub)
    if fn is None:
        raise SystemExit(f"Unknown subcommand: {sub}")

    # Call exactly once (no retries)
    sig = inspect.signature(fn)
    rc = fn(None) if len(sig.parameters) else fn()
    raise SystemExit(0 if (rc is None or rc == 0) else int(rc))