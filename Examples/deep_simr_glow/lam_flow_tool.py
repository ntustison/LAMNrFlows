#!/usr/bin/env python3
"""
lam_flow_tool.py — Sample M×N image grids from trained LAM-Flow (Glow 2D) checkpoints.

v0.3.0 (2025-11-06)
- Add ANTs resampling by physical spacing (--resample-spacing SxT) and by voxel size (--resample-size HxW).
- Add --native-spacing override.
- Ensure priming happens at checkpoint-native size to avoid internal shape mismatches.
- Prefer EMA weights if available; robust checkpoint loader.
- Save a metadata JSON next to the PNG output for reproducibility.

Usage examples
--------------
# 6×8 grid, 192×192 tiles, sample at ckpt-native size, then resample tiles to 0.8×0.8 mm (ANTs), then save:
python lam_flow_tool.py \
  --ckpt runs/t1_t2_fa_256x256_vicreg/training_state.pt \
  --view-index 1 \
  --grid-size 6x8 \
  --image-size 192x192 \
  --resample-spacing 0.8x0.8
  --out samples_view1.png

# If native spacing is not in the checkpoint, provide it:
python lam_flow_tool.py \
  --ckpt runs/t1_t2_fa_256x256_vicreg/training_state.pt \
  --view-index 1 \
  --grid-size 6x8 \
  --image-size 192x192 \
  --native-spacing 1.0x1.0 \
  --resample-spacing 0.7x0.7

# Resample by voxel count (use_voxels=True) to 192×192 before final grid save:
python lam_flow_tool.py \
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
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torchvision as tv
from PIL import Image
import numpy as np

# Ensure headless save works
import matplotlib
matplotlib.use("Agg")

__version__ = "0.3.7"

# ---------------- antstorch / model factory -----------------
import antstorch  # provided by env
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

def to01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def try_import_ants():
    try:
        import ants  # type: ignore
        return ants
    except Exception as e:
        raise RuntimeError(
            "ANTsPy is required for resampling. Install with `pip install antspyx` or ensure it's on PYTHONPATH."
        ) from e

@torch.no_grad()
def resample_with_ants_spacing(x: torch.Tensor,
                               native_spacing: Tuple[float, float],
                               target_spacing: Tuple[float, float]) -> torch.Tensor:
    """
    Resample (N,C,H,W) to a target physical spacing using ANTsPy (use_voxels=False).
    If C>1, channels are resampled independently and stacked back.
    """
    ants = try_import_ants()
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
    ants = try_import_ants()
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
    """
    Save a grid image. Accepts (N,C,H,W) tensor in [0,1]. If target_hw is given, tiles are resized for saving.
    """
    x = _coerce_nchw_4d(x, target_hw=target_hw)
    x = torch.clamp(x, 0.0, 1.0)
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
        ants = try_import_ants()
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
    ).to(device).float()
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

    if prefer_ema and isinstance(blob.get("ema"), (list, tuple)) and len(blob["ema"]) > 0:
        vidx = max(0, min(view_idx, len(blob["ema"]) - 1))
        ok, note = try_load(blob["ema"][vidx])
        return ok, ("ema", note)
    if isinstance(blob.get("models"), (list, tuple)) and len(blob["models"]) > 0:
        vidx = max(0, min(view_idx, len(blob["models"]) - 1))
        ok, note = try_load(blob["models"][vidx])
        return ok, ("models", note)
    if "state_dict" in blob and isinstance(blob["state_dict"], dict):
        ok, note = try_load(blob["state_dict"])
        return ok, ("state_dict", note)
    if isinstance(blob, dict) and all(isinstance(k, str) for k in blob.keys()) and any("." in k for k in blob.keys()):
        ok, note = try_load(blob)
        return ok, ("raw", note)
    return False, ("none", "no recognizable weights in checkpoint")

# --------------------------- main ---------------------------
def main():
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

    print(f"[info] lam_flow_tool {__version__}")
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

if __name__ == "__main__":
    main()



