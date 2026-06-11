
import argparse
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision as tv

from tqdm.auto import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

import ants
import antstorch
import antsnormflows as nf

from contextlib import nullcontext
from multiprocessing import Value  # optional but recommended if num_workers>0

from datetime import datetime
import json, platform

class AugSchedulerWrapper:
    """Wrapper global pour permettre la sérialisation (pickling) du scheduler."""
    def __init__(self, sched):
        self.sched = sched

    def __call__(self, step: int):
        return self.sched.step(step)

from concurrent.futures import ThreadPoolExecutor

class GlowStepWrapper(nn.Module):
    """
    Contient la logique d'entraînement à exécuter sur chaque GPU en parallèle.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Cette partie s'exécutera sur GPU 0 ET GPU 1 en parallèle
        z, logdet = self.model.inverse_and_log_det(x)
        
        # Calcul du Prior (Local à chaque GPU)
        m = self.model
        if hasattr(m, "q0s"): bases = m.q0s
        elif hasattr(m, "q0"): bases = m.q0
        if not isinstance(bases, (list, tuple, torch.nn.ModuleList)): bases = [bases]
        
        if isinstance(z, (list, tuple)):
            if len(bases) == 1 and len(z) > 1: bases = list(bases) * len(z)
            base_lp = sum(b.log_prob(zi) for b, zi in zip(bases, z))
        else:
            base_lp = bases[0].log_prob(z)
        
        log_prob = base_lp + logdet
        
        # Aplatissement local (Version 2D simple)
        z_flat = _flatten_latents(z)
        return log_prob, z_flat

    # Redirections
    def inverse_and_log_det(self, x): return self.model.inverse_and_log_det(x)
    def log_prob(self, x): return self.model.log_prob(x)
    def sample(self, *args, **kwargs): return self.model.sample(*args, **kwargs)

class GlowDataParallel(nn.DataParallel):
    # Laissez forward vide (il utilise celui de nn.DataParallel par défaut)
    
    # Redirections manuelles obligatoires pour que le code extérieur voit les méthodes internes
    def log_prob(self, x): 
        return self.module.log_prob(x)
    
    def inverse_and_log_det(self, x): 
        return self.module.inverse_and_log_det(x)
    
    def sample(self, *args, **kwargs): 
        return self.module.sample(*args, **kwargs)

def screen_dump_run_config(args, out_dir: Path, note: str = "", dataset_info: dict | None = None):
    """
    Pretty-print the effective CLI + a few env bits. Also saves JSON/TXT to out_dir.
    Call once right after args are finalized (post resume/ckpt overrides),
    and optionally again after dataloaders are built with dataset_info.
    """

    def _fmt_bool(x): return "true" if bool(x) else "false"

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(args))  # argparse Namespace -> dict (includes defaults)
    
    # ---- grad accumulation (derived + explicit) ----
    cfg["grad_accum"] = int(cfg.get("grad_accum", 1))
    cfg["effective_batch"] = int(cfg.get("batch", 0)) * cfg["grad_accum"]
    # ----------------------------------------------
    
    # Lightweight env/context
    env = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    if dataset_info:
        cfg["dataset_info"] = dataset_info

    # Save JSON (machine-readable)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump({"env": env, "config": cfg, "note": note}, f, indent=2)

    # Pretty TXT (human-readable)
    rows = []
    rows.append(
        f"[run] {env['timestamp']} | Py {env['python']} | torch {env['torch']} "
        f"| cuda={_fmt_bool(env['cuda_available'])} (n={env['cuda_device_count']})"
    )
    if note:
        rows.append(f"[note] {note}")

    def add(k, v):
        if v is None:
            v = "None"
        rows.append(f"{k:>24}: {v}")

    # Core architecture (2D Specific)
    add("out_dir", cfg.get("out_dir"))
    add("views", getattr(args, "num_views", None))
    add("H×W", f"{cfg.get('H')}×{cfg.get('W')}")
    add("L / K / hidden", f"{cfg.get('L')} / {cfg.get('K')} / {cfg.get('hidden')}")
    add("precision / amp_dtype", f"{cfg.get('precision')} / {cfg.get('amp_dtype')}")
    add("devices", cfg.get("devices"))
    add("num_workers", cfg.get("num_workers"))
    add("seed", cfg.get("seed"))

    # Training & Optimization
    add("batch", cfg.get("batch"))
    add("grad_accum", cfg.get("grad_accum"))
    add("effective_batch", cfg.get("effective_batch"))
    add("max_iter / extra", f"{cfg.get('max_iter')} / {cfg.get('extra_iters')}")
    add("lr / warmup", f"{cfg.get('lr')} / {cfg.get('warmup_iters')}")
    add("grad_clip", cfg.get("grad_clip"))
    add("weight_decay", cfg.get("weight_decay"))
    add("ema / decay", f"{_fmt_bool(cfg.get('ema'))} / {cfg.get('ema_decay')}")
    
    # LR Scheduling & Plateau
    add("lr_decay_gamma/steps", f"{cfg.get('lr_decay_gamma')} / {cfg.get('lr_decay_steps')}")
    add("plateau (fac/pat/thr)", f"{cfg.get('plateau_factor')} / {cfg.get('plateau_patience')} / {cfg.get('plateau_threshold')}")

    # Data & Augmentation (2D Specific)
    add("slice_idx", cfg.get("slice_idx"))
    add("val_frac", cfg.get("val_frac"))
    add("train / val samples", f"{cfg.get('train_samples')} / {cfg.get('val_samples')}")
    add("disable_aug_anneal", _fmt_bool(cfg.get("disable_aug_anneal")))
    add("aug_schedules", cfg.get("aug_schedules"))

    # Alignment & VICReg
    add("align", cfg.get("align"))
    add("weighting", cfg.get("weighting"))
    add("align_weight/warmup", f"{cfg.get('align_weight')} / {cfg.get('align_warmup')}")
    add("vicreg (i/v/c/g)", f"{cfg.get('vicreg_inv')}/{cfg.get('vicreg_var')}/{cfg.get('vicreg_cov')}/{cfg.get('vicreg_gamma')}")

    # CCA Screening
    add("screen", cfg.get("screen"))
    add("screen_frac", cfg.get("screen_frac"))
    add("screen_warmup / refresh", f"{cfg.get('screen_warmup')} / {cfg.get('screen_refresh')}")
    add("cca_ridge", cfg.get("cca_ridge"))
    add("prefilter_frac", cfg.get("prefilter_frac"))

    # Glow / Sampling Specifics
    add("sample_mode / temp", f"{cfg.get('sample_mode')} / {cfg.get('sample_temp')}")
    add("smooth_alpha", cfg.get("smooth_alpha"))
    add("scale_map / scale_cap", f"{cfg.get('scale_map')} / {cfg.get('scale_cap')}")
    add("glowbase (min/max log)", f"{cfg.get('glowbase_min_log')} / {cfg.get('glowbase_max_log')}")

    # Dataset summary if available
    if dataset_info:
        rows.append("-" * 60)
        for k, v in dataset_info.items():
            add(k, v)

    txt = "\n".join(rows) + "\n"
    print("\n" + txt)
    with open(out_dir / "run_config.txt", "a") as f:
        f.write(txt)

# ------------------------- small utils -------------------------

def set_deterministic(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _check_hw_divisible(H: int, W: int, L: int):
    r = 2 ** L
    if (H % r) or (W % r):
        raise ValueError(f"H and W must be divisible by 2**L={r}. Got H={H}, W={W}, L={L}")

def to01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # 1. Élimination pure et simple des NaNs/Infs issus de l'augmentation ANTs
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    # 2. Normalisation Min-Max
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    norm = (x - x_min) / (x_max - x_min + eps)    
    # 3. Sécurité Logit (empécher le 0.0 et 1.0 absolus)
    return torch.clamp(norm, 1e-5, 1.0 - 1e-5)

def bits_per_dim(logp: torch.Tensor, num_dims: int) -> torch.Tensor:
    return -logp / (np.log(2.0) * float(num_dims))  # [B]

def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

@torch.no_grad()
def _copy_actnorm_state(src, dst):
    for ms, md in zip(src.modules(), dst.modules()):
        if "actnorm" in ms.__class__.__name__.lower():
            for fld in ("logs","log_scale","scale","weight"):
                if hasattr(ms, fld) and hasattr(md, fld):
                    getattr(md, fld).data.copy_(getattr(ms, fld).data)
            for fld in ("bias","b"):
                if hasattr(ms, fld) and hasattr(md, fld):
                    getattr(md, fld).data.copy_(getattr(ms, fld).data)
            for fld in ("initialized","is_initialized","inited"):
                if hasattr(ms, fld) and hasattr(md, fld):
                    try: getattr(md, fld).data.copy_(getattr(ms, fld).data)
                    except Exception: setattr(md, fld, bool(getattr(ms, fld)))

@torch.no_grad()
def _prime_if_needed(model, x_view_1bhw: torch.Tensor):
    """
    Ensure MultiscaleFlow has cached latent shapes for .sample() by running a real forward.
    Avoids probing with sample(1), which can be a false negative in normflows.
    """
    x1 = x_view_1bhw[:1]
    if x1.ndim == 3:  # (B,H,W) -> (B,1,H,W)
        x1 = x1.unsqueeze(1)
    p = next(model.parameters(), None)
    dev = (p.device if p is not None else x1.device)
    x1 = x1.to(dev, dtype=torch.float32)
    # Prefer inverse_and_log_det to guarantee multiscale shapes are established
    try:
        _ = model.inverse_and_log_det(x1)
    except Exception:
        _ = model.log_prob(x1)

# Robust, version-agnostic exact log p(x)
def log_prob_exact(model, x: torch.Tensor) -> torch.Tensor:
    """
    Exact log p(x) = Σ_i log p_i(z_i) + log|det J|.
    Works for nf.MultiscaleFlow (z is list) and single-scale (z is tensor).
    """
    z, logdet = model.inverse_and_log_det(x)

    def bases_of(m):
        if hasattr(m, "q0s"):
            q0s = getattr(m, "q0s")
            if isinstance(q0s, (list, tuple, nn.ModuleList)):
                return list(q0s)
        if hasattr(m, "q0"):
            q0 = getattr(m, "q0")
            if isinstance(q0, (list, tuple, nn.ModuleList)):
                return list(q0)
            if q0 is not None:
                return [q0]
        raise RuntimeError("No base distribution(s) on model (q0/q0s)")

    if isinstance(z, (list, tuple)):
        bases = bases_of(model)
        if len(bases) == 1 and len(z) > 1:
            bases = bases * len(z)
        if len(bases) != len(z):
            raise RuntimeError(f"bases ({len(bases)}) != latents ({len(z)})")
        base_lp = sum(b.log_prob(zi) for b, zi in zip(bases, z))
    else:
        base_lp = bases_of(model)[0].log_prob(z)

    return base_lp + logdet  # shape [B]

def make_warmup(optimizer, warmup_iters: int, decay_gamma: float, decay_steps: int):
    if warmup_iters <= 0 and (decay_gamma == 1.0 or decay_steps <= 0):
        return None
    def lr_lambda(step):
        s = max(1, step)
        scale = 1.0
        if warmup_iters > 0 and s < warmup_iters:
            scale *= s / float(warmup_iters)
        if decay_gamma != 1.0 and decay_steps > 0:
            scale *= (decay_gamma ** (s / float(decay_steps)))
        return scale
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ------------------------- alignment helpers -------------------------
class Projector(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):  # [B, D]
        return self.net(x)

def _flatten_latents(z, target_pool_size=2):
    """
    Aplatit les latents pour l'alignement (VICReg/CCA/HSIC).
    
    STRATÉGIE LAMNr (Latent-Aligned Multiview) : 
    On extrait uniquement le niveau latent le plus profond (zs[-1]). 
    Cela permet d'aligner la sémantique globale des vues sans détruire 
    les gradients des convolutions spatiales haute fréquence des niveaux supérieurs.
    """
    zs = z if isinstance(z, (list, tuple)) else [z]
    
    # 1. Isolation exclusive du niveau le plus profond
    deepest_z = zs[-1]
    
    # 2. Pooling adaptatif pour contrôler la taille du Projector MLP
    if deepest_z.ndim == 5:
        # Volume 3D (N, C, D, H, W)
        z_pooled = F.adaptive_avg_pool3d(deepest_z, (target_pool_size, target_pool_size, target_pool_size))
        return z_pooled.flatten(1)
        
    elif deepest_z.ndim == 4:
        # Image 2D (N, C, H, W)
        # target_pool_size peut être augmenté à 4 pour la 2D via l'appel de fonction si désiré
        z_pooled = F.adaptive_avg_pool2d(deepest_z, (target_pool_size, target_pool_size))
        return z_pooled.flatten(1)
        
    else:
        # Tenseur déjà plat (N, D)
        return deepest_z.flatten(1)

# ------------------------- screening helpers (CCA / HSIC) -------------------------

from typing import Optional, Dict, Literal, Tuple as _Tuple
Method = Literal["none","cca","hsic"]

@dataclass
class ScreenState:
    method: Method = "none"
    proj_dim: int = 0
    keep_dim: int = 0
    n_views: int = 0
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    projectors: Optional[List[torch.Tensor]] = None  # for CCA (D,r)
    masks: Optional[List[torch.Tensor]] = None       # for HSIC (D,)
    meta: Optional[Dict] = None

def _whiten(F: torch.Tensor, ridge: float = 1e-3) -> _Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = F.mean(dim=0, keepdim=True)
    X = F - mu
    cov = (X.T @ X) / max(1, X.shape[0] - 1)
    cov = cov + ridge * torch.eye(cov.shape[0], device=F.device, dtype=F.dtype)
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.clamp(evals, min=1e-12)
    inv_sqrt = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
    return X @ inv_sqrt, mu, inv_sqrt

@torch.no_grad()
def _cca_pair(A: torch.Tensor, B: torch.Tensor, ridge: float = 1e-3):
    Xa, _, Wa = _whiten(A, ridge=ridge)
    Xb, _, Wb = _whiten(B, ridge=ridge)
    M = Xa.T @ Xb / max(1, A.shape[0] - 1)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    Ua = Wa @ U
    Vb = Wb @ Vh.T
    return Ua, S, Vb

@torch.no_grad()
def _screen_cca(feats: List[torch.Tensor], keep_dim: int, ridge: float = 1e-3):
    n = len(feats)
    B, D = feats[0].shape
    accum = [torch.zeros(D, D, device=feats[0].device, dtype=feats[0].dtype) for _ in range(n)]
    spectra = []
    for i in range(n):
        for j in range(i+1, n):
            Ui, S, Vj = _cca_pair(feats[i], feats[j], ridge=ridge)
            ui = Ui[:, :keep_dim]
            vj = Vj[:, :keep_dim]
            accum[i] = accum[i] + ui @ ui.T
            accum[j] = accum[j] + vj @ vj.T
            spectra.append(S.detach().cpu())
    projectors = []
    for i in range(n):
        A = accum[i] / max(1, (n-1)) + 1e-6 * torch.eye(accum[i].shape[0], device=accum[i].device, dtype=accum[i].dtype)
        ev, evc = torch.linalg.eigh(A)
        idx = torch.argsort(ev, descending=True)[:keep_dim]
        Pi = evc[:, idx]
        projectors.append(Pi)
    info = {"cca_keep_dim": int(keep_dim), "mean_spectrum": (torch.stack(spectra).mean(dim=0).tolist() if len(spectra) else None)}
    return projectors, info

def _rbf_kernel(x: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
    B = x.shape[0]
    x_norm = (x * x).sum(1).view(-1, 1)
    dist = x_norm + x_norm.T - 2.0 * (x @ x.T)
    if gamma is None:
        vals = dist.detach()
        median = torch.median(vals[~torch.eye(B, dtype=torch.bool, device=x.device)])
        if median <= 0: median = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        gamma = 1.0 / (2.0 * median)
    K = torch.exp(-gamma * dist)
    H = torch.eye(B, device=x.device, dtype=x.dtype) - (1.0/B) * torch.ones(B, B, device=x.device, dtype=x.dtype)
    return H @ K @ H

@torch.no_grad()
def _hsic_unbiased(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    K = _rbf_kernel(x); L = _rbf_kernel(y)
    mask = ~torch.eye(B, dtype=torch.bool, device=x.device)
    K_off = K[mask].view(B, B-1); L_off = L[mask].view(B, B-1)
    term1 = (K_off * L_off).sum() / (B * (B - 3))
    K1 = K.sum(dim=1) - torch.diagonal(K); L1 = L.sum(dim=1) - torch.diagonal(L)
    term2 = (K1 * L1).sum() / (B * (B - 3) * (B - 1))
    return term1 - term2

@torch.no_grad()
def _screen_hsic(feats: List[torch.Tensor], keep_frac: float, prefilter_frac: float = 0.5):
    n = len(feats)
    B, D = feats[0].shape
    r = max(1, int(round(D * keep_frac)))
    k_pref = max(1, int(round(D * prefilter_frac)))
    Z = []
    for Fv in feats:
        Zv = (Fv - Fv.mean(dim=0, keepdim=True)) / (Fv.std(dim=0, keepdim=True) + 1e-6)
        Z.append(Zv)
    pearson_scores = [torch.zeros(D, device=feats[0].device, dtype=feats[0].dtype) for _ in range(n)]
    for v in range(n):
        others = [Z[u] for u in range(n) if u != v]
        Zcat = torch.cat(others, dim=1) if len(others) else None
        if Zcat is None or Zcat.shape[1] == 0:
            continue
        zmean = Zcat.mean(dim=1, keepdim=True)
        a = Z[v]
        num = (a * zmean).sum(dim=0)
        den = (a.pow(2).sum(dim=0).sqrt() * (zmean.pow(2).sum(dim=0).sqrt().squeeze(0) + 1e-8))
        corr = (num / (den + 1e-8)).abs()
        pearson_scores[v] = corr
    hsic_scores = [torch.zeros(D, device=feats[0].device, dtype=feats[0].dtype) for _ in range(n)]
    for v in range(n):
        top_idx = torch.topk(pearson_scores[v], k=k_pref, largest=True).indices
        others = [Z[u] for u in range(n) if u != v]
        Zcat = torch.cat(others, dim=1) if len(others) else None
        if Zcat is None or Zcat.shape[1] == 0:
            continue
        y = Zcat.mean(dim=1, keepdim=True)
        for d in top_idx.tolist():
            x = Z[v][:, d:d+1]
            hs = _hsic_unbiased(x, y)
            hsic_scores[v][d] = hs
    masks = []
    kept_counts = []
    for v in range(n):
        idx = torch.topk(hsic_scores[v], k=r, largest=True).indices
        mask = torch.zeros(D, dtype=torch.bool, device=feats[0].device)
        mask[idx] = True
        masks.append(mask)
        kept_counts.append(int(mask.sum().item()))
    info = {"keep_dim": r, "prefilter_dim": k_pref, "kept_per_view": kept_counts}
    return masks, info

def update_screen(feats: List[torch.Tensor], state: Optional[ScreenState], method: Method="none",
                  keep_frac: float=0.5, ridge: float=1e-3, refresh: bool=False, prefilter_frac: float=0.5) -> ScreenState:
    if method == "none":
        return ScreenState(method="none")
    assert 0.0 < keep_frac <= 1.0
    B, D = feats[0].shape
    device, dtype = feats[0].device, feats[0].dtype
    n_views = len(feats)
    r = max(1, int(round(D * keep_frac)))
    if state is None or (state.method != method or state.proj_dim != D or state.n_views != n_views or state.keep_dim != r):
        state = ScreenState(method=method, proj_dim=D, keep_dim=r, n_views=n_views, device=device, dtype=dtype, projectors=None, masks=None, meta={})
    if not refresh:
        return state
    if method == "cca":
        projectors, info = _screen_cca(feats, keep_dim=r, ridge=ridge)
        state.projectors = [P.to(device=device, dtype=dtype) for P in projectors]
        state.masks = None
        state.meta = {"cca_info": info, "keep_dim": r}
    elif method == "hsic":
        masks, info = _screen_hsic(feats, keep_frac=keep_frac, prefilter_frac=prefilter_frac)
        state.masks = [m.to(device=device, dtype=dtype) for m in masks]
        state.projectors = None
        state.meta = {"hsic_info": info, "keep_dim": r}
    return state

@torch.no_grad()
def apply_screen(feats: List[torch.Tensor], state: Optional[ScreenState]) -> List[torch.Tensor]:
    """
    Apply the learned screening transform if it exists.
    If screening hasn't been computed yet, this safely falls back to identity.
    """
    if state is None or state.method == "none":
        return feats

    if state.method == "cca":
        # Projectors not ready yet → skip screening
        if state.projectors is None:
            return feats
        return [f @ P for f, P in zip(feats, state.projectors)]

    if state.method == "hsic":
        # Masks not ready yet → skip screening
        if state.masks is None:
            return feats
        return [f[:, m] for f, m in zip(feats, state.masks)]

    return feats



# ------------------------- viz helpers -------------------------

def _extract_views_from_batch(batch, num_views: int | None = None):
    """
    Normalize a multi-view batch into a list [x_view0, x_view1, ...], each (B,C,H,W).

    Supported input forms:
      - dict with 'x' or 'views' (list/tuple of tensors)
      - list/tuple of per-view tensors
      - tuple like (x, y, ...) where x is a tensor or multi-view container
      - torch.Tensor of shape:
          (B, V, C, H, W)  -> unstack along V
          (B, C_total, H, W) with C_total % num_views == 0 -> split channels
    """
    import torch

    if isinstance(batch, tuple) and len(batch) > 0 and (
        torch.is_tensor(batch[0]) or isinstance(batch[0], (list, tuple, dict))
    ):
        return _extract_views_from_batch(batch[0], num_views=num_views)

    if isinstance(batch, dict):
        if 'x' in batch:
            return _extract_views_from_batch(batch['x'], num_views=num_views)
        if 'views' in batch:
            vs = batch['views']
            if isinstance(vs, (list, tuple)) and len(vs) > 0 and torch.is_tensor(vs[0]):
                return list(vs)
            raise ValueError("Batch['views'] not in expected list/tuple[tensor] format.")
        for v in batch.values():
            if isinstance(v, (list, tuple)) and len(v) > 0 and torch.is_tensor(v[0]):
                return list(v)
        raise ValueError("Batch dict format not recognized for multi-view data.")

    if isinstance(batch, (list, tuple)) and len(batch) > 0 and torch.is_tensor(batch[0]):
        return list(batch)

    if torch.is_tensor(batch):
        if batch.ndim == 5:
            B, V, C, H, W = batch.shape
            return [batch[:, vi, :, :, :] for vi in range(V)]
        elif batch.ndim == 4:
            if num_views is None or num_views <= 1:
                return [batch]
            B, Ctot, H, W = batch.shape
            if Ctot % num_views != 0:
                raise ValueError(
                    f"Cannot split (B,C,H,W)=({B},{Ctot},{H},{W}) into {num_views} views: "
                    f"C ({Ctot}) not divisible by num_views."
                )
            Cpv = Ctot // num_views
            return [batch[:, vi*Cpv:(vi+1)*Cpv, :, :] for vi in range(num_views)]
        else:
            raise ValueError(f"Unsupported tensor ndim={batch.ndim}; expected 4 or 5.")

    raise ValueError(f"Unsupported batch type for multi-view extraction: {type(batch)}")

def _save_grid_from_tensor(x, out_path: Path, nrow: int, target_hw=None, value_range=None):
    x = x.detach().cpu()
    if target_hw is not None and (x.shape[-2] != target_hw[0] or x.shape[-1] != target_hw[1]):
        x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
    grid = tv.utils.make_grid(x, nrow=nrow, normalize=(value_range is None), value_range=value_range)
    tv.utils.save_image(grid, str(out_path))

def save_coordinated_input_grids(val_loader, num_views: int, out_dir: Path,
                                 fallback_loader=None,
                                 n: int = 100, nrow: int = 10, target_hw=None, device="cpu"):
    import torch
    import gc

    def collect_from_loader(loader):
        samples_per_view = [[] for _ in range(num_views)]
        collected = 0

        for batch in loader:
            xs = _extract_views_from_batch(batch, num_views=num_views)
            if len(xs) != num_views:
                return None, f"Expected {num_views} views, got {len(xs)}."
            B = xs[0].shape[0]
            take = min(n - collected, B)
            if take > 0:
                for vi in range(num_views):
                    xvi = xs[vi][:take].to(dtype=torch.float32, device=device, non_blocking=True)
                    samples_per_view[vi].append(xvi)
                collected += take

            del xs
            del batch 

            if collected >= n:
                break
            
        gc.collect()

        if collected == 0:
            return None, "loader yielded no samples."
        stacked = [torch.cat(vs, dim=0)[:n] for vs in samples_per_view]
        return stacked, None

    try:
        result, err = collect_from_loader(val_loader)
        if result is None:
            if fallback_loader is not None:
                result, err_fb = collect_from_loader(fallback_loader)
                if result is None:
                    return False, f"val+fallback loaders failed: {err}; {err_fb}"
            else:
                return False, f"val loader failed: {err}"
    except Exception as e:
        if fallback_loader is not None:
            try:
                result, err_fb = collect_from_loader(fallback_loader)
                if result is None:
                    return False, f"val+fallback loaders failed: {e}; {err_fb}"
            except Exception as e2:
                return False, f"val+fallback loaders exception: {e}; {e2}"
        else:
            return False, f"val loader exception: {e}"

    for vi in range(num_views):
        x = result[vi]  # (n,C,H,W)
        out_path = out_dir / f"input_data_view{vi}.png"
        _save_grid_from_tensor(x, out_path, nrow=nrow, target_hw=target_hw)
    return True, None

def _make_grid_canvas(x, nrow=10):
    assert torch.is_tensor(x) and x.dim() == 4, "x must be (N,C,H,W) tensor"
    N, C, H, W = x.shape
    cols = int(nrow)
    rows = (N + cols - 1) // cols
    canvas = x.new_zeros(C, rows * H, cols * W)
    for idx in range(N):
        r = idx // cols
        c = idx % cols
        canvas[:, r*H:(r+1)*H, c*W:(c+1)*W] = x[idx]
    return canvas

def _coerce_nchw_4d(x, target_hw=None):
    if isinstance(x, (list, tuple)):
        cands = [t for t in x if torch.is_tensor(t) and t.dim() in (3,4)]
        if not cands:
            raise ValueError("No tensor candidates in sample output.")
        areas, fixed = [], []
        for t in cands:
            if t.dim() == 3:
                if t.shape[-1] in (1,3) and (t.shape[0] not in (1,3)):
                    t = t.permute(2,0,1).contiguous()
                t = t.unsqueeze(0)
            elif t.dim() == 4:
                if t.shape[-1] in (1,3) and t.shape[1] not in (1,3):
                    t = t.permute(0,3,1,2).contiguous()
            fixed.append(t)
            areas.append(int(t.shape[-1]) * int(t.shape[-2]))
        x = fixed[int(torch.tensor(areas, dtype=torch.float32).argmax().item())]
    if not torch.is_tensor(x):
        raise ValueError(f"Sample output is not a tensor: {type(x)}")
    if x.dim() == 3:
        if x.shape[-1] in (1,3) and x.shape[0] not in (1,3):
            x = x.permute(2,0,1).contiguous()
        x = x.unsqueeze(0)
    if x.dim() == 4 and x.shape[-1] in (1,3) and x.shape[1] not in (1,3):
        x = x.permute(0,3,1,2).contiguous()
    if x.size(1) not in (1,3):
        x = x.mean(dim=1, keepdim=True)
    x = torch.clamp(x, 0, 1).float()
    if target_hw is not None:
        Ht, Wt = int(target_hw[0]), int(target_hw[1])
        H, W = int(x.shape[-2]), int(x.shape[-1])
        if (H, W) != (Ht, Wt):
            x = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)
    return x

@torch.no_grad()
def _save_samples_grid(model, n, temp, out_prefix, nrow=10, target_hw=None, warm_x=None, which_type="to01"):
    temp_tensor = torch.tensor(temp, dtype=torch.float32)
    device_original = next(model.parameters()).device
    try:
        try:
            s = model.sample(n, temperature=temp_tensor)
        except TypeError:
            s = model.sample(n)
    except Exception as e:
        msg = str(e).lower()
        if "latent shapes unknown" in msg and warm_x is not None:
            _prime_if_needed(model, warm_x)
            try:
                try:
                    s = model.sample(n, temperature=temp_tensor)
                except TypeError:
                    s = model.sample(n)
            except Exception as e2:
                # REPLI CPU : Si l'échantillonnage échoue (ex. erreur float64 sur MPS)
                try:
                    model.to('cpu')
                    temp_tensor_cpu = temp_tensor.to('cpu')
                    try:
                        s = model.sample(n, temperature=temp_tensor_cpu)
                    except TypeError:
                        s = model.sample(n)
                    # Déplacer les échantillons générés vers le périphérique d'origine
                    if isinstance(s, (list, tuple)):
                        s = [item.to(device_original) if isinstance(item, torch.Tensor) else item for item in s]
                    elif isinstance(s, torch.Tensor):
                        s = s.to(device_original)
                    model.to(device_original)
                except Exception as e_cpu:
                    model.to(device_original)
                    return False, f"MPS failed: {e2}. CPU fallback also failed: {e_cpu}"
        else:
             # REPLI CPU (même logique si pas d'erreur 'latent shapes unknown')
             try:
                 model.to('cpu')
                 temp_tensor_cpu = temp_tensor.to('cpu')
                 try:
                     s = model.sample(n, temperature=temp_tensor_cpu)
                 except TypeError:
                     s = model.sample(n)
                 if isinstance(s, (list, tuple)):
                     s = [item.to(device_original) if isinstance(item, torch.Tensor) else item for item in s]
                 elif isinstance(s, torch.Tensor):
                     s = s.to(device_original)
                 model.to(device_original)
             except Exception as e_cpu:
                 model.to(device_original)
                 return False, f"MPS failed: {e}. CPU fallback also failed: {e_cpu}"

    try:
        x = s[0] if isinstance(s, (list, tuple)) else s
        x = _coerce_nchw_4d(x, target_hw=target_hw)
        try:
            _std = x.std().item()
        except Exception:
            _std = 0.0
        if _std < 1e-5:
            try:
                x = _manual_prior_sample(model, n, temp_tensor, x_template=None)
                x = _coerce_nchw_4d(x, target_hw=target_hw)
            except Exception:
                pass
        if torch.isfinite(x).all():
            _std = x.std().item()
            if _std < 1e-5:
                try:
                    x = _manual_prior_sample(model, n, temp_tensor)
                    x = _coerce_nchw_4d(x, target_hw=target_hw)
                except Exception:
                    pass

        assert torch.isfinite(x).all(), "non-finite in sample grid"
                
        valid = {"to01", "clamp", "both"}
        if which_type not in valid:
            raise ValueError(f"_save_samples_grid: unrecognized which_type={which_type}")

        x_to01 = to01(x) if which_type in ("to01", "both") else None
        x_clamp = x.clamp(0, 1) if which_type in ("clamp", "both") else None

        if x_to01 is not None:
            if x_to01.shape[0] < n:
                reps = (n + x_to01.shape[0] - 1) // x_to01.shape[0]
                x_to01 = x_to01.repeat(reps, 1, 1, 1)
            x_to01 = x_to01[:n]
            grid = _make_grid_canvas(x_to01, nrow=nrow)
            tv.utils.save_image(grid, str(out_prefix) + "_to01.png")
        if x_clamp is not None:
            if x_clamp.shape[0] < n:
                reps = (n + x_clamp.shape[0] - 1) // x_clamp.shape[0]
                x_clamp = x_clamp.repeat(reps, 1, 1, 1)
            x_clamp = x_clamp[:n]
            grid = _make_grid_canvas(x_clamp, nrow=nrow)
            tv.utils.save_image(grid, str(out_prefix) + "_clamp.png")

        return True, None
    except Exception as e:
        return False, str(e)

def _save_metric_plots(csv_path: Path, out_dir: Path, remove_spikes: bool = False):
    if not csv_path.exists():
        return
    iters, losses, bpds = [], [], []
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                # Le bloc try/except évite un crash si le script lit la ligne exactement 
                # au moment où l'écrivain CSV est en train de l'enregistrer.
                try:
                    it, loss, bpd = int(float(row[0])), float(row[1]), float(row[2])
                    iters.append(it); losses.append(loss); bpds.append(bpd)
                except ValueError:
                    continue
                    
        if len(iters) < 2:
            return
            
        if remove_spikes and len(losses) > 10:
            s_losses = pd.Series(losses)
            
            # Fenêtre adaptative (max 50, s'ajuste si le fichier est petit)
            w = min(50, max(5, len(losses) // 10))
            
            # Calcul de la médiane glissante locale
            rolling_med = s_losses.rolling(window=w, center=True, min_periods=1).median()
            
            # Calcul de l'écart absolu de chaque point par rapport à la médiane
            diff = np.abs(s_losses - rolling_med)
            
            # Calcul du MAD (Median Absolute Deviation) local
            rolling_mad = diff.rolling(window=w, center=True, min_periods=1).median()
            
            # Un 'spike' est un point qui dévie de plus de 5 fois le MAD (avec une tolérance minimale de sécurité)
            is_spike = diff > (5 * rolling_mad + 1e-6)
            
            # Remplacer les valeurs par NaN pour créer une cassure visuelle sur le graphique
            losses = np.where(is_spike, np.nan, losses)
            bpds = np.where(is_spike, np.nan, bpds)

        plt.figure()
        plt.plot(iters, losses)
        plt.xlabel("iter"); plt.ylabel("loss"); plt.title("Training loss")
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png"); plt.close()
        
        plt.figure()
        plt.plot(iters, bpds)
        plt.xlabel("iter"); plt.ylabel("sum_bpd"); plt.title("Sum BPD (training batches)")
        plt.tight_layout()
        plt.savefig(out_dir / "bpd_curve.png"); plt.close()
        
    except Exception as e:
        pass

def cleanup_checkpoints(run_dir: Path, keep_every: int = 10000):
    """
    Supprime les points de contrôle versionnés qui ne sont pas des jalons.
    Conserve uniquement les fichiers dont l'itération est un multiple de keep_every.
    """
    for f in run_dir.glob("training_state_it*.pt"):
        try:
            # Extrait le numéro d'itération du nom de fichier
            it_num = int(f.stem.split('it')[-1])
            if it_num % keep_every != 0:
                f.unlink()  # Supprime le fichier
        except (ValueError, IndexError):
            continue

# ------------------------- data -------------------------

import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
import torchvision.io as io
from pathlib import Path
import numpy as np

class PNGMultiViewDataset(Dataset):
    def __init__(self, images_list, target_size=(128, 128), do_aug=False):
        """
        images_list: Liste de listes. Chaque élément est un sujet contenant N vues.
                     Les vues peuvent être des chemins de fichiers (str/Path) ou des tenseurs.
        """
        self.images_list = images_list
        self.do_aug = do_aug
        
        # Base : Redimensionnement et conversion des pixels en [0.0, 1.0]
        self.base_transform = v2.Compose([
            v2.Resize(target_size, antialias=True),
            v2.ToDtype(torch.float32, scale=True)
        ])
        
        # Augmentations spatiales (réplique de affineAndDeformation)
        if self.do_aug:
            self.spatial_transforms = v2.Compose([
                v2.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                v2.ElasticTransform(alpha=50.0, sigma=5.0)
            ])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        views = self.images_list[idx]
        tensors = []
        
        for v in views:
            if isinstance(v, (str, Path)):
                # Lecture native du PNG multi-canaux -> shape: (C, H, W)
                img = io.read_image(str(v), mode=io.ImageReadMode.RGB)
            elif isinstance(v, np.ndarray):
                # Conversion numpy (H, W, C) -> tensor (C, H, W)
                img = torch.from_numpy(v).permute(2, 0, 1)
            else:
                img = v
                
            img = self.base_transform(img)
            tensors.append(img)
            
        # Empiler [V, C, H, W] garantit que torchvision applique la MÊME 
        # transformation géométrique à toutes les vues du sujet.
        stacked_views = torch.stack(tensors)
        
        if self.do_aug:
            stacked_views = self.spatial_transforms(stacked_views)
            
            # Réplique de additivegaussian noise (indépendant par canal/vue)
            noise = torch.randn_like(stacked_views) * 0.05
            stacked_views = torch.clamp(stacked_views + noise, 0.0, 1.0)
            
        return stacked_views

def build_loaders_from_globs(view_specs, H, W, train_samples, val_samples, batch, num_workers,
                             slice_idx: int, val_frac: float,
                             subject_limit: int | None,
                             do_aug=True, aug_schedules=None, disable_aug_anneal=False, seed: int = 0):

    def _scan_globbed_views(view_specs, H, W, slice_idx, subject_limit=None):
        """
        Returns images_by_subject = [ [ [view0_k, view1_k, ...], ... ] for each subject ], where each element is an ANTs 2D image.
        """
        import ants

        def _group_by_subject(per_view_files):
            """
            Groups file lists by subject.
            Extracts the subject ID (e.g., 'sub-001') from the BIDS-compliant filename or path.
            Ensures all views share the same subject set; returns mapping subj -> list[list[Path]] per view index within subject.
            """
            from collections import defaultdict
            import re
            
            per_view_by_subj = []
            subj_sets = []
            for files in per_view_files:
                d = defaultdict(list)
                for f in files:
                    # Extraction de l'ID du sujet via expression régulière
                    # Cherche le motif "sub-" suivi de caractères alphanumériques
                    # match = re.search(r'(sub-[a-zA-Z0-9]+)', str(f))
                    # subj = match.group(1) if match else f.parent.name
                    subj = f.name.split('_')[0]
                    
                    d[subj].append(f)
                
                # sort within subject for determinism
                for k in d:
                    d[k] = sorted(d[k])
                per_view_by_subj.append(d)
                subj_sets.append(set(d.keys()))
                
            common_subj = set.intersection(*subj_sets) if subj_sets else set()
            if not common_subj:
                raise RuntimeError("No common subjects found across views.")
            
            # Ensure consistent counts per subject across views
            per_subj = {}
            for s in sorted(common_subj):
                counts = [len(d[s]) for d in per_view_by_subj]
                if len(set(counts)) != 1:
                    raise RuntimeError(f"Subject {s} has different file counts across views: {counts}")
                # collect per-index across views
                M = counts[0]
                per_subj[s] = [[per_view_by_subj[v][s][k] for v in range(len(per_view_by_subj))] for k in range(M)]
            return per_subj
            
        def _expand_globs_per_view(view_specs):
            """
            view_specs: list[list[str]]; each inner list are glob patterns for that view
            Returns: list[list[Path]] per view (sorted)
            """
            import glob, os
            per_view_files = []
            for specs in view_specs:
                paths = []
                for pat in specs:
                    pat = os.path.expanduser(pat)
                    paths.extend(glob.glob(pat))
                # unique + sort
                paths = sorted({str(p) for p in paths})
                per_view_files.append([Path(p) for p in paths])
            return per_view_files

        def _read_slice(path: Path, idx: int, H: int, W: int):
            # Si c'est un PNG, on retourne juste le chemin pour Torchvision
            if path.suffix.lower() == '.png':
                return path
                
            # Sinon, on applique la logique ANTs standard pour les NIfTI/NRRD
            im = ants.image_read(str(path))
            if im.dimension == 3:
                slc = ants.slice_image(im, axis=1, idx=idx, collapse_strategy=1)
            else:
                slc = im
            resize_factor = min(float(H)/float(slc.shape[0]), 
                                float(W)/float(slc.shape[1]))
            spacing = (slc.spacing[0] / resize_factor, 
                       slc.spacing[1] / resize_factor)   
            slc = ants.resample_image(slc, spacing, use_voxels=False, interp_type=0)
            slc = ants.pad_or_crop_image_to_size(slc, (H, W))
            return slc
        
        per_view_files = _expand_globs_per_view(view_specs)
        per_subj = _group_by_subject(per_view_files)
        subjects = list(sorted(per_subj.keys()))
        if subject_limit and subject_limit > 0:
            subjects = subjects[:int(subject_limit)]
        images_by_subject = []

        def _load_subject_data(s):
            subj_samples = []
            for sample in per_subj[s]:
                views = []
                for f in sample:
                    # _read_slice est défini plus haut dans votre code
                    views.append(_read_slice(Path(f), slice_idx, H, W))
                subj_samples.append(views)
            return subj_samples

        # --- CORRECTION : Gestion du cas num_workers=0 ---
        if num_workers > 0:
            print(f"[info] Loading {len(subjects)} subjects using {num_workers} threads...")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                images_by_subject = list(tqdm(
                    executor.map(_load_subject_data, subjects), 
                    total=len(subjects), 
                    desc="Loading volumes"
                ))
        else:
            # Si num_workers=0, on charge séquentiellement dans le thread principal
            print(f"[info] Loading {len(subjects)} subjects sequentially (main thread)...")
            images_by_subject = []
            for s in tqdm(subjects, desc="Loading volumes"):
                images_by_subject.append(_load_subject_data(s))
            # -------------------------------------------------

        if len(images_by_subject) == 0:
            raise RuntimeError("No images assembled from provided --view globs.")
        return images_by_subject

    images_by_subject = _scan_globbed_views(view_specs, H, W, slice_idx, subject_limit=subject_limit)
    n_subj = len(images_by_subject)
    rng = np.random.default_rng(seed)
    idx = np.arange(n_subj); rng.shuffle(idx)
    n_val = int(round(float(val_frac) * n_subj))
    # ensure we always leave at least one subject for training
    if n_val >= n_subj:
        n_val = max(0, n_subj - 1)

    val_idx = set(idx[:n_val])

    images_train = [s for si, subj in enumerate(images_by_subject) if si not in val_idx for s in subj]
    images_val   = [s for si, subj in enumerate(images_by_subject) if si in val_idx for s in subj]

    if len(images_train) == 0:
        raise RuntimeError("Split produced an empty training set. Decrease --val-frac or increase subjects.")

    tmpl = images_train[0][0]

    if aug_schedules and not disable_aug_anneal:
        sched = antstorch.MultiParamScheduler(antstorch.parse_schedules(aug_schedules))
        aug_sched_fn = AugSchedulerWrapper(sched)
    else:
        aug_sched_fn = None

    global_step = Value('i', 0)

    if isinstance(tmpl, ants.core.ants_image.ANTsImage):
        train_ds = antstorch.ImageDataset(
            images=images_train,
            template=tmpl,
            do_data_augmentation=do_aug,
            data_augmentation_transform_type="affineAndDeformation",
            data_augmentation_sd_affine=0.05,
            data_augmentation_sd_deformation=10.0,
            data_augmentation_noise_model="additivegaussian",
            data_augmentation_noise_parameters=(0.0, 0.05),
            data_augmentation_sd_simulated_bias_field=0.00000001,
            data_augmentation_sd_histogram_warping=0.025,
            number_of_samples=int(train_samples),
            aug_scheduler=aug_sched_fn,
        )
        train_ds.global_step_ref = global_step

        val_ds = antstorch.ImageDataset(
            images=(images_val if len(images_val) > 0 else images_train[:1]),
            template=tmpl,
            do_data_augmentation=True,
            data_augmentation_transform_type="affineAndDeformation",
            data_augmentation_sd_affine=0.0,
            data_augmentation_sd_deformation=0.0,
            data_augmentation_noise_model="additivegaussian",
            data_augmentation_noise_parameters=(0.0, 0.0),
            data_augmentation_sd_simulated_bias_field=0.0,
            data_augmentation_sd_histogram_warping=0.0,
            number_of_samples=int(val_samples),
        )

    else:
        
        train_ds = PNGMultiViewDataset(
            images_list=images_train,
            target_size=(H, W),
            do_aug=False
        )
        # L'attribut global_step_ref peut toujours être attaché dynamiquement
        train_ds.global_step_ref = global_step

        val_ds = PNGMultiViewDataset(
            images_list=(images_val if len(images_val) > 0 else images_train[:1]),
            target_size=(H, W),
            do_aug=False
        )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:    
        device = torch.device("cpu")

    use_pin_memory = (device.type == "cuda")

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  
                              num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=min(16, batch), shuffle=False, 
                              num_workers=max(1, num_workers // 2), pin_memory=use_pin_memory)
    return train_loader, val_loader, global_step

def ensure_shapes_cached(model, x_template: torch.Tensor):
    try:
        _ = model.sample(1)  # if cache exists this is a no-op
        return
    except Exception:
        pass
    with torch.no_grad():
        x1 = x_template[:1].to(next(model.parameters()).device).to(torch.float32)
        try:
            _ = model.log_prob(x1)
        except Exception:
            _ = model.inverse_and_log_det(x1)

@torch.no_grad()
def warmup_actnorm_with_real_batch(model, x_real: torch.Tensor):
    dev = next(model.parameters()).device
    x1 = x_real[:1].to(dev, torch.float32)
    for fn in ("log_prob", "inverse_and_log_det", "__call__"):
        if hasattr(model, fn):
            try:
                getattr(model, fn)(x1)
                break
            except Exception:
                continue

@torch.no_grad()
def _manual_prior_sample(model, n: int, temp: float = 1.0, x_template: torch.Tensor = None):
    p = next(model.parameters())
    dev, dt = p.device, torch.float32
    if x_template is None:
        H = W = 64
        if hasattr(model, "input_shape") and isinstance(model.input_shape, (tuple, list)) and len(model.input_shape) >= 3:
            H, W = int(model.input_shape[-2]), int(model.input_shape[-1])
        x_template = torch.randn(1, 1, H, W, device=dev, dtype=dt) * 0.1
    z_tmpl, _ = model.inverse_and_log_det(x_template[:1].to(dtype=dt, device=dev))
    if isinstance(z_tmpl, torch.Tensor):
        z_tmpl = [z_tmpl]
    z_list = [torch.randn(n, *z.shape[1:], device=dev, dtype=z.dtype) * float(temp) for z in z_tmpl]
    for fn in ("forward_from_latents", "forward", "sample_from_latents", "_forward"):
        if hasattr(model, fn):
            out = getattr(model, fn)(z_list)
            return out[0] if isinstance(out, (list, tuple)) else out
    s = model.sample(n, T=temp) if hasattr(model, "sample") and ("T" in model.sample.__code__.co_varnames) else model.sample(n)
    return s[0] if isinstance(s, (list, tuple)) else s

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser("Glow 2D (builder) trainer")
    
    # --- Input Data & Geometry ---
    ap.add_argument("--view", action="append", nargs="+", required=True,
                help="Path patterns for each imaging modality (e.g., T1, FA). Use one --view per modality. Files are paired across views by subject folder.")
    ap.add_argument("--H", type=int, default=128, 
                    help="Target image height (pixels) for resizing 2D inputs.")
    ap.add_argument("--W", type=int, default=128, 
                    help="Target image width (pixels) for resizing 2D inputs.")
    
    # --- Glow Architecture ---
    ap.add_argument("--L", type=int, default=4, 
                    help="Number of resolution levels in the multi-scale architecture.")
    ap.add_argument("--K", type=int, nargs="+", default=[32], 
                    help="Number of flow steps per resolution level. Accepts a single int or a list of ints matching L.")    
    ap.add_argument("--hidden", type=int, nargs="+", default=[96], 
                    help="Number of hidden channels in the convolutional coupling networks.")
    ap.add_argument("--base", type=str, default="glow", choices=["glow","diag"], 
                    help="Base distribution type for the latent space (Standard Gaussian vs. Learned Diagonal).")
    ap.add_argument("--glowbase-logscale-factor", type=float, default=3.0, 
                    help="Scaling factor applied to the log-scale output of the base distribution.")
    ap.add_argument("--glowbase-min-log", type=float, default=-5.0, 
                    help="Minimum clamp value for log-scale parameters (prevents numerical underflow).")
    ap.add_argument("--glowbase-max-log", type=float, default=5.0, 
                    help="Maximum clamp value for log-scale parameters (prevents numerical explosion).")
    ap.add_argument("--scale-map", type=str, default="tanh", choices=["tanh","exp","sigmoid","sigmoid_inv"], 
                    help="Activation function applied to the scale parameters (s) in the affine coupling layers.")
    ap.add_argument("--scale-cap", type=float, default=2.0, 
                    help="Hard limit on the scaling factor when using exp or tanh mapping to ensure stability.")
    ap.add_argument("--net-actnorm", action="store_true", 
                    help="If set, inserts ActNorm layers inside the coupling subnetworks.")

    # --- Training Loop & Logistics ---
    ap.add_argument("--batch", type=int, default=32, 
                    help="Per-GPU batch size. Reduce this if encountering CUDA Out Of Memory errors.")
    ap.add_argument("--train-samples", type=int, default=6000, 
                    help="Number of samples (or iterations) constituting one logical training epoch.")
    ap.add_argument("--val-samples", type=int, default=256, 
                    help="Number of samples used during the validation/evaluation phase.")
    ap.add_argument("--max-iter", type=int, default=30000, 
                    help="Absolute total number of training iterations before the script terminates.")
    ap.add_argument("--extra-iters", type=int, default=0, 
                    help="If >0, overrides --max-iter to train for exactly this many iterations past the resumed checkpoint.")
    ap.add_argument("--eval-interval", type=int, default=1000, 
                    help="Frequency (in iterations) to run validation, calculate NLL, and update the LR scheduler.")
    ap.add_argument("--plot-interval", type=int, default=1000, 
                    help="Frequency (in iterations) to generate and save sample/reconstruction preview grids.")
    ap.add_argument("--num-workers", type=int, default=4, 
                    help="Number of CPU subprocesses used for dataloading and ANTs augmentation.")

    # --- Hardware & Precision ---
    ap.add_argument("--devices", type=str, default="cuda:0", 
                    help="PyTorch device string (e.g., 'cuda:0', 'cpu', 'mps').")
    ap.add_argument("--precision", type=str, default="mixed", choices=["double","float","mixed"], 
                    help="Floating point precision. 'mixed' uses AMP for faster training.")
    ap.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16","fp16"], 
                    help="Data type for Automatic Mixed Precision. bf16 is recommended for Ampere+ GPUs.")
    ap.add_argument("--seed", type=int, default=0, 
                    help="Global random seed for PyTorch, NumPy, and dataloading determinism.")

    # --- Optimizer & Scheduler ---
    ap.add_argument("--lr", type=float, default=1e-4, 
                    help="Initial learning rate for the AdamW optimizer.")
    ap.add_argument("--weight-decay", type=float, default=1e-5, 
                    help="L2 regularization penalty applied to network weights (excluding ActNorm parameters).")
    ap.add_argument("--warmup-iters", type=int, default=800, 
                    help="Number of iterations over which the learning rate linearly scales from 0 to --lr.")
    ap.add_argument("--lr-decay-gamma", type=float, default=1.0, 
                    help="Multiplicative factor for StepLR decay. 1.0 disables step decay.")
    ap.add_argument("--lr-decay-steps", type=int, default=0, 
                    help="Number of iterations before applying the lr-decay-gamma reduction. 0 disables step decay.")
    ap.add_argument("--plateau-factor", type=float, default=0.5, 
                    help="Factor by which the learning rate is reduced when validation NLL plateaus.")
    ap.add_argument("--plateau-patience", type=int, default=4, 
                    help="Number of eval-intervals with no NLL improvement before the learning rate drops.")
    ap.add_argument("--plateau-threshold", type=float, default=1e-4, 
                    help="Minimum change in NLL to qualify as an improvement for the plateau scheduler.")
    ap.add_argument("--plateau-cooldown", type=int, default=0, 
                    help="Number of eval-intervals to wait after an LR reduction before resuming normal plateau monitoring.")
    ap.add_argument("--min-lr", type=float, default=1e-6, 
                    help="Absolute minimum learning rate floor for the plateau scheduler.")

    # --- Gradients & EMA ---
    ap.add_argument("--grad-clip", type=float, default=2.0, 
                    help="Maximum L2 norm for gradient clipping. Crucial for stability in Normalizing Flows.")
    ap.add_argument("--grad-accum", type=int, default=1,
                     help="Accumulate gradients over N micro-batches before optimizer.step(). Effective batch = batch * grad_accum.")
    ap.add_argument("--ema", action="store_true", 
                    help="Maintain an Exponential Moving Average of model weights for stabler sampling/evaluation.")
    ap.add_argument("--ema-decay", type=float, default=0.9995, 
                    help="Decay rate for the EMA weights. Higher values mean slower updates but more stability.")

    # --- Checkpointing & I/O ---
    ap.add_argument("--resume", type=str, default="", 
                    help="Explicit path to a checkpoint .pt file to resume training from.")
    ap.add_argument("--auto-resume", action="store_true", 
                    help="If set, automatically tries to load <out-dir>/training_state.pt when --resume is not provided.")
    ap.add_argument("--out-dir", type=str, default="runs_glow2d_builder", 
                    help="Directory where checkpoints, logs, and preview images will be saved.")

    # --- Dataset & Cohort Options ---
    ap.add_argument("--slice-idx", type=int, default=120, 
                    help="Z-axis slice index to extract consistently across the cohort/template.")
    ap.add_argument("--val-frac", type=float, default=0.10, 
                    help="Fraction of subjects held out for validation in cohort mode (e.g., 0.10 = 10%).")
    ap.add_argument("--subject-limit", type=int, default=0, 
                    help="(Debug) limit the number of subjects loaded; 0 means use all available subjects.")
    ap.add_argument("--smooth-alpha", type=float, default=0.1, 
                    help="EMA smoothing factor in (0,1] for metric logging; higher = faster adaptation to new values.")

    # --- Multimodal Latent Alignment ---
    ap.add_argument("--align", choices=["none","infonce","barlow","vicreg","hsic","pearson","mse"], default="none", 
                    help="Latent alignment loss function to synchronize representations across different views.")
    ap.add_argument("--align-weight", type=float, default=0.05, 
                    help="Fixed weight for alignment loss (used if --weighting=fixed).")
    ap.add_argument("--align-warmup", type=int, default=500, 
                    help="Number of iterations to gradually scale up alignment loss from 0 to its target weight.")
    ap.add_argument("--proj-dim", type=int, default=256, 
                    help="Output dimensionality of the projection head used before alignment.")
    ap.add_argument("--proj-hidden", type=int, default=512, 
                    help="Hidden layer dimensionality of the projection MLP.")
    ap.add_argument("--temperature", type=float, default=0.1, 
                    help="Temperature parameter for the InfoNCE contrastive loss.")
    ap.add_argument("--barlow-lambda", type=float, default=5e-3, 
                    help="Off-diagonal weight penalty (lambda) for Barlow Twins loss.")
    ap.add_argument("--weighting", choices=["fixed","kendall"], default="fixed", 
                    help="Loss weighting strategy. 'kendall' learns the balance between NLL and Alignment dynamically.")
    ap.add_argument("--init-logvar-nll", type=float, default=0.0, 
                    help="Initial log variance (s) for NLL when using Kendall weighting.")
    ap.add_argument("--init-logvar-align", type=float, default=0.0, 
                    help="Initial log variance (s) for Alignment when using Kendall weighting.")
    
    # --- VICReg Hyperparameters ---
    ap.add_argument("--vicreg-inv", type=float, default=25.0, 
                    help="VICReg invariance weight (pulls representations of the same subject closer).")
    ap.add_argument("--vicreg-cov", type=float, default=1.0,  
                    help="VICReg covariance weight (penalizes off-diagonal correlations to prevent collapse).")
    ap.add_argument("--vicreg-var", type=float, nargs="+", default=[25.0], 
                    help="VICReg variance weight. Pass multiple values for asymmetric weighting per view.")
    ap.add_argument("--vicreg-gamma", type=float, nargs="+", default=[1.0], 
                    help="VICReg variance floor. Pass multiple values for asymmetric targets per view.")    

    # --- HSIC Hyperparameters ---
    ap.add_argument("--hsic-sigma", type=float, default=0.0, 
                    help="RBF bandwidth for HSIC. 0 -> use the median pairwise distance heuristic per batch.")

    ap.add_argument("--use-ckpt-config", action="store_true",
                help="When resuming, override command-line architectural args with those saved in the checkpoint.")

    # --- Data Augmentation ---
    ap.add_argument("--aug-schedules", type=str,
        default=(
            "noise_std:cos:0.05->0.00@150k,"
            "sd_affine:linear:0.05->0.00@80k,"
            "sd_deformation:cos:0.20->0.00@100k,"
            "sd_simulated_bias_field:cos:1.00->0.00@120k,"
            "sd_histogram_warping:exp:0.05->0.00@120k"
        ),
        help="Multi-parameter anneal spec for ANTs data augmentation intensities (param:curve:start->end@iters).")
    ap.add_argument("--disable-aug-anneal", action="store_true", 
                    help="If set, uses static augmentation values from the dataset constructor instead of annealing.")

    # --- Previews & Grids ---
    ap.add_argument("--sample-mode", type=str, choices=["model","data","off"], default="model",
                    help="How to produce preview grids during eval: 'model' sampling from prior, random 'data' batch, or 'off'.")
    ap.add_argument("--sample-temp", type=float, default=1.0,
                help="Sampling temperature: scales prior noise (z = T·ε) when generating images. Lower T = sharper/less diverse.")
    ap.add_argument("--sample-grid-norm", type=str, choices=["to01","clamp","both"], default="to01",
                    help="How to normalize pixel intensities before saving preview grids. 'to01' min-max scales, 'clamp' strictly clips to [0,1].")

    # --- Screening (Shared Subspace Discovery) ---
    ap.add_argument("--screen", type=str, default="none", choices=["none","cca","hsic"],
                    help="Optional subspace screening method to filter out unshared information before applying alignment loss.")
    ap.add_argument("--screen-warmup", type=int, default=1000,
                    help="Iterations to wait before running the first screening pass.")
    ap.add_argument("--screen-refresh", type=int, default=0,
                    help="Recompute screening projection matrix every N iterations (0 = compute only once).")
    ap.add_argument("--screen-frac", type=float, default=0.5,
                    help="Fraction of total projected dimensions to retain as 'shared' (0,1].")
    ap.add_argument("--cca-ridge", type=float, default=1e-3,
                    help="Ridge regularization penalty for Canonical Correlation Analysis (CCA) numerical stability.")
    ap.add_argument("--prefilter-frac", type=float, default=0.5,
                help="Fraction of dimensions to keep during the preliminary Pearson filter step of HSIC screening.")
    
    args = ap.parse_args()
    args.num_views = len(args.view)

    if isinstance(args.K, list):
        if len(args.K) == 1:
            args.K = args.K[0]  # Rétrocompatibilité si une seule valeur est passée
        elif len(args.K) != args.L:
            raise ValueError(f"La longueur de K ({len(args.K)}) doit correspondre à L ({args.L}).")
        else:
            args.K = tuple(args.K)  # Conversion en tuple pour antstorch/normflows

    if isinstance(args.hidden, list):
        if len(args.hidden) == 1:
            args.hidden = args.hidden[0]  # Rétrocompatibilité si une seule valeur est passée
        elif len(args.hidden) != args.L:
            raise ValueError(f"La longueur de hidden ({len(args.hidden)}) doit correspondre à L ({args.L}).")
        else:
            args.hidden = tuple(args.hidden)  # Conversion en tuple pour antstorch/normflows

    # Device + precision
    set_deterministic(args.seed)
    dev = torch.device("mps" if args.devices == "mps" and torch.backends.mps.is_available() else "cpu")
    print(f"Utilisation du périphérique : {dev}")

    if args.precision == "double":
        model_dtype = torch.float64
        amp_enabled = False
        amp_dtype = None
    elif args.precision == "float":
        model_dtype = torch.float32
        amp_enabled = False
        amp_dtype = None
    else:
        model_dtype = torch.float32  # keep params in FP32
        amp_enabled = True
        if args.amp_dtype == "bf16" and dev.type == "cuda" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16

    scaler = torch.amp.GradScaler(
        enabled=(amp_enabled and amp_dtype == torch.float16),
        init_scale=2.0**12, growth_factor=2.0, backoff_factor=0.5, growth_interval=200
    )

    if not args.disable_aug_anneal:
        schedules = list(antstorch.parse_schedules(args.aug_schedules))
        sched = antstorch.MultiParamScheduler(schedules)
        def aug_sched_fn(step: int):
            return sched.step(step)
    else:
        aug_sched_fn = None

    _check_hw_divisible(args.H, args.W, args.L)

    # ---------------- Data ----------------
    
    try:
        train_loader, val_loader, global_step = build_loaders_from_globs(
            view_specs=args.view, H=args.H, W=args.W,
            train_samples=args.train_samples, val_samples=args.val_samples,
            batch=args.batch, num_workers=args.num_workers,
            slice_idx=args.slice_idx, val_frac=float(args.val_frac),
            subject_limit=(args.subject_limit if args.subject_limit > 0 else None),
            do_aug=True, aug_schedules=(args.aug_schedules if not args.disable_aug_anneal else None),
            disable_aug_anneal=args.disable_aug_anneal, seed=args.seed)
    except Exception as e:
        import traceback
        print("[data] failed to build loaders:", repr(e))
        traceback.print_exc()
        raise

    sample_batch = next(iter(train_loader))
    if sample_batch.ndim == 5: # Forme: (Batch, View, Channel, Height, Width)
        C = sample_batch.shape[2]
    else:                      # Forme: (Batch, Channel, Height, Width)
        C = sample_batch.shape[1]

    input_shape = (C, args.H, args.W)
    n_dims = int(np.prod(input_shape))
    print(f"[info] Modèle instancié avec {C} canal/canaux.")

    input_data_sampled = False

    # Build models using the builder
    from antstorch import create_glow_normalizing_flow_model_2d
    models: List[nf.Flow] = []
    for vi in range(args.num_views):
        m = create_glow_normalizing_flow_model_2d(
            input_shape=input_shape,
            L=args.L, K=args.K, hidden_channels=args.hidden,
            base=args.base,
            glowbase_logscale_factor=args.glowbase_logscale_factor,
            glowbase_min_log=args.glowbase_min_log,
            glowbase_max_log=args.glowbase_max_log,
            split_mode="channel",
            scale=True,
            scale_map=args.scale_map,
            leaky=0.0,
            net_actnorm=bool(args.net_actnorm),
            scale_cap=args.scale_cap,
        )
        # --- AJOUT CRITIQUE POUR LA VITESSE ---
        # 1. Envoyer sur GPU
        m = m.to(dtype=torch.float32, device=dev).float().train()
        
        # 2. Forcer l'init ActNorm MAINTENANT (sur GPU) avec un faux batch
        with torch.no_grad():
            print(f"[init] Initializing ActNorm for view {vi} on GPU...")
            dummy = torch.randn((1, *input_shape), device=dev, dtype=torch.float32)
            try:
                _ = m.log_prob(dummy)
            except Exception as e:
                print(f"[warn] ActNorm init warning: {e}")
        # --------------------------------------

        if not hasattr(m, 'input_shape'):
            m.input_shape = input_shape
            
        if torch.cuda.device_count() > 1 and len(args.devices.split(',')) > 1:
            print(f"[info] Wrapping model view {len(models)} in DataParallel on {args.devices}")
            # Utilisation du Wrapper que vous avez déjà ajouté
            m_step = GlowStepWrapper(m)
            m_parallel = GlowDataParallel(m_step, device_ids=[int(d.split(':')[-1]) for d in args.devices.split(',')])
            models.append(m_parallel)
        else:
            models.append(m)

    # ---------------- EMA (lazy init) ----------------
    ema_models = None  # created after first optimizer step

    # ---- One-time ActNorm warmup on REAL data (before projectors) ----
    with torch.no_grad():
        try:
            warm_batch = next(iter(train_loader))
            xs = _extract_views_from_batch(warm_batch, num_views=len(models))
            with torch.no_grad():
                # base
                for vi, m in enumerate(models):
                    _prime_if_needed(m, xs[vi])
                # ema
                if ema_models is not None:
                    for vi, em in enumerate(ema_models):
                        _prime_if_needed(em, xs[vi])
        except StopIteration:
            pass

    # --- projection heads for alignment ---
    projectors = None
    if args.align != "none":
        with torch.no_grad():
            x_tmpl = to01(xs[0][:1].to(dtype=torch.float32, device=dev))
            z_probe, _ = models[0].inverse_and_log_det(x_tmpl)
            flat_dim = _flatten_latents(z_probe).size(1)
        projectors = nn.ModuleList([
            Projector(flat_dim, args.proj_hidden, args.proj_dim).to(dtype=torch.float32, device=dev).train()
            for _ in range(len(models))
        ])

    # --- Kendall & Gal weighting scalars ---
    s_nll = s_align = None
    if args.weighting == "kendall" and args.align != "none":
        s_nll   = nn.Parameter(torch.tensor([args.init_logvar_nll], device=dev, dtype=torch.float32))
        s_align = nn.Parameter(torch.tensor([args.init_logvar_align], device=dev, dtype=torch.float32))

    # Optimizer + schedulers
    param_groups = [{"params": [p for m in models for p in m.parameters()]}]
    if projectors is not None:
        param_groups.append({"params": [p for p in projectors.parameters()]})
    if s_nll is not None:
        param_groups.append({"params": [s_nll, s_align], "weight_decay": 0.0})
    opt = torch.optim.Adamax(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    warm = make_warmup(opt, args.warmup_iters, args.lr_decay_gamma, args.lr_decay_steps)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=args.plateau_factor,
        patience=args.plateau_patience if hasattr(args, "plateau_patience") else 4,
        threshold=args.plateau_threshold, cooldown=args.plateau_cooldown, min_lr=args.min_lr
    )

    run_dir = Path(args.out_dir); run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "training_state.pt"
    csv_path   = run_dir / "metrics.csv"

    start_iter = 1
    resume_path = None
    if args.resume:
        rp = Path(args.resume)
        if rp.exists():
            resume_path = rp
        else:
            raise FileNotFoundError(f"--resume specified but file not found: {rp}")
    elif args.auto_resume and state_path.exists():
        resume_path = state_path

    ckpt_cfg = None
    if resume_path is not None:

        blob = torch.load(resume_path, map_location="cpu")
        ckpt_cfg = blob.get("config", {})
        # backward-compat: handle old checkpoints that used \"modalities\" list
        if ckpt_cfg and "num_views" not in ckpt_cfg and "modalities" in ckpt_cfg:
            try:
                args.num_views = len(ckpt_cfg.get("modalities") or [])
            except Exception:
                pass

        arch_keys = ["num_views","H","W","L","K","hidden","base",
                    "glowbase_logscale_factor","glowbase_min_log","glowbase_max_log",
                    "scale_map","scale_cap","net_actnorm"]  # add split_mode if exposed
        if args.use_ckpt_config and ckpt_cfg:
            for k in arch_keys:
                if k in ckpt_cfg:
                    setattr(args, k, ckpt_cfg[k])

        # optional: warn if you’re overriding user-provided values
        mismatches = [k for k in arch_keys if k in ckpt_cfg and getattr(args,k)!=ckpt_cfg[k]]
        if args.use_ckpt_config and mismatches:
            print("[resume] using checkpoint arch; overrides:", {k:(getattr(args,k), ckpt_cfg[k]) for k in mismatches})

        blob = torch.load(resume_path, map_location=dev, weights_only=False)
        start_iter = int(blob.get("iter", 1))
        try:
            opt.load_state_dict(blob["opt"])
        except Exception as e:
            print(f"[resume] optimizer state not loaded ({e}); using fresh optimizer.")
            # Optionally preserve LR/betas from ckpt group 0:
            try:
                g0 = blob["opt"]["param_groups"][0]
                for k in ("lr","betas","eps","weight_decay"):
                    if k in g0:
                        for g in opt.param_groups:
                            g[k] = g0[k]
            except Exception:
                pass

        if scaler is not None:
            if "scaler" in blob:
                try:
                    scaler.load_state_dict(blob["scaler"])
                    print("[resume] restored GradScaler state")
                except Exception as e:
                    print(f"[resume] GradScaler not loaded ({e}); starting fresh")
            else:
                print("[resume] no GradScaler state in checkpoint; starting fresh")
                
        if warm and blob.get("warm") is not None:
            warm.load_state_dict(blob["warm"])

        if blob.get("models") is not None:
            for m, sd in zip(models, blob["models"]):
                # --- CHARGEMENT ROBUSTE (SINGLE <-> MULTI GPU + WRAPPER) ---
                if isinstance(m, GlowDataParallel):
                    # Structure: DataParallel -> StepWrapper -> Glow
                    target_model = m.module.model
                    
                    # Nettoyage des clés
                    clean_sd = {}
                    for k, v in sd.items():
                        new_k = k.replace("module.", "").replace("model.", "")
                        clean_sd[new_k] = v
                    
                    target_model.load_state_dict(clean_sd)
                else:
                    # Cas Single-GPU
                    clean_sd = {}
                    for k, v in sd.items():
                        new_k = k.replace("module.", "").replace("model.", "")
                        clean_sd[new_k] = v
                    m.load_state_dict(clean_sd)

        if args.ema and blob.get("ema") is not None:
            import copy
            # Note : em est maintenant un GlowDataParallel car m l'est aussi
            ema_models = [copy.deepcopy(m).eval().to(dtype=torch.float32, device=dev) for m in models]
            for em in ema_models:
                for p in em.parameters():
                    p.requires_grad_(False)
            
            for em, sd in zip(ema_models, blob["ema"]):
                # --- CORRECTION EMA ---
                if isinstance(em, GlowDataParallel):
                    target_model = em.module.model
                    clean_sd = {}
                    for k, v in sd.items():
                        new_k = k.replace("module.", "").replace("model.", "")
                        clean_sd[new_k] = v
                    target_model.load_state_dict(clean_sd)
                else:
                    clean_sd = {}
                    for k, v in sd.items():
                        new_k = k.replace("module.", "").replace("model.", "")
                        clean_sd[new_k] = v
                    em.load_state_dict(clean_sd)

        if blob.get("proj") is not None and projectors is not None:
            try:
                projectors.load_state_dict(blob["proj"])
                tqdm.write("[resume] restored projectors")
            except Exception as e:
                tqdm.write(f"[resume] warning: could not load projectors: {e}")
        if blob.get("kendall") is not None and s_nll is not None:
            try:
                kd = blob["kendall"]
                if kd.get("s_nll") is not None:  s_nll.data.fill_(float(kd["s_nll"]))
                if kd.get("s_align") is not None: s_align.data.fill_(float(kd["s_align"]))
                tqdm.write(f"[resume] restored Kendall s_nll={float(s_nll.item()):.3f}, s_align={float(s_align.item()):.3f}")
            except Exception as e:
                tqdm.write(f"[resume] warning: could not load Kendall scalars: {e}")
        tqdm.write(f"[resume] from {str(resume_path)} @ iter {start_iter}")


    # ---- Prime latent-shape caches for ALL views (base + EMA) ----
    def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
        # Expect (B, C, H, W). If (B,H,W), add channel; if (H,W), add batch+channel.
        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        return x

    def _prime_latent_shapes(models, ema_models, loader, device):
        try:
            batch = next(iter(loader))
        except StopIteration:
            return  # nothing to prime
        xs = _extract_views_from_batch(batch, num_views=len(models))
        with torch.no_grad():
            for vi, m in enumerate(models):
                xb = xs[vi][:1]                      # 1 sample
                xb = _ensure_4d(xb).to(dtype=torch.float32).to(dtype=torch.float32, device=device, non_blocking=True)
                # match dtype to model (avoid half/float mismatch under mixed precision)
                p = next(m.parameters(), None)
                if p is not None and xb.dtype != p.dtype:
                    xb = xb.to(p.dtype)
                # disable autocast here; we just need to cache shapes cheaply
                with torch.amp.autocast(device_type=dev.type, enabled=False):
                    try:
                        _ = m.log_prob(xb)           # primes base model
                    except Exception as e:
                        print(f"[prime] base view{vi} log_prob failed: {e}")
                    if ema_models is not None:
                        try:
                            _ = ema_models[vi].log_prob(xb)  # primes EMA model too
                        except Exception as e:
                            print(f"[prime] ema  view{vi} log_prob failed: {e}")

    # Call it once after weights are loaded and loaders are built:
    _prime_latent_shapes(models, ema_models, train_loader, dev)

    if args.extra_iters > 0:
        args.max_iter = (start_iter - 1) + args.extra_iters

    if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "global_step_ref"):
        try:
            train_loader.dataset.global_step_ref.value = start_iter
        except Exception:
            pass
    if hasattr(val_loader, "dataset") and hasattr(val_loader.dataset, "global_step_ref"):
        try:
            val_loader.dataset.global_step_ref.value = start_iter
        except Exception:
            pass

    try:
        dataset_info = {
            "subjects_total": getattr(train_loader.dataset, "subjects_total", "n/a"),
            "train_images_list_len": len(getattr(train_loader.dataset, "images", [])),
            "val_images_list_len": len(getattr(val_loader.dataset, "images", [])),
            "effective_train_samples": args.train_samples,
            "effective_val_samples": args.val_samples,
            "batch_size": args.batch,
            "grad_accum": int(getattr(args, "grad_accum", 1)),
            "effective_batch_size": int(args.batch) * int(getattr(args, "grad_accum", 1)),
        }
    except Exception:
        dataset_info = {"note": "dataset stats unavailable (non-ANTs dataset type)"}

    screen_dump_run_config(args, Path(args.out_dir), note="post-dataset build", dataset_info=dataset_info)

    if not csv_path.exists():
            with open(csv_path, "w") as f:
                f.write("iter,loss,sum_bpd,lr\n")
    else:
        # Troncature du CSV pour éviter la corruption du graphique lors d'un --resume
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            # Ne conserver que les historiques strictement antérieurs au point de reprise
            df = df[df['iter'] < start_iter]
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"[warn] Impossible de nettoyer le CSV : {e}")

    with global_step.get_lock():
        global_step.value = int(start_iter)
    train_iter = iter(train_loader)

    # ------------------------- train loop -------------------------
    n_views = len(models)

    # screening state
    screen_state: ScreenState = None
    tqdm.write(f"[info] training {n_views} view(s); params per view: {[n_params(m) for m in models]}")

    alpha = float(args.smooth_alpha)
    ema_loss_disp = None
    ema_sum_bpd_disp = None
    ema_bpd_views_disp = [None] * n_views

    pbar = tqdm(total=args.max_iter, initial=start_iter - 1, dynamic_ncols=True, desc="train")

    for it in range(start_iter, args.max_iter + 1):
        grad_accum = max(1, int(getattr(args, "grad_accum", 1)))
        opt.zero_grad(set_to_none=True)

        # Accumulators for logging (average over micro-batches)
        loss_total_acc = torch.tensor(0.0, device=dev, dtype=torch.float32)
        L_align_acc    = torch.tensor(0.0, device=dev, dtype=torch.float32)
        sum_bpd_acc    = 0.0
        bpd_views_acc  = None  # lazily init after first micro-batch
        w_nll = 1.0
        w_align = float(args.align_weight if args.align != "none" else 0.0)

        bad_update = False
        x_last = None

        for micro in range(grad_accum):
            try:
                x = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x = next(train_iter)

                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            x_last = x

            L_nll = torch.tensor(0.0, device=dev, dtype=torch.float32)
            curr_bpd_views = []
            sum_bpd = 0.0
            lat_flat = []

            if amp_enabled:
                from contextlib import nullcontext
                ctx = torch.amp.autocast(dev.type, dtype=amp_dtype)
            else:
                from contextlib import nullcontext
                ctx = nullcontext()

            xs_train = _extract_views_from_batch(x, num_views=len(models))

            with ctx:
                bad_batch = False
                for vi, m in enumerate(models):
                    x_v = to01(xs_train[vi].to(dtype=torch.float32, device=dev))
                    
                    # --- GESTION MULTI-GPU ---
                    if isinstance(m, GlowDataParallel):
                        logp_v, zflat = m(x_v.float())
                    else:
                        logp_v = m.log_prob(x_v.float())
                        z_v, _ = m.inverse_and_log_det(x_v.float())
                        # Note: pour la 2D, votre _flatten_latents actuel est suffisant
                        # pas besoin du pooling adaptatif du 3D sauf si OOM
                        zflat = _flatten_latents(z_v)
                    # -------------------------

                    if not torch.isfinite(logp_v).all():
                        tqdm.write(f"[nan] non-finite logp in view {vi}...")
                        bad_batch = True
                        break

                    # 1. Calcul du Bits Per Dimension (BPD) pour la vue courante
                    n_dims = float(np.prod(x_v.shape[1:]))
                    bpd_v = bits_per_dim(logp_v, n_dims).mean()
                    
                    # 2. Accumulation de la perte NLL globale 
                    # L'absence de '.item()' est cruciale ici pour préserver le graphe de gradient
                    L_nll = L_nll + bpd_v
                    
                    # 3. Mise à jour des compteurs pour l'affichage (tqdm)
                    curr_bpd_views.append(bpd_v.item())
                    sum_bpd += bpd_v.item()

                    # ... suite du calcul BPD ...
                    
                    # ATTENTION : Ne recalculez pas zflat ici !
                    lat_flat.append(torch.nan_to_num(zflat))

            if bad_batch or (not torch.isfinite(L_nll)) or abs(float(L_nll.item())) > 100.0:
                tqdm.write(f"[anomaly] skipping update at iter {it} (bad_batch={bad_batch}, L_nll={float(L_nll.item()):.2f})")
                bad_update = True
                break

            L_align = torch.tensor(0.0, dtype=torch.float32, device=dev)
            if args.align != "none" and it >= args.align_warmup:
                # 1) Build per-view features (post-projector) BEFORE screening
                feats = [projectors[i](lat_flat[i]) for i in range(len(lat_flat))]
                feats = [f.float() for f in feats]

                # 2) Optional shared-subspace screening
                if args.screen != "none" and it >= args.screen_warmup:
                    # Semantics:
                    #   screen_refresh == 0  → discover once at first eligible iter
                    #   screen_refresh > 0   → recompute every screen_refresh iters
                    if screen_state is None:
                        do_refresh = True
                    else:
                        do_refresh = (
                            args.screen_refresh > 0
                            and (it - args.screen_warmup) % args.screen_refresh == 0
                        )

                    screen_state = update_screen(
                        feats,
                        state=screen_state,
                        method=args.screen,           # 'cca' | 'hsic'
                        keep_frac=args.screen_frac,   # e.g., 0.5
                        ridge=args.cca_ridge,         # CCA ridge
                        prefilter_frac=args.prefilter_frac,
                        refresh=do_refresh,
                    )
                    feats_screened = apply_screen(feats, screen_state)

                else:
                    feats_screened = feats

                # 3) Alignment loss
                if args.align == "barlow":
                    L_align = antstorch.barlow_twins_multi(feats_screened, lam=float(args.barlow_lambda))
                elif args.align == "vicreg":
                    # 1. Obtenir les composantes Invariance et Covariance (symétriques)
                    L_inv_cov = antstorch.vicreg_multi(
                        feats_screened,
                        w_inv=float(args.vicreg_inv),
                        w_var=0.0, # Désactivé temporairement
                        w_cov=float(args.vicreg_cov),
                        gamma=1.0
                    )
                    
                    # 2. Calculer la Variance asymétrique
                    L_var = torch.tensor(0.0, device=dev)
                    for vi, feat in enumerate(feats_screened):
                        # Fallback au premier élément si l'utilisateur ne fournit qu'un seul scalaire
                        w_var_v = args.vicreg_var[vi] if vi < len(args.vicreg_var) else args.vicreg_var[0]
                        gamma_v = args.vicreg_gamma[vi] if vi < len(args.vicreg_gamma) else args.vicreg_gamma[0]
                        
                        std = torch.sqrt(feat.var(dim=0) + 1e-04)
                        L_var += w_var_v * torch.mean(F.relu(gamma_v - std))
                        
                    L_align = L_inv_cov + L_var
                elif args.align == "infonce":
                    L_align = antstorch.info_nce_multi(feats_screened, T=float(args.temperature))
                elif args.align == "hsic":
                    L_align = antstorch.hsic_multi(feats_screened, sigma=float(args.hsic_sigma))
                elif args.align == "pearson":
                    L_align = antstorch.pearson_multi(feats_screened)
                elif args.align == "mse":
                    L_align = antstorch.lpnorm_multi(feats, p=2.0)

            if args.weighting == "fixed" or args.align == "none":
                loss_total = L_nll + (args.align_weight * L_align if args.align != "none" else 0.0)
                w_nll = 1.0
                w_align = float(args.align_weight if args.align != "none" else 0.0)
            else:
                # On limite uniquement les paramètres de pondération adaptative (s_nll, s_align)
                s_nll_eff   = torch.clamp(s_nll, -5.0, 5.0)
                s_align_eff = torch.clamp(s_align, -5.0, 5.0)
                
                # Calcul de la perte totale SANS utiliser nan_to_num sur les tenseurs du modèle
                loss_total = torch.exp(-s_nll_eff) * L_nll + s_nll_eff
                loss_total = loss_total + torch.exp(-s_align_eff) * L_align + s_align_eff

            # Vérification de sécurité (déjà présente dans votre code mais on l'assure)
            if not torch.isfinite(loss_total):
                tqdm.write(f"[nan] loss_total non-finite at iter {it}; skipping update")
                bad_update = True
                break

            # Backprop (scaled by grad_accum so effective step matches larger batch)
            loss_scaled = loss_total / float(grad_accum)
            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # Logging accumulators (micro-batch averages)
            loss_total_acc = loss_total_acc + loss_total.detach().float()
            L_align_acc    = L_align_acc + L_align.detach().float()
            sum_bpd_acc   += float(sum_bpd)
            if bpd_views_acc is None:
                bpd_views_acc = [0.0 for _ in range(len(curr_bpd_views))]
            for _i in range(len(curr_bpd_views)):
                bpd_views_acc[_i] += float(curr_bpd_views[_i])

        if bad_update:
            opt.zero_grad(set_to_none=True)
            continue

        # Step optimizer once per accumulated update
        if scaler.is_enabled():
            scaler.unscale_(opt)
        params_to_clip = []
        for g in opt.param_groups:
            params_to_clip.extend(g["params"])
        torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=float(getattr(args, "grad_clip", 2.0)))
        if scaler.is_enabled():
            scaler.step(opt); scaler.update()
        else:
            opt.step()

        # Use last micro-batch for any "real batch" operations downstream (e.g., EMA ActNorm warmup)
        x = x_last

        # Publish averaged tensors/metrics under the names the rest of the trainer expects
        loss_total = loss_total_acc / float(grad_accum)
        L_align    = L_align_acc / float(grad_accum)
        sum_bpd    = float(sum_bpd_acc) / float(grad_accum)
        curr_bpd_views = [float(v) / float(grad_accum) for v in (bpd_views_acc or [])]
        if args.ema and ema_models is None:
            import copy
            ema_models = [copy.deepcopy(m).eval().to(dtype=torch.float32, device=dev) for m in models]
            for em in ema_models:
                for p in em.parameters():
                    p.requires_grad_(False)

            with torch.no_grad():
                xs_ema = _extract_views_from_batch(x, num_views=len(models))
                for vi, (m, em) in enumerate(zip(models, ema_models)):
                    _copy_actnorm_state(m, em)
                    xv_real = to01(xs_ema[vi].to(dtype=torch.float32, device=dev)).float()
                    warmup_actnorm_with_real_batch(em, xv_real)

            tqdm.write("[ema] initialized from base after first update")

        if ema_models is not None:
            with torch.no_grad():
                for em, m in zip(ema_models, models):
                    for p_em, p in zip(em.parameters(), m.parameters()):
                        p_em.data.mul_(args.ema_decay).add_(p.data, alpha=1.0 - args.ema_decay)

        if warm is not None and it <= args.warmup_iters:
            warm.step()

        with global_step.get_lock():
            global_step.value += 1

        lr_now = opt.param_groups[0]["lr"]

        curr_loss = float(loss_total.detach().cpu().item())
        if ema_loss_disp is None:
            ema_loss_disp = curr_loss
            ema_sum_bpd_disp = sum_bpd
            for i in range(n_views):
                ema_bpd_views_disp[i] = curr_bpd_views[i]
        else:
            a = alpha
            ema_loss_disp = (1.0 - a) * ema_loss_disp + a * curr_loss
            ema_sum_bpd_disp = (1.0 - a) * ema_sum_bpd_disp + a * sum_bpd
            for i in range(n_views):
                ema_bpd_views_disp[i] = (1.0 - a) * ema_bpd_views_disp[i] + a * curr_bpd_views[i]

        postfix = {
            "iter": it, "loss": f"{curr_loss:.4f}", "loss~": f"{ema_loss_disp:.4f}",
            "bpd": f"{sum_bpd:.3f}", "bpd~": f"{ema_sum_bpd_disp:.3f}", "lr": f"{lr_now:.2e}",
            "align": f"{float(L_align.detach().cpu().item()):.4f}", "mode": args.align,
            "w_nll": f"{w_nll:.2f}", "w_aln": f"{w_align:.2f}",
        }
        for i in range(n_views):
            postfix[f"v{i}"] = f"{curr_bpd_views[i]:.3f}/{ema_bpd_views_disp[i]:.3f}"
        pbar.set_postfix(postfix); pbar.update(1)

        if not input_data_sampled:
            with torch.no_grad():
                eval_models = ema_models if ema_models is not None else models
                num_views = len(eval_models)
                ok, err = save_coordinated_input_grids(
                    train_loader, num_views=num_views, out_dir=run_dir, fallback_loader=train_loader,
                    n=100, nrow=10, target_hw=(args.H, args.W), device=dev,
                )
                if ok:
                    tqdm.write(f"[samples] saved coordinated input data grids @ iter {it}")
                    input_data_sampled = True
                else:
                    tqdm.write(f"[warn] input data grid failed @ iter {it}: {err}")

        if it % args.eval_interval == 0:
            with torch.no_grad():
                eval_models = ema_models if ema_models is not None else models
                bpd_acc = []
                tmpl_by_view = [None] * len(eval_models)
                vbar = tqdm(total=10, leave=False, dynamic_ncols=True, desc=f"val@{it}")

                for j, batch_val in enumerate(val_loader):
                    xs_val = _extract_views_from_batch(batch_val, num_views=len(eval_models))
                    bpd_views = []
                    for vi, m in enumerate(eval_models):
                        xv = to01(xs_val[vi].to(dtype=torch.float32, device=dev))

                        tmpl_by_view[vi] = xv
                        lp = m.log_prob(xv.float())
                        lp = torch.nan_to_num(lp, nan=-1e9, posinf=-1e9, neginf=-1e9)
                        bpd_views.append(bits_per_dim(lp, n_dims).mean().item())
                    bpd_acc.append(np.mean(bpd_views))
                    vbar.update(1)
                    if len(bpd_acc) >= 10:
                        break
                vbar.close()
                avg_bpd = float(np.mean(bpd_acc)) if bpd_acc else float("nan")
            plateau.step(avg_bpd)
            lr_now = opt.param_groups[0]["lr"]

            tqdm.write(f"[eval] iter={it} avg_bpd={avg_bpd:.4f} lr={lr_now:.2e}")

            with torch.no_grad():
                eval_models = ema_models if ema_models is not None else models
                if args.sample_mode == "model":
                    any_ok = False
                    n_samples, nrow = 100, 10
                    shared_seed = int(getattr(args, "seed", 12345)) + int(it)
                    for vi, m in enumerate(eval_models):
                        if tmpl_by_view[vi] is None:
                            tqdm.write(f"[warn] no real template available for view {vi}; skipping model samples this eval")
                            continue
                        _prime_if_needed(m, tmpl_by_view[vi])
                        warmup_actnorm_with_real_batch(m, tmpl_by_view[vi])
                        cpu_state = torch.random.get_rng_state()
                        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                        try:
                            torch.manual_seed(shared_seed)
                            ok, err = _save_samples_grid(
                                m, n_samples, args.sample_temp,
                                run_dir / f"samples_view{vi}_it{it:06d}",
                                nrow=nrow, target_hw=(args.H, args.W),
                                warm_x=tmpl_by_view[vi],  
                                which_type=args.sample_grid_norm, 
                            )
                        finally:
                            torch.random.set_rng_state(cpu_state)
                            if cuda_states is not None:
                                torch.cuda.set_rng_state_all(cuda_states)
                        if not ok:
                            import traceback
                            tqdm.write(f"[warn] model sampling failed for view {vi} at iter {it}: {err}")
                            traceback.print_exc()
                        any_ok = any_ok or ok
                    if any_ok:
                        tqdm.write(f"[samples] saved *coordinated* model sample grids @ iter {it}")
                elif args.sample_mode == "data":
                    ok, err = save_coordinated_input_grids(
                        val_loader, num_views=len(eval_models), out_dir=run_dir,
                        n=100, nrow=10, target_hw=(args.H, args.W), device=dev,
                    )
                    if ok:
                        tqdm.write(f"[samples] saved *coordinated* validation-batch grids @ iter {it}")
                    else:
                        tqdm.write(f"[warn] val-batch grid failed at iter {it}: {err}")
                else:
                    tqdm.write("[samples] skipping previews (--sample-mode off)")
            _save_metric_plots(csv_path, run_dir, remove_spikes=True)

        with open(csv_path, "a") as f:
            f.write(f"{it},{curr_loss:.6f},{sum_bpd:.6f},{lr_now:.6g}\n")

        if it % args.eval_interval == 0:
            blob = {
                "iter": it + 1,
                "opt": opt.state_dict(),
                "warm": (warm.state_dict() if warm else None),
                "models": [m.state_dict() for m in models],
                "ema": ([em.state_dict() for em in ema_models] if ema_models is not None else None),
                "proj": (projectors.state_dict() if projectors is not None else None),
                "kendall": ({
                    "s_nll": float(s_nll.detach().cpu()) if s_nll is not None else None,
                    "s_align": float(s_align.detach().cpu()) if s_align is not None else None,
                }),
                "config": vars(args),
                "scaler": (scaler.state_dict() if scaler is not None and scaler.is_enabled() else None),
            }
            
            # 1. Sauvegarde du fichier 'latest' pour l'auto-resume
            torch.save(blob, state_path)
            
            # 2. Sauvegarde d'une version horodatée
            iter_state_path = run_dir / f"training_state_it{it:06d}.pt"
            torch.save(blob, iter_state_path)
            
            # 3. Nettoyage : ne garder que les jalons (ex: tous les 10 000 itérations)
            cleanup_checkpoints(run_dir, keep_every=10000)
            
            tqdm.write(f"[ckpt] saved: {str(iter_state_path)} (and updated latest)")

    pbar.close()
    print("Done. Run dir:", str(run_dir))

if __name__ == "__main__":
    main()


