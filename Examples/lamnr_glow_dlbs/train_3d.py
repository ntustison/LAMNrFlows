
import argparse
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
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
import normflows as nf

from contextlib import nullcontext
from multiprocessing import Value  # optional but recommended if num_workers>0

from datetime import datetime
import json, platform

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
        
        # Aplatissement local
        z_flat = _flatten_latents(z)
        return log_prob, z_flat

    # Redirections pour le priming et l'échantillonnage
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

    # Core architecture & training knobs
    add("out_dir", cfg.get("out_dir"))
    add("views", getattr(args, "num_views", None))
    add("H×WxD", f"{cfg.get('H')}×{cfg.get('W')}×{cfg.get('D')}")
    add("L / K / hidden", f"{cfg.get('L')} / {cfg.get('K')} / {cfg.get('hidden')}")
    add("align", cfg.get("align"))
    add("weighting", cfg.get("weighting"))
    add("batch", cfg.get("batch"))
    add("grad_accum", cfg.get("grad_accum"))
    add("effective_batch", cfg.get("effective_batch"))
    add("max_iter", cfg.get("max_iter"))
    add("extra_iters", cfg.get("extra_iters"))
    add("lr / warmup", f"{cfg.get('lr')} / {cfg.get('warmup_iters')}")
    add("ema / decay", f"{_fmt_bool(cfg.get('ema'))} / {cfg.get('ema_decay')}")
    add("precision", cfg.get("precision"))
    add("devices", cfg.get("devices"))
    add("slice_idx", cfg.get("slice_idx"))
    add("val_frac", cfg.get("val_frac"))
    add("train_samples / val_samples", f"{cfg.get('train_samples')} / {cfg.get('val_samples')}")
    add("num_workers", cfg.get("num_workers"))
    add("seed", cfg.get("seed"))
    add("smooth_alpha", cfg.get("smooth_alpha"))
    add("sample_mode / temp", f"{cfg.get('sample_mode')} / {cfg.get('sample_temp')}")
    add("disable_aug_anneal", _fmt_bool(cfg.get("disable_aug_anneal")))
    add("aug_schedules", cfg.get("aug_schedules"))

    # --- NEW: screening configuration ---
    add("screen", cfg.get("screen"))
    add("screen_frac", cfg.get("screen_frac"))
    add("screen_warmup / refresh",
        f"{cfg.get('screen_warmup')} / {cfg.get('screen_refresh')}")
    add("cca_ridge", cfg.get("cca_ridge"))
    add("prefilter_frac", cfg.get("prefilter_frac"))
    # ------------------------------------

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

def _check_hw_divisible(
    H: int,
    W: int,
    L: int,
    D: int | None = None,
    spatial_dims: int = 2,
):
    """
    Ensure spatial dims are divisible by 2**L.

    For 2D, checks H and W.
    For 3D, also checks D.
    """
    r = 2 ** L
    if (H % r) or (W % r):
        raise ValueError(f"H and W must be divisible by 2**L={r}. Got H={H}, W={W}, L={L}")
    if spatial_dims == 3:
        if D is None:
            raise ValueError("D must be provided when spatial_dims=3.")
        if D % r:
            raise ValueError(f"D must be divisible by 2**L={r}. Got D={D}, L={L}")


def to01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # 1. Élimination pure et simple des NaNs/Infs issus de l'augmentation ANTs
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Sécurité : on ignore les tenseurs qui n'ont pas de dimensions spatiales
    if x.ndim < 4:
        return x
        
    # 2. Détection dynamique des axes spatiaux (ex: (2, 3) en 2D, (2, 3, 4) en 3D)
    spatial_dims = tuple(range(2, x.ndim))
    
    # 3. Normalisation Min-Max
    x_min = x.amin(dim=spatial_dims, keepdim=True)
    x_max = x.amax(dim=spatial_dims, keepdim=True)
    norm = (x - x_min) / (x_max - x_min + eps)
    
    # 4. Sécurité Logit (empêcher le 0.0 et 1.0 absolus)
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
    Aplatit les latents pour l'alignement (Projector).
    Applique d'abord un pooling adaptatif pour éviter une explosion de la mémoire 
    avec des volumes 3D de haute résolution (ex: 128^3).
    
    Args:
        z: Tenseur ou liste de tenseurs (sortie de model.inverse_and_log_det)
        target_pool_size (int): Taille spatiale cible (ex: 4 -> 4x4 ou 4x4x4).
                                Réduit la dimensionnalité tout en gardant une 
                                information structurelle grossière.
    """
    zs = z if isinstance(z, (list, tuple)) else [z]
    flattened_list = []
    
    for zi in zs:
        # Si c'est un volume 3D (N, C, D, H, W)
        if zi.ndim == 5:
            # Réduit à (N, C, 4, 4, 4) peu importe la taille d'entrée
            zi_pooled = F.adaptive_avg_pool3d(zi, (target_pool_size, target_pool_size, target_pool_size))
            flattened_list.append(zi_pooled.flatten(1))
            
        # Si c'est une image 2D (N, C, H, W)
        elif zi.ndim == 4:
            # Réduit à (N, C, 4, 4)
            zi_pooled = F.adaptive_avg_pool2d(zi, (target_pool_size, target_pool_size))
            flattened_list.append(zi_pooled.flatten(1))
            
        # Si c'est déjà plat ou autre (N, D)
        else:
            flattened_list.append(zi.flatten(1))

    return torch.cat(flattened_list, dim=1)  # [B, Total_Reduced_Features]

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
        state.masks = [m.to(device=device) for m in masks]
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
    Normalize a multi-view batch into a list [x_view0, x_view1, ...].

    Supported input forms:
      - dict with 'x' or 'views' (list/tuple of tensors)
      - list/tuple of per-view tensors
      - tuple like (x, y, ...) where x is a tensor or multi-view container
      - torch.Tensor of shape:
          (B, V, *spatial)                     -> unstack along V (views along dim=1)
          (B, C_total, *spatial) with C_total % num_views == 0 -> split channels
    """
    import torch

    # Unwrap simple containers
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
        if num_views is None or num_views <= 1:
            return [batch]

        if batch.ndim == 5:
            # 3D or channel-augmented tensors: (B, V_or_C, D, H, W)
            B, C_or_V, D, H, W = batch.shape
            if C_or_V == num_views:
                # treat dim 1 as views
                return [batch[:, vi:vi+1, ...] for vi in range(num_views)]
            if C_or_V % num_views != 0:
                raise ValueError(
                    f"Cannot split (B,C,D,H,W)=({B},{C_or_V},{D},{H},{W}) into {num_views} views: "
                    f"dim1 ({C_or_V}) not divisible by num_views."
                )
            Cpv = C_or_V // num_views
            return [batch[:, vi*Cpv:(vi+1)*Cpv, ...] for vi in range(num_views)]

        elif batch.ndim == 4:
            # 2D tensors: (B, V_or_C, H, W)
            B, C_or_V, H, W = batch.shape
            if C_or_V == num_views:
                return [batch[:, vi:vi+1, ...] for vi in range(num_views)]
            if C_or_V % num_views != 0:
                raise ValueError(
                    f"Cannot split (B,C,H,W)=({B},{C_or_V},{H},{W}) into {num_views} views: "
                    f"dim1 ({C_or_V}) not divisible by num_views."
                )
            Cpv = C_or_V // num_views
            return [batch[:, vi*Cpv:(vi+1)*Cpv, :, :] for vi in range(num_views)]

        else:
            raise ValueError(f"Unsupported tensor ndim={batch.ndim}; expected 4 or 5.")

    raise ValueError(f"Unsupported batch type for multi-view extraction: {type(batch)}")

from pathlib import Path
import torch
import torch.nn.functional as F

def save_input_grids_any(
    train_loader,
    val_loader,
    num_views: int,
    out_dir: Path,
    max_per_view: int,
    grid_cols: int,
    global_step: int,
    logger,
):
    """
    Save per-view input grids for both 2D and 3D runs.

    - For 2D, behaves like the original coordinated input grids.
    - For 3D, uses the center slice along D for each view.
    """
    from torchvision.utils import make_grid, save_image

    def _get_first_batch(loader):
        try:
            return next(iter(loader))
        except StopIteration:
            return None

    def _coerce_batch_to_tensor(batch):
        # Mirror what the training loop expects
        if isinstance(batch, (list, tuple)):
            # most likely (images, *meta)
            return batch[0]
        if isinstance(batch, dict) and "image" in batch:
            return batch["image"]
        return batch

    def _save_for_loader(loader, split_name: str):
        if loader is None:
            return
        batch = _get_first_batch(loader)
        if batch is None:
            logger.warning(f"[input-grids] no batch available for {split_name}")
            return

        x = _coerce_batch_to_tensor(batch)
        xs = _extract_views_from_batch(x, num_views=num_views)  # uses same logic as training

        split_dir = out_dir / f"grids_input_{split_name}"
        split_dir.mkdir(parents=True, exist_ok=True)

        # For each view, make a small grid of center slices
        for vi in range(min(num_views, len(xs))):
            x_v = xs[vi]  # shape: (N, C, H, W) or (N, C, D, H, W)
            if not torch.is_tensor(x_v):
                continue

            # normalize to [0,1] over spatial dims
            x_v = to01(x_v.clone())

            # Limit number of examples per view
            x_v = x_v[:max_per_view]

            # _coerce_nchw_4d handles 3D volumes by taking center slice along D
            imgs = _coerce_nchw_4d(x_v)

            # Make a grid and save
            grid = make_grid(
                imgs,
                nrow=max(1, grid_cols),
                padding=2,
                normalize=False,
            )
            out_path = split_dir / f"input_data_view{vi}_it{global_step:06d}.png"
            save_image(grid, out_path)
            logger.info(
                f"[input-grids] wrote {out_path} "
                f"(split={split_name}, view={vi}, n={imgs.size(0)})"
            )

    _save_for_loader(train_loader, "train")
    _save_for_loader(val_loader, "val")
    return True, None

from pathlib import Path
import torch

from pathlib import Path
import torch

def save_coordinated_input_grids(
    loader,
    num_views,
    out_dir,
    fallback_loader=None,
    n=100,
    nrow=10,
    target_hw=None,
    device="cpu",
):
    """
    Build one grid per view by collecting up to `n` examples across
    one or more batches of `loader` (or `fallback_loader`).

    Works for:
      - 2D: tensors shaped (N, C, H, W)
      - 3D: tensors shaped (N, C, D, H, W); _coerce_nchw_4d will take
             the center slice along the last spatial axis to get (N, C, H, W).

    Writes:
      out_dir/input_data_view0.png
      out_dir/input_data_view1.png
      ...
    """
    import torch
    from torchvision.utils import make_grid, save_image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def collect_from_loader(ld):
        samples_per_view = [[] for _ in range(num_views)]
        collected = 0
        for batch in ld:
            # unwrap batch to a tensor x
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            elif isinstance(batch, dict) and "image" in batch:
                x = batch["image"]
            else:
                x = batch

            xs = _extract_views_from_batch(x, num_views=num_views)
            if len(xs) != num_views:
                return None, f"Expected {num_views} views, got {len(xs)}."

            B = xs[0].shape[0]
            take = min(n - collected, B)
            if take > 0:
                for vi in range(num_views):
                    # keep raw tensors on CPU for now
                    xvi = xs[vi][:take].detach().cpu()
                    samples_per_view[vi].append(xvi)
                collected += take
            if collected >= n:
                break

        if collected == 0:
            return None, "loader yielded no samples."
        stacked = [torch.cat(vs, dim=0)[:n] for vs in samples_per_view]
        return stacked, None

    # try main loader, then fallback if needed
    try:
        result, err = collect_from_loader(loader)
        if result is None:
            if fallback_loader is not None:
                result, err_fb = collect_from_loader(fallback_loader)
                if result is None:
                    return False, f"val+fallback loaders failed: {err}; {err_fb}"
            else:
                return False, f"loader failed: {err}"
    except Exception as e:
        if fallback_loader is not None:
            try:
                result, err_fb = collect_from_loader(fallback_loader)
                if result is None:
                    return False, f"loader+fallback failed: {e}; {err_fb}"
            except Exception as e2:
                return False, f"loader+fallback exception: {e}; {e2}"
        else:
            return False, f"loader exception: {e}"

    # now build and save grids per view
    for vi in range(num_views):
        x_v = result[vi].to(device)

        # normalize to [0,1] and coerce:
        #   - for 5D (N,C,D,H,W) → center-slice to (N,C,H,W)
        #   - for C>3 → grayscale
        x_v = to01(x_v)
        imgs = _coerce_nchw_4d(x_v, target_hw=target_hw)

        grid = make_grid(
            imgs,
            nrow=max(1, nrow),
            padding=2,
            normalize=False,
        )
        out_path = out_dir / f"input_data_view{vi}.png"
        save_image(grid, out_path)

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
    """
    Coerce sample outputs to (N, C, H, W) for grid saving.

    Supports:
      - 2D tensors: (N, C, H, W), (C, H, W), (H, W, C)
      - 3D volumes: (N, C, S0, S1, S2) by taking the center slice along
        the *last* spatial dimension S2 -> (N, C, S0, S1).
      - Lists/tuples of such tensors: pick the candidate with the largest
        spatial area of the last two dims.
    """
    import torch

    # If we got a list/tuple, pick the largest spatial candidate
    if isinstance(x, (list, tuple)):
        cands = [t for t in x if torch.is_tensor(t) and t.dim() in (3, 4, 5)]
        if not cands:
            raise ValueError("No tensor candidates in sample output.")
        areas, fixed = [], []
        for t in cands:
            if t.dim() == 5:
                # Treat as (N, C, S0, S1, S2); project to 2D via center slice on last dim
                mid = t.shape[-1] // 2
                t = t[..., mid]  # (N, C, S0, S1)
            elif t.dim() == 3:
                # (C, H, W) or (H, W, C)
                if t.shape[-1] in (1, 3) and (t.shape[0] not in (1, 3)):
                    t = t.permute(2, 0, 1).contiguous()
                t = t.unsqueeze(0)  # (1, C, H, W)
            elif t.dim() == 4:
                # (N, H, W, C) -> (N, C, H, W) if needed
                if t.shape[-1] in (1, 3) and t.shape[1] not in (1, 3):
                    t = t.permute(0, 3, 1, 2).contiguous()
            fixed.append(t)
            H, W = int(t.shape[-2]), int(t.shape[-1])
            areas.append(H * W)
        x = fixed[int(torch.tensor(areas).argmax().item())]

    if not torch.is_tensor(x):
        raise ValueError(f"Sample output is not a tensor: {type(x)}")

    # Direct 5D volumes: (N, C, S0, S1, S2)
    if x.dim() == 5:
        mid = x.shape[-1] // 2  # center slice along last spatial dim
        x = x[..., mid]         # -> (N, C, S0, S1)

    # Standard 2D coercion from here
    if x.dim() == 3:
        # (C, H, W) or (H, W, C)
        if x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
            x = x.permute(2, 0, 1).contiguous()
        x = x.unsqueeze(0)  # (1, C, H, W)
    if x.dim() == 4 and x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

    # If channel count is not 1 or 3, average to grayscale
    if x.dim() == 4 and x.size(1) not in (1, 3):
        x = x.mean(dim=1, keepdim=True)

    x = torch.clamp(x, 0, 1).float()

    if target_hw is not None:
        Ht, Wt = int(target_hw[0]), int(target_hw[1])
        H, W = int(x.shape[-2]), int(x.shape[-1])
        if (H, W) != (Ht, Wt):
            x = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)
    return x

@torch.no_grad()
def _save_samples_grid(model, n, temp, out_path, nrow=10, target_hw=None, warm_x=None):
    
    try:
        try:
            s = model.sample(n, temperature=temp)
        except TypeError:
            s = model.sample(n)
    except Exception as e:
        msg = str(e).lower()
        if "latent shapes unknown" in msg and warm_x is not None:
            _prime_if_needed(model, warm_x)
            try:
                try:
                    s = model.sample(n, temperature=temp)
                except TypeError:
                    s = model.sample(n)
            except Exception as e2:
                return False, str(e2)
        else:
            return False, str(e)

    try:
        try:
            s = model.sample(n, temperature=temp)   
        except TypeError:
            s = model.sample(n)                     
        x = s[0] if isinstance(s, (list, tuple)) else s
        x = _coerce_nchw_4d(x, target_hw=target_hw)
        try:
            _std = x.std().item()
        except Exception:
            _std = 0.0
        if _std < 1e-5:
            try:
                x = _manual_prior_sample(model, n, temp, x_template=None)
                x = _coerce_nchw_4d(x, target_hw=target_hw)
            except Exception:
                pass
        if torch.isfinite(x).all():
            _std = x.std().item()
            if _std < 1e-5:
                try:
                    x = _manual_prior_sample(model, n, temp)
                    x = _coerce_nchw_4d(x, target_hw=target_hw)
                except Exception:
                    pass
        x = to01(x)
        assert torch.isfinite(x).all(), "non-finite in sample grid"
        if x.shape[0] < n:
            reps = (n + x.shape[0] - 1) // x.shape[0]
            x = x.repeat(reps, 1, 1, 1)
        x = x[:n]
        grid = _make_grid_canvas(x, nrow=nrow)
        tv.utils.save_image(grid, str(out_path))
        return True, None
    except Exception as e:
        return False, str(e)

def _save_metric_plots(csv_path: Path, out_dir: Path):
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
                it, loss, bpd = int(float(row[0])), float(row[1]), float(row[2])
                iters.append(it); losses.append(loss); bpds.append(bpd)
        if len(iters) < 2:
            return
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
    except Exception:
        pass

import shutil

def cleanup_checkpoints_3d(run_dir: Path, keep_every: int = 20000):
    """
    Nettoie les points de contrôle 3D pour économiser l'espace disque.
    Vérifie également l'espace restant sur la partition.
    """
    # 1. Suppression des versions intermédiaires
    for f in run_dir.glob("training_state_it*.pt"):
        try:
            it_num = int(f.stem.split('it')[-1])
            if it_num % keep_every != 0:
                f.unlink()
        except (ValueError, IndexError):
            continue

    # 2. Alerte si l'espace disque est critique (< 50 Go pour le 3D)
    total, used, free = shutil.disk_usage(run_dir)
    free_gb = free // (2**30)
    if free_gb < 50:
        print(f"[ALERTE DISQUE] Espace restant critique : {free_gb} Go.")

# ------------------------- data -------------------------

def build_loaders_from_globs_3d(
    view_specs,
    H,
    W,
    D,
    train_samples,
    val_samples,
    batch,
    num_workers,
    val_frac: float,
    subject_limit: int | None,
    do_aug: bool = True,
    aug_schedules=None,
    disable_aug_anneal: bool = False,
    seed: int = 0,
):
    """
    3D variant of build_loaders_from_globs.

    Expects per-view glob specs pointing to 3D ANTs images. For each subject and
    sample index, we assemble a list of views [v0, v1, ...] as full 3D volumes.
    """
    import ants
    import antstorch
    from pathlib import Path

    def _expand_globs_per_view(view_specs):
        """Expand per-view glob patterns for 3D loader.

        Supports absolute ("/..."), home-relative ("~/..."), and relative
        patterns. Uses glob.glob instead of Path().glob to avoid the
        NotImplementedError on non-relative patterns.
        """
        import glob, os
        per_view_files = []
        for spec in view_specs:
            files = []
            for g in spec:
                g = os.path.expanduser(g)
                files.extend(sorted(glob.glob(g)))
            per_view_files.append([Path(f) for f in files])
        return per_view_files

    def _group_by_subject(per_view_files):
        from collections import defaultdict
        import re

        subj_map = defaultdict(list)
        n_views = len(per_view_files)
        
        # Extraction de l'ID BIDS (ex: sub-001) au lieu du dossier parent (anat)
        def _key(p: Path):
            match = re.search(r'(sub-[a-zA-Z0-9]+)', str(p))
            return match.group(1) if match else p.parent.name

        subj_to_files = [defaultdict(list) for _ in range(n_views)]
        for vi, files in enumerate(per_view_files):
            for f in files:
                subj_to_files[vi][_key(f)].append(f)

        common_subjects = set(subj_to_files[0].keys())
        for vi in range(1, n_views):
            common_subjects &= set(subj_to_files[vi].keys())

        for s in sorted(common_subjects):
            per_view_lists = []
            for vi in range(n_views):
                flist = sorted(subj_to_files[vi][s])
                if len(flist) == 0:
                    break
                per_view_lists.append(flist)
            if len(per_view_lists) == n_views:
                # zip over samples
                for sample_files in zip(*per_view_lists):
                    subj_map[s].append(sample_files)
        return subj_map

    def _read_volume(path: Path, H: int, W: int, D: int):
        img = ants.image_read(str(path))
        resize_factor = min(float(H)/float(img.shape[0]), 
                            float(W)/float(img.shape[1]),
                            float(D)/float(img.shape[2]))
        spacing = (img.spacing[0] / resize_factor, 
                   img.spacing[1] / resize_factor,
                   img.spacing[2] / resize_factor)   
        img = ants.resample_image(img, spacing, use_voxels=False, interp_type=0)
        img = ants.pad_or_crop_image_to_size(img, (H, W, D))
        return img

    per_view_files = _expand_globs_per_view(view_specs)
    per_subj = _group_by_subject(per_view_files)
    subjects = list(sorted(per_subj.keys()))
    if subject_limit and subject_limit > 0:
        subjects = subjects[: int(subject_limit)]

    def _load_subject_data(s):
        subj_samples = []
        for sample in per_subj[s]:
            views = []
            for f in sample:
                views.append(_read_volume(Path(f), H, W, D))
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
        raise RuntimeError("No 3D images assembled from provided --view globs.")

    n_subj = len(images_by_subject)
    rng = np.random.default_rng(seed)
    idx = np.arange(n_subj)
    rng.shuffle(idx)
    n_val = int(round(float(val_frac) * n_subj))
    if n_val >= n_subj:
        n_val = max(0, n_subj - 1)

    val_idx = set(idx[:n_val])

    images_train = [s for si, subj in enumerate(images_by_subject) if si not in val_idx for s in subj]
    images_val = [s for si, subj in enumerate(images_by_subject) if si in val_idx for s in subj]

    if len(images_train) == 0:
        raise RuntimeError("Split produced an empty training set. Decrease --val-frac or increase subjects.")

    tmpl = images_train[0][0]

    if aug_schedules and not disable_aug_anneal:
        sched = antstorch.MultiParamScheduler(antstorch.parse_schedules(aug_schedules))
        def aug_sched_fn(step: int):
            return sched.step(step)
    else:
        aug_sched_fn = None

    global_step = Value('i', 0)

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

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(16, batch),
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=False,
    )
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
def _manual_prior_sample(
    model,
    n: int,
    temp: float = 1.0,
    x_template: torch.Tensor = None,
):
    """
    Fallback prior sampling when model.sample is unavailable or unprimed.

    Uses model.input_shape to construct a dummy template with the right spatial
    dimensions, supporting both 2D (C,H,W) and 3D (C,D,H,W).
    """
    p = next(model.parameters())
    dev, dt = p.device, torch.float32
    if x_template is None:
        if hasattr(model, "input_shape") and isinstance(
            getattr(model, "input_shape"), (tuple, list)
        ):
            inp = tuple(int(d) for d in model.input_shape)
            if len(inp) == 3:
                C, H, W = inp
                x_shape = (1, C, H, W)
            elif len(inp) == 4:
                C, D, H, W = inp
                x_shape = (1, C, D, H, W)
            else:
                x_shape = (1, 1, 64, 64)
        else:
            x_shape = (1, 1, 64, 64)
        x_template = torch.randn(*x_shape, device=dev, dtype=dt) * 0.1
    z_tmpl, _ = model.inverse_and_log_det(x_template[:1].to(dev, dt))
    if isinstance(z_tmpl, torch.Tensor):
        z_tmpl = [z_tmpl]
    z_list = [
        torch.randn(n, *z.shape[1:], device=dev, dtype=z.dtype) * float(temp)
        for z in z_tmpl
    ]
    for fn in ("forward_from_latents", "forward", "sample_from_latents", "_forward"):
        if hasattr(model, fn):
            out = getattr(model, fn)(z_list)
            return out[0] if isinstance(out, (list, tuple)) else out
    s = (
        model.sample(n, T=temp)
        if hasattr(model, "sample") and ("T" in model.sample.__code__.co_varnames)
        else model.sample(n)
    )
    return s[0] if isinstance(s, (list, tuple)) else s


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser("Glow 2D (builder) trainer")
    ap.add_argument("--view", action="append", nargs="+", required=True,
                help="Repeat per view. Each view takes one or more glob patterns (full paths). Files are paired across views by subject folder.")
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--D", type=int, default=128, help="Depth for 3D volumes")
    ap.add_argument(
        "--spatial-dims",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of spatial dimensions: 2 for 2D slices, 3 for 3D volumes",
    )
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=96)

    ap.add_argument("--base", type=str, default="glow", choices=["glow","diag"])
    ap.add_argument("--glowbase-logscale-factor", type=float, default=3.0)
    ap.add_argument("--glowbase-min-log", type=float, default=-5.0)
    ap.add_argument("--glowbase-max-log", type=float, default=5.0)
    ap.add_argument("--scale-map", type=str, default="tanh", choices=["tanh","exp","sigmoid","sigmoid_inv"])
    ap.add_argument("--scale-cap", type=float, default=2.0)
    ap.add_argument("--net-actnorm", action="store_true", help="ActNorm in coupling subnets")

    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train-samples", type=int, default=6000)
    ap.add_argument("--val-samples", type=int, default=256)
    ap.add_argument("--max-iter", type=int, default=30000, help="Target total iterations for this run")
    ap.add_argument("--extra-iters", type=int, default=0, help="If >0, ignore --max-iter and run this many more iterations from the resume point")
    ap.add_argument("--eval-interval", type=int, default=1000)
    ap.add_argument("--plot-interval", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--devices", type=str, default="cuda:0")
    ap.add_argument("--precision", type=str, default="mixed", choices=["double","float","mixed"])
    ap.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16","fp16"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--warmup-iters", type=int, default=800)
    ap.add_argument("--lr-decay-gamma", type=float, default=1.0)
    ap.add_argument("--lr-decay-steps", type=int, default=0)
    ap.add_argument("--plateau-factor", type=float, default=0.5)
    ap.add_argument("--plateau-patience", type=int, default=4)
    ap.add_argument("--plateau-threshold", type=float, default=1e-4)
    ap.add_argument("--plateau-cooldown", type=int, default=0)
    ap.add_argument("--min-lr", type=float, default=1e-6)

    ap.add_argument("--grad-clip", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=1,
                     help="Accumulate gradients over N micro-batches before optimizer.step(). Effective batch = batch * grad_accum.")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.9995)

    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt to resume from")
    ap.add_argument("--auto-resume", action="store_true", help="If set, try <out-dir>/training_state.pt when --resume is not provided")
    ap.add_argument("--out-dir", type=str, default="runs_glow2d_builder")

    # NEW cohort options
    ap.add_argument("--slice-idx", type=int, default=120, help="Z slice index to extract across cohort/template")
    ap.add_argument("--val-frac", type=float, default=0.10, help="Fraction of subjects held out for validation in cohort mode")
    ap.add_argument("--subject-limit", type=int, default=0, help="(Debug) limit number of subjects; 0 means all")

    ap.add_argument("--smooth-alpha", type=float, default=0.1, help="EMA smoothing factor in (0,1]; higher = faster")

    # Alignment & weighting
    ap.add_argument("--align", choices=["none","infonce","barlow","vicreg","hsic","pearson"], default="none", help="Latent alignment loss across views")
    ap.add_argument("--align-weight", type=float, default=0.05, help="Fixed weight for alignment loss (if --weighting=fixed)")
    ap.add_argument("--align-warmup", type=int, default=500, help="Number of warm-up alignment its")
    ap.add_argument("--proj-dim", type=int, default=256, help="Projection head output dim")
    ap.add_argument("--proj-hidden", type=int, default=512, help="Projection MLP hidden dim")
    ap.add_argument("--temperature", type=float, default=0.1, help="InfoNCE temperature")
    ap.add_argument("--barlow-lambda", type=float, default=5e-3, help="Off-diagonal weight (lambda) for Barlow Twins")
    ap.add_argument("--weighting", choices=["fixed","kendall"], default="fixed", help="Loss weighting strategy")
    ap.add_argument("--init-logvar-nll", type=float, default=0.0, help="Init log variance (s) for NLL in Kendall weighting")
    ap.add_argument("--init-logvar-align", type=float, default=0.0, help="Init log variance (s) for ALIGN in Kendall weighting")
    # VICReg hyperparameters
    ap.add_argument("--vicreg-inv", type=float, default=1.0, help="VICReg invariance weight (MSE between views)")
    ap.add_argument("--vicreg-var", type=float, default=1.0, help="VICReg variance weight (keep per-dim std above gamma)")
    ap.add_argument("--vicreg-cov", type=float, default=1.0,  help="VICReg covariance weight (penalize off-diagonals)")
    ap.add_argument("--vicreg-gamma", type=float, default=1.0, help="VICReg variance floor (target std per feature)")
    # HSIC hyperparameters (RBF kernel)
    ap.add_argument("--hsic-sigma", type=float, default=0.0, help="RBF bandwidth; 0 -> median heuristic per batch")

    ap.add_argument("--use-ckpt-config", action="store_true",
                help="When resuming, override arch args with those saved in the checkpoint.")

    ap.add_argument("--aug-schedules", type=str,
        default=(
            "noise_std:cos:0.05->0.00@150k,"
            "sd_affine:linear:0.05->0.00@80k,"
            "sd_deformation:cos:0.20->0.00@100k,"
            "sd_simulated_bias_field:cos:1.00->0.00@120k,"
            "sd_histogram_warping:exp:0.05->0.00@120k"
        ),
        help="Multi-parameter anneal spec for ANTs data_augmentation knobs.")
    ap.add_argument("--disable-aug-anneal", action="store_true", help="If set, uses static augmentation values from dataset ctor.")

    # Preview grids
    ap.add_argument("--sample-mode", type=str, choices=["model","data","off"], default="model",
                    help="How to produce preview grids during eval: model sampling, random val batch, or skip")
    ap.add_argument("--sample-temp", type=float, default=1.0,
                help="Sampling temperature: scales prior noise (z = T·ε) when --sample-mode model")


    # --- Screening (shared subspace discovery) ---
    ap.add_argument("--screen", type=str, default="none", choices=["none","cca","hsic"],
                    help="Optional subspace screening before alignment.")
    ap.add_argument("--screen-warmup", type=int, default=1000,
                    help="Iterations before first screening pass.")
    ap.add_argument("--screen-refresh", type=int, default=0,
                    help="Recompute screening every N iters (0 = one-shot).")
    ap.add_argument("--screen-frac", type=float, default=0.5,
                    help="Fraction of projected dims to keep as shared (0,1].")
    ap.add_argument("--cca-ridge", type=float, default=1e-3,
                    help="CCA ridge regularization (stability).")
    ap.add_argument("--prefilter-frac", type=float, default=0.5,
                help="HSIC Pearson prefilter fraction (0,1].")

    args = ap.parse_args()
    args.num_views = len(args.view)

    # Device + precision
    set_deterministic(args.seed)
    dev = torch.device("cpu") if args.devices.lower() == "cpu" else torch.device(args.devices.split(",")[0])

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

    # spatial shape + dimensionality
    if args.spatial_dims == 2:
        _check_hw_divisible(args.H, args.W, args.L)
        C = 1
        input_shape = (C, args.H, args.W)
    else:
        _check_hw_divisible(args.H, args.W, args.L, D=args.D, spatial_dims=3)
        C = 1
        input_shape = (C, args.H, args.W, args.D)
    n_dims = int(np.prod(input_shape))

    # ---------------- Data ----------------

    try:
        train_loader, val_loader, global_step = build_loaders_from_globs_3d(
            view_specs=args.view,
            H=args.H,
            W=args.W,
            D=args.D,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            batch=args.batch,
            num_workers=args.num_workers,
            val_frac=float(args.val_frac),
            subject_limit=(args.subject_limit if args.subject_limit > 0 else None),
            do_aug=True,
            aug_schedules=(args.aug_schedules if not args.disable_aug_anneal else None),
            disable_aug_anneal=args.disable_aug_anneal,
            seed=args.seed,
        )
    except Exception as e:
        import traceback
        print("[data] failed to build loaders:", repr(e))
        traceback.print_exc()
        raise


    input_data_sampled = False

    # Build models using the builder
    from antstorch import create_glow_normalizing_flow_model_2d
    try:
        from antstorch import create_glow_normalizing_flow_model_3d
    except Exception:
        create_glow_normalizing_flow_model_3d = None

    models: List[nf.Flow] = []
    for vi in range(args.num_views):
        if args.spatial_dims == 2:
            m = create_glow_normalizing_flow_model_2d(
                input_shape=input_shape,
                L=args.L,
                K=args.K,
                hidden_channels=args.hidden,
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
        else:
            if create_glow_normalizing_flow_model_3d is None:
                raise RuntimeError(
                    "antstorch.create_glow_normalizing_flow_model_3d is not available; "
                    "please update antstorch or set --spatial-dims=2."
                )
            m = create_glow_normalizing_flow_model_3d(
                input_shape=input_shape,
                L=args.L,
                K=args.K,
                hidden_channels=args.hidden,
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
        m = m.to(dev).float().train()  # FP32 params
        for name, p in m.named_parameters():
            if p.dtype != torch.float32:
                print(f"[warn] casting param {name} from {p.dtype} -> float32")
                p.data = p.data.float()

        # ... (code existant) ...
        if not hasattr(m, "input_shape"):
            m.input_shape = input_shape
            
        # 1. ENVOYER SUR GPU IMMÉDIATEMENT
        m = m.to(dev).float()
        
        # 2. VERROUILLER SUR GPU (Initialisation ActNorm forcée)
        # On crée un faux batch de données sur le GPU pour forcer ActNorm à s'initialiser ICI et MAINTENANT
        with torch.no_grad():
            dummy_shape = (1, *input_shape) # (1, 1, 64, 64, 64)
            dummy_data = torch.randn(dummy_shape, device=dev, dtype=torch.float32)
            print(f"[init] Initializing ActNorm for view {vi} on GPU...")
            try:
                # Cela force ActNorm à calculer ses stats sur le GPU
                _ = m.log_prob(dummy_data) 
            except Exception as e:
                print(f"[warn] ActNorm init warning: {e}")

        # --- MODIFICATION CORRIGÉE MULTI-GPU ---
        if torch.cuda.device_count() > 1 and len(args.devices.split(',')) > 1:
            print(f"[info] Wrapping model view {len(models)} in DataParallel on {args.devices}")
            
            # 1. On enveloppe la logique (StepWrapper)
            m_step = GlowStepWrapper(m)
            
            # 2. On distribue (DataParallel)
            # ids corrects
            device_ids = [int(d.split(':')[-1]) for d in args.devices.split(',')]
            m_parallel = GlowDataParallel(m_step, device_ids=device_ids)
            
            models.append(m_parallel)
        else:
            models.append(m)
        # ---------------------------------------


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
            x_tmpl = to01(warm_batch[:, 0:1].to(dev)).to(torch.float32)
            z_probe, _ = models[0].inverse_and_log_det(x_tmpl[:1])
            flat_dim = _flatten_latents(z_probe).size(1)
        projectors = nn.ModuleList([
            Projector(flat_dim, args.proj_hidden, args.proj_dim).to(dev).train()
            for _ in range(len(models))
        ])

    # --- Kendall & Gal weighting scalars ---
    s_nll = s_align = None
    if args.weighting == "kendall" and args.align != "none":
        s_nll   = nn.Parameter(torch.tensor([args.init_logvar_nll], device=dev))
        s_align = nn.Parameter(torch.tensor([args.init_logvar_align], device=dev))

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
                # Si c'est notre modèle parallèle
                if isinstance(m, GlowDataParallel):
                    print(f"[info] Loading checkpoint into Multi-GPU StepWrapper...")
                    # Le vrai modèle est au fond : m (DataParallel) -> module (StepWrapper) -> model (Glow)
                    target_model = m.module.model
                    
                    # Nettoyage des clés du checkpoint
                    clean_sd = {}
                    for k, v in sd.items():
                        # On retire tous les préfixes possibles pour retrouver les clés brutes
                        new_k = k.replace("module.", "").replace("model.", "")
                        clean_sd[new_k] = v
                    
                    target_model.load_state_dict(clean_sd)
                
                # Si c'est un modèle simple (Single GPU)
                else:
                    clean_sd = {}
                    for k, v in sd.items():
                        new_k = k.replace("module.", "").replace("model.", "")
                        clean_sd[new_k] = v
                    m.load_state_dict(clean_sd)

        if args.ema and blob.get("ema") is not None:
            import copy
            # Note : em est maintenant un GlowDataParallel car m l'est aussi
            ema_models = [copy.deepcopy(m).eval().to(dev) for m in models]
            for em in ema_models:
                for p in em.parameters():
                    p.requires_grad_(False)
            
            for em, sd in zip(ema_models, blob["ema"]):
                # --- CORRECTION MULTI-GPU ROBUSTE (AVEC STEP WRAPPER) ---
                if isinstance(em, GlowDataParallel):
                    print(f"[info] Loading EMA checkpoint into Multi-GPU StepWrapper...")
                    # Cible : em (DataParallel) -> module (StepWrapper) -> model (Glow)
                    target_model = em.module.model
                    
                    # Nettoyage des clés (retire 'module.' et 'model.')
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
                    em.load_state_dict(clean_sd)
                # --------------------------------------------------------

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
                xb = _ensure_4d(xb).to(device, non_blocking=True)
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
            "grad_accum": args.grad_accum,
            "effective_batch_size": int(args.batch) * int(getattr(args, "grad_accum", 1)),
        }
    except Exception:
        dataset_info = {"note": "dataset stats unavailable (non-ANTs dataset type)"}

    screen_dump_run_config(args, Path(args.out_dir), note="post-dataset build", dataset_info=dataset_info)

    if not csv_path.exists():
        with open(csv_path, "w") as f:
            f.write("iter,loss,sum_bpd,lr\n")

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
        opt.zero_grad(set_to_none=True)

        # For logging (average over micro-batches)
        loss_total_for_log = 0.0
        sum_bpd_for_log = 0.0
        align_for_log = 0.0
        bpd_views_for_log = np.zeros(n_views, dtype=np.float64)

        bad_update = False

        for micro in range(args.grad_accum):
            try:
                x = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x = next(train_iter)

            L_nll = torch.tensor(0.0, device=dev, dtype=torch.float32)
            curr_bpd_views = []
            sum_bpd = 0.0
            lat_flat = []

            ctx = torch.amp.autocast(dev.type, dtype=amp_dtype) if amp_enabled else nullcontext()

            with ctx:
                bad_batch = False
                for vi, m in enumerate(models):
                    x_v = to01(x[:, vi:vi+1, ...].to(dev))

                    # 1. Forward Pass
                    if isinstance(m, GlowDataParallel):
                        logp_v, zflat = m(x_v.float())
                    else:
                        logp_v = m.log_prob(x_v.float())
                        z_v, _ = m.inverse_and_log_det(x_v.float())
                        zflat = _flatten_latents(z_v)

                    # 2. Vérification NaN
                    if not torch.isfinite(logp_v).all():
                        bad_batch = True
                        break 
                    
                    # 3. Stockage des latents (UNE SEULE FOIS)
                    lat_flat.append(torch.nan_to_num(zflat))

                    # 4. Métriques
                    bpd_v = -logp_v / (np.log(2.0) * float(n_dims))
                    bpd_mean = float(bpd_v.mean().detach().cpu().item())
                    bpd_views_for_log[vi] += bpd_mean
                    curr_bpd_views.append(bpd_mean)
                    sum_bpd += bpd_mean
                    L_nll = L_nll - logp_v.mean()

            for i, z in enumerate(lat_flat):
                first = next(m for m in projectors[i].net.modules() if isinstance(m, torch.nn.Linear))

            if bad_batch or (not torch.isfinite(L_nll)):
                bad_update = True
                break

            L_align = torch.tensor(0.0, device=dev)
            if args.align != "none" and it >= args.align_warmup:
                feats = [projectors[i](lat_flat[i]) for i in range(len(lat_flat))]
                feats = [f.float() for f in feats]

                if args.screen != "none" and it >= args.screen_warmup:
                    do_refresh = (screen_state is None) or (
                        args.screen_refresh > 0 and (it - args.screen_warmup) % args.screen_refresh == 0
                    )
                    screen_state = update_screen(
                        feats, state=screen_state, method=args.screen,
                        keep_frac=args.screen_frac, ridge=args.cca_ridge,
                        refresh=do_refresh, prefilter_frac=args.prefilter_frac,
                    )
                    feats = apply_screen(feats, screen_state)

                if args.align == "vicreg":
                    L_align = antstorch.vicreg_multi(
                        feats, w_inv=float(args.vicreg_inv), w_var=float(args.vicreg_var),
                        w_cov=float(args.vicreg_cov), gamma=float(args.vicreg_gamma),
                    )
                elif args.align == "barlow":
                    L_align = antstorch.barlow_twins_multi(feats, lam=float(args.barlow_lambda))
                elif args.align == "infonce":
                    L_align = antstorch.info_nce_multi(feats, T=float(args.temperature))
                elif args.align == "hsic":
                    L_align = antstorch.hsic_multi(feats, sigma=float(args.hsic_sigma))
                elif args.align == "pearson":
                    L_align = antstorch.pearson_multi(feats)

            if args.weighting == "fixed" or args.align == "none":
                loss_total = L_nll + (args.align_weight * L_align if args.align != "none" else 0.0)
            else:
                # keep your existing kendall block here
                ...

            # scale so accumulated grads match a true big batch
            loss_scaled = loss_total / float(args.grad_accum)

            if scaler.is_enabled():
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # logging (use unscaled loss)
            loss_total_for_log += float(loss_total.detach().cpu().item())
            sum_bpd_for_log += float(sum_bpd)
            align_for_log += float(L_align.detach().cpu().item())

        if bad_update:
            opt.zero_grad(set_to_none=True)
            continue

        # clip + step once per accumulated update
        if scaler.is_enabled():
            scaler.unscale_(opt)
            params_to_clip = []
            for g in opt.param_groups:
                params_to_clip.extend(g["params"])
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=float(args.grad_clip))
            scaler.step(opt)
            scaler.update()
        else:
            params_to_clip = []
            for g in opt.param_groups:
                params_to_clip.extend(g["params"])
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=float(args.grad_clip))
            opt.step()

        if warm is not None:
            warm.step()

        curr_loss = loss_total_for_log / float(args.grad_accum)
        sum_bpd = sum_bpd_for_log / float(args.grad_accum)
        L_align_log = align_for_log / float(args.grad_accum)
        curr_bpd_views = (bpd_views_for_log / float(args.grad_accum)).tolist()

        w_nll = 1.0
        w_align = float(args.align_weight if args.align != "none" else 0.0)

        # keep the rest of your bookkeeping (ema update, eval, ckpt, csv) the same

        if args.ema and ema_models is None:
            import copy
            ema_models = [copy.deepcopy(m).eval().to(dev) for m in models]
            for em in ema_models:
                for p in em.parameters():
                    p.requires_grad_(False)
            with torch.no_grad():
                for vi, (m, em) in enumerate(zip(models, ema_models)):
                    _copy_actnorm_state(m, em)
                    xv_real = to01(x[:, vi:vi+1, ...].to(dev)).float()
                    warmup_actnorm_with_real_batch(em, xv_real)
            tqdm.write("[ema] initialized from base after first update")

        if ema_models is not None:
            with torch.no_grad():
                for em, m in zip(ema_models, models):
                    for p_em, p in zip(em.parameters(), m.parameters()):
                        p_em.data.mul_(args.ema_decay).add_(p.data, alpha=1.0 - args.ema_decay)

        with global_step.get_lock():
            global_step.value += 1

        lr_now = opt.param_groups[0]["lr"]

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
            "align": f"{L_align_log:.4f}", "mode": args.align,
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
                    train_loader,
                    num_views=num_views,
                    out_dir=run_dir,
                    fallback_loader=train_loader,
                    n=100,
                    nrow=10,
                    target_hw=(args.H, args.W),
                    device=dev,
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
                    bpd_views = []
                    for vi, m in enumerate(eval_models):
                        xv = to01(batch_val[:, vi:vi+1, ...].to(dev))
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
            tqdm.write(f"[eval] iter={it} avg_bpd={avg_bpd:.4f} lr={lr_now:.2e}")

            with torch.no_grad():
                eval_models = ema_models if ema_models is not None else models
                if args.sample_mode == "model":
                    any_ok = False
                    n_samples, nrow = 25, 5
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
                                run_dir / f"samples_view{vi}_it{it:06d}.png",
                                nrow=nrow, target_hw=(args.H, args.W),
                                warm_x=tmpl_by_view[vi],   # <— add this
                            )
                        finally:
                            torch.random.set_rng_state(cpu_state)
                            if cuda_states is not None:
                                torch.cuda.set_rng_state_all(cuda_states)
                        if not ok:
                            tqdm.write(f"[warn] model sampling failed for view {vi} at iter {it}: {err}")
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
            _save_metric_plots(csv_path, run_dir)

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
            # Sauvegarde 'latest' pour --auto-resume
            torch.save(blob, state_path)
            
            # Sauvegarde du jalon horodaté
            iter_state_path = run_dir / f"training_state_it{it:06d}.pt"
            torch.save(blob, iter_state_path)
            
            # Nettoyage spécifique 3D (jalons toutes les 20k itérations)
            cleanup_checkpoints_3d(run_dir, keep_every=20000)
            
            tqdm.write(f"[ckpt 3D] Jalon it{it} sauvegardé. Espace disque vérifié.")

    pbar.close()
    print("Done. Run dir:", str(run_dir))

if __name__ == "__main__":
    main()


