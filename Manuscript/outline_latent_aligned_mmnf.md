# Latent-Aligned Multimodal Normalizing Flows for Medical Images

---

## Abstract

- Motivation: invertible models for multimodal medical imaging (exact
  likelihoods, faithful inverses, multiscale latents).  
- Contributions (high level): hardened 2D/3D Glow + ANTsTorch integration;
  per-level latent alignment family with CCA safety; Kendall–Gal–style aleatoric
  weighting; conditional Gaussian modeling (CGM) for imputation.  
- Results teaser (to be filled after experiments).  

---

## 1. Introduction

- **Problem setting.** Co-registered modalities (e.g., T1, T2, FA). Desire a
  single invertible backbone for generation, cross-modal synthesis, and (later)
  imputation.  
- **Practical gaps.** 3D Glow brittleness; under/over-alignment risks when
  coupling views; manual loss balancing across noisy modalities.  
- **Uncertainty motivation.** Following Kendall–Gal, treat alignment as
  heteroscedastic auxiliary objectives; learn aleatoric weights instead of
  hand-tuning.  
- **Contributions.**
  1. Hardened 2D/3D Glow in *normflows* with ANTsTorch IO & tests.  
  2. **Per-level latent alignment** with projector heads and **CCA-guided
     clamp** across multiple objective families.  
  3. **Aleatoric-aware loss balancing** (Kendall–Gal–style) for alignment.  
  4. **Conditional Gaussian Modeling** (CGM) for multimodal imputation in latent
     space with CCA subspace control and uncertainty-aware sampling.  
- **Manuscript scope.** Methods (§§3–5); experiments for §5 to be added.

---

## 2. Related Work

- **Normalizing flows.** RealNVP/Glow; multiscale squeeze/split; 3D extensions;
  invertibility vs. diffusion trade-offs.  
- **Multiview alignment.** Barlow Twins, VICReg, InfoNCE, HSIC; relation to
  medical translation and registration.  
- **Uncertainty in deep learning.** Kendall–Gal: epistemic vs. aleatoric;
  heteroscedastic task weighting; multi-task learning connections.  
- **Positioning.** Robust 3D flows + principled, uncertainty-aware latent
  alignment + CGM for imputation in a unified pipeline.

---

## 3. Library Fixes & Extensions (normflows + ANTsTorch)

### 3.1 Architecture corrections & 3D enablement

- **Correct multiscale pipeline.** Fix **squeeze/unsqueeze** and **split/merge**
  ordering; explicit shape asserts; stable log-det tracking.  
- **3D invertible components.** `GlowBlock3d`, `Invertible1x1x1Conv`,
  `ActNorm3d`; optional spectral norm; gradient clipping.  

### 3.2 Training stability & performance

- **NNL** Negative log-likelihood (NLL): The key training objective for flows.
  For $z=f(x)$ and a standard normal prior, 
  $\text{NLL}(x) = \tfrac12(|z|^2 + d\log(2\pi)) - \log |\det J_f(x)|$; 
  expressed in nats or bits-per-dimension, $\mathrm{bpd} = \text{NLL}/(D \ln 2)$ .
- **AMP + EMA**, LR warmup, **jitter** (with annealable jitter-alpha) as an
  aleatoric proxy; deterministic seeds.  
- **Resumable training.** Checkpoints (model/optimizer/EMA), TQDM progress;
  consistent metric logging (bpd/NLL, alignment diagnostics, grad norms).  

### 3.3 Data & IO

- ANTsTorch paired-modality loaders; resampling/cropping; reproducible splits;
  intensity standardization.  

### 3.4 Reproducibility assets

- Single-entry `train.py` with flags for: alignment family, per-level taps, CCA
  clamp, aleatoric weighting.  
- **pytest**: round-trip/inversion; log-det consistency; level-wise shape
  invariants; 3D path tests.  
- Minimal docs/API; CI notes.

---

## 4. Latent-Aligned Training (with Aleatoric-Aware Weighting)

### 4.1 Setup and notation

Let $V$ modalities $\lbrace x^{(v)}\rbrace_{v=1}^V$. Flow $f$ factorizes into
levels $f_\ell$, yielding per-level latents $Z_\ell^{(v)}=f_\ell(x^{(v)})$.
Lightweight projector $P_\ell$ (shared or per-view) produces
$\tilde Z_\ell^{(v)}=P_\ell Z_\ell^{(v)}$.  Unified training objective:

$$\mathcal{L} = \mathrm{NLL}(x) + \sum_{\ell=0}^{L-1}\sum_{t\in\mathcal{T}} \lambda_{\ell,t}\,\mathcal{R}_{\ell,t}\big(\{\tilde Z_\ell^{(v)}\}_v\big).$$

### 4.2 Alignment objective family (unified view)

- **Pearson (multi).** Maximize mean pairwise correlation; low overhead; robust
  at small batch.  
- **Barlow Twins (multi).** Push cross-correlation toward identity; penalize
  off-diagonals; decorrelate.  
- **VICReg (multi).** Invariance + variance floor + covariance shrinkage to
  prevent collapse.  
- **InfoNCE (multi).** Contrastive with in-batch negatives; temperature control;
  batch-size sensitive.  
- **HSIC (biased).** Kernel dependence measure capturing higher-order relations.  

| Method                   | Objective (sketch)                                                                        | Encourages                           | Typical hyperparams                                              | Batch size need | Notes                                                                                                                |
| ------------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------ | ---------------------------------------------------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Pearson (multi)**      | Maximize mean pairwise corr: maximize $\mathrm{corr}(\tilde Z^{(i)},\tilde Z^{(j)})$      | Linear shared structure              | projector dim; feature normalization (pre-BN/L2)                 | **Low**         | Simple, fast, stable at small batch; second-order only—can over-smooth fine texture if over-weighted at high levels. |
| **Barlow Twins (multi)** | Cross-corr to identity: $\mathcal{L}=\sum_i (1-C_{ii})^2+\lambda!\sum_{i\neq j} C_{ij}^2$ | Invariance + decorrelation           | $\lambda$ (off-diag weight); covariance shrinkage; projector dim | **Med**         | No negatives; good default. Needs decent batch to estimate $C$; shrinkage helps stability.                           |
| **VICReg (multi)**       | $\alpha,\text{Inv}+\beta,\text{Var}+\gamma,\text{Cov}$                                    | Invariance while preserving variance | $\alpha,\beta,\gamma$; var margin; projector dim                 | **Med–Low**     | Collapse-resistant; tunable trade-offs. More knobs; match var margin to feature dim.                                 |
| **InfoNCE (multi)**      | Contrastive: $\mathcal{L}=-\log \frac{\exp(s/\tau)}{\sum \exp(s'/\tau)}$                  | Discriminative cross-view alignment  | temperature $\tau$; projector dim; (optional) aug strength       | **High**        | Strong signal with large batches; sensitive to batch/negatives; heavier compute.                                     |
| **HSIC (biased)**        | Maximize kernel dependence: $\mathrm{HSIC}(X,Y)$ (e.g., RBF)                              | Non-linear shared structure          | kernel type; bandwidth $\sigma$ (median heuristic); reg          | **Med**         | Captures beyond second-order; $O(B^2)$ cost; bandwidth selection matters.                                            |

**Implementation notes.** Feature normalization; projector depth 1–2;
temperature schedules; covariance shrinkage for Barlow/VICReg.

### 4.3 Per-level extraction & CCA-guided safety

- **Why per-level?** Coarse structure (lower levels) vs. fine texture (higher
  levels). Avoids one-size-fits-all pressure and blur.  
- **CCA-safe clamp.** At each level, compute minibatch CCA across views; scale
  top-$k$ canonical directions by $\alpha\in(0,1]$ to prevent runaway
  spikes/collapse. Modes: `perlevel` or `global` aggregation.

### 4.4 Aleatoric-aware weighting (Kendall–Gal–style)

Replace fixed $\lambda_{\ell,t}$ with learned log-variances:

$$\mathcal{L} = \mathrm{NLL}(x) + \sum_{\ell,t} \Big[ \frac{\mathcal{R}_{\ell,t}}{2\sigma_{\ell,t}^2} + \log \sigma_{\ell,t} \Big],$$

interpreting $\sigma_{\ell,t}$ as alignment **aleatoric** noise.  
**Recipe.** Initialize $\log\sigma=0$; exclude from EMA; optional L2 prior on
$\log\sigma$; joint or delayed warmup; anneal jitter-alpha; schedule InfoNCE
temperature.

### 4.5 Hyper-parameters & optimization

- Architecture: $L,K$, hidden channels; projector width/depth.  
- Optimization: LR/WD, gradient clipping, AMP+EMA, batch size; covariance
  shrinkage for Barlow/VICReg.  

---

## 5. Conditional Gaussian Modeling (CGM) for Multimodal Imputation

### 5.1 Problem setup

For a subject with observed set $S$ and missing set $M$, operate **per level**
in latent space. Concatenate projected latents across views:  
$\tilde Z_\ell = [\tilde Z_\ell^{(1)};\dots;\tilde Z_\ell^{(V)}]$. Assume
dataset-level Gaussianity after flow+alignment: 
$\tilde Z_\ell \sim\mathcal{N}(\mu_\ell,\Sigma_\ell)$. 
Partition into observed $X$ and missing $Y$ blocks and use the standard conditional:

$$\mu_{Y|X} = \mu_Y + \Sigma_{YX}\Sigma_{XX}^{-1}(x-\mu_X), \quad
\Sigma_{Y|X} = \Sigma_{YY}-\Sigma_{YX}\Sigma_{XX}^{-1}\Sigma_{XY}.$$

### 5.2 Estimation of $\mu_\ell,\Sigma_\ell$ (robust)

- Centering + optional per-feature scaling.  
- **Regularized covariance:** ridge/diagonal loading $\widehat\Sigma+\varepsilon
  I$ (flag `--jitter`) and/or **Ledoit–Wolf** shrinkage.  
- **CCA subspace control:** project $X,Y$ into rank-$k$ shared subspace (`--cca
  perlevel --cca-k k`) before covariance; **CCA-safe clamp** with strength
  $\alpha$.  
- **SPD numerics:** Cholesky solves; SVD fallback; auto-jitter retries.

### 5.3 Inference pipeline (per level; vectorized)

1. Encode observed modalities: $Z_\ell^{(v)}=f_\ell(x^{(v)})$ → $\tilde
   Z_\ell^{(v)}=P_\ell Z_\ell^{(v)}$ for $v\in S$.  
2. Build block means/covariances
   $(\mu_Y,\mu_X,\Sigma_{YY},\Sigma_{YX},\Sigma_{XX})$.  
3. Compute $(\mu_{Y|X},\Sigma_{Y|X})$. 4. **Posterior mean** or samples $y_\ell
\sim \mathcal{N}(\mu_{Y|X},\,\tau^2\Sigma_{Y|X})$ (temperature $\tau$).  
5. Replace missing latents for $v\in M$; **invert flow** to reconstruct $\hat
   y$.  

**Engineering:** chunked latent indexing; batched solves; arbitrary missingness
supported.

### 5.4 Control knobs (noise ↔ variance)

- **Temperature $\tau$:** scales $\Sigma_{Y|X}$.  
- **CCA rank $k$:** subspace dimensionality.  
- **Clamp $\alpha$:** limit top-$k$ canonical directions.  
- **Jitter $\varepsilon$:** SPD stability vs. bias.  
- **Per-level ranks $k_\ell$:** larger at coarse levels, smaller at texture levels.

### 5.5 Diagnostics & planned reports

- **Calibration:** coverage vs. $\Sigma_{Y|X}$; Mahalanobis residuals.  
- **Fidelity:** PSNR/SSIM; structure correlation; intensity bias.  
- **Uncertainty maps:** trace$(\Sigma_{Y|X})$ per voxel.  
- **Ablations:** $\varepsilon,k,\alpha,\tau$, mean vs. sampling; EMA on/off
  during encode/decode.  
- **Efficiency:** wall-clock, memory; chunk-size sensitivity.

### 5.6 Interface flags (current defaults)

```
--use-ema --gauss-samples 10000 --eval-samples 256
--jitter 1e-3 --cca perlevel --cca-k 16 --cca-safe clamp --cca-alpha 0.5
--eval-tag cca
```

Notes: fixed seed; EMA weights optional during encode/decode.

### 5.7 Limitations & future work

- Approximate Gaussianity; consider **mixture models** / **graphical shrinkage**.  
- Covariance across subjects at fixed spatial indices; consider **local spatial banding**.  
- Data-driven selection of $k_\ell$ (eigengap, held-out likelihood).
