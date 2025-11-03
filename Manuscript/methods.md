
\clearpage

# Methods

Here’s a crisp, itemized list of the major technical contributions.

* **Robust 2D/3D Glow backbone (PyTorch)**

  * Canonical Glow step ordering with strict forward/inverse assertions.
  * **ActNorm-3D**, **invertible 1×1×1 conv (LU)**, and 3D coupling nets with exact log-det bookkeeping.

* **Multiscale reshape/split corrections**

  * Fixed squeeze/unsqueeze and split/merge orderings across levels (2D/3D), eliminating channel-mismatch and bad caching during inversion.

* **Stable and fast log-det computation**

  * Per-layer formulations (ActNorm, 1×1×1 conv, coupling) with numerically stable accumulation; verified with unit tests.

* **Per-level latent access (“taps”)**

  * Clean exposure of latents at each scale for analysis, alignment, and conditional modeling—critical for principled imputation.

* **Cross-modal latent alignment module**

  * Lightweight per-level projector heads supporting **Pearson, Barlow Twins, VICReg, InfoNCE, HSIC**; works for **>2 modalities**.

* **CCA-guided subspace + clamping**

  * Optional projection into a canonical-correlation subspace with eigenvalue clamping to stabilize alignment and downstream moment estimation.

* **Uncertainty-aware alignment weighting**

  * **Kendall–Gal** style learned weights to temper alignment losses for modalities/levels with higher aleatoric noise.

* **Conditional Gaussian Modeling (CGM) over per-level latents**

  * Estimation of per-level Gaussian moments (with **shrinkage** and **jitter**) and exact **joint** conditionals for any observed/missing split.

* **One-pass, joint M→N imputation**

  * From the CGM posterior, a **single exact inverse** produces all missing contrasts simultaneously, ensuring cross-modal coherence.

* **Reproducible training stack**

  * **AMP**, **EMA**, **resumable checkpoints** (model/optimizer/EMA/RNG/augmentation state), deterministic seeds, and shape-cache priming to avoid sampling warnings.

* **Augmentation scheduling**

  * Configurable, logged schedules (e.g., cosine/linear anneals) for spatial and intensity transforms tailored to multimodal volumes.

* **CLI + test suite**

  * A reproducible CLI (YAML/argparse) and unit tests covering forward$\leftrightarrow$inverse consistency, per-layer and cumulative log-dets, and 2D/3D shape invariants.

* **Exact-likelihood evaluation for medical volumes**

  * Calibrated **bpd/NLL** alongside synthesis/imputation metrics (PSNR/SSIM), enabling apples-to-apples comparisons and uncertainty reporting.

* **Integration with ANTsTorch**

  * Seamless I/O, preprocessing, and augmentation for medical imaging workflows; practical scaling to 3D volumes with parallel inverse decoding.

* **Practical guidance**

  * Empirical ablations and heuristics for choosing alignment objectives, CCA dimensionality, shrinkage/jitter, and augmentation schedules in multimodal settings.


## Overview
We model co-registered, multi-contrast medical images using a discrete, invertible **Glow**-style normalizing flow that exposes **per-level latents** and supports **closed-form conditional inference**. The method is implemented in PyTorch by **extending `normflows`** with **ANTsTorch** integration for data I/O, augmentation, and reproducibility tooling. Training maximizes exact log-likelihood and can optionally include **per-level cross‑modal latent alignment** objectives with **uncertainty-aware weighting**. For missing contrasts, we fit **Conditional Gaussian Modeling (CGM)** on per-level latents to compute **closed-form posteriors** followed by a **single exact inverse** to image space.

## Data and preprocessing
We assume subject-wise tuples of co-registered modalities \(\{x^{(m)} \in \mathbb{R}^{C_m \times H \times W ( \times D )}\}_{m=1}^M\). Preprocessing follows standard ANTsX/ANTsTorch pipelines: bias correction (N4), intensity standardization per modality, and optional spatial resampling/cropping to a common lattice [@avants2011ants]. When evaluating partial data, we mark a subset \(\mathcal{O} \subset \{1,\dots,M\}\) as observed and its complement \(\mathcal{M}\) as missing.

## Glow backbone (2‑D/3‑D)
We adopt a multiscale Glow architecture [@kingma2018glow] with \(L\) levels and \(K\) flow steps per level. Each step applies (i) **ActNorm** with data-dependent initialization, (ii) **invertible \(1\times 1(\times 1)\) convolution** parameterized via LU factorization, and (iii) **affine coupling**. Between levels we use **squeeze** (reshape) and **split** operations to expose multiscale latents. We correct the **reshape ordering** to ensure consistent channel/voxel layouts in 2‑D/3‑D and add strict forward–inverse assertions. The change-of-variables objective is
\[
\log p_X(x) \;=\; \log p_Z\!\big(f_\theta(x)\big) \;+\; \sum_{k=1}^{K\cdot L} \log \left|\det J_{f_k}(h_{k-1})\right| ,
\]
with base \(p_Z=\mathcal{N}(0,I)\) and \(h_{k}\) the activations after step \(k\) [@dinh2016realnvp; @papamakarios2021nfreview].

### Efficient log‑determinant terms
- **ActNorm (2D/3D).** Channel-wise scale \(s_c\) and shift \(t_c\); the log‑det is \((\text{num voxels}) \sum_c \log|s_c|\).
- **Invertible \(1\times 1(\times 1)\) conv.** Per-voxel linear mixing with weight \(W\in\mathbb{R}^{C\times C}\); \(\log|\det J| = (\text{num voxels}) \log|\det W|\). LU factorization yields stable \(\sum_i \log|U_{ii}|\) [@kingma2018glow].
- **Affine coupling.** Triangular Jacobian; \(\log|\det J| = \sum \log|\text{scale outputs}|\).

### 3‑D extensions and stability
We implement **ActNorm‑3D**, **Invertible \(1\!\times\!1\!\times\!1\)** with LU, and 3‑D coupling networks (ConvNet3D backbones), together with **exact log‑det bookkeeping** and strong **shape assertions** to preempt inversion and caching errors. We maintain **mixed precision (AMP)**, **EMA** of parameters, and **resumable checkpoints** that include optimizer/rng/EMA states.

## Training objective
The primary loss is **negative log-likelihood (NLL)**:
\[
\mathcal{L}_\text{NLL}(\theta) \;=\; - \mathbb{E}_{x}\big[ \log p_X(x;\theta) \big].
\]
Optionally, when multiple modalities are available per subject, we add **per‑level alignment** between latent projections to encourage shared structure across modalities while preserving invertibility:
\[
\mathcal{L}(\theta,\phi) \;=\; \mathcal{L}_\text{NLL}(\theta) \;+\; \sum_{\ell=1}^{L} \sum_{(m,n)\in\mathcal{P}} \lambda_{\ell}^{(m,n)} \; \mathcal{A}\!\left( g_{\phi_\ell}^{(m)}\big(z_\ell^{(m)}\big),\; g_{\phi_\ell}^{(n)}\big(z_\ell^{(n)}\big) \right) ,
\]
where \(z_\ell^{(m)}\) is the latent at level \(\ell\) for modality \(m\), \(g_{\phi_\ell}^{(m)}\) is a lightweight projector head (MLP or \(1\times1(\times1)\) conv), \(\mathcal{P}\) is a set of modality pairs (or all pairs), and \(\mathcal{A}\) is one of the alignment objectives below.

### Alignment objectives
We support several choices, all operating on mini-batch latent tensors (with spatial pooling as appropriate):

- **Pearson correlation loss.** \(\mathcal{A}_\text{Pearson} = 1 - \tfrac{1}{d}\sum_i \mathrm{corr}\big(u_i, v_i\big)\) (maximize correlation across features).  
- **Barlow Twins** [@zbontar2021barlow]. Cross-correlation matrix \(C\) between \(u\) and \(v\); loss \(\sum_i (1-C_{ii})^2 + \lambda \sum_{i\neq j} C_{ij}^2\).  
- **VICReg** [@bardes2021vicreg]. Invariance, variance, and covariance terms on \(u,v\); loss \( \alpha \|u-v\|_2^2 + \mu\, \sum \mathrm{penalize\; low\; std}(u,v) + \nu\, \sum \mathrm{offdiag}\big(\mathrm{Cov}(u),\mathrm{Cov}(v)\big)^2\).  
- **InfoNCE / CPC** [@oord2018cpc]. Contrastive objective with temperature \(\tau\): aligns positives across modalities and repels negatives within the batch.  
- **HSIC (biased)** [@gretton2005hsic]. Kernel covariance dependence measure on \((u,v)\); we use Gaussian or linear kernels with bandwidth set by median heuristic or per-level scale.

### CCA‑guided subspace and clamping
To stabilize estimation, we optionally compute a **CCA subspace** between modalities at each level using the *observed* latents during training and CGM fitting [@hotelling1936; @andrew2013dcca]. Let \(U_\ell\) denote the top \(k\) canonical directions (shared across a modality pair or multi-view generalization). We project \(u,v\) onto \(U_\ell\) before alignment or CGM moment estimation and optionally **clamp** singular values/eigenvalues to \([\epsilon, \gamma]\) to avoid ill-conditioned inverses. Hyperparameters \(k,\epsilon,\gamma\) are selected via validation.

### Uncertainty‑aware weighting (Kendall–Gal)
Because modalities may exhibit **aleatoric noise**, we use **learned task weights** \(\sigma_{\ell}^{(m,n)}\) following Kendall & Gal [@kendall2018mtl; @kendall2017uncertainties]:
\[
\sum_{\ell,(m,n)} \left(\frac{1}{2(\sigma_{\ell}^{(m,n)})^{2}} \mathcal{A}_{\ell}^{(m,n)} + \frac{1}{2}\log (\sigma_{\ell}^{(m,n)})^{2} \right).
\]
This down‑weights noisy alignment channels while regularizing the scale via the log term.

## Conditional Gaussian Modeling (CGM) for imputation
We model per‑level concatenated latents across modalities as **Gaussian**:
\[
z_\ell \;=\; \big[ z_\ell^{(1)};\dots; z_\ell^{(M)} \big] \sim \mathcal{N}\!\big(\mu_\ell,\; \Sigma_\ell \big).
\]
We estimate \((\mu_\ell,\Sigma_\ell)\) from a held‑out training cache of latents (with optional **CCA projection** first). To ensure well‑conditioned estimates, we use **shrinkage** and **jitter**:
\[
\widehat{\Sigma}_\ell \;=\; (1-\lambda)\,S_\ell \;+\; \lambda\,\mathrm{diag}(S_\ell) \;+\; \alpha I ,
\]
with \(\lambda\in[0,1]\), \(\alpha>0\) (small), and \(S_\ell\) the empirical covariance (block‑structured across modalities).

Given an observed set \(\mathcal{O}\) and missing \(\mathcal{M}\), we partition \((\mu_\ell,\Sigma_\ell)\) as
\[
\mu_\ell=\begin{bmatrix}\mu_{\ell,O}\\ \mu_{\ell,M}\end{bmatrix},\quad
\Sigma_\ell=\begin{bmatrix}\Sigma_{\ell,OO} & \Sigma_{\ell,OM}\\ \Sigma_{\ell,MO} & \Sigma_{\ell,MM}\end{bmatrix},
\]
and compute the **joint conditional** for missing latents via standard Gaussian identities:
\[
z_{\ell,M}\mid z_{\ell,O} \sim \mathcal{N}\!\Big(
\mu_{\ell,M} + \Sigma_{\ell,MO}\Sigma_{\ell,OO}^{-1}(z_{\ell,O}-\mu_{\ell,O}),\;\;
\Sigma_{\ell,MM}-\Sigma_{\ell,MO}\Sigma_{\ell,OO}^{-1}\Sigma_{\ell,OM}
\Big).
\]
We impute either the **posterior mean** (for deterministic reconstructions) or **posterior samples** (for uncertainty visualization). The **exact inverse** of the flow then produces all requested image-space contrasts in **one pass** (\(M\!\rightarrow\!N\) imputation), with cross‑modal coherence arising from the joint posterior.

## Data augmentation and schedules
We integrate ANTsTorch spatial and intensity transforms (e.g., small affine, elastic deformation, bias field, histogram warping, additive noise). Each transform has a **schedule** \(s(t)\) over training steps \(t\) (e.g., linear or cosine anneal):  
`noise_std:cos:0.02->0.00@150k, sd_affine:linear:0.05->0.00@80k, ...`  
Schedules are parsed and applied deterministically per step; complete configuration is logged to checkpoints for reproducibility.

## Optimization and training details
We use **Adam** with typical settings (\(\beta_1{=}0.9,\;\beta_2{=}0.999\)), learning rate selected by validation (e.g., \(1\mathrm{e}{-4}\)–\(2\mathrm{e}{-4}\)), and gradient clipping when needed. **AMP** is enabled via PyTorch GradScaler; an **EMA** of weights (decay \(0.999\)–\(0.9999\)) is maintained for evaluation. Batch size is adapted to memory; training proceeds for a fixed number of iterations with early stopping on validation likelihood. We **prime** shape caches by a dummy `log_prob` call before sampling to avoid unknown-latent‑shape warnings. All experiments fix seeds and record RNG states.

## Evaluation
We report:
- **Log-likelihood / bits‑per‑dimension (bpd)** on held‑out data for calibration [@papamakarios2021nfreview].
- **Synthesis quality** via PSNR/SSIM on cross‑modal reconstructions and imputation targets; optionally perceptual or task-specific metrics.
- **Imputation accuracy** (MAE/MSE) and **coherence** across jointly imputed contrasts (e.g., correlation or structural consistency).
- **Ablations** over alignment objectives, CCA dimension \(k\), shrinkage \(\lambda\), jitter \(\alpha\), and augmentation schedules.
- **Runtime & memory**: wall‑clock for 3‑D inverse vs. alternative methods.

## Software and reproducibility
The codebase provides a **reproducible CLI** with YAML/argparse configuration, deterministic seeds, saved **checkpoints** (model, optimizer, EMA, RNG, augmentation state), and **unit tests** covering forward/inverse consistency (tolerance \(<10^{-6}\) in \(L_\infty\)), log‑det correctness (per‑layer and cumulative), and shape invariants across 2‑D/3‑D. Experiments can be resumed from checkpoints, and all metrics, schedules, and hyperparameters are stored for auditability. The implementation builds on `normflows` with added 3‑D layers and integrates ANTsTorch I/O and augmentation.

## Relation to alternative generative families
**Diffusion/score‑based models** excel at perceptual quality but typically trade away **exact** likelihoods and **one‑shot** inversion [@ho2020ddpm; @song2021score]. **Autoregressive/transformer flows** offer high capacity at token resolution but decode sequentially, which is costly for large volumes. Our Glow‑style backbone yields **parallel inverse decoding**, explicit log‑det bookkeeping, and **per‑level taps** that directly support alignment and CGM—priorities for within-subject, multi‑modal medical imaging.

## Notation summary
- \(x^{(m)}\): image of modality \(m\).  
- \(f_\theta\): invertible flow; \(z=f_\theta(x)\); \(g_\theta=f_\theta^{-1}\).  
- \(z_\ell^{(m)}\): latent at level \(\ell\) for modality \(m\).  
- \(\mu_\ell,\Sigma_\ell\): per‑level Gaussian moments (optionally in a CCA subspace).  
- \(\mathcal{A}\): alignment objective; \(\lambda_\ell^{(m,n)}\): alignment weight; \(\sigma_\ell^{(m,n)}\): Kendall–Gal uncertainty parameter.


## References
Citations in text use standard citekeys: NICE/RealNVP/Glow [@dinh2016realnvp; @kingma2018glow], reviews [@papamakarios2021nfreview], contrastive and redundancy‑reduction aligners [@oord2018cpc; @zbontar2021barlow; @bardes2021vicreg], HSIC [@gretton2005hsic], CCA [@hotelling1936; @andrew2013dcca], uncertainty weighting [@kendall2018mtl; @kendall2017uncertainties], diffusion [@ho2020ddpm; @song2021score], and ANTs registration/processing [@avants2011ants].
