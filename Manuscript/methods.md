
\clearpage

# Methods

Our proposed imputation and synthesis framework comprises two main components:
training and inference.  We describe each of these components below.
In addition, we detail our open-source PyTorch-based implementation for the
benefit of the research community.

## Training

### Architecture

We adopt a multiscale Glow architecture [@kingma2018glow] with $L$ levels and $K$
flow steps per level. Each step applies (i) ActNorm with data-dependent
initialization (first batch), (ii) an invertible 1×1(×1) convolution
parameterized via an LU factorization for stable log-det computation, and (iii)
an affine coupling transform whose scale/shift predictor is a small ConvNet with
internal width (“hidden”) that operates on the transformed half of the channels
at that level. Between levels we apply squeeze (space-to-depth) and split
operations, which expose multiscale latents for analysis, alignment, and
conditional modeling.  This is in contrast to many transformer-based flow 
variants which do not naturally expose multiscale access to latents making 
mutual alignment (described below) much more difficult.  This is in addition
to the well-known difficulty of 3-D scaling.

### Single-flow optimization

Given an image \(x\), a flow learns an invertible function \(f_\theta\) that maps
\(x\) to a latent representation \(z=f_\theta(x)\). We place a simple base density
on the latents, typically the standard normal \(p_Z=\mathcal{N}(0,I)\). The model
optimizes the exact change-of-variables objective

\[
\log p_X(x)=\log p_Z\!\big(f_\theta(x)\big)+\sum_{k=1}^{K\cdot L}\log\big|\det J_{f_k}(h_{k-1})\big|,
\]

where \(f=f_{K\cdot L}\circ\cdots\circ f_{1}\), \(h_0=x\), and \(h_k=f_k(h_{k-1})\).
The first term is the log-probability of the latent under the base density; the
second term is the log-determinant Jacobian that corrects for local
volume change introduced by each invertible step. Maximizing \(\log p_X(x)\) over
\(\theta\) (the trainable parameters) encourages images to map to
high-probability latents while the summed log determinant terms keep the induced
density over images mathematically consistent.

### Latent alignment 

Independently trained flows learn invertible maps, but their latent coordinates
do not necessarily overlap and therefore not mutually informative.  For our
multimodal framework, we seek a latent space that captures shared anatomy at
multiple scales while ignoring modality-specific effects. This motivates the
inclusion of per-level latent alignment constraints between flows to encourage
coordinated latent spaces to encode corresponding structure. This simplifies and
better conditions cross-modal relationships permitting our conditional Gaussian
modeling inference framework (explained below) and yields joint $M
\leftrightarrow N$ imputations that are coherent across all outputs. Crucially,
alignment is an auxiliary objective (via projector heads) that preserves exact
likelihood training and invertibility.

Assuming alignment, $\mathcal{A}$, the multimodal (i.e., multiple views) objective 
function becomes

\[
\mathcal{L}(\theta,\phi) \;=\; \mathcal{L}_\text{NLL}(\theta) \;+\; \sum_{\ell=1}^{L} \sum_{(m,n)\in\mathcal{P}} \lambda_{\ell}^{(m,n)} \; \mathcal{A}\!\left( g_{\phi_\ell}^{(m)}\big(z_\ell^{(m)}\big),\; g_{\phi_\ell}^{(n)}\big(z_\ell^{(n)}\big) \right) ,
\]

where 

\[
\mathcal{L}_{\text{NLL}}(\Theta)
= -\,\mathbb{E}_{i\sim\mathcal{D}}
\Bigg[ \sum_{m\in \mathcal{O}(i)}
\log p_{X^{(m)}}\!\big(x_i^{(m)};\,\theta^{(m)}\big) \Bigg].
\]

\(z_\ell^{(m)}\) is the latent at level \(\ell\) for modality \(m\), 
$\lambda_{\ell}^{(m,n)}$ is the (non-negative) alignment weight for a given level $\ell$ 
and modality pair $(m,n)$, 
\(g_{\phi_\ell}^{(m)}\) is a lightweight projector head (MLP or 
\(1\times1(\times1)\) conv), \(\mathcal{P}\) is a set of modality pairs 
(or all pairs), and \(\mathcal{A}\) is one of the following alignment 
objectives:.

- **Pearson correlation loss.** \(\mathcal{A}_\text{Pearson} = 1 -
  \tfrac{1}{d}\sum_i \mathrm{corr}\big(u_i, v_i\big)\) (maximize correlation
  across features).  

- **Barlow Twins** [@zbontar2021barlow]. Cross-correlation matrix \(C\) between
  \(u\) and \(v\); loss \(\sum_i (1-C_{ii})^2 + \lambda \sum_{i\neq j}
  C_{ij}^2\).  

- **VICReg** [@bardes2021vicreg]. Invariance, variance, and covariance terms on
  \(u,v\); loss \( \alpha \|u-v\|_2^2 + \mu\, \sum \mathrm{penalize\; low\;
  std}(u,v) + \nu\, \sum
  \mathrm{offdiag}\big(\mathrm{Cov}(u),\mathrm{Cov}(v)\big)^2\).  

- **InfoNCE / CPC** [@oord2018cpc]. Contrastive objective with temperature
  \(\tau\): aligns positives across modalities and repels negatives within the
  batch.  

- **HSIC (biased)** [@gretton2005hsic]. Kernel covariance dependence measure on
  \((u,v)\); we use Gaussian or linear kernels with bandwidth set by median
  heuristic or per-level scale.


### Modeling aleatoric uncertainty

Because modalities may exhibit **aleatoric noise**, we use **learned task weights** \(\sigma_{\ell}^{(m,n)}\) following Kendall & Gal [@kendall2018mtl; @kendall2017uncertainties]:

\[
\sum_{\ell,(m,n)} \left(\frac{1}{2(\sigma_{\ell}^{(m,n)})^{2}} \mathcal{A}_{\ell}^{(m,n)} + \frac{1}{2}\log (\sigma_{\ell}^{(m,n)})^{2} \right).
\]

This down‑weights noisy alignment channels while regularizing the scale via the
log term.

<!-- 
### CCA‑guided subspace and clamping
To stabilize estimation, we optionally compute a **CCA subspace** between
modalities at each level using the *observed* latents during training and CGM
fitting [@hotelling1936; @andrew2013dcca]. Let \(U_\ell\) denote the top \(k\)
canonical directions (shared across a modality pair or multi-view
generalization). We project \(u,v\) onto \(U_\ell\) before alignment or CGM
moment estimation and optionally **clamp** singular values/eigenvalues to
\([\epsilon, \gamma]\) to avoid ill-conditioned inverses. Hyperparameters
\(k,\epsilon,\gamma\) are selected via validation. 
-->


## Inference via Conditional Gaussian Modeling

We model per‑level concatenated latents across modalities as Gaussian:

\[
z_\ell \;=\; \big[ z_\ell^{(1)};\dots; z_\ell^{(M)} \big] \sim \mathcal{N}\!\big(\mu_\ell,\; \Sigma_\ell \big).
\]

We estimate \((\mu_\ell,\Sigma_\ell)\) from a held‑out training cache of latents
(with optional CCA projection first). To ensure well‑conditioned estimates, we
use shrinkage and jitter:

\[
\widehat{\Sigma}_\ell \;=\; (1-\lambda)\,S_\ell \;+\; \lambda\,\mathrm{diag}(S_\ell) \;+\; \alpha I ,
\]

with \(\lambda\in[0,1]\), \(\alpha>0\) (small), and \(S_\ell\) the empirical
covariance (block‑structured across modalities).

Given an observed set \(\mathcal{O}\) and missing \(\mathcal{M}\), we partition \((\mu_\ell,\Sigma_\ell)\) as

\[
\mu_\ell=\begin{bmatrix}\mu_{\ell,O}\\ \mu_{\ell,M}\end{bmatrix},\quad
\Sigma_\ell=\begin{bmatrix}\Sigma_{\ell,OO} & \Sigma_{\ell,OM}\\ \Sigma_{\ell,MO} & \Sigma_{\ell,MM}\end{bmatrix},
\]

and compute the joint conditional for missing latents via standard Gaussian identities:

\[
z_{\ell,M}\mid z_{\ell,O} \sim \mathcal{N}\!\Big(
\mu_{\ell,M} + \Sigma_{\ell,MO}\Sigma_{\ell,OO}^{-1}(z_{\ell,O}-\mu_{\ell,O}),\;\;
\Sigma_{\ell,MM}-\Sigma_{\ell,MO}\Sigma_{\ell,OO}^{-1}\Sigma_{\ell,OM}
\Big).
\]

We impute either the posterior mean (for deterministic reconstructions) or
posterior samples (for uncertainty visualization). The exact inverse of the flow
then produces all requested image-space contrasts in one pass
(\(M\!\rightarrow\!N\) imputation), with cross‑modal coherence arising from the
joint posterior.


## Implementation









### Data augmentation and schedules
We integrate ANTsTorch spatial and intensity transforms (e.g., small affine, elastic deformation, bias field, histogram warping, additive noise). Each transform has a **schedule** \(s(t)\) over training steps \(t\) (e.g., linear or cosine anneal):  
`noise_std:cos:0.02->0.00@150k, sd_affine:linear:0.05->0.00@80k, ...`  
Schedules are parsed and applied deterministically per step; complete configuration is logged to checkpoints for reproducibility.

### Optimization and training details
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

