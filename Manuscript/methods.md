
\clearpage

# Methods

Our proposed imputation and synthesis framework comprises two main components:
training and inference. We describe each below and summarize implementation
details for reproducibility.

## Training

### Architecture

We train one Glow model per modality, sharing architecture but not parameters.
Each model uses a multiscale Glow design [@kingma2018glow] with \(L\) levels and
\(K\) flow steps per level. Each step applies (i) ActNorm with data-dependent
initialization on the first batch, (ii) an invertible \(1\times1(\times1)\)
convolution parameterized via LU for stable log-det computation, and (iii) an
affine coupling transform whose scale/shift predictor is a small ConvNet with
internal width (“hidden”) that operates on the transformed half of the channels
at that level. Between levels we apply squeeze (space-to-depth) and split
operations, which expose multiscale latents for analysis, alignment, and
conditional modeling. In contrast, many transformer-based flow variants decode
sequentially and do not naturally expose per-level latent access, making our
multi-scale alignment and CGM machinery less direct in addition to the usual 3-D
scaling challenges.

### Single-flow optimization

Given an image \(x\), a flow learns an invertible map \(f_\theta\) that transforms
\(x\) to a latent space \(z=f_\theta(x)\). We place a simple base density on latents,
typically \(p_Z=\mathcal{N}(0,I)\), and optimize the exact change-of-variables
objective

\[
\log p_X(x)=\log p_Z\!\big(f_\theta(x)\big)+\sum_{k=1}^{K\cdot L}\log\big|\det J_{f_k}(h_{k-1})\big|,
\]

where \(f=f_{K\cdot L}\circ\cdots\circ f_1\), \(h_0=x\), and
\(h_k=f_k(h_{k-1})\). The first term is the log-probability of the latent under
the base density; the second is the log-det Jacobian that corrects for local
volume change. Maximizing \(\log p_X(x)\) over \(\theta\) encourages images to
map to high-probability latents while keeping the induced image-space density
consistent.

### Latent alignment

When trained separately, each flow’s latent coordinate system is arbitrary,
i.e., it can be rotated, scaled, or permuted without changing the likelihood so
different flows do not necessarily align after optimization, thus necessitating
explicit alignment. For multimodal imaging we seek a shared, multi-scale
scaffold that captures common anatomy while leaving modality-specific variation.
We therefore add per-level latent alignment between flows via lightweight
projector heads, which simplifies cross-modal relations and improves
conditioning for our conditional-Gaussian inference; this yields joint
\(M\!\to\!N\) imputations that are coherent across outputs. Alignment is
auxiliary and preserves exact likelihood training and invertibility.

Including alignment, the multi-view objective is

\[
\mathcal{L}(\Theta,\Phi)
= \mathcal{L}_\text{NLL}(\Theta)
+ \sum_{\ell=1}^{L}\;\sum_{(m,n)\in\mathcal{P}}
\lambda_{\ell}^{(m,n)}\;
\mathcal{A}\!\left(
g_{\phi_\ell}^{(m)}\!\big(z_\ell^{(m)}\big),\;
g_{\phi_\ell}^{(n)}\!\big(z_\ell^{(n)}\big)
\right),
\]

with per-modality NLL

\[
\mathcal{L}_{\text{NLL}}(\Theta)
= -\,\mathbb{E}_{i\sim\mathcal{D}}
\Bigg[\sum_{m\in\mathcal{O}(i)}
\log p_{X^{(m)}}\!\big(x_i^{(m)};\,\theta^{(m)}\big)\Bigg].
\]

Here \(\mathcal{O}(i)\) is the set of observed modalities for sample \(i\);
\(\Theta=\{\theta^{(m)}\}\) are flow parameters; \(\Phi=\{\phi_\ell^{(m)}\}\)
are projector parameters; \(\mathcal{P}\) denotes the set of unordered modality
pairs (e.g., all \(m<n\)); and \(\lambda_{\ell}^{(m,n)}\!\ge 0\) weights
alignment strength (we do **not** weight the per-modality NLLs).

We support the following alignment objectives \(\mathcal{A}\) (operating on
projector outputs \(u,v\)):

- **Pearson correlation.** \(\mathcal{A}_\text{Pearson}=1-\tfrac{1}{d}\sum_i \mathrm{corr}(u_i,v_i)\).

- **Barlow Twins** [@zbontar2021barlow]. Cross-correlation \(C\) between \(u\) and \(v\); loss \(\sum_i (1-C_{ii})^2+\beta\sum_{i\ne j} C_{ij}^2\), with \(\beta>0\).

- **VICReg** [@bardes2021vicreg]. Invariance–variance–covariance: \(\alpha\|u-v\|_2^2 + \mu\,\mathrm{VarPenalty}(u,v)+\nu\,\mathrm{OffDiagCov}(u,v)\).

- **InfoNCE / CPC** [@oord2018cpc]. Temperature-scaled contrastive objective aligning positives across modalities and repelling in-batch negatives.

- **HSIC (biased)** [@gretton2005hsic]. Kernel dependence measure (Gaussian or linear kernels; bandwidth via median heuristic or per-level scale).

### Modeling aleatoric uncertainty (Kendall–Gal weighting)

Different modality pairs can exhibit different levels of measurement noise
(e.g., FA vs. T2), so a fixed alignment weight can over- or under-penalize some
pairs. Following Kendall & Gal [@kendall2017uncertainties;@kendall2018mtl], 
we replace the fixed alignment weight with a
learned log-variance per level and modality pair. Let \(s_{\ell}^{(m,n)}=\log
(\sigma_{\ell}^{(m,n)})^{2}\) be unconstrained parameters. The total training
objective becomes

$$
\begin{aligned}
\mathcal{L}(\Theta,\Phi,\mathbf{s})
=& -\,\mathbb{E}_{i\sim\mathcal{D}}\!\Bigg[\sum_{m\in\mathcal{O}(i)} \log p_{X^{(m)}}\!\big(x_i^{(m)};\theta^{(m)}\big)\Bigg] \\ \nonumber
&+ \mathbb{E}_{i\sim\mathcal{D}}\!\Bigg[\sum_{\ell=1}^{L}\sum_{(m,n)\in \mathcal{P}\cap \mathcal{O}(i)^{2}}
\underbrace{\Big(\tfrac{1}{2}e^{-s_{\ell}^{(m,n)}} \,\mathcal{A}_{\ell}^{(m,n)} + \tfrac{1}{2}s_{\ell}^{(m,n)}\Big)}_{\text{uncertainty-weighted alignment}}
\Bigg].
\end{aligned}
$$

Here the first term is the (unweighted) multi-view NLL from above; the second
term **down-weights** alignment for noisy pairs via \(e^{-s}\) while the
\(\tfrac{1}{2}s\) term regularizes the scale to prevent trivial solutions. In
practice, this is equivalent to using
\(\lambda_{\ell}^{(m,n)}=\tfrac{1}{2}e^{-s_{\ell}^{(m,n)}}\) in the earlier
alignment objective and adding the \(\tfrac{1}{2}s_{\ell}^{(m,n)}\) penalty.

**Implementation notes.** We parameterize \(s_{\ell}^{(m,n)}\) directly (no
positivity constraint needed) and initialize \(s{=}0\) (\(\sigma^2{=}1\)). For
stability, we optionally clamp \(s\in[\log \sigma_{\min}^2,\log
\sigma_{\max}^2]\) (e.g., \(\sigma_{\min}{=}0.3,\ \sigma_{\max}{=}3\)). Only
alignment terms are uncertainty-weighted; the likelihood remains unweighted
across modalities. 

<!-- 

This learned balancing pairs naturally with the optional CCA
subspace/clamp used for alignment and CGM.

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

