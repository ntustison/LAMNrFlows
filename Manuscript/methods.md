
\clearpage

# Methods

Our proposed imputation and synthesis framework comprises two main components:
training and inference. We describe each below and then summarize open-source
implementation details.

## Training

### Architecture

We train one Glow model per modality, sharing architectural configuration but
not parameters. Each model uses a multiscale design with \(L\) levels and \(K\)
flow steps per level. Each step applies (i) ActNorm with data-dependent
initialization on the first batch, (ii) an invertible \(1\times1(\times1)\)
convolution parameterized via LU for stable log-det computation, and (iii) an
affine coupling transform whose scale/shift predictor is a small ConvNet with
internal width (``hidden'') that operates on the transformed half of the
channels at that level. Between levels we apply squeeze (space-to-depth) and
split operations, which expose multiscale latents for analysis, alignment, and
conditional modeling [@kingma2018glow]. In contrast to Glow, transformer-based flow
variants often rely on sequential (token-wise) decoding and typically lack
explicit per-level latent access, making per-level alignment and
conditional-Gaussian modeling less straightforward and further challenging 3-D
scaling.

### Single-flow optimization

Given an image \(x\), a flow learns an invertible map \(f_\theta\) that
transforms \(x\) to a latent space \(z=f_\theta(x)\). A simple base density on
latents, typically \(p_Z=\mathcal{N}(0,I)\), is used to optimize the exact
change-of-variables objective

\[
\log p_X(x)=\log p_Z\!\big(f_\theta(x)\big)+\sum_{k=1}^{K\cdot L}\log\big|\det J_{f_k}(h_{k-1})\big|,
\]

where \(f=f_{K\cdot L}\circ\cdots\circ f_1\), \(h_0=x\), and
\(h_k=f_k(h_{k-1})\). The first term is the log-probability of the latent under
the base density; the second is the log-det Jacobian that corrects for local
volume change. Maximizing \(\log p_X(x)\) over \(\theta\) encourages images to
map to high-probability latents while keeping the induced image-space density
consistent.

### Multimodal latent alignment

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
alignment strength (we do not weight the per-modality NLLs).

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

Here the first term is the (unweighted) multi-view NLL from above.  The second
term down-weights alignment for noisy pairs via \(e^{-s}\) while the
\(\tfrac{1}{2}s\) term regularizes the scale to prevent trivial solutions. In
practice, this is equivalent to using
\(\lambda_{\ell}^{(m,n)}=\tfrac{1}{2}e^{-s_{\ell}^{(m,n)}}\) in the earlier
alignment objective and adding the \(\tfrac{1}{2}s_{\ell}^{(m,n)}\) penalty.
We parameterize \(s_{\ell}^{(m,n)}\) directly (no
positivity constraint needed) and initialize \(s{=}0\) (\(\sigma^2{=}1\)). For
stability, we optionally clamp \(s\in[\log \sigma_{\min}^2,\log
\sigma_{\max}^2]\) (e.g., \(\sigma_{\min}{=}0.3,\ \sigma_{\max}{=}3\)). Only
alignment terms are uncertainty-weighted; the likelihood remains unweighted
across modalities. 


### Optimization

 We integrate ANTsTorch spatial and intensity transforms (e.g., small affine,
elastic deformation, bias field, histogram warping, additive
noise)[@Tustison:2024aa]. Each transform has a schedule \(s(t)\) over training
steps \(t\) (e.g., linear, exponential or cosine anneal). We use Adam with
typical settings (\(\beta_1{=}0.9,\;\beta_2{=}0.999\)), learning rate selected
by validation (e.g., \(1\mathrm{e}{-4}\)–\(2\mathrm{e}{-4}\)), and gradient
clipping when needed. AMP is enabled via PyTorch GradScaler.  An EMA of weights
(decay \(0.999\)–\(0.9999\)) is maintained for evaluation. Batch size is adapted
to memory.  Training proceeds for a fixed number of iterations with early
stopping on validation likelihood. 


## Inference via Conditional Gaussian Modeling

Following the per-level multimodal training above, we treat the concatenated
latents at each level as approximately Gaussian across the training cohort:
\[
z_\ell = \big[z_\ell^{(1)};\dots; z_\ell^{(M)}\big] \sim \mathcal{N}\!\big(\mu_\ell,\; \Sigma_\ell\big).
\]
We estimate \((\mu_\ell,\Sigma_\ell)\) from a held-out cache of training latents
(with covariance shrinkage and a small diagonal jitter for numerical stability).

For any subject with observed views \(O\) and missing views \(M\), we partition
the moments as
\[
\mu_\ell=\begin{bmatrix}\mu_{\ell,O}\\ \mu_{\ell,M}\end{bmatrix},\qquad
\Sigma_\ell=\begin{bmatrix}\Sigma_{\ell,OO} & \Sigma_{\ell,OM}\\[2pt]
\Sigma_{\ell,MO} & \Sigma_{\ell,MM}\end{bmatrix},
\]
and compute the conditional Gaussian \(p(z_{\ell,M}\mid z_{\ell,O})\) with mean
and covariance
\[
\mathbb{E}[z_{\ell,M}\mid z_{\ell,O}]
= \mu_{\ell,M} + \Sigma_{\ell,MO}\,\Sigma_{\ell,OO}^{-1}\,(z_{\ell,O}-\mu_{\ell,O}),
\]
\[
\mathrm{Cov}(z_{\ell,M}\mid z_{\ell,O})
= \Sigma_{\ell,MM} - \Sigma_{\ell,MO}\,\Sigma_{\ell,OO}^{-1}\,\Sigma_{\ell,OM}.
\]
Intuitively, in a jointly Gaussian vector, observing one subset yields the
minimum mean-squared-error estimate of the remaining subset; the covariance
above is the Schur complement of \(\Sigma_{\ell,OO}\) in the joint covariance, 
so uncertainty shrinks most along directions
best predicted by the observed subset.
For deterministic reconstructions we use the posterior mean.  For uncertainty
estimation we draw posterior samples. In either case, a single exact inverse
pass then produces all requested image-space contrasts, enabling flexible
\(M \to N\) imputation with cross-modal coherence.[@bishop2006prml; @murphy2012mlpp]

Optionally, at each level we fit
CCA on training latents from observed modalities and keep the top \(k\)
directions \(U_\ell\) (choose \(k\) by validation) [@hotelling1936;
@andrew2013dcca]. For alignment and CGM, we project latents to this subspace; at
inference we project observed latents, compute the conditional in-subspace, then
map back with \(U_\ell U_\ell^\top\). To avoid ill-conditioning, we add a small
\(\varepsilon I\) to covariances and clip canonical correlations/eigenvalues to
\([\epsilon,\gamma]\). We save \(U_\ell\) (and whitening stats) with checkpoints
for consistent evaluation.

## Open-source availability

Code and documentation is found across the following GitHub repositories:

**ANTsTorch** (``ANTsX/ANTsTorch``) serves as the medical-imaging
layer. It provides I/O and preprocessing (e.g., N4 bias correction,
resampling/cropping, intensity standardization), volumetric dataloaders for
multi-view studies, and spatial/intensity data augmentation with schedulable
ranges. All latent-alignment machinery lives here: lightweight projector heads,
the family of alignment objectives (Pearson, Barlow Twins, VICReg, InfoNCE/CPC,
HSIC), optional Kendall–Gal uncertainty weighting, and the optional CCA-guided
subspace with clamping. This repository also contains unit tests that exercise
alignment objectives and numerical sanity checks.

**normalizing-flows** (``ntustison/normalizing-flows``) is the flow
backbone. We selected this codebase after surveying common PyTorch flow
libraries; several alternatives are strong for 2-D or tabular settings but
lacked stable, multiscale Glow with exact log-det bookkeeping in 3-D, or a
mature path to ActNorm-3D and invertible $1\times1\times1$ convolutions. The chosen
repository already offered a clean design and probability-centric interfaces.
Building on that foundation, we contributed features needed for medical-volume
work: canonical Glow step ordering with strict forward/inverse assertions;
corrected multiscale squeeze/split/reshape orderings; ActNorm in 2-D/3-D with
data-dependent initialization; invertible 1×1(×1) convolutions parameterized via
LU for stable log-det computation; 2-D/3-D coupling networks; and exact,
numerically stable log-det accumulation. We added tests that catch
shape-mismatch regressions and verify per-layer and cumulative log-dets. In
short, the original repo was good; we hardened and extended it for 3-D and
multimodal analytics without changing its overall philosophy.

**MultimodalNormalizingFlows**
(``ntustison/MultimodalNormalizingFlows``) is the experiment and
manuscript layer. It contains the trainer and evaluation pipelines that
orchestrate multi-flow training (one flow per modality), hooks to ANTsTorch
augmentations and alignment, conditional-Gaussian modeling utilities for
per-level moment fitting and M→N imputation, and scripts for likelihood,
PSNR/SSIM, and imputation-coherence evaluations. It also includes example
configurations, reproducible command lines for the HCP study, and the manuscript
sources.

The codebase provides a reproducible CLI with YAML/argparse configuration,
deterministic seeds, saved checkpoints (model, optimizer, EMA, RNG,
augmentation state), and unit tests covering forward/inverse consistency
(tolerance \(<10^{-6}\) in \(L_\infty\)), log‑det correctness (per‑layer and
cumulative), and shape invariants across 2‑D/3‑D. Experiments can be resumed
from checkpoints, and all metrics, schedules, and hyperparameters are stored. 
