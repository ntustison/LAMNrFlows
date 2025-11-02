
\clearpage

# Introduction

Medical imaging data and their representative latent spaces are essential for
insight into biological structure and function.  Deep learning workflows have
become foundational for modern approaches to investigating and leveraging such
spaces.  Many existing approaches, however, are opaque to data likelihoods and
lack invertibility complicating transformations between image space and their
latent counterparts. In practice, multimodal medical imaging research is often
characterized by incomplete contrasts and other constraints which affect
downstream processing.  Specifically, downstream analyses often require
cross-modal imputation and synthesis, exact likelihoods for principled model
comparison and uncertainty modeling, and multiscale latent representations that
can be aligned and analyzed across modalities.  Normalizing flows
[@papamakarios2021nfreview] are well-suited for addressing these concerns by
coupling expressive latent spaces with exact likelihoods and single-pass
inversion, yielding multiscale latents that can be aligned across modalities and
precisely decoded back to image space.

## Normalizing flows 

Normalizing flows emerged as a practical class of invertible generative models
approximately a decade ago.  Although other classes of invertible (or
approximately invertible) architectures were developed in parallel
[@gomez2017revnet;@jacobsen2018irevnet], such networks were not designed for
density modeling with exact likelihoods.  An early pioneer Non-linear
Independent Components Estimation (NICE) [dinh2014nice] demonstrated that
features can be split into two parts with one half "nudging" the other with a
learned shift.  This keeps density computation simple while guaranteeing an
exact inverse. Variational flows broadened this idea by stacking small,
invertible "warps" that are easy to compute [@rezende2015variational]. RealNVP
then added learned scaling in addition to shifting and arranged the model across
multiple resolutions, improving modeling while keeping computations efficient
[@dinh2016realnvp]. In parallel, Inverse Autoregressive Flow (IAF) and Masked
Autoregressive Flow (MAF) explored autoregressive flows that set the direction
of computation to trade off fast sampling versus fast likelihood evaluation
[@kingma2016iaf;@papamakarios2017maf]. 

The Glow architecture consolidated these ideas for large images with
data-dependent ActNorm, invertible $1\times1$ convolutions, and a clean
multiscale design, yielding strong likelihoods and single-pass inversion
[@kingma2018glow]. Subsequent work broadened the family: Flow++ improved sample
quality via variational dequantization and richer coupling transforms
[@ho2019flowpp]; Neural Spline Flows replaced affine transforms with monotonic
splines for greater flexibility [@durkan2019nsf]; Residual Flows enforced
Lipschitz constraints for stability in deep stacks [@behrmann2019resflow]; and
FFJORD introduced continuous-time flows with unbiased likelihood estimates via
Hutchinson trace estimators [@grathwohl2019ffjord]. Continuous-time variants
(e.g., continuous normalizing flows, flow-matching) are related but generally
lack the same one-shot inverse and straightforward multiscale architectures 
leveraged for analytics and imputation.  Surveys synthesize these developments and
map the trade-offs across density estimation, sampling, and invertibility
[@kobyzev2020nfsurvey;@papamakarios2021nfreview].

More recent work proposes flow-based models that operate at the same resolution
and scale that popularized diffusion models
[@croitoru2023diffusion_vision_survey].  TarFlow (Transformer Autoregressive
Flow) shows that normalizing flows can achieve state-of-the-art image
likelihoods and diffusion-comparable sample quality using autoregressive
Transformers and a few key training recipes [@zhai2024tarflow]. STARFlow builds
on this with a scalable latent-space design and guidance mechanisms, reporting
competitive high-resolution synthesis (class-conditional and text-conditional)
that explicitly benchmarks against diffusion while retaining exact likelihood
training [@gu2025starflow].

## Contribution

### Why Glow (vs. TarFlow/STARFlow)

While recent work revisits flows at with transformer backbones and shows diffusion-level sample quality, our setting
prioritizes calibrated densities, exact inversion, and analyzable latents for
multi-modal medical volumes. We therefore adopt a Glow-style architecture
because it provides single-pass exact inverse decoding, explicit log-likelihoods
with clean per-layer bookkeeping, and straightforward multiscale taps that
expose per-level latents for cross-modal alignment and conditional Gaussian
imputation. In practice, this design scales predictably to 2D/3D data
(ActNorm-3D, invertible $1\times1\times1$ convolutions, corrected reshape
order), supports per-scale uncertainty-aware alignment (Pearson/Barlow
Twins/VICReg/InfoNCE/HSIC with optional CCA constraints), and integrates cleanly
with ANTsTorch tooling. Compared with autoregressive flow variants, which decode
sequentially over tokens/variables, Glow’s parallel inverse is compute-efficient
for large volumes and enables per-level statistics needed for imputation. Our
goal is not web-scale text-conditional synthesis, but fit-for-purpose modeling
of within-subject multimodal data where exactness, interpretability, and
reproducibility matter most.


### Robust 2-D/3-D Glow backbone (normflows + ANTsTorch)
We provide a tested PyTorch backbone that restores canonical Glow step ordering
and extends it to 3-D: ActNorm-2D/3D with data-dependent initialization;
invertible \(1\times1(\times1)\) convolutions with LU factorization; affine
coupling with strict forward/inverse assertions; and exact log-det bookkeeping.
We correct multiscale **reshape** orderings, add 3-D invertible layers, and
package training utilities—mixed precision, EMA, resumable checkpoints,
augmentation scheduling—behind a reproducible CLI with unit tests. The result is
a stable, image-centric flow stack that exposes clean per-level latents.

### Per-level latent alignment and uncertainty-aware weighting
To relate modalities in latent space, we attach lightweight projector heads at
each scale and support multiple alignment objectives—Pearson correlation, Barlow
Twins, VICReg, InfoNCE, and HSIC—optionally constrained by a CCA-guided subspace
with clamping for numerical safety. Because modalities exhibit modality-specific
noise (aleatoric variability), alignment losses can be weighted à la Kendall–Gal
to temper the influence of higher-uncertainty channels. The design naturally
extends to settings with more than two modalities.

### Conditional Gaussian Modeling (CGM) for principled imputation
For missing contrasts, we estimate per-level means and covariances (with
shrinkage/jitter and an optional CCA subspace), compute closed-form posteriors
for observed/missing splits, and use the exact inverse to decode posterior
summaries or samples back to any requested image space. Discrete flows pair
naturally with this pipeline: exact likelihoods calibrate estimates; single-pass
inversion makes 3-D decoding practical; and multiscale taps yield structured
statistics across resolutions.

