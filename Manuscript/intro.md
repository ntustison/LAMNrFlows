
\clearpage

# Introduction

Insight into biological structure and function is enhanced by medical imaging
and the associated data latent spaces.  Deep learning has become foundational
for modern research approaches to investigating and leveraging such spaces.



Simplifying inference, deep learning has significantly facilitated the search for latent spaces 
associated with modern medical image analysis.
the working currency of modern medical image
analysis, yet most representations are only indirectly tied to data likelihoods
and are difficult to invert. In multimodal medical imaging—where subjects often
have incomplete contrasts and downstream tasks require calibrated
uncertainty—these limitations matter. Normalizing flows address this gap by
coupling expressive latent spaces with exact likelihoods and one-shot inversion,
yielding multiscale latents that can be aligned across modalities and decoded
back to image space without approximation.


Modern neuroimaging and biomedical studies routinely acquire complementary
modalities—e.g., T1/T2/FA in structural MRI, PET/MRI in humans, histology/MRI in
animal models—often with partial or missing contrasts for a subset of
participants or time points. Downstream analyses increasingly need three things
at once: (i) faithful cross-modal synthesis and imputation, (ii) calibrated
likelihoods for principled model comparison and uncertainty reporting, and (iii)
multiscale latent representations that can be aligned and analyzed across
modalities. Normalizing flows are attractive in this setting because they are
exactly invertible and train by maximizing a tractable likelihood via the
change-of-variables formula:

$$
\log p_X(x)=\log p_Z\!\big(f(x)\big)\;+\;\log\left|\det J_f(x)\right|.
$$

This yields bits-per-dimension metrics and fast, one-pass decoding from latent
to image space.

## Historical context and positioning: flows and diffusion as complementary toolkits
Diffusion/score-based models have recently dominated image synthesis through
robust training and strong perceptual quality, while the lineage from reversible
networks to NICE/RealNVP/Glow established a parallel path grounded in exact
likelihoods and bijective mappings. For multi-modal medical images—often 3-D
volumes where exact inversion, calibrated densities, and interpretable
multiscale latents matter—these families are best viewed as complementary. We
focus on discrete, convolutional Glow-style flows to exploit parallel inversion,
explicit log-determinant bookkeeping, and per-level latent access.

## Conceptual comparison for imaging (objective, likelihood, inversion cost, 3-D scalability)

- **Training objective.** Diffusion trains a denoiser/score across noise levels;
  discrete flows perform exact maximum likelihood.  
- **Likelihood access.** Diffusion likelihoods are often indirect or
  intractable; flows provide exact log-likelihoods and bpd.  
- **Sampling/inversion cost.** Diffusion typically requires many function
  evaluations; flows invert in a single parallel pass.  
- **3-D scalability.** For volumes, single-pass inversion and explicit Jacobians
  make flows practical for decoding and analysis.  

Continuous-time variants (CNFs, flow-matching) are related but generally lack
the same one-shot inverse and straightforward multiscale “taps” we exploit for
analytics and imputation.

## Practical challenges and our design choices

### Implementation pitfalls that block 3-D adoption
“Glow-lite” implementations—single-scale setups; inconsistent squeeze/unsqueeze
and split/merge orderings; missing ActNorm; absent invertible
\(1\times1(\times1)\) convolutions; ad-hoc log-det tracking—are brittle. Common
failures include channel mismatches on inverse, unstable log-det accumulation,
and poorly structured latents that undermine downstream estimation and
imputation.

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

## Contributions
- A robust 2-D/3-D Glow backbone in normflows with ANTsTorch integration,
  including corrected multiscale reshape orderings, 3-D invertible layers,
  stabilized log-det tracking, and a reproducible CLI with AMP/EMA, resumable
  checkpoints, augmentation scheduling, and tests.  
- Per-level latent alignment across modalities via projector heads, supporting
  Pearson/Barlow Twins/VICReg/InfoNCE/HSIC, with an optional CCA-guided subspace
  and aleatoric-aware (Kendall–Gal) weighting.  
- A conditional Gaussian modeling pipeline that performs closed-form imputation
  over aligned latents and decodes exactly to one or more target image spaces.

## Datasets, modalities, and generality
We demonstrate the framework on young-adult HCP T1-weighted, T2-weighted, and FA
images, and emphasize that the workflow is modality-agnostic by construction.
The same per-level alignment and CGM machinery apply to human PET/MRI,
histology/MRI in animal models, and settings with more than two modalities or
non-uniform availability. Experiments include likelihood/bpd, PSNR/SSIM for
synthesis, and imputation accuracy, plus ablations over alignment objectives,
CCA options, and augmentation schedules.

## Paper organization
We next review background on discrete flows and related generative models;
detail the Glow backbone, training utilities, and per-level alignment
objectives; present the conditional Gaussian modeling pipeline; and report
experiments on HCP with discussion of generality to other modality combinations
and limitations.
