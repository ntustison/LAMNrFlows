
\clearpage

# Abstract {-}

Normalizing flows are invertible, exact-likelihood generative models whose
multiscale latent representations are well suited to multimodal medical imaging
applications. While diffusion models have dominated recent research efforts in
image synthesis, the developmental trajectory of normalizing flows from
reversible networks through NICE/RealNVP to Glow and beyond offers a
complementary path that provides a tractable, exact bijective mapping between
images and their representative latent spaces. Building on this foundational
work, we propose the use of single multiflow, multiscale latent systems for
medical image imputation and synthesis. Specifically, using Glow architectures
with explicit per-level latent access, we fit per-level Gaussian statistics
across the multi-view cohort and, for any observed subset of modalities, compute
a closed-form joint posterior over the missing latents that captures cross-modal
covariance.  The exact inverse then yields joint imputations that are
coherent across outputs while preserving estimated likelihoods for
comparison and uncertainty analyses. We further introduce per-level latent
alignment across modalities under alignment constraints provided by a family of
possible objectives: Pearson, Barlow Twins, VICReg, InfoNCE, and HSIC.  Optional
features include CCA-guided subspaces and uncertainty-aware weighting to account
for aleatoric variability. Together, these components provide a flexible
framework for multimodal image modeling that scales to medical volumes and
emphasizes exactness, interpretability, and reproducibility. We provide
open-source implementations and demonstrate cross-modal synthesis and imputation
in various medical imaging contexts.
