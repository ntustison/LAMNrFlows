
\clearpage

# Abstract {-}

Normalizing flows are invertible, exact-likelihood generative models whose
multiscale latent representations are well suited to multimodal medical imaging
applications. While diffusion models have dominated recent research efforts in
image synthesis, the developmental trajectory of normalizing flows from
reversible networks through NICE/RealNVP to Glow offers a complementary path
that provides a tractable, exact bijective mapping between image and latent
spaces. Building on this perspective, we treat modalities as a single multiflow,
multiscale latent system. Using Glow architectures with explicit per-level
latent access, we fit per-level Gaussian statistics across the multi-view cohort
and, for any observed subset of modalities, compute a closed-form joint
posterior over the missing latents that captures cross-modal covariance.  One
exact inverse then yields \(M \to N\) imputations that are jointly coherent
across outputs while preserving calibrated likelihoods for principled comparison
and uncertainty analyses. We further introduce per-level latent alignment
across modalities via lightweight projector heads under alignment constraints
provided by a family of possible objectives:  Pearson, Barlow Twins, VICReg,
InfoNCE, HSIC, with optional CCA-guided subspaces and uncertainty-aware
weighting to account for aleatoric variability. Together, these components
provide a flexible framework for within-subject multimodal modeling that scales
to 2-D/3-D medical volumes and emphasizes exactness, interpretability, and
reproducibility. We release open-source implementations and demonstrate
cross-modal synthesis and imputation on various medical imaging benchmarks.
