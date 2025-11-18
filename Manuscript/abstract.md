
\clearpage

# Abstract {-}

Normalizing flows provide exact-likelihood, bijective mappings between images
and latents, making them highly suitable for multimodal representation learning.
We introduce a general framework, Latent-Aligned Multiview Normalizing Flows,
that learns per-level shared and private latent structure across modalities
while retaining Glow-style, multiscale latent access for analysis and editing.
Given subject-matched batches, we impose latent-alignment constraints (e.g., Pearson,
Barlow Twins, VICReg, InfoNCE, HSIC) to learn shared multiscale latent spaces
and optionally discover shared subspaces via CCA/HSIC screening. On top of
maximum-likelihood training, we provide a conditional Gaussian inference framework
that estimates per-level moments and yields closed-form posteriors over arbitrary
subsets of latents. This enables principled cross-view imputation and, more
generally, latent manipulations that preserve anatomy while modulating
modality-specific contrast or confounders. A practical consequence is the
construction of shared-latent images (SLIs), i.e., reconstructions in which private
latents are replaced by their conditional means.  This provides contrast-robust
surrogates for downstream tasks such as cross-modal registration where transforms
are estimated on SLIs and then applied to the original data. The same machinery
supports synthesis, harmonization, uncertainty analysis, and interventional
latent edits (i.e., model-based counterfactuals) within a single, exact, and
interpretable model. We release open-source implementations and illustrate the
framework on multimodal MRI cohorts, highlighting how latent alignment enables
general-purpose reasoning and editing across views.
