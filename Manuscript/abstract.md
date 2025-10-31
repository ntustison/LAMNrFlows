
\clearpage

# Abstract {-}

Normalizing flows provide invertible, exact-likelihood generative models whose
multiscale latent representations are well suited to multimodal medical imaging
applications. While diffusion models have recently dominated image
synthesis, the lineage from reversible networks through NICE/RealNVP to Glow
offers a complementary path that emphasizes tractable change-of-variables and
exact inversion between image and latent spaces. Here, we present multimodal
imaging workflows within the ANTsX ecosystem that make this approach practical.
Specifically, we: 1) extend the PyTorch-based normflows and ANTsTorch libraries
supporting 2D/3D Glow implementations, correcting multiscale reshape orderings,
stabilizing log-determinants, and providing a reproducible command-line
interface with mixed precision, EMA, resumable checkpoints, and augmentation
scheduling; 2) introduce per-level latent alignment via lightweight projector
heads and an optional CCA-guided subspace/clamp, with uncertainty-aware
(Kendall–Gal) weighting across objectives including Pearson correlation, Barlow
Twins, VICReg, InfoNCE, and HSIC; and 3) propose conditional-Gaussian modeling
for image imputation that estimates per-level means and covariances (with
shrinkage/jitter as needed), computes closed-form posteriors for
observed/missing splits, and inversely maps precisely from the imputed latent
space to the corresponding image space. We demonstrate these workflows on the
young-adult HCP T1-weighted, T2-weighted, and FA images, yielding a tested,
open-source template for cross-modal synthesis, exact likelihood estimation, and
principled imputation, along with practical guidance for selecting alignment
objectives and regularization strategies across modalities.
