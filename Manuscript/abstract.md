
\clearpage

# Abstract {-}

In modeling complex probability distributions, normalizing flows provide
exact-likelihood, bijective mappings between empirical data and tractable latent
spaces. Building on this foundation, latent-aligned multiview normalizing
(LAMNr) flows leverage these bijective properties to learn shared latent
subspaces across heterogeneous, multimodal datasets. Using subject-matched
batches, we apply latent alignment constraints (e.g., VICReg, InfoNCE, or HSIC)
to mathematically isolate shared structural features from view-specific,
idiosyncratic noise while simultaneously linearizing the sampled anatomical
manifold. This geometric linearization establishes a potential basis for a deep
learning interpretation of foundational computational anatomy concepts, such as
the population template, latent distances, and geodesic interpolation.
Consequently, the framework enables closed-form conditional modeling for exact
cross-view imputation and latent space editing. Evaluations on multimodal MRI
and imaging-derived phenotypes (IDPs) demonstrate that LAMNr flows improve
calibrated likelihoods and downstream clinical predictions compared to linear
baselines. Furthermore, shared-latent reconstructions act as contrast-robust
population representatives that effectively suppress idiosyncratic variations,
significantly reducing the effort required for subsequent diffeomorphic image
registration. We provide a robust and comprehensive, 2D- and 3D-capable
open-source implementation in PyTorch, natively integrated with the ANTsX
ecosystem for streamlined data handling and computational anatomy utilities.
