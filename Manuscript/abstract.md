
\clearpage

# Abstract {-}

In modeling complex probability distributions, normalizing flows sdprovide
exact-likelihood, bijective mappings between empirical data and tractable latent
spaces. Building on this foundation, latent-aligned multiview normalizing
(LAMNr) flows leverage these salient properties to learn shared latent subspaces
across heterogeneous, multimodal datasets. Using subject-matched batches, we
apply formal latent-alignment constraints to mathematically model shared
structural features from view-specific variations while simultaneously
topologically unfolding the sampled anatomical manifold into a continuous vector
space. This geometric transformation establishes a potential basis for a deep
learning interpretation of foundational computational anatomy concepts, such as
the population template, latent distances, and geodesic interpolation.
Consequently, the proposed framework enables closed-form conditional modeling
for exact cross-view imputation and latent space editing. Evaluations and
illustrations on both imaging-derived phenotypes (IDPs) and multimodal MRI
illustrate the proposed framework and potential applications such as
anatomically-detailed interpolative reconstructions which significantly
facilitate traditional image registration scenarios. To complement our work, we
provide a robust and comprehensive, 2D- and 3D-capable open-source
implementation in PyTorch, natively integrated with the ANTsX ecosystem (i.e.,
ANTsTorch) for efficient training and subsequent data transformation,
manipulation, and analysis.
