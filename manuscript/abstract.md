
\clearpage

# Abstract {-}

In modeling complex probability distributions, normalizing flows provide
exact-likelihood, bijective mappings between empirical data and tractable latent
spaces. Building on this foundation, latent-aligned multiview normalizing
(LAMNr) flows leverage these salient properties to learn shared latent subspaces
across heterogeneous, multimodal datasets. Training with subject-matched
batches, formal latent-alignment constraints are used to statistically model
shared structural features from view-specific variations while simultaneously
topologically unfolding the sampled data manifold into a continuous vector
space. In the context of biological imaging, this transformation establishes a
potential basis for a deep learning interpretation of foundational computational
anatomy concepts, such as the population template, latent distances, and
geodesic pairwise image interpolation. Consequently, the proposed framework
enables closed-form conditional modeling for exact cross-view imputation and
other latent space manipulations. Evaluations and illustrations on both
imaging-derived phenotypes (IDPs) and multimodal MRI demonstrate the proposed
framework and potential applications such as anatomically-detailed interpolative
reconstructions which, for example, facilitate traditional image registration
scenarios. To further motivate our work, we provide a robust and comprehensive,
2D- and 3D open-source implementation in PyTorch, natively integrated with the
ANTsX ecosystem (i.e., ANTsTorch) for efficient training and subsequent data
transformation, manipulation, and analysis.
