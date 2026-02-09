
\clearpage

# Abstract {-}

Normalizing flows induce exact-likelihood, bijective mappings between data and
latents, providing the necessary framework for Latent-Aligned
Multiview Normalizing (LAMNr) flows. LAMNr flows learn shared latent subspaces
across multimodal views and, in the context of medical imaging, linearize the
sampled anatomical manifold, thus establishing a rigorous, deep learning
foundation for Computational Anatomy. By anchoring the population to a
centered Gaussian base distribution, the latent origin ($z=0$) serves as a
principled approximation of the population Fréchet mean. This construction
enables an approximate geodesic linearity where shared-latent reconstructions
act as contrast-robust population representatives. Using subject-matched
batches, we apply alignment objectives (e.g., Pearson, Barlow Twins, VICReg,
InfoNCE, HSIC) to isolate shared anatomical features from view-specific
variation. After maximum-likelihood training, closed-form conditional modeling
supports multiview queries, principled cross-view imputation, and
identity-preserving latent edits.  In imaging contexts, these shared-latent
images effectively suppress idiosyncratic variations, significantly reducing
diffeomorphic registration effort. This provides a direct connection between
probabilistic modeling and classical template construction. The same machinery
applies to tabular imaging-derived phenotypes (IDPs), enabling a unified
treatment of multimodal datasets. We provide an open-source implementation built
on PyTorch, integrating with the ANTsX ecosystem for data handling and
registration utilities. Evaluations on multimodal MRI and multiview IDP datasets
show improvements in calibrated likelihoods, dependence structure, and
downstream prediction. Our framework facilitates likelihood-calibrated,
interpretable multiview reasoning within a single, exact model that bridges deep
learning with foundational principles of computational anatomy.
