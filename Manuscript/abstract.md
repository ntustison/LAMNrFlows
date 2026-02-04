
\clearpage

# Abstract {-}

Normalizing flows provide exact-likelihood, bijective mappings between data and
latents, making them well suited to multiview representation learning. We
introduce Latent-Aligned Multiview Normalizing (LAMNr) flows, a general
framework that learns shared latent subspaces across views while treating the
orthogonal complement as view-specific variation. Using subject-matched batches,
we apply alignment objectives (e.g., Pearson, Barlow Twins, VICReg, InfoNCE,
HSIC) and optionally restrict alignment to statistically shared directions via
short CCA or HSIC screens. After maximum-likelihood training, per-level latent
moments are estimated and, under Gaussian base distributions, closed-form
conditional modeling supports multiview queries. This permits principled
cross-view imputation and identity-preserving latent edits, including
shared-latent reconstructions that suppress view-specific factors while
preserving anatomy or identity. In imaging contexts, these shared-latent images
act as contrast-robust representatives that empirically reduce diffeomorphic
registration effort, connecting our probabilistic population distribution
construction to template concepts from computational anatomy. The same machinery
applies to tabular views (e.g., imaging-derived phenotypes), enabling a unified
treatment across heterogeneous data types. We provide an open-source
implementation with 2D and 3D architectures built on PyTorch, using ANTsTorch
for data handling, augmentation, and registration utilities, and normflows for
flow primitives and training scaffolds. Evaluations on multimodal MRI and
multiview IDP datasets against strong linear baselines show improvements in
calibrated likelihoods, dependence structure, imputation accuracy, and
downstream prediction. Our release facilitates likelihood-calibrated multiview
reasoning and editing within a single, exact, and interpretable model.