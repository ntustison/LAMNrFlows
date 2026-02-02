
\clearpage

# Abstract {-}

Normalizing flows provide exact-likelihood, bijective mappings between data and
latents, making them well suited to multiview representation learning. We
introduce Latent-Aligned Multiview Normalizing (LAMNr) flows, a general
framework that learns shared latent subspaces across views while treating the
orthogonal complement as view-specific variation. Using subject-matched batches,
we apply alignment objectives (for example Pearson, Barlow Twins, VICReg,
InfoNCE, HSIC) and optionally restrict alignment to statistically shared
directions identified by short CCA or HSIC screens. After maximum-likelihood
training, we estimate per-level Gaussian moments of the joint latents and use
closed-form conditionals to answer multiview queries. This yields principled
cross-view imputation and identity-preserving latent edits, including
shared-latent reconstructions that suppress view-specific factors while
preserving anatomy or identity. In imaging contexts, these shared-latent images
act as contrast-robust representatives that empirically reduce registration
effort, connecting our probabilistic construction to template concepts from
Computational Anatomy. The same machinery applies to tabular views such as
imaging-derived phenotypes, enabling a unified treatment across heterogeneous
data types. We evaluate on multimodal MRI and multiview IDP datasets, comparing
against strong linear baselines, and observe improvements in calibrated
likelihoods, dependence structure, imputation accuracy, and downstream
prediction. We release open-source implementations to facilitate
likelihood-calibrated multiview reasoning and editing within a single, exact,
and interpretable model.