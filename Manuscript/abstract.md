
\clearpage

# Abstract {-}

Normalizing flows provide exact-likelihood, bijective mappings between data and
their latent representations, making them well-suited to multiview learning. In
this context, we introduce Latent-Aligned Multiview Normalizing (LAMNr) Flows, a
general framework that learns a shared latent subspace across views, thereby
treating the orthogonal complement as view-specific variation. Using
subject-matched batches, we implement a library of latent-alignment constraints
(Pearson, Barlow Twins, VICReg, InfoNCE, HSIC) and optionally use CCA (linear)
or HSIC (kernel) to identify latent directions that are statistically shared
across views, restricting alignment to those coordinates. After
maximum-likelihood training, we model the joint latents as Gaussian, estimate
per-level moments, and use the conditional Gaussian formulation to obtain
closed-form posteriors for any subset. This enables principled cross-view
imputation and, more generally, latent manipulations that preserve anatomy or
identity while modulating modality- or view-specific factors; for images,
replacing private components by their conditional means produces shared-latent
images that act as contrast-robust surrogates. We evaluate on multimodal MRI
cohorts and multiview imaging-derived phenotype datasets, comparing against
strong linear multiview baselines, and observe improvements in calibrated
likelihoods, cross-view dependence structure, imputation accuracy, and
downstream predictive transfer. We release open-source implementations and
illustrate how latent alignment enables general-purpose reasoning and editing
across heterogeneous views within a single, exact, and interpretable model.
