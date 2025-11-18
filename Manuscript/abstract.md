
\clearpage

# Abstract {-}

Normalizing flows provide exact-likelihood, bijective mappings between data and
latents, making them well suited to multiview representation learning across
images and tabular imaging-derived phenotypes. We introduce Latent-Aligned
Multiview Normalizing Flows, a framework that learns per-view shared and private
latent structure while supporting precise analysis and editing. For image views
we retain Glow-style, multiscale latent access; for tabular views we employ
per-view flows with the same alignment machinery. Using subject-matched batches,
we impose latent-alignment constraints (e.g., Pearson, Barlow Twins, VICReg,
InfoNCE, HSIC) and optionally use CCA (linear) or HSIC (kernel) to identify
per-level latent directions that are statistically shared across views,
restricting alignment to those coordinates. On top of maximum-likelihood
training, we add a conditional Gaussian layer that estimates per-level moments
and yields closed-form posteriors over arbitrary subsets of latents. This
enables principled cross-view imputation and, more generally, latent
manipulations that preserve anatomy or identity while modulating modality or
view-specific factors. For images, we construct shared-latent images
(SLIs)—reconstructions in which private latents are replaced by their
conditional means—providing contrast-robust surrogates for tasks such as
cross-modal registration; transforms are estimated on SLIs and then applied to
the original data. For tabular IDP blocks, the same conditional inference
supports calibrated queries, harmonization, and counterfactual edits. We
evaluate on multimodal MRI cohorts and multiview IDP datasets, comparing against
strong linear multiview baselines, and report improvements in calibrated
likelihoods, dependence structure, imputation accuracy, and downstream
predictive transfer. We release open-source implementations and illustrate how
latent alignment enables general-purpose reasoning and editing across
heterogeneous views within a single, exact, and interpretable model.
