
\clearpage

# Abstract {-}

Normalizing flows provide exact-likelihood, bijective, and non-linear mappings
between data and their latent representations, making them well-suited for
multiview learning and generative modeling. We introduce Latent-Aligned
Multiview Normalizing (LAMNr) flows, a flexible framework designed to learn a
shared latent subspace across disparate data views. By employing subject-matched
batches and a library of alignment constraints, we identify and restrict latent
alignment to statistically shared coordinates, treating the orthogonal
complement as view-specific variation. Post-training, we model the joint latent
space as Gaussian, utilizing conditional Gaussian formulations to derive
closed-form posteriors. This facilitates principled cross-view imputation and
identity-preserving latent manipulations. We evaluate LAMNr flows using
multimodal MRI and imaging-derived phenotype datasets, comparing performance
against established linear multiview baselines. Our results demonstrate that
while linear methods remain effective for simpler associations, LAMNr Flows
provide improved likelihood calibration and capture complex non-linear
dependencies, leading to gains in imputation accuracy and downstream prediction
for high-dimensional data. Our open-source implementation enables the
construction of unified, multiview models for general-purpose reasoning and
editing across heterogeneous data types.
