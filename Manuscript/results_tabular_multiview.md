

### Nonlinear LAMNr Extension of the NNHEmbed Framework

To assess whether a nonlinear, latent-aligned normalizing-flow model offers any
practical advantage over the linear SiMLR/NNHEmbed framework used in previous
work [@Avants:2025aa], we performed a targeted comparison on the same UK Biobank
M3RI IDPs. We treated the T1, diffusion (DTI), and resting-state fMRI (rsfmri)
IDP blocks as three views for 8,361 subjects, retaining all features within each
block (51 T1, 77 DTI, 484 rsf). For each view, we applied the same preprocessing
pipeline used in the main NNH analyses (winsorization and z-scoring), and then
learned either (i) a linear Gaussian baseline equivalent to NNHEmbed/SiMLR
(per-view PCA to $k = 31$ components, where 31 is the minimum number of
principal components required to explain at least $95%$ of the variance in the
least variable modality) or (ii) a shallow LAMNr multiview normalizing flow with
the same 31-dimensional GaussianPCA base distribution per view. In the LAMNr
setting, each view’s IDPs are mapped to a shared isotropic Gaussian base via a
small number of RealNVP-style coupling layers (for example, $K = 1-4$ steps per
view with scale regularization to keep the transformations close to identity),
together with a cross-view alignment penalty that encourages corresponding
latent dimensions across T1, DTI, and rsf to share structure while preserving
exact invertibility and a fully specified joint density. All models were trained
on the same UKB training split and evaluated on held-out UKB test subjects using
identical demographic covariates derived from the accompanying table of
non-imaging variables (age at assessment, sex, and assessment centre/site) in
downstream linear models. We then compared (a) predictive performance for age
and selected physical measures such as grip strength and waist circumference
from the learned latent representations, and (b) the empirical Gaussianity and
cross-modal alignment of the resulting latents (marginal histograms, skewness
and kurtosis, and inter-view similarity scores such as the RV coefficient),
thereby directly testing whether a lightly nonlinear, generative LAMNr model
offers any measurable improvement over the linear SiMLR limit in this
near-Gaussian M3RI IDP regime.
