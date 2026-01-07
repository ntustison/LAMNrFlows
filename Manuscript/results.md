
\clearpage

# Results

We structure the results to progress from single-view tabular modeling to
multiview tabular alignment and finally to image-based flows. We begin with
single-view experiments on UK Biobank imaging-derived phenotypes (IDPs) produced
by three standard processing suites (FSL, FreeSurfer, ANTsX) reported in our
previous work [@Tustison:2024aa]. Within each IDP block, we train RealNVP
tabular flows across a grid of $K$ coupling steps and coupling-MLP width, using a
Gaussian–PCA base with a fixed latent rank to standardize comparisons. Model
selection is based on validation bits per dimension averaged over multiple
random seeds, with ties broken by lower variance across seeds. This phase yields
one well-calibrated configuration per IDP source and establishes a
likelihood-calibrated baseline for downstream comparisons.

We then hold those single-view hyperparameters fixed and move to multiview IDPs
by jointly training per-view flows with latent-alignment objectives on
subject-matched batches. Here the only additional degrees of freedom are the
alignment choice and, when enabled, CCA or HSIC screening to restrict alignment
to directions that are statistically shared across views. We evaluate both
generative fit and utility: validation bits per dimension aggregated across
views, inter-view dependence in latent space, cross-view imputation accuracy and
calibration under controlled missingness, and simple predictive transfer where a
linear readout fit on one view is applied to another. Throughout, we compare to
strong linear multiview baselines to quantify gains beyond correlation-based
projections.

Finally, we evaluate Glow-based LAMNr models on paired MRI. We reuse the
selection principles above, report per-level likelihoods and stability
diagnostics, and exploit the exact inverse to construct shared-latent images by
replacing private coordinates with their conditional means. These
reconstructions provide contrast-robust surrogates that simplify downstream
operations such as cross-modal registration. Imaging results include synthesis
quality, calibration of conditional uncertainty, and registration performance
when transforms are estimated on shared-latent images and applied back to the
originals. All experiments use consistent subject-wise train, validation, and
test splits, identical normalization and input imputation policies, and multiple
seeds with mean and interval reporting to facilitate reproducibility.