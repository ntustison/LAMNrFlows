
\clearpage

# Introduction

Medical imaging data and their representative latent spaces are essential for
gaining insight into biological structure and function. While deep learning
workflows have become foundational for leveraging these spaces, many widely used
approaches lack tractable data likelihoods, optimize only surrogate bounds, or
fail to provide invertible latent mappings. These deficiencies complicate
calibration, objective comparison, and precise latent manipulations. For
instance, Generative Adversarial Networks (GANs) are implicit samplers trained
with divergence surrogates rather than likelihoods, precluding calibration by
exact probabilities [@papamakarios2021nfreview]. Similarly, Variational
Autoencoders (VAEs) optimize an evidence lower bound rather than the exact
log-likelihood [@kobyzev2020nfsurvey], while diffusion and score-based models
rely on denoising or score-matching objectives with likelihoods obtained only
indirectly [@croitoru2023diffusion_vision_survey]. Furthermore, although
autoregressive decoders offer exact likelihoods, they do not yield a one-shot
invertible latent representation [@papamakarios2021nfreview]. Such limitations
become particularly acute in multimodal and multiview settings, where
heterogeneous or missing data views necessitate calibrated comparisons and
coherent cross-view reconstructions.

## Related work in multiview learning

A "view" represents a set of measurements on a common set of subjects derived
from a distinct acquisition or feature space (e.g., disparate image modalities
or tabular blocks of imaging-derived phenotypes). These views typically vary in
scale, noise characteristics, and confounding factors. Multiview analysis
leverages two complementary principles: first, that each view contributes
unique, view-specific information and second, that shared information across
views can be distilled into lower-dimensional projections to improve calibration
and cross-cohort comparability. Traditionally, these shared projections have
been estimated using classical correlation-based methods such as Canonical
Correlation Analysis (CCA) [@Hotelling1936CCA; @Hardoon2004CCAOverview]. More
recently, kernel-based measures like the Hilbert–Schmidt Independence Criterion
(HSIC) [@gretton2005hsic] and learned alignment objectives, such as Barlow Twins,
VICReg, and InfoNCE [@zbontar2021barlow; @bardes2021vicreg; @oord2018cpc], have
expanded these capabilities to accommodate complex, incomplete data patterns
[@bishop2006prml; @Murphy2012ML].


### Similarity-driven multilinear reconstruction

Similarity-driven multilinear reconstruction (SiMLR) formalizes this
decomposition in a linear, low-rank setting by factorizing multiview data into
shared and view-specific components under subject-level similarity constraints
[@Avants2021NatCompSci]. In the SiMLR framework, each view is modeled as the sum
of a low-rank shared representation and a private residual. These shared factors
are regularized to respect an external similarity structure often derived from
clinical, cognitive, or exposure variables, which encourages the emergence of
components that are both statistically coherent across views and aligned with
downstream phenotypes. Consequently, private components capture contrast- or
modality-specific variation. This separation supports robust cross-view
harmonization, visualization, and prediction by isolating common effects from
idiosyncratic noise (e.g., [@Stone2020BreachersNeuroimaging; @Stone2024USSOCOM]).

LAMNr flows extend the SiMLR framework into a deep, likelihood-based
architecture capable of modeling nonlinear, invertible latent spaces. Instead of
an explicit linear factorization in the observation domain, LAMNr maps each view
into a shared multiscale latent space using normalizing flows, ensuring exact
log-likelihoods and bijective mappings. Latent-alignment objectives (e.g.,
VICReg, InfoNCE) identify specific coordinates that function as the shared
components, while the remaining coordinates serve as view-specific latents. By
modeling the joint latents with a Gaussian distribution and utilizing
conditional Gaussian formulations, LAMNr flows recover the interpretability of
SiMLR’s shared/private decomposition while providing nonlinear representational
capacity, calibrated likelihoods, and the ability to generate high-fidelity
reconstructions in the original data space.

### Shared and private representations

Beyond linear methods, various multiview representation-learning approaches
target shared and private structures. In medical imaging, cross-modal
translation and imputation have been explored via supervised CNNs and
adversarial or cycle-consistent mappings [@han2017dcnn; @florkow2020mrm;
@yang2018structurecyclegan; @lei2019densecyclegan]. Frameworks like HeMIS
(Hetero-Modal Image Segmentation) learn latent spaces that can be averaged
across modalities to handle missing views [@havaei2016hemis], while
diffusion-based models provide strong priors for imputation with uncertainty
quantification [@yuan2024remind; @webber2024bjrai]. Other efforts aim to
disentangle shared and private factors using autoencoders and contrastive losses
[@Chartsias2018MILR; @Chartsias2019SDNet], or link paired modalities via
conditional couplings in flow-based models [@sun2019dualglow]. While these
methods emphasize the separation of view-invariant content, they generally lack
the unique combination of exact likelihoods, one-shot invertible mapping, and
explicit Gaussian latent structure that enables the closed-form conditional
queries for both image and tabular data provided by LAMNr flows.

## Normalizing flows for latent-aligned multiview modeling

Normalizing flows model complex data distributions by composing invertible
transformations that map inputs to tractable base distributions. This
architecture yields three properties that are seldom obtained simultaneously in
other model families: exact likelihoods via the change-of-variables formula,
single-pass inversion, and direct access to latent variables that can be
manipulated and decoded without approximation [@papamakarios2021nfreview;
@kobyzev2020nfsurvey]. Early developments established both invertible networks
and flow-based density models [@Gomez2017RevNet; @Jacobsen2018iRevNet;
@dinh2014nice; @rezende2015variational; @dinh2016realnvp; @kingma2016iaf;
@papamakarios2017maf]. Notably, the Glow architecture introduced data-dependent
normalization, invertible $1 \times 1 (\times 1)$ convolutions, and a multiscale
structure specifically suited for high-resolution imaging [@kingma2018glow].
Subsequent variants have further refined coupling transforms, stability, and
parameterization while maintaining exact likelihoods [@ho2019flowpp;
@durkan2019nsf; @behrmann2019resflow; @grathwohl2019ffjord]. Recent advances
demonstrate that flows now scale to resolutions and sample qualities comparable
to other state-of-the-art generative models, establishing them as robust,
probabilistic options [@croitoru2023diffusion_vision_survey;
@zhai2024tarflow; @gu2025starflow].  For our framework, these bijective
properties render flows a natural foundation for multiview learning. A single
flow provides a precise mapping between an observed view and a latent
representation with a known base density. This allows for exact log-likelihood
estimation, principled comparison of subjects or cohorts, and closed-form
Gaussian conditioning once the joint latent model is established. Furthermore,
because flows can be flexibly parameterized using convolutional architectures
for images or multilayer perceptrons for tabular variables, the same underlying
machinery can be seamlessly applied across multimodal imaging, imaging-derived
phenotypes (IDPs), and broader multiview tabular datasets.

## Contributions

We introduce Latent-Aligned Multiview Normalizing (LAMNr) flows, a general
framework that learns shared and private latent structures across multiple views
while preserving exact likelihoods and invertibility. Within LAMNr flows, each view is
equipped with a dedicated flow that maps observations to a latent space.
Multiview structure is encoded by aligning selected latent coordinates across
views, while allowing the remaining coordinates to capture view-specific
variation. For imaging data, we adopt Glow-style multiscale architectures to
retain spatial detail and access latent features at multiple resolutions
[@kingma2018glow]. For imaging-derived phenotypes (IDPs) and other tabular
blocks, we utilize per-view flows integrated with the same alignment and
inference machinery, ensuring a unified treatment of all continuous-valued
views. Specifically, we employ a Gaussian-based PCA base distribution to enforce
a consistent latent dimensionality and geometric structure across views,
providing a shared coordinate system that facilitates direct alignment.

Using subject-matched batches, we encourage shared representations via a library
of alignment losses. We optionally identify candidate shared coordinates through
a short CCA [@Hotelling1936CCA] or HSIC [@gretton2005hsic] screen, restricting
alignment to those directions and leaving the orthogonal complement as
view-specific. Beyond maximum-likelihood training, we estimate per-level moments
of the joint latents and incorporate a conditional Gaussian layer. This provides
closed-form posteriors over arbitrary latent subsets, yielding a nonlinear,
invertible extension of the shared/private decomposition and conditional
reasoning found in SiMLR [@Avants2021NatCompSci].

For images, substituting private latents with their conditional means produces
shared-latent reconstructions that preserve anatomy or identity while
suppressing view-specific contrast. These shared-latent images serve as
contrast-robust surrogates for downstream tasks. For tabular IDPs and other
multiview blocks, the same conditional layer facilitates calibrated queries,
harmonization, and model-based counterfactuals. The framework is inherently
general, applying to multiple imaging contrasts, multiview IDP blocks, and
multimodal tabular datasets.

In summary, our contributions are:

1) __Nonlinear Extension of Shared/Private Decompositions:__ We extend linear
models like SiMLR to nonlinear, multiscale normalizing flows, providing a deep,
likelihood-based multiview model with explicit shared and view-specific latent
coordinates.

2) __Practical Training and Inference Recipe:__ We develop a robust training
protocol that combines per-view flows, latent alignment losses, and optional
CCA/HSIC screening with a Gaussian latent layer for closed-form conditional
posteriors.

3) __Unified Framework for Heterogeneous Data:__ We demonstrate that
shared-latent reconstructions act as contrast-robust surrogates for images,
while conditional queries enable calibrated harmonization and counterfactual
reasoning for IDPs.

4) __Open-Source Implementation and Validation:__ We provide an open-source
implementation featuring 2D and 3D architectures. We evaluate LAMNr flows on
multimodal MRI and multiview IDP datasets, comparing performance against linear
baselines with an emphasis on predictability, interpretability, and
reproducibility [@papamakarios2021nfreview; @kobyzev2020nfsurvey;
@Tustison:2024aa].

