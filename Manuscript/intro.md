
\clearpage

# Introduction

Medical imaging data and their representative latent spaces are essential for
insight into biological structure and function. Deep learning workflows have
become foundational for learning and leveraging such spaces. Many widely used
approaches either provide no tractable data likelihood, optimize only a
surrogate bound, or lack an invertible latent mapping, which complicates
calibration, comparison, and precise latent edits. GANs, for example, are
implicit samplers trained with divergence surrogates rather than likelihoods,
so calibration by exact probabilities is unavailable [@papamakarios2021nfreview];
VAEs optimize an evidence lower bound rather than the exact log likelihood
[@kobyzev2020nfsurvey]; and diffusion and score models train via denoising or
score-matching objectives, with likelihoods obtained indirectly
[@croitoru2023diffusion_vision_survey]. Although autoregressive decoders
provide exact likelihoods, they do not yield a one-shot invertible latent
representation [@papamakarios2021nfreview]. These limitations become acute in
multimodal and multiview settings, where contrasts or views may be missing or
heterogeneous and downstream analyses depend on calibrated comparisons and
coherent cross-view reconstructions.

## Related work in "multiview" learning

A view is a set of measurements on the same subjects that arises from a distinct
acquisition or feature space (for example, distinct image types or tabular
blocks of imaging-derived phenotypes). Views typically differ in scale, noise,
and confounders. Multiview analysis exploits two complementary notions. First,
each view contributes view-specific information that should be retained. Second,
the overlap of information across views can be distilled into lower-dimensional
shared projections that improve calibration and cross-cohort comparability.
These shared projections can be estimated with classical correlation-based
methods such as CCA [@Hotelling1936CCA; @Hardoon2004CCAOverview]. Kernel
dependence measures such as the Hilbert–Schmidt Independence Criterion (HSIC)
[@gretton2005hsic] or learned alignment objectives such as Barlow Twins, VICReg,
and InfoNCE [@zbontar2021barlow; @bardes2021vicreg; @oord2018cpc] can also be
used for broader application to missing-data patterns [@bishop2006prml;
@Murphy2012ML].

### Similarity-driven multilinear reconstruction

Similarity-driven multilinear reconstruction (SiMLR) makes this decomposition
explicit in a linear, low-rank setting by factorizing multiview data into
shared and view-specific components under subject-level similarity constraints
[@Avants2021NatCompSci]. In SiMLR, each view is expressed as the sum of a
low-rank shared representation and a private residual, with the shared factors
regularized to respect an external similarity structure (for example, derived
from clinical, cognitive, or exposure variables). This coupling to a
subject-similarity kernel encourages shared components that are both
statistically coherent across views and aligned with downstream phenotypes,
while private components capture contrast- or modality-specific variation. The
resulting embeddings support tasks such as cross-view harmonization,
visualization, and prediction in a way that cleanly separates common and
idiosyncratic effects (e.g., [@Stone2020BreachersNeuroimaging; @Stone2024USSOCOM]).

LAMNr Flows can be viewed as a deep, likelihood-based extension of this SiMLR
framework to nonlinear, invertible latent spaces. Rather than performing an
explicit linear factorization in the observation domain, LAMNr first maps each
view into a shared multiscale latent space using normalizing flows with exact
log-likelihoods and bijective mappings. Latent-alignment objectives (e.g.,
VICReg, InfoNCE) identify a subset of coordinates that play the role of SiMLR’s
shared component, while remaining coordinates act as view-specific latents.
Modeling the joint latents with a Gaussian and using conditional Gaussian
formulas recovers the same style of shared/private decomposition, cross-view
imputation, and phenotype-guided reasoning as SiMLR, but now with nonlinear
representational capacity, calibrated likelihoods, and the ability to generate
realistic reconstructions in the original data space, including images.

### Shared and private representations

Beyond linear methods, a broad range of multiview representation-learning
approaches also target shared and private structure. In medical imaging,
cross-modal translation and imputation have been studied with supervised CNNs
and adversarial or cycle-consistent mappings [@han2017dcnn; @florkow2020mrm;
@yang2018structurecyclegan; @lei2019densecyclegan]. Methods such as HeMIS learn
latent spaces that can be averaged across available modalities to obtain robust
predictions under missing views [@havaei2016hemis], and diffusion-based models
have recently been adapted to imputation and reconstruction with strong priors
and uncertainty summaries [@yuan2024remind; @webber2024bjrai]. Parallel efforts
in multiview representation learning aim to disentangle explicit shared and
private factors using autoencoders and contrastive losses [@Chartsias2018MILR;
@Chartsias2019SDNet], while flow-based multimodal models have linked latent
spaces across paired modalities via conditional couplings [@sun2019dualglow].
These methods highlight the importance of separating view-invariant content from
view-specific variation, but they typically lack the combination of exact
likelihoods, a one-shot invertible map, and explicit Gaussian latent structure
that enables closed-form conditional queries for both image and multiview
tabular data.

## Normalizing flows for latent-aligned multiview modeling

Normalizing flows model data by composing invertible transformations that map
inputs to tractable base distributions. This yields three properties that are
difficult to obtain together in other families: exact likelihoods via the
change-of-variables formula, single-pass inversion, and direct access to latent
variables that can be manipulated and decoded without approximation
[@papamakarios2021nfreview; @kobyzev2020nfsurvey]. Early work developed both
invertible networks and flow-based density models
[@Gomez2017RevNet; @Jacobsen2018iRevNet; @dinh2014nice; @rezende2015variational;
@dinh2016realnvp; @kingma2016iaf; @papamakarios2017maf]. Glow combined
data-dependent normalization, invertible $1 \times 1 (\times 1)$ convolutions, and a multiscale
architecture suited to large images [@kingma2018glow]. Subsequent variants
improved coupling transforms, stability, and parameterization while keeping
exact likelihoods [@ho2019flowpp; @durkan2019nsf; @behrmann2019resflow;
@grathwohl2019ffjord]. Recent results show that flows scale to resolutions and
sample quality comparable to popular generative models, supporting their use as
first-class probabilistic backbones [@croitoru2023diffusion_vision_survey;
@zhai2024tarflow; @gu2025starflow].

For our purposes, these properties make flows a natural foundation for multiview
learning. A single flow provides a bijection between an observed view and a
latent representation with a known base density, which allows exact
log-likelihoods, principled comparison of subjects and cohorts, and closed-form
Gaussian conditioning once a latent Gaussian model is determined. Because flows can
be parameterized with convolutional architectures for images or multilayer
perceptrons for tabular variables, the same machinery can be applied across
multimodal imaging, imaging-derived phenotypes, and broader multiview tabular
settings.

## Contributions

We introduce latent-aligned multiview normalizing (LAMNr) flows, a general
framework that learns shared and private latent structure across multiple views
while preserving exact likelihoods and invertibility. Each view is equipped with
its own flow that maps observations to a latent space.  Multiview structure is
encoded by aligning selected latent coordinates across views while allowing the
remaining coordinates to remain view-specific. For images we adopt Glow-style
multiscale architectures to retain spatial detail and access to latent features at
multiple resolutions. For imaging-derived phenotypes and other tabular blocks we
use per-view flows with the same alignment and inference machinery, so that
continuous-valued views are treated in a unified way.

Given subject-matched batches, we encourage shared representations with
alignment losses. We optionally identify candidate shared coordinates with a
short CCA or HSIC screen, then restrict alignment to those directions, leaving
the orthogonal complement as view-specific. On top of maximum-likelihood
training, we estimate per-level moments of the joint latents and add a
conditional Gaussian layer, which provides closed-form posteriors over arbitrary
latent subsets. This combination of aligned flows and Gaussian latent structure
yields the same style of shared/private decomposition and conditional reasoning
as SiMLR, but in a nonlinear, invertible latent space.

For images, replacing private latents by their conditional means produces
shared-latent reconstructions that preserve anatomy or identity while
suppressing view-specific contrast. These shared-latent images can act as
contrast-robust surrogates for downstream tasks. For tabular IDPs and other
multiview blocks, the same conditional layer supports calibrated queries,
harmonization, and model-based counterfactuals. The framework is designed to be
general: it applies to multiple imaging contrasts, multiview IDP blocks, and
multimodal tabular datasets where each view comprises a coherent set of
variables.

In summary, our contributions are:
1) We extend linear shared/private decompositions such as SiMLR to nonlinear,
   multiscale normalizing flows, yielding a deep, likelihood-based multiview
   model with explicit shared and view-specific latent coordinates.
2) We develop a practical training recipe that combines per-view flows, latent
   alignment losses, and optional CCA or HSIC screening with a Gaussian latent
   layer, providing closed-form conditional posteriors for arbitrary subsets of
   views and latent variables.
3) We show how the same framework applies to images and multiview tabular data:
   shared-latent reconstructions act as contrast-robust surrogates for image
   views, while conditional queries enable calibrated harmonization and
   counterfactual reasoning for imaging-derived phenotypes and related IDPs.
4) We provide an open-source implementation with 2D and 3D architectures and
   evaluate on multimodal MRI and multiview IDP datasets, comparing against
   linear baselines emphasizing predictability, interpretability,
   and reproducibility [@papamakarios2021nfreview; @kobyzev2020nfsurvey;
   @kingma2018glow; @Avants2021NatCompSci; @Tustison:2024aa].
