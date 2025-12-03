
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
idiosyncratic effects [@Stone2020BreachersNeuroimaging; @Stone2024USSOCOM].

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
realistic reconstructions in the original imaging space.


_Edit below_


### Shared and private representations

Cross-modal image translation and imputation have been studied with supervised
CNNs and with adversarial or cycle-consistent approaches [@han2017dcnn;
@florkow2020mrm; @yang2018structurecyclegan; @lei2019densecyclegan]. Methods
that accept missing modalities at inference without explicit synthesis, such as
HeMIS, offered robust alternatives by averaging in a learned latent space
[@havaei2016hemis]. Diffusion-based approaches have recently been adapted to
imputation and reconstruction with strong priors and uncertainty summaries
[@yuan2024remind; @webber2024bjrai]. Parallel efforts in medical representation
learning targeted explicit shared and private factors with autoencoders and
contrastive learning [@Chartsias2018MILR; @Chartsias2019SDNet]. Flow-based
multimodal modeling has also been explored for paired image tasks, for example
with conditional couplings linking latent spaces across contrasts
[@sun2019dualglow]. These works highlight the value of disentangling
view-invariant content from view-specific variations but generally lack exact
likelihoods and a one-shot invertible map to and from images.

### Tabular and IDP modeling at cohort scale

In large cohorts, multiview analyses often use linear embeddings, canonical
correlation, and screening strategies for representation, harmonization, and
transfer. A recent example is similarity-driven multiview embeddings for
high-dimensional biomedical data, which demonstrated coherent cross-view
structure discovery with rigorous validation across tasks
[@Avants2021NatCompSci]. Our prior UK Biobank work on imaging-derived phenotypes
used linear models as strong baselines and practical scorecards across cohorts
[@Tustison:2024aa]. LAMNr flows complement this landscape by providing an
exact-likelihood generative approach to multiview IDPs that exposes an
invertible latent, supports shared-subspace alignment, and enables closed-form
conditional queries.

### Applied multiview cohorts in operational blast-exposure populations

Operational and training environments provide natural multiview datasets mixing
imaging, cognitive, clinical, and molecular measures. Career breachers exposed
to repeated low-level blast have been profiled with multimodal MRI and serum
biomarkers, showing functional and structural correlates that benefit from joint
multiview analysis [@Stone2020BreachersNeuroimaging]. Follow-on work in Special
Operations Forces reported altered inflammatory signatures and extracellular
vesicle readouts alongside neurobehavioral assessment, again motivating
calibrated cross-view comparisons [@Stone2024USSOCOM]. Additional studies in
breacher training cohorts reported changes in glial fibrillary acidic protein
and longitudinal serum panels, reinforcing the multiview nature of these data
and the need to separate shared signal from view-specific factors
[@Tschiffely2020GFAPBreachers; @Kamimori2018BreachersSerum].











## Normalizing flows as a foundation

Normalizing flows model data by composing invertible transformations that map
inputs to tractable base distributions. This gives three properties that are
difficult to obtain together in other families: exact likelihoods, single-pass
inversion, and direct access to latent variables that can be manipulated and
decoded without approximation [@papamakarios2021nfreview; @kobyzev2020nfsurvey].
Early work developed both invertible networks and flow-based density models
[@Gomez2017RevNet; @Jacobsen2018iRevNet; @dinh2014nice; @rezende2015variational;
@dinh2016realnvp; @kingma2016iaf; @papamakarios2017maf]. Glow combined
data-dependent normalization, invertible 1×1 convolutions, and a multiscale
architecture suited to large images [@kingma2018glow]. Subsequent variants
improved coupling transforms, stability, and parameterization while keeping
exact likelihoods [@ho2019flowpp; @durkan2019nsf; @behrmann2019resflow;
@grathwohl2019ffjord]. Recent results show that flows scale to resolutions and
sample quality comparable to popular generative models, which supports their use
as first-class probabilistic backbones [@croitoru2023diffusion_vision_survey;
@zhai2024tarflow; @gu2025starflow].

## Latent-aligned multiview flows

We introduce latent-aligned multiview normalizing (LAMNr) flows, a general
framework that learns shared and private latent structure across multiple views
while preserving exact likelihoods and invertibility. For images we adopt
Glow-style models to retain multiscale access. For imaging-derived phenotypes
and other tabular blocks we use per-view flows with the same alignment and
inference machinery. Given subject-matched batches, we learn shared
representations with alignment losses such as Barlow Twins, VICReg, InfoNCE,
Pearson correlation, or HSIC [@zbontar2021barlow; @bardes2021vicreg;
@oord2018cpc; @gretton2005hsic]. We optionally identify shared coordinates with
a short CCA or HSIC screen, then restrict alignment to those directions. On top
of maximum-likelihood training, we estimate per-level moments of latents and use
a conditional Gaussian layer to compute closed-form posteriors over arbitrary
latent subsets. This enables principled cross-view imputation and targeted
latent manipulations with exact decoding.

For images, replacing private latents by their conditional means yields
shared-latent images that preserve anatomy while suppressing view-specific
contrast. These shared-latent images can serve as robust surrogates for
downstream tasks when desired. For tabular IDPs, the same conditional layer
supports calibrated queries, harmonization, and model-based counterfactuals. The
framework is intended to be general. It applies to multiple imaging contrasts
and also to multiview IDP blocks where each block comprises a coherent set of
variables.


## Contributions

We present an exact, invertible multiview framework that learns shared and
private latent structure across views using alignment losses and optional CCA or
HSIC screening, fits per-level Gaussian statistics on latents and provides
closed-form conditional posteriors for arbitrary subsets, supports shared-latent
reconstructions for image views and calibrated conditional queries for IDPs, and
integrates into open-source tooling for 2D and 3D data. We evaluate on
multimodal MRI and on multiview IDP datasets, and we compare against strong
linear multiview baselines. Throughout we emphasize exactness, interpretability,
and reproducibility [@papamakarios2021nfreview; @kobyzev2020nfsurvey;
@kingma2018glow; @Tustison:2024aa].
