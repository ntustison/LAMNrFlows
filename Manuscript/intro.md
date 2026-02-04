
\clearpage

# Introduction

Medical imaging data and their representative latent spaces are essential for
gaining insight into biological structure and function. While deep learning
workflows have become foundational for leveraging these spaces, many widely used
approaches lack tractable data likelihoods, optimize only surrogate bounds, or
fail to provide invertible latent mappings. These deficiencies complicate
calibration, objective comparison, and precise latent manipulations. For
instance, Generative Adversarial Networks are implicit samplers trained with
divergence surrogates rather than likelihoods, which precludes calibration by
exact probabilities [@papamakarios2021nfreview]. Variational Autoencoders
optimize an evidence lower bound rather than the exact log likelihood
[@kobyzev2020nfsurvey]. Diffusion and score-based models rely on denoising or
score-matching objectives with likelihoods obtained only indirectly
[@croitoru2023diffusion_vision_survey]. Autoregressive decoders offer exact
likelihoods but do not yield a one-shot invertible latent representation
[@papamakarios2021nfreview]. These limitations are acute in multimodal and
multiview settings, where heterogeneous or missing views require calibrated
comparisons and coherent cross-view reconstructions.

## Normalizing flows

Normalizing flows model complex data distributions by composing invertible
transformations that map input data to their corresponding latents.  This design
simultaneously yields three salient properties: 1) exact likelihoods via the
change-of-variables formula, 2) single-pass inversion, and
3) direct access to latent variables that can be manipulated and decoded without
approximation [@papamakarios2021nfreview; @kobyzev2020nfsurvey]. Early
developments established invertible networks and flow-based density models
[@Gomez2017RevNet; @Jacobsen2018iRevNet; @dinh2014nice; @rezende2015variational;
@dinh2016realnvp; @kingma2016iaf; @papamakarios2017maf]. Glow architectures
added data-dependent normalization, invertible $1 \times 1 (\times 1)$
convolutions, and a multiscale structure suited to high-resolution imaging
[@kingma2018glow], with later variants improving coupling transforms, stability,
and parameterization while preserving exact likelihoods [@ho2019flowpp;
@durkan2019nsf; @behrmann2019resflow; @grathwohl2019ffjord]. Recent work shows
that flows now scale to resolutions and sample qualities comparable to other
state-of-the-art generative models [@croitoru2023diffusion_vision_survey;
@zhai2024tarflow; @gu2025starflow]. 

In parallel, latent diffusion and flow matching achieve strong sample quality
but optimize denoising or continuous-transport objectives rather than exact log
likelihoods and require time-consuming multi-step sampling or ODE integration at
inference [@lipman2022flowmatching; @croitoru2023diffusion_vision_survey;
@ho2020ddpm]. By contrast, normalizing flows use discrete flows with single-pass
inversion and exact likelihood, exposing multiscale latents for per-level
alignment and enabling closed-form conditional queries. We therefore propose
flows as an attractive alternative providing an exact, interpretable framework
for likelihood-calibrated multiview modeling.

## Multiview learning with LAMNr flows

Normalizing flows with exposed latents provide a natural foundation for
multiview learning.  A "view" represents a set of measurements on a common set
of subjects derived from a distinct acquisition or feature space (e.g.,
disparate image modalities or tabular blocks of imaging-derived phenotypes).
These views typically vary in scale, noise characteristics, and other
confounding factors. Multiview analysis leverages two complementary principles:
first, that each view contributes unique, view-specific information and second,
that shared information across views can be distilled into lower-dimensional
projections to improve calibration and cross-cohort comparability.
Traditionally, these shared projections have been estimated using classical
correlation-based methods such as Canonical Correlation Analysis (CCA)
[@Hotelling1936CCA; @Hardoon2004CCAOverview]. More recently, kernel-based
measures like the Hilbert–Schmidt Independence Criterion (HSIC)
[@gretton2005hsic] and learned alignment objectives, such as Barlow Twins,
VICReg, and InfoNCE [@zbontar2021barlow; @bardes2021vicreg; @oord2018cpc], have
expanded these capabilities to accommodate complex, incomplete data patterns
[@bishop2006prml; @Murphy2012ML].

Similarity-driven multilinear reconstruction (SiMLR) instantiates this
decomposition in a linear, low-rank setting by factorizing multiview data into
shared and view-specific components under subject-level similarity constraints
[@Avants2021NatCompSci]. In the SiMLR framework, each view is modeled as the sum
of a low-rank shared representation and a private residual. These shared factors
are regularized to respect an external similarity structure often derived from
clinical, cognitive, or exposure variables, which encourages the construction of
components that are both statistically coherent across views and aligned with
downstream phenotypes. Consequently, private components capture contrast- or
modality-specific variation. This separation supports robust cross-view
harmonization, visualization, and prediction by isolating common effects from
idiosyncratic noise (e.g., [@Stone2020BreachersNeuroimaging; @Stone2024USSOCOM]).

Beyond linear methods, other multiview representation-learning approaches target
shared and private structures. In medical imaging, cross-modal translation and
imputation have been explored via supervised CNNs and adversarial or
cycle-consistent mappings [@han2017dcnn; @florkow2020mrm;
@yang2018structurecyclegan; @lei2019densecyclegan]. Frameworks like HeMIS
(Hetero-Modal Image Segmentation) learn latent spaces that can be averaged
across modalities to handle missing views [@havaei2016hemis], while
diffusion-based models provide strong priors for imputation with uncertainty
quantification [@yuan2024remind; @webber2024bjrai]. Other efforts aim to
disentangle shared and private factors using autoencoders and contrastive losses
[@Chartsias2018MILR; @Chartsias2019SDNet], or link paired modalities via
conditional couplings in flow-based models [@sun2019dualglow]. 

While these methods emphasize the separation of view-invariant content, they
generally lack the unique combination of exact likelihoods, one-shot invertible
mapping, and explicit Gaussian latent structure that enables the closed-form
conditional queries for both image and tabular data provided by our proposed
LAMNr flows approach. Specifically, LAMNr flows extend the SiMLR framework into
a deep, likelihood-based architecture capable of modeling nonlinear, invertible
latent spaces.  Instead of an explicit linear factorization in the observation
domain, LAMNr flows map each view into a shared multiscale latent space using
normalizing flows, ensuring exact log-likelihoods and bijective mappings.
Latent-alignment objectives (e.g., VICReg, InfoNCE) identify specific
coordinates that function as the shared components, while the remaining
coordinates serve as view-specific latents. By modeling the joint latents with a
Gaussian distribution and utilizing conditional Gaussian formulations, LAMNr
flows recover the interpretability of SiMLR’s shared/private decomposition while
providing nonlinear representational capacity, calibrated likelihoods, and the
ability to generate high-fidelity reconstructions in the original data space.


## LAMNr flows:  a computational anatomy perspective

In the image setting, a single Glow-style normalizing flow yields a
likelihood-calibrated single template via the latent mean $f^{-1}(0)$.  LAMNr
flows extend this to multiple views.  By aligning shared latent coordinates and
placing a conditional Gaussian over per-level latents, one can construct
contrast-robust shared-latent images and move analytically between
modality-specific representations. This demonstrates a potential link to the
field of computational anatomy (CA) [@GrenanderMiller1998CA], where such shared
representatives function as practical proxies for population templates and,
among other utilities, reduce diffeomorphic effort in registration.

CA treats anatomical variability as the action of diffeomorphisms on a template
with a Riemannian metric on velocity fields that induces a geodesic distance
between images [@Trouve1998DiffeoPatternMatching; @GrenanderMiller1998CA].
Within this framework, a population template is a Fréchet mean, i.e., the image
that minimizes the sum of squared geodesic distances to all subjects under that
metric [@Avants:2010aa]. Foundational work formalizes this orbit model and links
statistical estimation to geometric deformation via flows on shape spaces,
motivating modern diffeomorphic registration and template construction
[@Miller2002LDDMMOverview]. In implementation, Large Deformation Diffeomorphic
Metric Mapping (LDDMM) [@Beg2005LDDMM] and Symmetric Normalization
[@Avants:2008aa] are two popular approaches.  Their population template estimates
are fixed points of a barycentric optimization with respect to the induced
geodesic distance.

Normalizing flows (via the Glow architecture) formalize from this geometric
viewpoint with an exact, invertible statistical modeling approach. Each image is
mapped bijectively to a latent vector $z$ whose distribution is characterized by
a centered Gaussian with unit variance (i.e., the base distribution). The
inverse $f^{-1}$ maps latent coordinates back to image space. Because the base
distribution is Gaussian, the origin $z=0$ is both the mean and the mode in
latent space. Mapping this origin inversely through the network yields a
latent-mean reconstruction $\hat{x}_0 = f^{-1}(0)$. This object concentrates
cohort-common signal under the model’s likelihood and suppresses idiosyncratic
variation that does not persist across subjects. When LAMNr flows use latent
alignment across views, the shared subspace is explicitly promoted, so
$\hat{x}_0$ reflects common population anatomy (as opposed to shape/intensity
outliers), and the resulting shared-latent image behaves like a contrast-robust
population representative.

The relationship to CA can be further elucidated by comparing objective
functions. CA-based templates minimize geodesic energy under a diffeomorphic
metric whereas LAMNr flows maximize model likelihood (equivalently, minimizes
negative log likelihood) under a pushforward of the Gaussian base distribution.
If the network is locally well-conditioned around the population and the cohort
lies in a near-linear region of latent space (as is often the case for
high-quality, healthy-control datasets), then Euclidean barycenters in $z$-space
provide first-order approximations to barycenters in image space with respect to
the pullback metric.[^gx] Under these conditions, $\hat{x}_0$ acts as a
practical proxy for a Fréchet mean as it is the image at the center of the
distribution induced by the learned bijection.[^CA]

[^gx]: The pullback metric $g_x = J_f(x)^\top J_f(x)$ is the flow-induced
Riemannian metric that makes latent straight lines locally length preserving.
Under this geometry, the inverse at the latent origin, $f^{-1}(0)$, is a natural
cohort center. This construction leverages the property that latent barycenters
approximate image-space Karcher means (stationary points of the sum-of-squared
distances) to first order. On a Riemannian manifold, the Fréchet mean is the
global minimizer of this functional.  By modeling the population via a centered
Gaussian in a bijective latent space, the origin provides a computationally
efficient approximation of this global mean, effectively linearizing the
underlying geodesic structure.


[^CA]: There are, however, important geometric caveats to our comparison. The Fréchet
mean is defined relative to a specific metric whereas LAMNr flows induces its own
pullback metric inversely through the network. Unless these metrics coincide (or
are close in the neighborhood of interest), equality of means is not guaranteed.
Furthermore, straight lines in latent space are geodesics for the latent
Euclidean metric, not necessarily for LDDMM in image space. Finally, when
modality-specific factors live in private latent coordinates, a naive latent
average can blend contrasts rather than anatomy. The shared-latent construction
mitigates this by replacing private coordinates with conditional means estimated
under the joint Gaussian latent model, which attenuates contrast and preserves
structural content.  These considerations suggest a principled empirical program to
substantiate the approximate relationship. First, one could compare template
energies by evaluating $\sum_i d_{\mathrm{LDDMM}}^2(I_i,\hat{x}_0)$ against the
same sum for a conventional SyN or LDDMM template as similar energies and
deformation statistics support geometric consistency. Second, one can initialize
a Fréchet-mean iteration under the registration metric at $\hat{x}_0$ and
measure the displacement to the fixed point since small corrections indicate
that the latent-mean sits near the CA barycenter. Third, one could assess path
consistency by contrasting deformations along decoded latent paths with geodesic
shooting toward the template. Fourth, for multicontrast cohorts, demonstrate
that shared-latent templates lower the variance of registration energies across
contrasts relative to raw-contrast templates, indicating that alignment has
concentrated anatomy and factored out modality.

Framed this way, LAMNr flows is a complementary analogical view of CA that
provides an exact probabilistic formulation that admits closed-form
conditioning, one-shot inversion, and multiscale latent access. The latent-mean
$\hat{x}_0$ is then best understood as a statistically grounded proxy for the
population center whose proximity to the CA Fréchet mean can be measured and,
under smoothness and local-linearity conditions, justified to first order. This
perspective also clarifies the role of shared-latent images i.e., by projecting
to coordinates that carry cross-subject, cross-view signal and neutralizing
private factors, they produce references that are more uniform under
diffeomorphic matching while remaining invertible to and from subject space. In
the Results section, we leverage this view to provide a novel strategy for
improved image registration through latent-space editing, a particularly useful
approach where common diffeomorphic anatomical assumptions are violated.

## Contributions

We introduce Latent-Aligned Multiview Normalizing flows, a general
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
views. Specifically, we employ a Gaussian-PCA base distribution to enforce
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

We provide an open-source implementation with 2D and 3D architectures built on
PyTorch, using ANTsTorch (and other libraries from the ANTsX ecosystem) for data
handling, augmentation, and registration utilities.  We also provide an extensively
updated PyTorch normflows library. We provide all evaluation scripts of LAMNr flows
on multimodal MRI and multiview IDP datasets, comparing performance against
linear baselines with an emphasis on predictability, interpretability, and
reproducibility [@papamakarios2021nfreview; @kobyzev2020nfsurvey;
@Tustison:2024aa; @stimper2023normflows].
