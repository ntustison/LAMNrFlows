
\clearpage

# Introduction

Medical imaging data and their representative latent spaces are fundamental to
gaining insight into biological structure and function. While deep learning has
become the standard for navigating these high-dimensional spaces, many
contemporary architectures lack the tractable likelihoods, exact invertibility,
or bijective mappings required for rigorous statistical analysis. These
deficiencies complicate probabilistic calibration, objective comparison, and the
precise latent manipulations necessary for computational anatomy. For instance,
Generative Adversarial Networks are implicit samplers trained with divergence
surrogates rather than likelihoods, which precludes calibration by exact
probabilities [@papamakarios2021nfreview]. Variational Autoencoders optimize an
evidence lower bound rather than the exact log likelihood
[@kobyzev2020nfsurvey]. Diffusion and score-based models rely on denoising or
score-matching objectives with likelihoods obtained only indirectly
[@croitoru2023diffusion_vision_survey]. Finally, while autoregressive decoders
offer exact likelihoods, they do not yield a one-shot invertible latent
representation [@papamakarios2021nfreview]. Such limitations are particularly
acute in multimodal and multiview settings, where heterogeneous or missing data
require calibrated cross-view comparisons and coherent anatomical
reconstructions.

## Normalizing flows

Normalizing flows model complex data distributions by composing invertible
transformations that map input data to their corresponding latents. This
bijective design simultaneously yields three salient properties: 1) exact
likelihoods via the change-of-variables formula, 2) single-pass inversion, and
3) direct access to latent variables that can be manipulated and decoded without
approximation [@papamakarios2021nfreview; @kobyzev2020nfsurvey]. Early
developments established the properties and advantages of invertible networks
and flow-based density models [@Gomez2017RevNet; @Jacobsen2018iRevNet;
@dinh2014nice; @rezende2015variational; @dinh2016realnvp; @kingma2016iaf;
@papamakarios2017maf]. Glow architectures introduced data-dependent
normalization, invertible $1 \times 1 (\times 1)$ convolutions, and a multiscale
structure optimized for high-resolution imaging [@kingma2018glow], with
subsequent variants improving coupling transforms and stability while preserving
exact likelihoods [@ho2019flowpp; @durkan2019nsf; @behrmann2019resflow;
@grathwohl2019ffjord]. Recent work has demonstrated that flows scale to
resolutions and sample qualities comparable to other state-of-the-art generative
models [@croitoru2023diffusion_vision_survey; @zhai2024tarflow;
@gu2025starflow].

Beyond density estimation, normalizing flows provide a geometric framework for
linearizing the anatomical manifold. By mapping complex imaging data to a
symmetric Gaussian base distribution, the flow-induced metric ensures that
latent straight lines approximate geodesic paths in the original data domain.
While latent diffusion and flow matching achieve high sample quality, they
optimize denoising or continuous-transport objectives rather than exact log
likelihoods, requiring multi-step sampling or ODE integration
[@lipman2022flowmatching; @croitoru2023diffusion_vision_survey; @ho2020ddpm]. By
contrast, normalizing flows offer an exact, interpretable framework with
single-pass inversion, exposing multiscale latents for per-level alignment and
enabling closed-form conditional queries. These advantages point to normalzing
flows as an attractive framework for likelihood-calibrated multiview modeling in 
deep computational anatomy.


## Multiview learning with LAMNr flows

Multiview learning operates on two complementary principles: first, that each
distinct acquisition or feature space ("view") contributes unique, view-specific
information, and second, that shared information across views can be distilled
into lower-dimensional projections to improve calibration and cross-cohort
comparability. Traditionally, these shared projections have been estimated using
classical correlation-based methods such as Canonical Correlation Analysis (CCA)
[@Hotelling1936CCA; @Hardoon2004CCAOverview]. More recently, kernel-based
measures like the Hilbert–Schmidt Independence Criterion (HSIC)
[@gretton2005hsic] and learned alignment objectives,including Barlow Twins,
VICReg, and InfoNCE [@zbontar2021barlow; @bardes2021vicreg; @oord2018cpc], have
expanded these capabilities to accommodate the complex, non-linear patterns
inherent in modern neuroimaging datasets.

Similarity-driven multilinear reconstruction (SiMLR) instantiates this
decomposition in a linear, low-rank setting by factorizing multiview data into
shared and view-specific components under subject-level similarity constraints
[@Avants2021NatCompSci]. In the SiMLR framework, each view is modeled as the sum
of a low-rank shared representation and a private residual. This separation
supports robust cross-view harmonization and prediction by isolating stable
population effects from idiosyncratic noise [@Stone2020BreachersNeuroimaging;
@Stone2024USSOCOM]. While deep learning approaches have explored cross-modal
translation and disentanglement using CNNs, VAEs, or Diffusion models
[@havaei2016hemis; @Chartsias2019SDNet; @yuan2024remind], they often lack the
unique combination of exact likelihoods and one-shot invertible mappings
required for rigorous computational anatomy. Recent works have also explored
normalizing flows for unsupervised MRI harmonization, but utilize the
flow purely as a test-time density estimator to iteratively adapt an auxiliary
translation network to an unknown target domain [@Beizaee2025].

Unlike test-time adaptation strategies that require iterative network updates
during inference, LAMNr flows bridge this gap by extending the SiMLR framework
into a deep, likelihood-based architecture that linearizes the anatomical manifold.
Instead of an explicit linear factorization in the observation domain, LAMNr maps each
view into a shared multiscale latent space using normalizing flows, ensuring
exact log-likelihoods and bijective mappings. By utilizing latent-alignment
objectives (e.g., VICReg, InfoNCE) to identify shared coordinates, the framework
recovers the interpretability of a shared/private decomposition within a
nonlinear, invertible space. Crucially, by modeling the joint latents with a
Gaussian distribution, LAMNr flows enable closed-form conditional queries and
high-fidelity reconstructions. This allows the shared subspace to function as a
geometrically-informed coordinate system, facilitating contrast-robust
population representatives and providing a direct path to the "latent-mean"
templates discussed below in the context of the Fréchet mean.

Additionally, the development of LAMNr flows represents a strategic evolution in
ensuring topological integrity within neural density estimators. Historically,
models like Deep Diffeomorphic Normalizing Flows (DDNF) [@salman2018deep]
enforced smoothness by integrating time-varying velocity fields via Ordinary
Differential Equations (ODEs). While this continuous formulation guarantees a
diffeomorphic mapping, the computational cost of ODE integration is often
prohibitive for large-scale medical imaging applications. To address this, LAMNr
transitions from the continuous "geodesic flow" of DDNF to the discrete,
efficient architecture of Glow [@kingma2018glow]. While coupling-based flows
like Glow are mathematically bijective, they lack the inherent temporal
continuity that prevents anatomically "jagged" deformations. 

By aligning disparate modalities and views into a shared latent representation,
the LAMNr flows model is steered to prioritizing robust, underlying anatomical
structures over idiosyncratic noise. This Latent-Alignment acts in synergy with
specific numerical safeguards, such as bounding the scale parameters within the
affine coupling layers, to mitigate gradient blow-ups during training.
Furthermore, the inclusion of training jitter serves as an additional
regularizer (i.e., "dequantization" [@ho2019flowpp]). By introducing stochastic
intensity- and shape-based perturbations during the learning phase, the model is
discouraged from over-fitting to local voxel intensities. Together, these
constraints force convergence on more generalized anatomical representations,
stabilizing the Jacobian determinant and ensuring that the discrete transitions
of the Glow architecture maintain the smooth, diffeomorphic properties required
for robust computational anatomy.

## A computational anatomy perspective

\begin{figure*}[!t]
    \centering
    \includegraphics[width=\textwidth]{Figures/lamnr_templates.pdf}
    \caption{ 
    (Left) Anatomical variation in the observed data space $\mathcal{X}$ forms a
    non-linear manifold, visualized by the warped coordinate grid and the
    curved, non-Euclidean paths connecting individual subjects. (Right) The
    LAMNr framework learns a bijective mapping $f_{\theta}: \mathcal{X}
    \rightarrow \mathcal{Z}$ that transforms this manifold into a symmetric,
    centered Gaussian latent space $\mathcal{Z}$. In this linearized geometry,
    the origin $z=0$ represents the population mode and mean. Inverting this
    origin yields the "latent-mean" template $\hat{x}_0 = f^{-1}(0)$ (central
    brain), which serves as a contrast-robust representative of the cohort's
    anatomy. The regular grid in $\mathcal{Z}$ illustrates how the flow
    "unfolds" anatomical complexity, allowing straight lines in latent space to
    approximate geodesic paths in the image domain.}
    \label{fig:lamnr_manifold}
\end{figure*}

Beyond the proposed technical advancements, another contribution of this work is
the exploratory bridge it establishes between deep statistical and generative
modeling and the foundational principles of computational anatomy (CA). The
mathematical foundation of CA has long relied on Riemannian geometry to define
population templates and statistical variations
[@GrenanderMiller1998CA;@Trouve1998DiffeoPatternMatching;@Miller2002LDDMMOverview].
Within this framework, a template is formally defined as the Fréchet mean, i.e.,
the point on a curved manifold that minimizes the sum of squared geodesic
distances to all subjects in a cohort [@Avants:2010aa]. However, as noted by
Fletcher et al. [@Fletcher2009aa], the intrinsic curvature of image spaces
typically causes a divergence between the Fréchet mean (the variance minimizer),
the Karcher mean (a local stationary point), and the mode (the most probable
individual). This necessitates complex, non-linear mathematical machinery to
preserve anatomical consistency.  In implementation, Large Deformation
Diffeomorphic Metric Mapping (LDDMM) [@Beg2005LDDMM] and Symmetric Normalization
[@Avants:2008aa] are two popular template construction approaches.  Their
population template estimates are fixed points of a barycentric optimization
with respect to the induced geodesic distance.

Normalizing flows offer a transformative perspective by effectively linearizing
nonlinear manifolds through a bijective mapping to a symmetric, centered
Gaussian base distribution. In this latent space, the properties of the Gaussian
prior ensure that the mean, mode, and median coincide at the origin ($z=0$).
Consequently, the inverse mapping of this origin, $f^{-1}(0)$, provides a
principled approximation of the population Fréchet mean in the image domain. By
anchoring the cohort to this "latent-mean" template, the framework establishes
an approximate geodesic linearity where the deformation path from any subject to
the latent center is represented as a straight line. This object effectively
concentrates cohort-common signal under the model’s likelihood while suppressing
idiosyncratic variations that do not persist across subjects. By explicitly
promoting a shared subspace through latent alignment, $\hat{x}_0$ reflects
stable population anatomy rather than shape or intensity outliers, allowing the
resulting shared-latent image to function as a contrast-robust population
representative across heterogeneous views. This perspective reconciles
high-dimensional anatomical complexity with the simplicity of Euclidean
statistics, providing a robust, likelihood-based alternative to traditional
iterative group-wise registration.

The relationship to CA can be further elucidated by comparing objective
functions. CA-based templates minimize geodesic energy under a diffeomorphic
metric whereas normalizing flows maximize model likelihood (equivalently, minimizes
negative log likelihood) under a pushforward of the Gaussian base distribution.
If the network is locally well-conditioned around the population and the cohort
lies in a near-linear region of latent space (as is often the case for
high-quality, healthy-control datasets), then Euclidean barycenters in $z$-space
provide first-order approximations to barycenters in image space with respect to
the pullback metric, $g_x$.[^gx] Under these conditions, $\hat{x}_0$ acts as a
practical proxy for a Fréchet mean as it is the image at the center of the
distribution induced by the learned bijection.[^CA]

<!--
To maintain the validity of the pullback metric $g_x = J_f(x)^\top J_f(x)$, we
implemented specific numerical constraints within the affine coupling layers. By
bounding the scale parameters, we prevent the Jacobian determinant from
approaching zero or exploding, which ensures that the induced Riemannian metric
remains well-conditioned. This stabilization is critical for the process: it
ensures that the latent origin ($z=0$) remains a stable, computationally
efficient approximation of the population Fréchet mean. By preventing extreme
local distortions in $g_x$, these numerical safeguards ensure that straight
lines in the latent space correspond to smooth, length-preserving paths in the
image space. This effectively linearizes the underlying geodesic structure of
the cohort, allowing for rigorous statistical analysis and representative
"atlas" reconstruction without the overhead of iterative Karcher mean
computations. 
-->

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

Framed this way, LAMNr flows is a complementary analogical view of CA captured
by multimodal imaging that provides an exact probabilistic formulation that
admits closed-form conditioning, one-shot inversion, and multiscale latent
access. The latent-mean $\hat{x}_0$ is then best understood as a statistically
grounded proxy for the population center whose proximity to the CA Fréchet mean
can be measured and, under smoothness and local-linearity conditions, justified
to first order. This perspective also clarifies the role of shared-latent images
i.e., by projecting to coordinates that carry cross-subject, cross-view signal
and neutralizing private factors, they produce references that are more uniform
under diffeomorphic matching while remaining invertible to and from subject
space. In the Results section, we leverage this view to provide a novel strategy
for improved image registration through latent-space editing, a particularly
useful approach where common diffeomorphic anatomical assumptions are violated.

## Contributions

We introduce Latent-Aligned Multiview Normalizing (LAMNr) flows, a general
framework for deep computational anatomy that learns shared and private latent
structures across multiple views while preserving exact likelihoods and
invertibility. Within LAMNr, each view is equipped with a dedicated flow that
maps observations to a structured latent space. By anchoring the population to a
known base distribution, the framework linearizes the anatomical manifold,
ensuring that the latent origin serves as a principled approximation of the
population Fréchet mean.

Key contributions of this work include:

1. **Unified Multiview Modeling:** We provide a shared coordinate system for
   heterogeneous data types, including 2D/3D images and tabular imaging-derived
   phenotypes (IDPs). For imaging, we adopt Glow-style multiscale architectures
   to retain spatial detail; for tabular blocks, we utilize integrated per-view
   flows.

2. **Latent Alignment and Linearization:** Using subject-matched batches, we
   identify shared anatomical features via a library of alignment losses (e.g.,
   VICReg, InfoNCE). We optionally employ CCA or HSIC screens to restrict
   alignment to statistically shared directions, leaving the orthogonal
   complement to capture view-specific variation.

3. **Closed-form Inference and Reconstruction:** We incorporate a conditional
   Gaussian layer to provide closed-form posteriors over arbitrary latent
   subsets. This yields a nonlinear, invertible extension of the shared/private
   decomposition found in SiMLR [@Avants2021NatCompSci].

4. **Contrast-Robust Surrogates:** We demonstrate that substituting private
   latents with conditional means produces shared-latent reconstructions that
   preserve identity while suppressing idiosyncratic contrast. These
   "latent-mean" images act as robust representatives that empirically reduce
   diffeomorphic registration effort.

5. **Open-source, 3D-capable Implementation:** Unlike many contemporary
   flow-based tools restricted to 2D slice-wise processing [@Beizaee2025], we
   provide a comprehensive, 2D and 3D PyTorch implementation.
   Integrated with the ANTsX ecosystem (via ANTsTorch) for robust data handling
   and registration, and accompanied by a significantly updated `normflows`
   library, our release ensures reproducible, volume-level computational anatomy
   [@Tustison:2024aa; @stimper2023normflows].

Evaluations on multimodal MRI and multiview IDP datasets demonstrate that LAMNr flows improve calibrated likelihoods and downstream prediction while providing a single, exact framework for likelihood-calibrated multiview reasoning.