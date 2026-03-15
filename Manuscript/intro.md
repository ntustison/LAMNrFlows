
\clearpage

# Introduction

Medical imaging data and their representative latent spaces are fundamental to
gaining insight into biological structure and function. While deep learning has
become the current standard for navigating these high-dimensional spaces,
certain aspects of contemporary architectures (e.g., intractable likelihoods,
lack of bijective mappings) limit rigorous statistical analysis. These
deficiencies can complicate probabilistic calibration, objective comparison, and
the precise latent manipulations necessary for computational anatomy. For
instance, Generative Adversarial Networks (GANs) are implicit samplers trained
with divergence surrogates rather than likelihoods, which precludes calibration
by exact probabilities [@papamakarios2021nfreview]. Variational Autoencoders
(VAEs) optimize an evidence lower bound rather than the exact log likelihood
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
topologically unfolding the complex anatomical manifold sampled by modern
medical imaging. By mapping complex imaging data to a symmetric Gaussian base
distribution, the flow-induced metric ensures that latent straight lines
approximate geodesic paths in the original data domain. While latent diffusion
and flow matching achieve high sample quality, they optimize denoising or
continuous-transport objectives rather than exact log likelihoods, requiring
multi-step sampling or ODE integration [@lipman2022flowmatching;
@croitoru2023diffusion_vision_survey; @ho2020ddpm]. By contrast, normalizing
flows offer an exact, interpretable framework with single-pass inversion,
exposing multiscale latents for per-level alignment and enabling closed-form
conditional queries. These advantages point to normalzing flows as an attractive
framework for likelihood-calibrated multiview modeling for computational
anatomy.


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

Similarity-driven multilinear reconstruction (SiMLR) captures this joint
variation in a linear, low-rank setting by projecting multiview data into a
shared subspace under subject-level similarity constraints
[@Avants2021NatCompSci]. However, while SiMLR isolates this shared
representation, it treats the remaining view-specific variation as an
unstructured residual rather than explicitly modeling a private component. This
separation supports robust cross-view harmonization and prediction by isolating
stable population effects from idiosyncratic noise
[@Stone2020BreachersNeuroimaging; @Stone2024USSOCOM]. While deep learning
approaches have explored cross-modal translation and disentanglement using
convolutional neural networks, VAEs, or diffusion models [@havaei2016hemis;
@Chartsias2019SDNet; @yuan2024remind], they often lack the unique combination of
exact likelihoods and one-shot invertible mappings required for rigorous
computational anatomy. Recent works have also explored normalizing flows for
unsupervised MRI harmonization, but utilize the flow purely as a test-time
density estimator to iteratively adapt an auxiliary translation network to an
unknown target domain [@Beizaee2025].

Unlike test-time adaptation strategies that require iterative network updates
during inference [@beizaee2025harmonizingflows], LAMNr flows bridge this gap by
analogizing the SiMLR framework into a deep, likelihood-based architecture that
topologically unfolds the complex anatomical manifold into a continuous vector
space. Instead of an explicit linear factorization in the observation domain,
LAMNr flows map each view into a shared multiscale latent space, ensuring exact
log-likelihoods and bijective mappings. By utilizing latent-alignment objectives
(e.g., VICReg, InfoNCE) to identify shared coordinates, the framework recovers
the interpretability of a shared/private decomposition within a nonlinear,
invertible space. Crucially, by modeling the joint latents with a Gaussian
distribution, LAMNr flows enable closed-form conditional reconstructions. This
allows the shared subspace to function as a geometrically-informed coordinate
system. 

Additionally, the development of LAMNr flows represents a strategic evolution in
ensuring topological integrity within neural density estimators. Historically,
models like Deep Diffeomorphic Normalizing Flows (DDNF) [@salman2018deep]
enforced smoothness by integrating time-varying velocity fields via Ordinary
Differential Equations (ODEs). While this continuous formulation guarantees a
diffeomorphic mapping, the computational cost of ODE integration is often
prohibitive for large-scale medical imaging applications. To address this, LAMNr
transitions from the continuous "geodesic flow" of DDNF to the discrete,
efficient architecture of Glow [@kingma2018glow]. By aligning disparate
modalities and views into a shared latent representation, the LAMNr flows model
is steered to prioritizing robust, underlying anatomical structures over
idiosyncratic noise. This Latent-Alignment acts in synergy with specific
numerical safeguards, such as bounding the scale parameters within the affine
coupling layers, to mitigate gradient blow-ups during training. Furthermore, the
inclusion of training jitter serves as an additional regularizer (i.e.,
"dequantization" [@ho2019flowpp]). By introducing stochastic intensity- and
shape-based perturbations during the learning phase, the model is discouraged
from over-fitting to local voxel intensities. Together, these constraints force
convergence on more generalized anatomical representations, stabilizing the
Jacobian determinant and ensuring that the discrete transitions of the Glow
architecture maintain the smooth, diffeomorphic properties required.



## Bridging computational anatomy and normalizing flows

\begin{figure}[!htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{Figures/lamnr_templates.png}
    \caption{Bijective mapping between image and latent spaces established by a
    normalizing flow for a single modality (e.g., T1-weighted MRI). The
    architecture projects complex anatomical data from the observation space
    $\mathcal{X}$ (left) to a tractable, symmetric Gaussian base distribution in
    the latent space $\mathcal{Z} \sim \mathcal{N}(0, I)$ (right). This
    single-pass encoding and decoding mechanism provides the exact-likelihood
    foundation upon which the LAMNr flows framework is built.}
    \label{fig:single_view_flow}
\end{figure}

Computational anatomy (CA) is a comprehensive mathematical discipline that
formalizes the study of biological shape and its variability through the action
of diffeomorphic transformation groups on anatomical manifolds. Within this
broader probabilistic and geometric framework, typical population structure is
represented by a deformable template. This template is formally established as
the Fréchet mean—a stationary point on a curved manifold that minimizes the sum
of squared geodesic distances across a cohort. Traditionally, the intrinsic
curvature of image spaces causes a divergence between the Fréchet mean, the
Karcher mean, and the statistical mode. This divergence necessitates complex,
computationally demanding, non-linear registration frameworks to construct
templates that preserve anatomical consistency.

Normalizing flows offer a transformative theoretical perspective by effectively
organizing these nonlinear manifolds through a bijective mapping to a symmetric,
centered Gaussian base distribution (Figure \ref{fig:single_view_flow}). In this
latent space, the mathematical properties of the Gaussian prior ensure that the
mean, mode, and median coincide precisely at the origin ($z=0$). Consequently,
the inverse mapping of this origin, $f^{-1}(0)$, provides a principled,
single-pass approximation of the population Fréchet mean in the image domain. By
anchoring the cohort to this "latent-mean" template, the framework establishes
an approximate geodesic linearity where the deformation path from any subject to
the latent center is represented as a straight line.

Beyond template construction, this continuous latent framework provides direct
analogues to the fundamental metric operations of traditional computational
anatomy. In classic diffeomorphic frameworks, the transformation between a
source and target anatomy is governed by integrating a time-varying velocity
field over a continuous time domain $t \in [0, 1]$. The length of this optimal,
continuous deformation path establishes the exact geodesic distance between the
two biological structures. In the LAMNr flows framework, this computationally
intensive temporal integration is bypassed in favor of algebraic interpolation
within the latent space. Traversing the latent manifold between two encoded
images, $z_0$ and $z_1$, using a scalar interpolation parameter $\alpha \in [0,
1]$ generates a continuous trajectory of decoded images that closely
approximates this diffeomorphic flow. Consequently, the distances computed
directly in the latent space—when properly evaluated via distribution-preserving
spherical metrics rather than naive Euclidean norms—serve as highly efficient
surrogates for the complex, deformation-based geodesic distances of traditional
computational anatomy.


## Contributions 

\textcolor{red}{Rework when closer to finishing.}

We introduce LAMNr flows, a general
framework for deep computational anatomy that learns shared and private latent
structures across multiple views while preserving exact likelihoods and
invertibility. Within LAMNr flows, each view is equipped with a dedicated flow that
maps observations to a structured latent space. 

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
   flow-based tools limited to 2D slice-wise processing
   [@Beizaee2025;@Wen:2023aa], we provide a comprehensive, open-source, 2D and
   3D PyTorch implementation, based on the `normflows` library
   [@stimper2023normflows], which is integrated with the ANTsX ecosystem (via
   ANTsTorch) for robust data handling and auxiliary functionality.

Evaluations on multimodal MRI and multiview IDP datasets demonstrate that LAMNr
flows improve calibrated likelihoods and downstream prediction while providing a
single, exact framework for likelihood-calibrated multiview analysis.

