
\clearpage

# Introduction

Medical imaging data and their representative latent spaces have become
fundamental for insight into biological structure and function. While deep
learning has become the current standard for navigating these high-dimensional
spaces, certain aspects of contemporary architectures (e.g., intractable
likelihoods, lack of bijective mappings) limit exploration, quantitative
analysis, and other potential applications through latent space evaluations and
manipulations. Generative Adversarial Networks (GANs), for instance, are
implicit samplers trained with divergence surrogates rather than likelihoods,
which precludes calibration by exact probabilities [@papamakarios2021nfreview].
Variational Autoencoders (VAEs) optimize an evidence lower bound rather than the
exact log likelihood [@kobyzev2020nfsurvey]. Diffusion and score-based models
rely on denoising or score-matching objectives with likelihoods obtained only
indirectly [@croitoru2023diffusion_vision_survey]. Finally, while autoregressive
decoders offer exact likelihoods, they do not yield a one-shot invertible latent
representation [@papamakarios2021nfreview]. Such limitations extend to
multimodal and multiview settings, where heterogeneous or missing data often
require cross-view comparisons and coherent anatomical reconstructions.

## Normalizing Flows

Normalizing flows model complex data distributions by composing invertible
transformations that map input data to their corresponding latents. This
bijective design simultaneously yields exact likelihoods via the
change-of-variables formula, single-pass inversion, and direct access to latent
variables that can be manipulated and decoded without approximation
[@papamakarios2021nfreview; @kobyzev2020nfsurvey]. Early developments
established the properties and advantages of invertible networks and flow-based
density models [@Gomez2017RevNet; @Jacobsen2018iRevNet; @dinh2014nice;
@rezende2015variational; @dinh2016realnvp; @kingma2016iaf;
@papamakarios2017maf]. Later, Glow architectures introduced data-dependent
normalization, invertible $1 \times 1 (\times 1)$ convolutions, and a multiscale
structure optimized for imaging [@kingma2018glow], with subsequent variants
improving coupling transforms and stability while preserving exact likelihoods
[@ho2019flowpp; @durkan2019nsf; @behrmann2019resflow; @grathwohl2019ffjord].
Recent work has demonstrated that flows scale to resolutions and sample
qualities comparable to other state-of-the-art generative models
[@croitoru2023diffusion_vision_survey; @zhai2024tarflow; @gu2025starflow].
Beyond density estimation, normalizing flows provide a geometric framework for
topologically unfolding the complex anatomical manifold sampled by modern
medical imaging. By mapping complex imaging data to a symmetric Gaussian base
distribution, the flow-induced metric ensures that latent paths approximate
geodesics in the original data domain [@arvanitidis2018latent;
@kobyzev2020nfsurvey]. Recent advancements have further refined these flow
trajectories by incorporating Semi-Discrete Optimal Transport (SDOT) during
training [@kong2025alignflow].  This approach establishes an explicit, optimal
alignment between the noise distribution and data points to ensure straighter
paths and more effective inference, even in high-dimensional settings.

The bijective formulation of these models also enables the synthesis of biological
variation through stochastic sampling, where latent vectors drawn from the
Gaussian prior are mapped back to the high-dimensional image space. While latent
diffusion and flow matching achieve high sample quality, they optimize denoising
or continuous-transport objectives rather than exact log likelihoods, requiring
multi-step sampling or ODE integration [@lipman2022flowmatching;
@croitoru2023diffusion_vision_survey; @ho2020ddpm]. By contrast, normalizing
flows offer an exact, interpretable framework with single-pass inversion,
exposing multiscale latents for per-level alignment and enabling closed-form
conditional queries. These advantages point to normalzing flows as an attractive
framework for likelihood-calibrated multiview modeling.


## Multiview Learning with LAMNr Flows

Multiview learning operates on two complementary principles: first, that each
distinct acquisition or feature space ("view") contributes unique, view-specific
information, and second, that shared information across views can be transformed
via projections (often to lower-dimensional space) to improve calibration and
cross-cohort comparability. Traditionally, these shared projections have been
estimated using classical correlation-based methods, such as Canonical
Correlation Analysis (CCA) [@Hotelling1936CCA; @Hardoon2004CCAOverview]. More
recently, kernel-based measures like the Hilbert–Schmidt Independence Criterion
(HSIC) [@gretton2005hsic] and learned alignment objectives, including Barlow
Twins [@zbontar2021barlow], VICReg [@bardes2021vicreg], and InfoNCE
[@oord2018cpc], have expanded these capabilities to accommodate more complex
patterns.

Similarity-driven multilinear reconstruction (SiMLR) captures this joint
variation in a linear, low-rank setting by projecting multiview data into a
shared subspace under subject-level similarity constraints
[@Avants2021NatCompSci]. While SiMLR isolates this shared representation, it
treats the remaining view-specific variation as an unstructured residual rather
than explicitly modeling a private component. This separation supports robust
cross-view harmonization and prediction by isolating stable population effects
from noise [@Stone2020BreachersNeuroimaging; @Stone2024USSOCOM]. While deep
learning approaches have explored cross-modal translation and disentanglement
using convolutional neural networks, VAEs, or diffusion models
[@havaei2016hemis; @Chartsias2019SDNet; @yuan2024remind], they often lack the
unique combination of exact likelihoods and one-shot invertible mappings. Recent
works have also explored normalizing flows for unsupervised MRI harmonization,
but utilize the flow purely as a test-time density estimator to iteratively
adapt an auxiliary translation network to an unknown target domain
[@beizaee2025harmonizingflows].

In contrast, LAMNr flows analogize the SiMLR framework into a deep,
likelihood-based architecture that topologically unfolds the potentially complex
manifold into a continuous vector space. Instead of an explicit linear
factorization in the observation domain, LAMNr flows map each view into a shared
latent space, ensuring exact log-likelihoods and bijective mappings. By
utilizing latent-alignment objectives (e.g., VICReg, InfoNCE) to identify shared
coordinates, the framework recovers the interpretability of a shared/private
decomposition within a nonlinear, invertible space. Crucially, by modeling the
joint latents with a Gaussian distribution, LAMNr flows enable closed-form
conditional reconstructions. This allows the shared subspace to function as a
geometrically-informed coordinate system. 

Additionally, the development of LAMNr flows provides a practical strategy in
ensuring topological integrity within neural density estimators. Historically,
models like Deep Diffeomorphic Normalizing Flows (DDNF) [@salman2018deep]
enforced smoothness by integrating time-varying velocity fields via Ordinary
Differential Equations (ODEs). While this continuous formulation guarantees a
diffeomorphic mapping, the computational cost of ODE integration is often
prohibitive for large-scale applications (e.g., medical imaging). To address
this, LAMNr flows use the discrete, efficient architectures of RealNVP
[@dinh2016realnvp] and Glow [@kingma2018glow]. By aligning disparate modalities
and views into a shared latent representation, the LAMNr flows model is steered
towards prioritizing robust, underlying anatomical structures over idiosyncratic
signal. Additionally, this latent-alignment employs specific numerical
safeguards, such as bounding the scale parameters within the affine coupling
layers, to mitigate gradient blow-ups during training. Furthermore, the
inclusion of training jitter serves as an additional regularizer (i.e.,
"dequantization" [@ho2019flowpp]). In the imaging context (i.e., Glow-based
models), by introducing stochastic intensity- and shape-based perturbations
during the learning phase, the model is discouraged from over-fitting to local
voxel intensities. Together, these constraints force convergence on more
generalized anatomical representations, stabilizing the Jacobian determinant and
ensuring that the discrete transitions of the Glow architecture maintain the
smooth, diffeomorphic properties required.  Analogous network architectural
features are also leveraged for IDP-based scenarios using RealNVP.  


## Computational Anatomy and Normalizing Flows

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
of diffeomorphic transformation groups on anatomical manifolds
[@GrenanderMiller1998CA;@Miller2002LDDMMOverview]. As one important example,
within this probabilistic and geometric framework, population structure is
typically represented by a population template, i.e., a reference space
formally established as the Fréchet mean that minimizes the sum of squared
geodesic distances across a cohort [@Avants:2010aa]. Traditionally, the
intrinsic curvature of image spaces causes a divergence between the Fréchet
mean, the Karcher mean, and the statistical mode [@Fletcher2009aa]. 

Normalizing flows offer an alternative, deep learning perspective by
topologically unfolding these nonlinear manifolds into a symmetric, centered
diagonal Gaussian base distribution (Figure \ref{fig:single_view_flow}). Within
this framework, one principled approach to template construction is the inverse
mapping of the latent origin, $\hat{x}_0 = f_{\theta}^{-1}(0)$, which leverages
the property that the Gaussian mean, mode, and median coincide precisely at the
origin (cf. Figure \ref{fig:single_view_flow}). While registration-based
templates (e.g., via Symmetric Normalization [@Avants:2010aa] or Large
Deformation Diffeomorphic Metric Mapping [@Miller2002LDDMMOverview]) typically
preserve high-frequency details through iterative normalization, spatial
averaging, and sharpening, the generative latent origin-based template exhibits
a visually smoother appearance. This smoothness is a direct consequence of
high-dimensional probabilistic modeling. As the exact mode of the latent
distribution, the origin averages out idiosyncratic, high-frequency anatomical
variations, such as individual cortical folding patterns, that do not strictly
persist across the cohort.

Furthermore, regarding the geometric interpretation of this template, in high
dimensions the concentration of measure phenomenon dictates that probability
mass concentrates within a thin spherical shell rather than at the origin
[@white2016sampling; @vershynin2018high; @blum2020foundations]. While classical
geometric morphometry often utilizes the Fréchet mean to compute a statistically
valid average on a Riemannian manifold [@Pennec2006], applying such spherical
mapping in a spatial normalizing flow to force the template onto the
high-probability "typical set" destroys the anatomical signal. Because LAMNr
flows preserve spatial dimensions, projecting the vector norm to this spherical
shell normalizes the spatial contrast energy, resulting in severe high-frequency
noise. Consequently, the latent origin $z=0$ is not a statistically typical
anatomical instance, but rather a barycentric geometric anchor representing the
central axis of symmetry for the learned bijection.

Beyond template construction, this continuous latent framework provides direct
analogues to the fundamental metric operations of traditional computational
anatomy. For example, in classic diffeomorphic frameworks, the transformation
between a source and target anatomy is governed by integrating a time-varying
velocity field over a continuous time domain $t \in [0, 1]$ . The length of this
optimal, continuous deformation path establishes the exact geodesic distance
between the two biological structures [@Miller2002LDDMMOverview;@Beg2005LDDMM].
In the LAMNr flows framework, this computationally intensive temporal
integration can be substituted with an algebraic interpolation within the latent
space. Traversing the latent manifold between two encoded images, $z_0$ and
$z_1$, using a scalar interpolation parameter $\alpha \in [0, 1]$ generates a
continuous trajectory of decoded images that closely approximates this
diffeomorphic flow. Consequently, the distances computed directly in the latent
space, when properly evaluated via distribution-preserving spherical metrics
rather than Euclidean norms, serve as highly efficient surrogates for the
complex, deformation-based geodesic distances of traditional computational
anatomy.  These conceptualizations, along with other illustrative results, are
discussed and provided below in the context of our proposed LAMNr flows
framework.  


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
   [@beizaee2025harmonizingflows;@Wen:2023aa], we provide a comprehensive, open-source, 2D and
   3D PyTorch implementation, based on the `normflows` library
   [@stimper2023normflows], which is integrated with the ANTsX ecosystem (via
   ANTsTorch) for robust data handling and auxiliary functionality.

Evaluations on multimodal MRI and multiview IDP datasets demonstrate that LAMNr
flows improve calibrated likelihoods and downstream prediction while providing a
single, exact framework for likelihood-calibrated multiview analysis.

