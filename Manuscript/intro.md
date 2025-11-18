
\clearpage

# Introduction

Medical imaging data and their representative latent spaces are essential for
insight into biological structure and function. Deep learning workflows have
become foundational for learning and leveraging such spaces. Many common
approaches, however, are opaque to data likelihoods.  They also lack
invertibility, complicating access between image space and latent space. These
limitations become acute in multimodal studies, where contrasts are potentially
missing or heterogeneous and downstream analyses depend on calibrated
comparisons and coherent cross-modal reconstructions.

## Normalizing flows as a foundation

Normalizing flows furnish exact-likelihood, bijective mappings between images
and latent variables, together with single-pass inversion and per-level access
to multiscale latents [@papamakarios2021nfreview; @kobyzev2020nfsurvey].
Among these, Glow-style architectures expose latent variables at each scale that
can be probed, constrained, and edited before precise decoding to image space
[@kingma2018glow]. This makes flows a natural substrate for multimodal
representation learning in which image- and latent-space reasoning remain
tightly coupled. Recent large-scale results further demonstrate that flows can
scale in likelihood and sample quality at resolutions relevant to modern
imaging, strengthening their role as first-class generative models rather than
mere surrogates [@croitoru2023diffusion_vision_survey; @zhai2024tarflow; @gu2025starflow].

## The LAM-Flow framework: learning shared multiscale latent spaces

We develop a general systems view—*Latent-Aligned Multiview Normalizing
Flows (LAM-Flow)*—that leverages the multiscale structure of flows to learn
*shared* and *private* latent components across modalities for matched
subjects. A subject’s multiscale latents are treated as structured random
variables: some channels encode anatomy/geometry that are expected to be common
across modalities, while others capture modality-specific factors (contrast,
site, artifacts, or pathology-driven appearance). Using matched subjects, we
impose latent-alignment constraints (e.g., Barlow Twins, VICReg, InfoNCE, HSIC)
on designated shared latent channels at each scale and optionally discover
shared subspaces via CCA/HSIC screening
[@zbontar2021barlow; @bardes2021vicreg; @oord2018cpc; @gretton2005hsic]. Because
flows are exactly bijective, alignment acts where it matters most—directly on
the latents that will be decoded back to images—rather than on proxy embeddings
that require separate decoders or heuristics.

Beyond maximum-likelihood training, we introduce a *conditional Gaussian
inference* layer that estimates per-level moments and yields closed-form
posteriors over arbitrary latent subsets
[@Murphy2012ML; @bishop2006prml]. This layer enables subject-specific latent
manipulations that preserve anatomy (by holding shared latents fixed) while
modulating modality-specific factors (by editing private latents). A concrete
instantiation is the construction of *shared-latent images (SLIs)*:
reconstructions in which private latents are replaced by their conditional means
given the shared latents. SLIs function as *contrast-robust surrogates* that
can simplify downstream tasks—most notably cross-modal registration—after which
the estimated transforms are faithfully applied to the original data to
preserve native intensities. The same machinery supports cross-view imputation
as a special case, scanner/site harmonization, uncertainty analysis via exact
log-likelihoods, and controlled “what-if” interventional latent edits
(counterfactuals in the model sense), all within a single, exact, and
interpretable framework that scales to 3-D volumes.

## Related work and positioning

### Shared/private multimodal representations (primarily VAE- and GAN-based)

A substantial multimodal literature explicitly splits representations into
*shared* versus *modality-specific* factors so that transferable information can
be exploited while preserving view-unique content. In medical imaging,
Chartsias *et al.* develop a modality-invariant latent fused from multiple MR
sequences and demonstrate disentanglement between anatomy and modality factors,
with downstream benefits to segmentation and cross-modality generation
[@Chartsias2018MILR; @Chartsias2019SDNet]. More general multimodal VAE
families, such as MMVAE (mixture-of-experts) and MoPoE-VAE (mixture-of-products
of experts), formalize coherent joint and conditional generation across
modalities, often with an implicit division between shared and private parts
[@Shi2019MMVAE; @Sutter2021MoPoE]. These approaches are directly relevant
conceptually, but they (i) are not *invertible* in pixel space, (ii) rely on
approximate likelihoods, and (iii) do not expose clean, per-level multiscale
latents that can be edited and then exactly decoded. LAM-Flow keeps the
shared/private spirit but instantiates it in a bijective model with exact
log-likelihoods and Glow-style multiscale access, which matters for per-level
conditioning and surgical latent manipulations that invert precisely. In
practice, we fit our Gaussian latent layer on *normals-only* cohorts, so that
“shared” structure is shaped jointly by all modalities and defines a normative
multiscale latent space that later edits can project back onto.

### Latent editing and counterfactual manipulation

A parallel line of work explores *editing* latent codes to obtain “cleaner”
images or controlled semantic changes. StyleFlow, for instance, maps entangled
StyleGAN codes through conditional continuous normalizing flows to achieve
attribute-conditioned edits (pose, illumination, age), a concrete precedent for
flow-based controlled latent traversal, even if the underlying generator is a
GAN rather than a flow in pixel space [@Abdal2021StyleFlow]. Medical imaging has
also investigated counterfactual or pathology-aware latent edits in GAN spaces,
often for augmentation or interpretability, though exact invertibility of the
full image model is typically absent.

LAM-Flow differs in that edits occur *inside* a fully invertible model whose
latents are tied to exact likelihoods, and edits can be per-scale and
per-subspace (shared vs private), enabling SLIs and uncertainty-aware
manipulations via closed-form conditional Gaussians. In the lesion setting, we
fit the Gaussian layer only on normal subjects, treat the resulting multiscale
Gaussian as a *normative prior*, and then encode lesion-bearing inputs through
the same flow. Lesion latents appear as outliers in this normative space;
PCA-based shrinkage of their coordinates back toward typical radii yields
“normative reconstructions” that suppress abnormal structure while preserving
subject-specific anatomy. Even when only a single modality (for example T1) is
edited and visualized, its latent coordinates were shaped during training by
multimodal alignment, so the notion of “normal” that guides edits implicitly
reflects all available contrasts rather than that modality in isolation.

### Flow-based cross-modal relations

Flow-based conditional modeling has been used for cross-modality transfer.
DUAL-GLOW couples two flows (for example MR and PET) with an auxiliary relation
network to model conditional distributions in latent space [@sun2019dualglow].
This is an architectural cousin to conditional reasoning in flows, but it does
not perform per-level alignment across modalities nor the shared/private
decomposition operationalized here for registration via SLIs. More broadly,
modern flow families (for example Flow++, Neural Spline Flows, residual and
continuous-time flows) have improved flexibility and stability
[@ho2019flowpp; @durkan2019nsf; @behrmann2019resflow; @grathwohl2019ffjord], and
recent scaling results (TarFlow, STARFlow) reinforce flows as viable foundations
when invertibility, exact likelihoods, and latent access are central
[@zhai2024tarflow; @gu2025starflow].

LAM-Flow sits between pure conditional flows and separate per-modality models:
we learn a *joint* multiscale flow over all modalities, align its latents
across views, and then fit a normals-only Gaussian layer over the resulting
aligned latents. Cross-modal imputation and lesion “inpainting” are both
handled as conditional Gaussian updates at each scale. Even if only one modality
is ultimately decoded, its latent block has been regularized by the presence of
the other modalities during training, so conditional edits respect a
multimodal, rather than unimodal, notion of anatomical plausibility.

### Cross-modal synthesis and missing-modality robustness (outside flows)

Historically, cross-modality synthesis progressed from atlas and patch-based
pipelines to supervised U-Nets and then to unpaired adversarial methods
(CycleGAN variants) with structure-consistency constraints to preserve anatomy
[@han2017dcnn; @florkow2020mrm; @yang2018structurecyclegan; @lei2019densecyclegan].
HeMIS is notable for operating with any subset of inputs at test time via a
learned “mean-of-modalities” fusion rather than explicit synthesis
[@havaei2016hemis]. Diffusion models add uncertainty-aware sampling for
imputation and reconstruction, though they typically forgo one-shot inversion
and per-level latent access
[@croitoru2023diffusion_vision_survey; @yuan2024remind; @webber2024bjrai].

LAM-Flow keeps the “reason across available views” philosophy but supplies a
*closed-form* conditional posterior for unobserved latents at each scale, a
capability specific to multiscale flows with alignment. Conditioning on observed
views while drawing or shrinking unobserved latent blocks under a normals-only
Gaussian supports both missing-modality imputation and pathology-aware edits in
the same framework. In lesion applications, this lets us treat abnormal tissue
as a latent-space deviation from the learned normal manifold and move it back
toward typical configurations, yielding lesion-suppressed surrogates without
training a separate inpainting network.

### Latent-space templates and relation to Fréchet means in computational anatomy

Classical computational anatomy defines a population “template” as a Fréchet mean in a nonlinear shape–intensity space. Given a set of images \(\{x_i\}\) and a diffeomorphism group \(\mathcal{G}\) acting on them, template estimation proceeds by finding a reference \(T\) and subject-specific deformations \(\{\phi_i\in\mathcal{G}\}\) that minimize an energy of the form  
\[
T^\star \approx \arg\min_T \sum_i d\big(x_i, T\circ\phi_i\big)^2,
\]
where \(d(\cdot,\cdot)\) is an image similarity metric (e.g., cross-correlation) and \(\mathcal{G}\) is endowed with a Riemannian structure. Algorithms such as SyGN/ANTs iteratively update both the current template and diffeomorphic transforms, yielding an estimator of the Fréchet mean in a quotient space of images modulo deformations. This template lives on the orbit of the data under \(\mathcal{G}\) and is interpreted as a mean shape with mean intensity in the chosen atlas space.

While this construction is geometrically natural, it depends on the choice of metric and group action, and the Fréchet mean in quotient spaces can exhibit bias and inconsistency when variability is high or the template is strongly folded by the group action. These issues have motivated alternative formulations, including Bayesian template estimation and explicit modeling of shape and intensity variability around the template.

In our framework, we propose an alternative but complementary notion of a template based on a generative latent representation. For each modality, we train a multiscale normalizing flow that provides a bijection between images and latents,  
\[
f^{(m)}: x^{(m)} \leftrightarrow z^{(m)},
\]
and we fit a Gaussian model to the multilevel latents across subjects,
\[
z = [z^{(1)};\dots;z^{(M)}] \sim \mathcal{N}(\mu,\Sigma),
\]
with parameters estimated from held-out training latents. This construction endows the cohort with an explicit Euclidean latent coordinate system in which the distribution is approximately Gaussian by design.

A **latent-space template** for modality \(m\) is then obtained by taking the Gaussian mean for that modality and decoding it through the flow,
\[
T_{\text{lat}}^{(m)} \;=\; \big(f^{(m)}\big)^{-1}\!\big(\mu^{(m)}\big).
\]
Here \(\mu^{(m)}\) is the concatenation of the mean latent blocks for modality \(m\) across all scales. This template can be viewed as a Fréchet mean with respect to the *Euclidean* metric in latent space, transported back to image space by the learned generative map. It differs from the classical Fréchet mean in image/shape space in two important ways:

1. **Geometry**: the averaging is performed in a linear latent space where distances reflect the inductive biases of the flow (e.g., multiscale structure, alignment constraints) rather than in the nonlinear quotient space of images modulo deformations.
2. **Population alignment**: if the training data are already mapped into a common atlas space (e.g., via ANTs), the template \(T_{\text{lat}}^{(m)}\) can be interpreted as a “mean atlas image” whose geometry is implicitly regularized by the flow’s likelihood and inductive structure, rather than by an explicit group action.

Because the flow is nonlinear, \(T_{\text{lat}}^{(m)} = f^{(m)-1}(\mu^{(m)})\) is **not** equal in general to the pixel-wise mean image \(\mathbb{E}[x^{(m)}]\). To bridge this gap, we also consider a **Monte Carlo template** defined by sampling latents from the fitted Gaussian and averaging their reconstructions:
\[
\tilde{T}_{\text{MC}}^{(m)} \;=\; \frac{1}{K}\sum_{k=1}^{K} \big(f^{(m)}\big)^{-1}\!\big(z_k^{(m)}\big), \quad z_k \sim \mathcal{N}(\mu,\Sigma).
\]
This estimator approximates \(\mathbb{E}[x^{(m)}]\) under the generative model and is invariant to the particular choice of latent coordinate system as long as the Gaussian fit is fixed. Comparing \(T_{\text{lat}}^{(m)}\) and \(\tilde{T}_{\text{MC}}^{(m)}\) provides a direct measure of how strongly nonlinear the decoder is in regions of high probability mass: if the model is close to linear around \(\mu\), the two templates are nearly indistinguishable; large discrepancies highlight directions where curvature in the decoder or heavy-tailed latent structure matter.

This latent view also naturally recovers **template modes of variation**. The eigen-decomposition of the Gaussian covariance yields principal directions \(\{q_k\}\) with eigenvalues \(\{\lambda_k\}\). For a chosen modality \(m\), we can construct deformations of the latent template
\[
z^{(m)}(\alpha,k) = \mu^{(m)} + \alpha \sqrt{\lambda_k}\, q_k^{(m)}, \quad \alpha \in \mathbb{R},
\]
and visualize \(\big(f^{(m)}\big)^{-1}\big(z^{(m)}(\alpha,k)\big)\) for \(\alpha=\pm 1,\pm 2\). These correspond to “±standard deviation” perturbations of the template along dominant directions of population variability, analogous to PCA-based modes in classical shape analysis or statistical atlases, but now operating in a generative latent coordinate system coupled across modalities via the joint Gaussian.

Placed in the broader context of template building, our approach does not replace diffeomorphic Fréchet means; rather, it offers a complementary, model-based notion of a template:

- When the input images are first mapped into a common atlas space using a traditional SyGN-style pipeline, the latent-space and Monte Carlo templates can be viewed as *statistical refinements* of that atlas, summarizing intensity and multiscale structure learned by the flow.
- Because the Gaussian is joint across modalities, the resulting templates and modes of variation for a single modality implicitly encode cross-modal covariances, which is difficult to achieve with purely image-space Fréchet means.
- The same Gaussian layer used for cross-view imputation and conditional editing thus also provides a principled mechanism for template construction and visualization of population variability.

In this sense, LAM-Flow leverages established ideas from computational anatomy—population templates as Fréchet means in atlas space—while re-casting them in a learned latent geometry where Euclidean operations (means, modes, Monte Carlo expectations) correspond to nonlinear, anatomically coherent templates in image space.


### Registration via modality-invariant spaces and synthesis-aided alignment

Several works aim to simplify multimodal registration by learning contrast-
invariant structural representations and registering in that space, or by
synthesizing a target-like image and registering there. For the first category,
recent work learns modality-agnostic structural image representations that
reduce cross-modal registration to a near-monomodal problem, which is resonant
with the SLI idea but implemented as a forward encoder rather than an
invertible multiscale flow [@Mok2024ModalityAgnosticRep]. Other approaches
enforce diffeomorphic, modality-invariant objectives in learned feature spaces
[@Qiu2021ModalityInvariantReg]. For the second category, many pipelines
synthesize CT from MR (or the reverse) and then run conventional registration;
others rely on contrast-agnostic metrics such as mutual information.

LAM-Flow’s shared-latent images sharpen the representation route. Instead of a
learned descriptor space, we generate an image-space surrogate reconstructed
from only the shared subspace, replacing private latents by conditional means
under a normals-only Gaussian. This preserves anatomical geometry while
dampening modality-specific contrast, site effects, or lesion-driven appearance
that might destabilize similarity metrics. Lesion-edited SLIs can be used as
registration targets for structurally abnormal subjects, after which estimated
transforms are applied back to the original data to preserve native
intensities. The result is a single, invertible system that links registration,
harmonization, missing-modality imputation, and lesion suppression through the
same aligned multiscale latent space.

## Flow modeling background in brief

The broader evolution of invertible models situates our foundation. Early
invertible networks (RevNets, i-RevNets) demonstrated reversibility but did not
target exact density modeling [@Gomez2017RevNet; @Jacobsen2018iRevNet]. NICE
introduced additive coupling with a tractable Jacobian [@dinh2014nice],
variational flows stacked simple invertible transformations
[@rezende2015variational], and RealNVP added affine coupling in a multiscale
architecture to improve expressivity while retaining computational efficiency
[@dinh2016realnvp]. IAF and MAF explored trade-offs between fast sampling and
fast likelihood evaluation [@kingma2016iaf; @papamakarios2017maf]. Glow
consolidated a practical design for large images using data-dependent ActNorm,
invertible $1{\times}1$ convolutions, and a clean multiscale layout
[@kingma2018glow]. Subsequent work augmented flexibility and stability
(Flow++ [@ho2019flowpp], Neural Spline Flows [@durkan2019nsf], Residual Flows
[@behrmann2019resflow], FFJORD [@grathwohl2019ffjord]); surveys provide broader
context [@papamakarios2021nfreview; @kobyzev2020nfsurvey]. In this landscape,
flows remain distinctive in offering exact likelihoods, a one-shot inverse, and
analyzable multiscale latents—properties we exploit not only for generation but
for multimodal reasoning and editing on shared multiscale latent spaces.

## Practical scope of this work

We present an application-agnostic system in which latent alignment is
first-class: a multiscale flow learns shared and private structure across
modalities for matched subjects; per-level moments are estimated to enable
closed-form, subject-specific latent inference; and edits in latent space are
decoded exactly back to images. Registration, harmonization, and imputation are
treated as uses of the same mechanism rather than separate models or objectives.
We provide an open-source implementation for 2-D and 3-D medical volumes with
Glow-style components (ActNorm, invertible $1{\times}1(\times1)$ convolutions),
corrected reshape orderings, and stable log-determinant bookkeeping, together
with evaluation protocols that emphasize exactness, interpretability, and
reproducibility. In doing so, the proposed *Latent-Aligned Multimodal
Normalizing Flows* framework positions normalizing flows not merely as
generators, but as systems for learning, exposing, and manipulating shared
multiscale latent spaces across modalities—thereby simplifying downstream
multimodal tasks while preserving a calibrated, invertible link to the
underlying data.
