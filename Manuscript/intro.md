
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

We develop a general systems view—*Latent-Aligned Multimodal Normalizing
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
modalities—often with an implicit division between shared and private parts
[@Shi2019MMVAE; @Sutter2021MoPoE]. These approaches are directly relevant
conceptually, but they (i) are not *invertible* in pixel space, (ii) rely on
approximate likelihoods, and (iii) do not expose clean, per-level multiscale
latents that can be edited and then exactly decoded. LAM-Flow keeps the
shared/private spirit but instantiates it in a bijective model with exact
log-likelihoods and Glow-style multiscale access, which matters for per-level
conditioning and surgical latent manipulations that invert precisely.

### Latent editing and counterfactual manipulation

A parallel line of work explores *editing* latent codes to obtain “cleaner”
images or controlled semantic changes. StyleFlow, for instance, maps entangled
StyleGAN codes through *conditional continuous normalizing flows* to achieve
attribute-conditioned edits (pose, illumination, age), a concrete precedent for
flow-based controlled latent traversal—even if the underlying generator is a
GAN rather than a flow in pixel space [@Abdal2021StyleFlow]. Medical imaging has
also investigated counterfactual or pathology-aware latent edits in GAN spaces,
often for augmentation or interpretability; however, exact invertibility of the
full image model is typically absent. LAM-Flow differs in that edits occur
*inside* a fully invertible model whose latents are tied to exact likelihoods,
and edits can be per-scale and per-subspace (shared vs private), enabling SLIs
and uncertainty-aware manipulations via closed-form conditional Gaussians.

### Flow-based cross-modal relations

Flow-based conditional modeling has been used for cross-modality transfer.
DUAL-GLOW couples two flows (e.g., MR and PET) with an auxiliary relation
network to model conditional distributions in latent space [@sun2019dualglow].
This is an architectural cousin to conditional reasoning in flows, but it does
not perform per-level alignment across modalities nor the shared/private
decomposition operationalized here for registration via SLIs. More broadly,
modern flow families (e.g., Flow++, Neural Spline Flows, residual and
continuous-time flows) have improved flexibility and stability
[@ho2019flowpp; @durkan2019nsf; @behrmann2019resflow; @grathwohl2019ffjord], and
recent scaling results (TarFlow, STARFlow) reinforce flows as viable foundations
when invertibility, exact likelihoods, and latent access are central
[@zhai2024tarflow; @gu2025starflow].

### Cross-modal synthesis and missing-modality robustness (outside flows)

Historically, cross-modality synthesis progressed from atlas/patch pipelines to
supervised U-Nets and then to unpaired adversarial methods (CycleGAN variants)
with structure-consistency constraints to preserve anatomy
[@han2017dcnn; @florkow2020mrm; @yang2018structurecyclegan; @lei2019densecyclegan].
HeMIS is notable for operating with any subset of inputs at test time via a
learned “mean-of-modalities” fusion rather than explicit synthesis
[@havaei2016hemis]. Diffusion models add uncertainty-aware sampling for
imputation and reconstruction, though they typically forgo one-shot inversion
and per-level latent access
[@croitoru2023diffusion_vision_survey; @yuan2024remind; @webber2024bjrai].
LAM-Flow keeps the “reason across available views” philosophy but supplies a
*closed-form* conditional posterior for unobserved latents *at each scale*, a
capability specific to the multiscale flow setting with alignment.

### Registration via modality-invariant spaces and synthesis-aided alignment

Several works aim to simplify multimodal registration by (i) learning contrast-
invariant structural representations and registering *in that space*, or (ii)
synthesizing a target-like image and registering *there*. For (i), recent work
learns modality-agnostic structural image representations to reduce cross-modal
registration to a near-monomodal problem—resonant with the SLI idea, but
implemented as a forward encoder rather than an invertible multiscale flow
[@Mok2024ModalityAgnosticRep]. Other approaches enforce diffeomorphic,
modality-invariant objectives in learned feature spaces
[@Qiu2021ModalityInvariantReg]. For (ii), many pipelines synthesize CT from MR
(or vice versa) and then run conventional registration; others rely on
contrast-agnostic metrics (e.g., mutual information). LAM-Flow’s shared-latent
images sharpen the representation route: instead of a learned *descriptor*
space, we generate an *image-space* surrogate reconstructed *from only the
shared subspace* (replacing private latents by conditional means). This
preserves anatomical geometry while dampening modality-specific contrast or
confounders—exactly what one wants for robust cross-modal registration—yet
remains fully invertible with calibrated likelihoods.

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
