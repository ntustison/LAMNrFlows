
\clearpage

# Introduction

Medical imaging data and their representative latent spaces are essential for
insight into biological structure and function. Deep learning workflows have
become foundational for learning and leveraging such spaces. Yet many common
approaches are opaque to data likelihoods and lack invertibility, complicating
access between image space and latent space. Consideration of these issues
is crucial in multimodal medical imaging studies, where contrasts are often
missing and downstream analyses depend on calibrated comparisons and coherent
cross-modal reconstructions. Normalizing flows [@papamakarios2021nfreview]
provide potential modeling approaches to address these needs by coupling
expressive latents with exact likelihoods and single-pass inversion. Certain
normalizing flow variants also yield accessible multiscale latents that can be
aligned across modalities and precisely decoded back to image space for 
further research queries.

## Medical image imputation / cross-modal synthesis

Early approaches pre-dating deep learning framed cross-modal synthesis (e.g.,
MR$\rightarrow$CT) and attenuation-correction as either
segmentation-/atlas-based mapping or patch-based learning from paired MR/CT
exemplars. Typical pipelines registered a subject to one or more atlases,
transferred tissue labels or Hounsfield surrogates, and then refined with local
patch regressors or random forests to better handle bone/air ambiguity and
intensity/tissue mismatch
[@andreasen2015patchpct;@torrado2016fastpatchpct;@yang2017rfpatchpct;@wu2016localdiffeo].
These methods set important baselines and established evaluation practices in
radiotherapy planning and PET/MR, but accuracy depended on registration quality,
hand-tuned features, and limited modeling flexibility for non-linear cross-modal
relationships.

With the advent of deep learning, supervised CNNs (typically U-Net) became the
default for synthetic CT generation and related imputation tasks, showing
improved performance with paired MR/CT data [@han2017dcnn;@florkow2020mrm].
Unpaired image-translation emerged via adversarial learning (CycleGAN and
structural-consistency variants) for MR$\leftrightarrow$CT and other modality
pairs, improving realism while explicitly encouraging anatomy preservation
[@lei2019densecyclegan;@yang2018structurecyclegan]. In parallel, proposed
workflows that accept missing modalities at inference without explicit synthesis
(e.g., HeMIS's latent ``mean-of-modalities'' fusion) provided robust
alternatives [@havaei2016hemis]. Comprehensive reviews summarize these deep
learning-based methods and their clinical contexts [@wang2021medimgsynth].

Most recently, diffusion models have been adapted to medical imputation
settings, offering strong generative priors and uncertainty handling. For
example, ReMiND (Recovery of Missing Neuroimaging using Diffusion Models)
targets longitudinal MRI recovery of missing visits via conditional diffusion
[@yuan2024remind].  Domain reviews in reconstruction discuss how diffusion-based
priors can mitigate domain shift and quantify uncertainty which are also
relevant to translation/imputation [@webber2024bjrai].

## Normalizing flows 

Normalizing flows emerged as a practical class of invertible generative models
approximately a decade ago.  Although other classes of invertible (or
approximately invertible) architectures were developed in parallel
[@gomez2017revnet;@jacobsen2018irevnet], such networks were not designed for
density modeling with exact likelihoods.  An early pioneer, Non-linear
Independent Components Estimation (NICE) [@dinh2014nice], demonstrated that
features can be split into two parts with one half "nudging" the other with a
learned shift.  This keeps density computation simple while guaranteeing an
exact inverse. Variational flows broadened this idea by stacking small,
invertible "warps" that are easy to compute [@rezende2015variational]. RealNVP
added learned scaling in addition to shifting and arranged the model across
multiple resolutions, improving modeling while keeping computations efficient
[@dinh2016realnvp]. In parallel, Inverse Autoregressive Flow (IAF) and Masked
Autoregressive Flow (MAF) explored flows which explored the trade-off between
fast sampling and fast likelihood evaluation
[@kingma2016iaf;@papamakarios2017maf].

The original Glow architecture consolidated these ideas for large images with
data-dependent ActNorm, invertible $1\times1$ convolutions, and a clean
multiscale design, yielding strong likelihoods and single-pass inversion
[@kingma2018glow]. Subsequent work broadened this architectural family. Flow++
improved sample quality via variational dequantization and richer coupling
transforms [@ho2019flowpp], Neural Spline Flows replaced affine transforms with
monotonic splines for greater flexibility [@durkan2019nsf], Residual Flows
enforced Lipschitz constraints for stability in deep stacks
[@behrmann2019resflow], and FFJORD introduced continuous-time flows with
unbiased likelihood estimates via Hutchinson trace estimators
[@grathwohl2019ffjord]. Continuous-time variants (e.g., continuous normalizing
flows, flow-matching) are related but generally lack the same one-shot inverse
and straightforward multiscale architectures that are leveraged for analytics
and imputation.  Surveys synthesize these developments and map the trade-offs
across density estimation, sampling, and invertibility
[@kobyzev2020nfsurvey;@papamakarios2021nfreview].

More recent work proposes flow-based models that operate at the same resolution
and scale that popularized diffusion models
[@croitoru2023diffusion_vision_survey].  TarFlow (Transformer Autoregressive
Flow) shows that normalizing flows can achieve state-of-the-art image
likelihoods and diffusion-comparable sample quality using autoregressive
Transformers and a few key training protocols [@zhai2024tarflow]. STARFlow builds
on this with a scalable latent-space design and guidance mechanisms, reporting
competitive high-resolution synthesis (class-conditional and text-conditional)
that explicitly benchmarks against diffusion while retaining exact likelihood
training [@gu2025starflow].

## Contribution

Prior published synthesis/imputation frameworks are configured as one-to-one or
many-to-one mappings.  They typically generate a single target contrast even
when trained for multiple targets and rarely model the joint conditional across
all missing contrasts. We instead contextualize modalities within a single
multiflow, multiscale latent system. Using the Glow architecture with multi-scale
access, we fit per-level Gaussian statistics and, given any observed subset,
compute a closed-form joint posterior over the missing latents that captures
cross-modal covariance. For $M$ modalities, a single, exact inverse then yields 
$\mathcal{O} \rightarrow \mathcal{U}$ imputations, where $\mathcal{O}$ is the 
set of observed modalities and $\mathcal{U}$ is the set of missing modalities.
These imputations are jointly coherent across all requested outputs, while
preserving calibrated likelihoods for principled comparison and uncertainty
reporting.

We adopt a Glow-style discrete flow [@kingma2018glow] because our setting
prioritizes exact inversion, explicit log-likelihoods, and analyzable multiscale
latents for medical images, in contrast to alternatives that typically forgo one
or more of these properties in the interest of alternative objectives.
Concretely, we provide a robust open-source implementation (2-D/3-D) with
ActNorm, invertible $1\times1(\times1)$ convolutions, corrected reshape
orderings, stable log determinant bookkeeping, and a comprehensive command-line
interface. We also enable per-level latent alignment across modalities via
multiple possible alignment modeling objectives (Pearson, Barlow Twins, VICReg,
InfoNCE, HSIC) with an optional CCA-guided subspace. Learned relative weighting
of the individual terms within the multimodal imputation objective is used  to
account for aleatoric variability. The result is a flexible, robust, and
open-source framework for within-subject multimodal modeling that scales
predictably to 2-D/3-D data and emphasizes exactness, interpretability, and
reproducibility.