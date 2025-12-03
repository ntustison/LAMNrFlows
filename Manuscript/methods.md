
\clearpage


# Methods

LAMNr Flows are built in two stages. First, each view is assigned a separate
normalizing flow that is trained by maximum likelihood, with additional
penalties that align designated shared latent coordinates across views.
Alignment can be applied directly in the latent space or after an optional CCA
or HSIC-based screening step that selects coordinates with empirical cross-view
dependence. Second, after training the flows, we fit a conditional Gaussian
model over the concatenated latents to enable closed-form posteriors, exact
decoding, and latent-space manipulations for imputation, harmonization,
counterfactual edits, and shared-latent reconstructions.

## Normalizing flows and the multiview training objective

For a single view \(v\), a normalizing flow with parameters \(\theta^{(v)}\) is
an invertible mapping
\[
f^{(v)}_{\theta} : \mathcal{X}^{(v)} \to \mathcal{Z}^{(v)}, \quad
z_n^{(v)} = f^{(v)}_{\theta}(x_n^{(v)}),
\]
that sends observed data \(x_n^{(v)}\) to a latent space
\(\mathcal{Z}^{(v)}\) with a simple base density, typically
\(p_Z(z) = \mathcal{N}(0, I)\). The induced density on \(\mathcal{X}^{(v)}\)
follows from the change-of-variables formula:
\[
\log p_{\theta}(x_n^{(v)}) =
\log p_Z\bigl(z_n^{(v)}\bigr)
+ \log \bigl|\det \partial f^{(v)}_{\theta} / \partial x_n^{(v)}\bigr|,
\]
which we can evaluate exactly for the Glow [@kingma2018glow] and RealNVP-style
[@dinh2016realnvp] architectures used here. Maximum-likelihood estimation
chooses \(\theta^{(v)}\) to maximize the sum of log-likelihoods over subjects,
or equivalently to minimize the average negative log-likelihood  [@kobyzev2020nfsurvey].

In our proposed multiview setting, each subject \(n\) has measurements
\(\{x_n^{(v)}\}_{v=1}^V\) across views \(v = 1,\dots,V\). We instantiate one
flow per view and train them jointly on subject-matched minibatches. If we
consider only the flow likelihoods, the pure maximum-likelihood objective is
\[
\mathcal{L}_{\text{like}}(\theta)
= \frac{1}{N} \sum_{n=1}^N \sum_{v=1}^V
\bigl[- \log p_{\theta}(x_n^{(v)})\bigr],
\]
where \(\theta = \{\theta^{(v)}\}_{v=1}^V\) collects all view-specific
parameters. This term ensures that each per-view flow is an exact-likelihood
model of its corresponding data distribution.

To expose multiview structure, we work in latent space. For each view and
subject we write
\[
z_n^{(v)} = \bigl[z^{(v)}_{S,n}, \; z^{(v)}_{P,n}\bigr],
\]
splitting the latent into a block \(z^{(v)}_{S,n}\) that is intended to carry
shared information across views and a block \(z^{(v)}_{P,n}\) that is
view-specific. The indices that define this split can be chosen a priori or via
the CCA/HSIC-based screening procedure described below.

We then attach a small projector network
\(\phi^{(v)}_{\psi} : \mathcal{Z}^{(v)} \to \mathbb{R}^D\) to each view, with a
matched output dimension \(D\) across views, and apply a multiview alignment
loss to the projected shared coordinates. For a subject-matched minibatch of
size \(N\), the full training objective becomes
\[
\mathcal{L}(\theta, \psi)
= \frac{1}{N} \sum_{n=1}^N \sum_{v=1}^V
\bigl[- \log p_{\theta}(x_n^{(v)})\bigr]
+ \lambda \, \mathcal{L}_{\text{align}}
\Bigl(\bigl\{\phi^{(v)}_{\psi}(z^{(v)}_{S,n})\bigr\}_{v,n}\Bigr),
\]
where \(\mathcal{L}_{\text{align}}\) is chosen from Barlow Twins, VICReg,
InfoNCE, Pearson correlation, or HSIC, and \(\lambda\) controls the strength of
alignment.[^Kendall-Gal] 

[^Kendall-Gal]: 
In some experiments we replaced the fixed weighting, \(\lambda\), by learned task weights
following the homoscedastic aleatoric uncertainty scheme of Kendall and Gal
[@kendall2018mtl]. In this formulation, each loss term \(L_i\) is scaled
as \(e^{-s_i} L_i + s_i\), where \(s_i = \log \sigma_i^2\) represents
task-dependent aleatoric (data) uncertainty, as opposed to epistemic (model)
uncertainty [@Hullermeier2021UncertaintyReview]. While this can automatically
balance losses with different units, in our multiview setting it tended to
inflate the alignment variance and drive the effective alignment weight
\(e^{-s_{\text{align}}}\) toward zero, effectively suppressing latent
alignment. For clarity and robustness we therefore report results using a fixed
\(\lambda\) schedule in the main experiments.


## Per-view flow backbones

### Image views (Glow-style multiscale flows)

For image views we adopt Glow-style discrete normalizing flows with \(L\) levels
and \(K\) coupling steps per level. Each step comprises: (i) ActNorm layers with
data-dependent initialization, (ii) invertible \(1 \times 1 (\times 1)\)
convolutions parameterized with LU factorization for efficient log-determinant
computation, and (iii) affine coupling layers whose scale and shift fields are
predicted by shallow convolutional subnetworks with a configurable number of
hidden channels. Squeeze and split operations provide a multiscale
representation in which shallower levels capture coarse structure while deeper
levels model fine texture. Our implementation follows the standard Glow
construction, instantiated via a model factory in ANTsTorch
(`create_glow_normalizing_flow_model_2d`/`3d`), with configurable image size
(both 2-D and 3-D), number of levels \(L\), steps per level \(K\), and hidden
channels. 

#### Tabular and IDP views (RealNVP/MAF whiteners)

For imaging-derived phenotypes (IDPs) and other tabular blocks, we use
single-scale flows based on RealNVP and masked autoregressive flows (MAF) with
affine couplings and masked multilayer perceptrons. Continuous variables are
preprocessed per view using dataset-owned normalization and imputation:
columns are coerced to numeric, NaNs are imputed (typically by the column mean),
and features are standardized to zero mean or rescaled to \([0,1]\) depending on
a user-selectable normalization mode. Very low-variance columns are stabilized
by floor-clamping the standard deviation. Positively skewed, non-negative
variables can optionally be log or log1p transformed before normalization to
reduce skewness.

We use two base distributions: a diagonal Gaussian and a Gaussian–PCA base
(`GaussianPCA`) that performs an additional linear whitening of the flow
latents. In the latter case, the flow acts as a learnable multiview “whitener”
that maps each tabular view to a standardized latent \(\varepsilon\) with
approximately independent components; both the raw flow latents \(z^{(v)}\) and
the whitened coordinates \(\varepsilon^{(v)}\) can be exported for downstream
Gaussian modeling and diagnostics. Each tabular view is typically modeled with a
stack of 8–12 coupling layers with fully connected subnetworks of width 256–512,
and the resulting latent vector \(z^{(v)}\) is then passed to the same
projector, screening, and alignment machinery as the image latents.

### Projector networks and latent alignment objectives

Let \(\phi^{(v)}_\psi : \mathcal{Z}^{(v)} \to \mathbb{R}^D\) be a small
projector head that maps the flattened latents of view \(v\) to a
\(D\)-dimensional feature vector with the same dimension across views. In
practice, the projector is a two-layer MLP for tabular flows or a linear head on
the concatenated multiscale image latents; the dimensionality \(D\) is chosen so
that we can apply multiview alignment losses efficiently. Given a
subject-matched minibatch \(\{\phi^{(v)}_\psi(z^{(v)}_{S,n})\}\), we apply one
of several alignment losses on the coordinates marked as shared:
Pearson correlation, Barlow Twins, VICReg, InfoNCE, or HSIC
[@zbontar2021barlow; @bardes2021vicreg; @oord2018cpc; @gretton2005hsic]. These
objectives trade off simplicity, batch-size requirements, and the type of
cross-view dependence they emphasize (linear vs. non-linear, second-order vs.
higher-order).

We summarize the main options in Table~\ref{tab:alignment}, which we use
interchangeably depending on the experiment. Pearson and VICReg are effective
when batch sizes are modest and we want stable, low-cost alignment. Barlow Twins
adds explicit redundancy reduction via cross-correlation decorrelation. InfoNCE
provides a strong discriminative signal when large batches and in-batch
negatives are available, while HSIC captures non-linear dependence at the cost
of \(O(B^2)\) kernel operations. In all cases, the alignment term acts only on
shared coordinates, leaving private coordinates free to capture
view-specific variation.

\input{latent_alignment_table.tex}

In practice, we choose the alignment objective based on computational budget and
the expected structure of the views. For small batches or limited compute,
Pearson or VICReg are attractive. When we want stronger redundancy reduction
without negatives, Barlow Twins is a good default. InfoNCE is most useful for
large-batch, discriminative alignment (e.g., retrieval-like settings), and HSIC
is reserved for settings where non-linear cross-modality relations are expected
to be dominant.

### Shared-subspace screening with CCA and HSIC

If we apply alignment losses to all latent coordinates, the model is implicitly
pushed toward making every feature shared across views. This can over-constrain
the flows, force view-specific structure into the shared space, and blur
contrast- or modality-specific information. It is also wasteful from a
statistical and computational perspective as many latent dimensions are weakly
related across views or primarily encode private variation, so aligning them
adds noise rather than useful signal. In analogy to SiMLR, we therefore seek a
lower-dimensional shared subspace that concentrates cross-view dependence while
leaving a complementary private subspace unconstrained.

To automatically identify shared coordinates, we perform a short screening pass
after an MLE warm-up phase. For CCA-based screening, we construct whitened
feature matrices for two views and perform an SVD of the cross-covariance
\(X_a^\top X_b\), retaining the top \(r\) canonical directions per view.
Averaging across all view pairs yields per-view projectors
\(P^{(v)} \in \mathbb{R}^{D \times r}\) defining the shared subspace. For
HSIC-based screening, we first prefilter coordinates using Pearson correlation,
then rank remaining dimensions by an unbiased HSIC estimate with RBF kernels
averaged over other views, and select the top \(r\) per view. Alignment losses
are applied only to these projected or masked coordinates, so that dependence
is enforced where cross-view signal is strongest and private dimensions remain
free to capture view-specific variation. Screening can be performed once after
warm-up or periodically refreshed during training; in our experiments we use a
single screening stage for simplicity.

In the tabular setting we additionally allow a coarse pre-training dependence
screen on the raw views. Using either normalized HSIC or maximum canonical
correlation on standardized input features, we estimate the average pairwise
dependence across views on a subsample of subjects. If this average falls below
a user-defined threshold, we disable the alignment penalty altogether and train
independent per-view flows. This avoids forcing alignment when views are only
weakly related or effectively independent, while still enabling shared-subspace
alignment when substantial cross-view structure is present.

## Conditional Gaussian model over latents

After training the per-view flows and projector alignment, we freeze the flow
parameters and collect latents for all subjects. For image views, we retain a
multiscale representation \(z^{(v)}_\ell\) at each level \(\ell \in
\{1,\dots,L\}\); for tabular views we have a single level. Concatenating across
views and levels yields a joint latent vector
\[
z = \bigl[z^{(1)}_1, \dots, z^{(1)}_L, \dots, z^{(V)}_1, \dots, z^{(V)}_L\bigr].
\]

We model this joint latent as Gaussian,
\[
z \sim \mathcal{N}(\mu, \Sigma),
\]
with mean \(\mu\) and covariance \(\Sigma\) estimated either per level or in a
merged representation, using either full covariance, shrinkage estimators, or a
low-rank-plus-diagonal parameterization depending on dimensionality.

Given an observed subset of coordinates \(O\) and an unobserved subset \(U\),
the posterior \(p(z_U \mid z_O)\) is Gaussian with closed-form mean and
covariance:
\[
\mu_{U\mid O}
= \mu_U + \Sigma_{UO}\Sigma_{OO}^{-1}(z_O - \mu_O),
\]
\[
\Sigma_{U\mid O}
= \Sigma_{UU} - \Sigma_{UO}\Sigma_{OO}^{-1}\Sigma_{OU}.
\]
We use shrinkage or low-rank regularization to ensure positive definiteness and
numerical stability during inversion of \(\Sigma_{OO}\).

Samples from this conditional Gaussian propagate uncertainty, while the
posterior mean provides a calibrated point estimate. Applying the inverse flows
to these posterior latents yields imputations, harmonized representations, and
latent edits in the original data space, with exact likelihoods available for
all configurations.

## Shared-latent reconstructions, templates, and operational edits

For image views, we define shared-latent reconstructions by holding the shared
coordinates fixed and replacing private coordinates by draws from the
conditional posterior mean (or samples) of \(z_P^{(v)}\) given all available
views. This produces images that preserve subject-specific anatomy while
suppressing view-specific contrast and noise. These shared-latent images can be
used either directly in downstream tasks or as contrast-robust surrogates for
estimating mappings that are later applied back to the original images.

For IDP and other tabular blocks, the same conditional layer provides a unified
mechanism for:

- imputing missing views under arbitrary missingness patterns,
- harmonizing across sites or acquisition protocols by conditioning on shared
  covariates, and
- answering model-based counterfactual queries (e.g., editing a subset of
  variables while conditioning on the remainder).

We also define latent-space templates as averages in latent space decoded back
to image space, or as Monte Carlo expectations under the learned latent
distribution. In the small-variance or locally linear regime, these
constructions coincide up to second-order terms, linking our latent templates to
Fréchet means in the induced metric on images. These tools are implemented via
the `recon` and `recon-template` subcommands of our `lamnr_flow_tool.py`
utility, which load trained checkpoints, apply Gaussian editing in latent space,
and render the resulting templates or edited reconstructions as images for
inspection.

## Implementation and training details

All flows and auxiliary functionality are implemented in PyTorch via ANTsTorch
and a modified fork of a publicly available version of a normalizing flows
library [@stimper2023normflows]. Training and validation splits are defined at
the subject level, and each minibatch contains aligned multiview slices from
matched subjects. Image data augmentation is performed on-the-fly using the
ANTsTorch-based `ImageDataset` with affine and diffeomorphic deformations, small
intensity perturbations (histogram warping and bias field simulation), and
additive Gaussian noise treated as dequantization rather than biological
variability. We control the overall augmentation strength by a scalar schedule
\(\alpha(t) \in [0,1]\) as a function of normalized training time \(t\), and
support linear, cosine, and exponential decay: for example, a linear schedule
reduces augmentation proportionally to \(t\), a cosine schedule keeps stronger
perturbations early and then decays smoothly, and an exponential schedule
reduces aggressive warps and noise most rapidly at the beginning of training.
This allows us to start with heavier augmentations to regularize the flows and
discourage overfitting to discrete templates, then gradually emphasize fidelity
to the true data distribution as training progresses. For validation, we use the
same spatial transforms but disable additional noise and histogram warping. This
design preserves anatomical variability while preventing overfitting to
discrete, noise-free templates that would otherwise cause flows to collapse onto
spiky background modes.  For tabular flows we apply a small additive “jitter”
noise to the features, treated as dequantization rather than biological
variation. The amplitude is controlled by a scalar schedule \(\alpha(t)\)
(linear, cosine, or exponential in training time), analogous to the image-domain
augmentation schedule. This helps regularize the tabular flows and prevents them
from overfitting to discrete patterns or exact repeated rows in large cohorts.

Glow models are initialized with data-dependent ActNorm, and we perform a
one-time warm-up pass with real images before starting training to stabilize
statistics. We train with Adamax, mixed precision, and optional exponential
moving averages of model parameters. Learning rates follow a warm-up plus decay
schedule with a plateau-based reducer. We monitor exact negative log-likelihood
in bits-per-dimension, view-wise breakdowns, and alignment loss values, and
periodically log reconstructions and samples for visual inspection.
The same training loop supports multiple views by instantiating one flow per
view, computing per-view log-likelihoods and latents for each minibatch,
flattening the multiscale latents, and applying the projector plus alignment and
screening logic described above. For IDP/tabular experiments, we replace the
image Glow backbones with RealNVP/MAF stacks while keeping the multiview
alignment and conditional Gaussian stages unchanged, which allows LAMNr Flows to
operate uniformly across image contrasts and multiview IDP blocks within the
same methodological framework.
