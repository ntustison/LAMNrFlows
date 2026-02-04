
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


## View-specific flow architectures

\begin{figure}
\centering
\begin{tabular}{cc}
\raisebox{1.25cm}{\includegraphics[width=0.4\textwidth]{Figures/realnvp.pdf}} &
\includegraphics[width=0.6\textwidth]{Figures/Glow.pdf} \\
(a) & (b)
\end{tabular}
\caption{
Overview of the LAMNr flows architectures.  
(a) Single-scale RealNVP architecture for tabular data. An
input vector $x \in \mathbb{R}^{B \times D}$ (e.g., imaging-derived phenotypes)
is processed through $K$ coupling steps to produce a latent representation $z_K$
of the same dimensionality. In addition to a diagonal Gaussian distribution, 
a $\texttt{GaussianPCA}$ base distribution is also supported where 
$z \sim \mathcal{N}(\mu, WW^\top + \sigma^2 I_D)$, which acts as a
learnable, geometrically-informed coordinate system. This unified approach
ensures exact invertibility and facilitates principled latent alignment across
heterogeneous views.
(b) Generalized multiscale Glow architecture for imaging data (2-D illustrated,
3-D also supported).  An input image $x \in \mathbb{R}^{B \times C_1 \times H_1
\times W_1}$ is processed through a sequence of levels $\ell = 1, \dots, N$. At
each level $\ell < N$, a squeeze operation trades spatial resolution for
channels, followed by a stack of Glow steps (ActNorm, invertible $1 \times 1$
convolution, and affine coupling). The output is split into a factored-out
latent block $z_\ell$ and a remaining block passed to the subsequent level. At
the final level $N$, a squeeze produces the remaining latent block $z_N$. The
complete latent representation $z = \{z_1, \dots, z_N\}$ preserves the original
image dimensionality. 
}
\label{fig:lamnr_diagrams}
\end{figure}

### Tabular/IDP views via RealNVP

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

We use two base distributions: a diagonal Gaussian and a Gaussian–PCA base that
performs an additional linear whitening of the flow latents (see Figure
\ref{fig:lamnr_diagrams}(a)). In the latter case, the flow acts as a learnable
multiview “whitener” that maps each tabular view to a standardized latent
\(\varepsilon\) with approximately independent components. Both the raw flow
latents \(z^{(v)}\) and the whitened coordinates \(\varepsilon^{(v)}\) can be
exported for downstream Gaussian modeling and diagnostics. This Gaussian–PCA
base is particularly useful when tabular views have different numbers of
columns. The per-view PCA yields an orthonormal, variance-ordered latent in
which we can select a common rank $r$ for alignment, producing matched-
dimension standardized coordinates \(\varepsilon^{(v)} \in \mathbb{R}^r\)
without altering the exact invertibility of the flow (truncation is used only
for the alignment head). Whitening also improves the conditioning of the
covariance estimates used in the conditional-Gaussian step by reducing
collinearity and stabilizing \(\Sigma_{OO}^{-1}\). Each tabular view is
typically modeled with a stack of 8–12 coupling layers with fully connected
subnetworks of width 256–512, and the resulting latent vector \(z^{(v)}\) is
then passed to the same projector, screening, and alignment machinery as the
image latents.

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

\input{latent_alignment_table.tex}

We summarize the main options in Table \ref{tab:alignment}, which we use
interchangeably depending on the experiment. Pearson and VICReg are effective
when batch sizes are modest and we want stable, low-cost alignment. Barlow Twins
adds explicit redundancy reduction via cross-correlation decorrelation. InfoNCE
provides a strong discriminative signal when large batches and in-batch
negatives are available, while HSIC captures non-linear dependence at the cost
of \(O(B^2)\) kernel operations, where $B =$ batch size. In all cases, the
alignment term acts only on shared coordinates, leaving private coordinates free
to capture view-specific variation.[^align] 

[^align]: In practice, we choose the alignment objective based on both computational
budget and the expected cross-view structure. For small batches or limited
compute, simple second-order methods such as Pearson correlation, or
non-contrastive objectives such as VICReg, are attractive because they are
stable and inexpensive to estimate. When stronger redundancy reduction is needed
while still avoiding negative pairs, Barlow Twins is a good default, explicitly
driving cross-correlation matrices towards identity. InfoNCE is most useful
when we can train with large batches and many in-batch negatives, for example in
discriminative multiview matching or retrieval-style scenarios where each sample
in one view must be matched against many candidates in another. HSIC-based
alignment is reserved for settings where we expect predominantly non-linear
cross-modality relations as it can capture richer dependencies than correlations
but carries an $O(B^2)$ kernel cost and is more sensitive to kernel and
bandwidth choices.

### Image views via Glow-based multiscale flows

For image views we adopt Glow-style discrete normalizing flows with \(L\) levels
and \(K\) coupling steps per level (see Figure \ref{fig:lamnr_diagrams}(b)). Each step
comprises: (i) ActNorm layers with data-dependent initialization, (ii)
invertible \(1 \times 1 (\times 1)\) convolutions parameterized with LU
factorization for efficient log-determinant computation, and (iii) affine
coupling layers whose scale and shift fields are predicted by shallow
convolutional subnetworks with a configurable number of hidden channels. Squeeze
and split operations provide a multiscale representation in which shallower
levels capture coarse structure while deeper levels model fine texture. Our
implementation follows the standard Glow construction, instantiated via a model
factory in ANTsTorch, with configurable image size (both 2-D and 3-D), number of
levels \(L\), steps per level \(K\), and hidden channels.[^starflow] 

[^starflow]: Recent transformer autoregressive flows such as STARFlow achieve
strong high-resolution synthesis by operating as a normalizing flow in the
latent space of a pretrained autoencoder [@gu2025starflow]. This design does not
provide an exact, per-sample bijection from pixel space to the flow’s latents or
multiscale per-level latents for analysis, both of which we require for
per-level alignment and post-hoc Gaussian conditioning.  We therefore adopt
Glow-style multiscale flows that offer single-pass, exact encoding/decoding in
image space with explicit latent access [@kingma2018glow].

Base distribution for image latents (Glow-style channel Gaussian) For image
views we use a channel-wise diagonal Gaussian (“Glow base”) with one mean and
one log-scale per channel, broadcast across spatial locations. Let \(z \in
\mathbb{R}^{C\times N_1\times \dots \times N_S}\) with \(S\in\{2,3\}\) spatial
dimensions and \(d=C\prod_{i=1}^S N_i\). The log density is

\[
\log p(z)
= -\tfrac12\, d\log(2\pi)
- \Big(\prod_{i=1}^S N_i\Big)\sum_{c=1}^C s_c
- \tfrac12 \sum_{c}\sum_{\mathbf{x}}\big[(z_{c,\mathbf{x}}-\mu_c)\,e^{-s_c}\big]^2,
\]

where \(\mu_c\) and \(s_c\) are per-channel parameters broadcast over all
spatial indices \(\mathbf{x}\). Compared to a conventional per-voxel diagonal
Gaussian, tying parameters within each channel reduces degrees of freedom,
matches Glow’s multiscale semantics, and avoids per-voxel scale collapse.


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

Normalizing flows give us an explicit bijection between data space and a latent
space with a simple base density (e.g., Gaussian). Once the per-view flows have been trained,
every multiview observation \(x = \{x^{(v)}\}_{v=1}^V\) can be mapped to a
collection of latents \(z = \{z^{(v)}\}_{v=1}^V\) with an exactly known
Jacobian. Any joint distribution placed on these latents induces a valid joint
distribution on the original data via the change-of-variables formula. In other
words, specifying a model \(p_Z(z)\) in latent space is equivalent to
specifying a generative model \(p_X(x)\) in data space, but with the advantage
that inference and conditioning can be carried out where the geometry is
simpler.

LAMNr Flows exploit this by choosing a multivariate Gaussian model on the
concatenated latents. This choice is deliberately simple: flows absorb the
complex, non-Gaussian aspects of each view into the invertible mappings
\(f^{(v)}\), so that the residual cross-view structure can be captured by a
Gaussian dependence model in \(z\). Under this construction, the joint density
factorizes as
\[
p_X(x) = p_Z(z)\,\prod_{v=1}^V \left|\det \frac{\partial f^{(v)}}{\partial x^{(v)}}\right|,
\]
with \(z = f(x)\). Because \(p_Z(z)\) is Gaussian, all conditionals
\(p_Z(z_U \mid z_O)\) are available in closed form, and exact conditional
inference in data space reduces to three steps: encode observed views to
latents, apply Gaussian conditioning in \(z\), and decode the resulting latents
back through the inverse flows. This yields closed-form posteriors, imputations,
and counterfactuals that are fully consistent with the learned flow model.


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

Our implementation builds on the \texttt{normflows} PyTorch package for
normalizing flows [@stimper2023normflows], which we adapt and extend for the
LAMNr setting. At the architectural level, we reconfigured the layer ordering to
match Glow-style multiscale flows (ActNorm $\rightarrow$ invertible
$1{\times}1(\times1)$ convolution $\rightarrow$ affine coupling).  We also
implemented 3-D variants of the core components (squeeze / unsqueeze, split /
merge, invertible $1{\times}1{\times}1$ convolutions, and 3-D coupling networks)
to support volumetric MRI data. These models are exposed through ANTsTorch as
configurable factories for both image and tabular views: Glow-style flows for
2-D/3-D images and RealNVP-style flows for IDPs and other tabular blocks. The
ANTsTorch interface handles dataset-level normalization and imputation, Gaussian
and Gaussian–PCA whiteners with optional application at train and test time.

Several empirical and survey works have noted that, compared to VAEs and
diffusion models, deep normalizing flows can be numerically sensitive on
high-dimensional data, with stability depending strongly on architectural
choices, scale parameterization, and Jacobian conditioning
[@papamakarios2021nfreview; @kobyzev2020nfsurvey; @Behrmann2019ResidualFlows;
@durkan2019nsf; @croitoru2023diffusion_vision_survey]. In light of this, we
introduced several additional stability-oriented modifications in both our
ANTsTorch builders and our \texttt{normflows} fork. These include bounded
coupling scales (via configurable \texttt{scale\_map} and \texttt{scale\_cap}
parametrizations), constrained base log-scales for Glow-style bases, optional
ActNorm inside coupling subnetworks, and gradient-norm clipping. We also
refactored computation of the Gaussian log likelihood for the base \(\Sigma =
W^\top W + \sigma^2 I\) using a Cholesky factorization with a small adaptive
jitter, evaluate \(\log|\Sigma|\) as \(2\sum \log \mathrm{diag}(L)\), and form
the quadratic term via triangular solves rather than explicit matrix inversion.
This avoids determinant and matrix-inverse calls that are unstable in high
dimensions and yields fewer NaNs during training. We initialize \(W\) at small
scale so that \(\Sigma\) is well conditioned at start, and we learn \(\log
\sigma\) to keep \(\sigma\) strictly positive.  These choices follow standard
numerical recommendations for stable positive-definite computations
[@higham2002accuracy] and pair naturally with shrinkage used elsewhere in our
conditional-Gaussian step [@ledoit2004well; @schafer2005shrinkage].
Collectively, these changes reduce log-det explosions and latent outliers in
deep multiscale flows while preserving exact likelihoods and invertibility.

Training and validation splits are defined at the subject level, and each
minibatch contains aligned multiview slices from matched subjects. Image data
augmentation is performed on-the-fly using the ANTsTorch-based `ImageDataset`
with affine and diffeomorphic deformations, small intensity perturbations
(histogram warping and bias field simulation), and additive Gaussian noise
treated as dequantization rather than biological variability. We control the
overall augmentation strength by a scalar schedule \(\alpha(t) \in [0,1]\) as a
function of normalized training time \(t\), and support linear, cosine, and
exponential decay: for example, a linear schedule reduces augmentation
proportionally to \(t\), a cosine schedule keeps stronger perturbations early
and then decays smoothly, and an exponential schedule reduces aggressive warps
and noise most rapidly at the beginning of training. This allows us to start
with heavier augmentations to regularize the flows and discourage overfitting to
discrete templates, then gradually emphasize fidelity to the true data
distribution as training progresses. For validation, we use the same spatial
transforms but disable additional noise and histogram warping. This design
preserves anatomical variability while preventing overfitting to discrete,
noise-free templates that would otherwise cause flows to collapse onto spiky
background modes.  

### Tabular-specific implementation details

For tabular flows we apply a small additive “jitter” noise to the features,
treated as dequantization rather than biological variation. The amplitude is
controlled by a scalar schedule \(\alpha(t)\) (linear, cosine, or exponential in
training time). In
addition, certain views can undergo a per-feature marginal transform prior to
normalization, such as an elementwise \(\operatorname{asinh}(x)\) for
heavy-tailed continuous variables or rank-based Gaussianization that maps the
empirical CDF of each feature to a standard normal. These monotone transforms
preserve rank information while making marginals more Gaussian and reducing
extreme tails. Together, marginal transforms and jitter regularize the tabular
flows and prevent them from overfitting to discrete patterns or exact repeated
rows in large cohorts.

### Glow-specific implementation details

Glow models are initialized with data-dependent ActNorm, and we perform a
one-time warm-up pass with real images before starting training to stabilize
statistics. We train with Adamax, mixed precision, and optional exponential
moving averages of model parameters. Learning rates follow a warm-up plus decay
schedule with a plateau-based reducer. We monitor exact negative log-likelihood
in bits-per-dimension, view-wise breakdowns, and alignment loss values, and
periodically log reconstructions and samples for visual inspection.

To accommodate 3-D volumes under constrained VRAM, we enable gradient
accumulation (microbatching). With an accumulation factor \(A\) and microbatch
size \(B_{\mu}\), the effective batch is \(B_{\mathrm{eff}} = A \cdot B_{\mu}\).
We compute per-sample losses on each microbatch, accumulate their gradients, and
perform a single optimizer step every \(A\) microbatches. Likelihood terms are
summed in natural units and normalized by the total number of samples across the
\(A\) microbatches.  Alignment losses (Pearson, Barlow Twins, VICReg, InfoNCE,
HSIC) are accumulated with microbatch-size weighting so the effective objective
matches the non-accumulated baseline. Under mixed precision, gradients are
accumulated in scaled form and unscaled once before a single global-norm clip
and optimizer step.  EMA updates and learning-rate schedulers advance once per
effective batch. ActNorm uses fixed statistics after warm-up, so accumulation
does not change normalization. For InfoNCE, negatives are limited to the current
microbatch by default. When larger negative sets are required, we optionally
maintain a cross-microbatch queue to approximate large-batch behavior. In
practice, \(A \in \{2,4,8\}\) trades memory for step latency while preserving
likelihood calibration and alignment strength for 3-D flows.

The same training loop supports multiple views by instantiating one flow per
view, computing per-view log-likelihoods and latents for each minibatch,
flattening the multiscale latents, and applying the projector plus alignment and
screening logic described above. For IDP/tabular experiments, we replace the
image Glow backbones with RealNVP/MAF stacks while keeping the multiview
alignment and conditional Gaussian stages unchanged, which allows LAMNr Flows to
operate uniformly across image contrasts and multiview IDP blocks within the
same methodological framework.

\begin{figure*}[htb]
  \centering
  \captionsetup[subfigure]{justification=centering}
  \begin{subfigure}[t]{0.19\textwidth}
    \includegraphics[width=\linewidth]{Figures/aug_step000.png}
    \caption{Step 1}
  \end{subfigure}
  \begin{subfigure}[t]{0.19\textwidth}
    \includegraphics[width=\linewidth]{Figures/aug_step001.png}
    \caption{Step 2}
  \end{subfigure}
  \begin{subfigure}[t]{0.19\textwidth}
    \includegraphics[width=\linewidth]{Figures/aug_step002.png}
    \caption{Step 3}
  \end{subfigure}
  \begin{subfigure}[t]{0.19\textwidth}
    \includegraphics[width=\linewidth]{Figures/aug_step003.png}
    \caption{Step 4}
  \end{subfigure}
  \begin{subfigure}[t]{0.19\textwidth}
    \includegraphics[width=\linewidth]{Figures/aug_step004.png}
    \caption{Step 5}
  \end{subfigure}
  \caption{
   Image augmentation schedule preview across five steps (left to right).
   Schedules used: \texttt{noise\_std:cos:0.06$\rightarrow$0.008},
   \texttt{sd\_affine:cos:0.05$\rightarrow$0.00},
   \texttt{sd\_deformation:linear:16.0$\rightarrow$1.0},
   \texttt{sd\_simulated\_bias\_field:cos:0.25$\rightarrow$0.05},
   \texttt{sd\_histogram\_warping:cos:0.05$\rightarrow$0.01}. Previews were
   generated with the ANTsTorch test driver
   (\texttt{test\_image\_dataset\_and\_scheduler.py}) using a $3\times3$ grid of
   $128\times128$ tiles. The source image is the HCP young-adult template
   constructed with ANTsX tools and distributed via ANTsTorch.
   }
  \label{fig:aug-schedule}
\end{figure*}

We apply lightweight, label-free augmentations during maximum-likelihood
training to improve robustness without changing the model’s exact likelihood
computation (augmentations act on inputs only) [@Tustison:2024aa;@Tustison:2025aa] (cf Figure
\ref{fig:aug-schedule}). For image views (2-D/3-D), we use geometric transforms
(linear and non-linear transformations) shared across all views of a subject to
preserve alignment targets, and per-view intensity-based transforms (noise,
simulated bias-field, histogram warping).  Similar to the tabular case, the amplitude is
controlled by a scalar schedule \(\alpha(t)\) (linear, cosine, or exponential in
training time).

