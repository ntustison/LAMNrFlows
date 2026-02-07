
\clearpage

# Methods

## Normalizing flows and the LAMNr flows multiview formulation

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
which we can evaluate exactly for the RealNVP-style [@dinh2016realnvp] and Glow
[@kingma2018glow] architectures used here. Maximum-likelihood estimation chooses
\(\theta^{(v)}\) to maximize the sum of log-likelihoods over subjects, or
equivalently to minimize the average negative log-likelihood
[@kobyzev2020nfsurvey].

Each subject \(n\) has measurements \(\{x_n^{(v)}\}_{v=1}^V\) across one or more 
views \(v = 1,\dots,V\). We instantiate one flow per view and train them jointly on
subject-matched minibatches. If we consider only the flow likelihoods, the pure
maximum-likelihood objective is
\[
\mathcal{L}_{\text{like}}(\theta)
= \frac{1}{N} \sum_{n=1}^N \sum_{v=1}^V
\bigl[- \log p_{\theta}(x_n^{(v)})\bigr],
\]
where \(\theta = \{\theta^{(v)}\}_{v=1}^V\) collects all view-specific
parameters. This term ensures that each per-view flow is an exact-likelihood
model of its corresponding data distribution.

The LAMNr flows multiview approach leverages the exposure of latent space. For
each view and subject we write
\[
z_n^{(v)} = \bigl[z^{(v)}_{S,n}, \; z^{(v)}_{P,n}\bigr],
\]
splitting the latent into a block \(z^{(v)}_{S,n}\) that is intended to carry
shared information across views and a block \(z^{(v)}_{P,n}\) that is
view-specific. The indices that define this split can be chosen a priori or via
the CCA/HSIC-based screening procedure described below.

We then attach a small projector network \(\phi^{(v)}_{\psi} : \mathcal{Z}^{(v)}
\to \mathbb{R}^D\) to each view, with a matched output dimension \(D\) across
views, and apply a multiview alignment loss to the projected shared coordinates.
Attaching a small projector $\phi_{\psi}^{(v)} : \mathcal{Z}^{(v)} \rightarrow
\mathbb{R}^{D}$ decouples the flow's latent dimensionality and arbitrary
coordinate system from the alignment space, letting each view learn a light
reparameterization (linear or shallow MLP) that canonizes rotations/scales
introduced by invertible mixing and harmonizes dimensions across views. By
restricting alignment to this matched, low-dimensional subspace $D$, we
stabilize dependence objectives (i.e., latent alignment constraints), avoid
forcing private directions to align, and integrate CCA/HSIC screening cleanly by
aligning only the coordinates deemed shared.  We summarize the main options in
Table \ref{tab:alignment}. In all cases, the alignment term acts only on shared
coordinates, leaving private coordinates free to capture view-specific
variation.[^align] 

\input{latent_alignment_table.tex}

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

To prevent over-constraining the flows and blurring view-specific information,
we perform a short screening pass after an MLE warm-up phase. For CCA-based
screening, we construct whitened feature matrices for two views and perform an
SVD of the cross-covariance \(X_a^\top X_b\), retaining the top \(r\) canonical
directions per view. Averaging across all view pairs yields per-view projectors
\(P^{(v)} \in \mathbb{R}^{D \times r}\) defining the shared subspace. For
HSIC-based screening, we first prefilter coordinates using Pearson correlation,
then rank remaining dimensions by an unbiased HSIC estimate with RBF kernels
averaged over other views, and select the top \(r\) per view. Alignment losses
are applied only to these projected or masked coordinates, so that dependence is
enforced where cross-view signal is strongest and private dimensions remain free
to capture view-specific variation. Screening can be performed once after
warm-up or periodically refreshed during training.

For a subject-matched minibatch of size \(N\), the full training objective becomes
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
collinearity and stabilizing \(\Sigma_{OO}^{-1}\). 


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


