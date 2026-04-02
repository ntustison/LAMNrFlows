
\clearpage

\clearpage

# Methods

## Normalizing Flows and the LAMNr Flows Multiview Formulation

\begin{figure}
   \centering
   \includegraphics[width=0.99\textwidth]{Figures/lamnr_flows_illustration.pdf} 
   \caption{LAMNr flows architecture and latent alignment. The model processes
   three data views, illustrated by the panels View 1,
   View 2, and View 3. For each view, the data distribution in the
   observation space $\mathcal{X}$ can be mapped to a simplified base
   distribution in the latent space $\mathcal{Z}\sim\mathcal{N}(0,1)$. This
   mapping is performed by the individual normalizing flows sequential bijections
   ($T_1$, $T_2$, $\dots$, $T_n$). Joint alignment optimization is performed on
   latent distributions to drive convergence towards a harmonized shared space
   through the application of the alignment loss function
   $\mathcal{L}_{align}(\{\phi_{v}^{(v)}(z_{S,n}^{(v)})\}_{v,n})$.}
   \label{fig:lamnr_flows_illustration}
\end{figure}

Given subject \(n\) with measurements \(\{x_n^{(v)}\}_{v=1}^V\) across one or
more views \(v = 1,\dots,V\), for a single view \(v\), a normalizing flow with
parameters \(\theta^{(v)}\) is an invertible mapping

\begin{equation}
f^{(v)}_{\theta} : \mathcal{X}^{(v)} \to \mathcal{Z}^{(v)}, \quad
z_n^{(v)} = f^{(v)}_{\theta}(x_n^{(v)}),
\end{equation}

that transforms observed data \(x_n^{(v)}\) to a latent space
\(\mathcal{Z}^{(v)}\) with a chosen base density, typically
\(p_Z(z) = \mathcal{N}(0, I)\). The induced density on \(\mathcal{X}^{(v)}\)
follows from the change-of-variables formula:

\begin{equation}
\log p_{\theta}(x_n^{(v)}) =
\log p_Z\bigl(z_n^{(v)}\bigr)
+ \log \bigl|\det \partial f^{(v)}_{\theta} / \partial x_n^{(v)}\bigr|,
\end{equation}

which can be evaluated exactly for the RealNVP [@dinh2016realnvp] and Glow
[@kingma2018glow] architectures used this proposed framework. The maximum-likelihood
estimation chooses \(\theta^{(v)}\) to maximize the sum of log-likelihoods over
subjects, or equivalently to minimize the average negative log-likelihood
[@kobyzev2020nfsurvey].

For multiple views in the LAMNr flows framework, we instantiate one flow per
view and train them jointly on subject-matched minibatches. If we consider only
the flow likelihoods, the pure maximum-likelihood objective is

\begin{equation}
\mathcal{L}_{\text{like}}(\theta)
= \frac{1}{N} \sum_{n=1}^N \sum_{v=1}^V
\bigl[- \log p_{\theta}(x_n^{(v)})\bigr],
\end{equation}

where \(\theta = \{\theta^{(v)}\}_{v=1}^V\) represents all view-specific
parameters. This term ensures that each per-view flow is an exact-likelihood
model of its corresponding data distribution. For each view and subject we write

\begin{equation}
z_n^{(v)} = \bigl[z^{(v)}_{S,n}, \; z^{(v)}_{P,n}\bigr],
\end{equation}

splitting the latent into a block \(z^{(v)}_{S,n}\) that is intended to carry
shared information across views and a block \(z^{(v)}_{P,n}\) that is
view-specific. The indices that define this split can be chosen a priori or via
the CCA/HSIC-based screening procedure (described below).

We attach a small projector network $\phi^{(v)}_{\psi} : \mathcal{Z}^{(v)} \to
\mathbb{R}^D$ to each view, utilizing a multi-layer perceptron (MLP)
architecture with a hidden layer width $H$ and a matched output dimensionality
$D$ across all views. Default values are $H=512$ and $D=256$.  Attaching this
projector decouples the flow’s internal latent dimensionality and arbitrary
coordinate system from the alignment space. This allows each view to learn a
light reparameterization of alignment of rotations and scales frequently
introduced by invertible mixing layers such as the $1 \times 1 (\times 1)$
convolutions in Glow, while harmonizing dimensions across disparate views.  The
role of this $D$ subspace can very strategically depending on the data type. For
tabular IDPs (i.e., RealNVP) this configuration represents a high-capacity
expansion of the lower-dimensional latent space. This expansion provides the
alignment constraints with sufficient degrees of freedom to operate without
inducing information loss or architectural bottlenecks.  For image
data (i.e., Glow), the projection acts as an intentional dimensionality
reduction and selective filter. By compressing the massive raw latent space into
the lower space $D$, we filter out view-specific high-frequency noise, ensuring
the alignment objective captures shared, global morphometric trends rather than
idiosyncratic imaging artifacts. By restricting alignment to this matched
subspace, we stabilize the specified alignment constraint and avoid the
instability of forcing private, view-specific anatomical directions to align. We
summarize the main options in Table \ref{tab:alignment}. In all cases, the
alignment term acts only on these shared coordinates, leaving private
coordinates free to capture independent variation.[^align] 

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
we perform a short screening pass after an initial ``warm-up'' phase. For
CCA-based screening, we construct whitened feature matrices for two views and
perform an SVD of the cross-covariance \(X_a^\top X_b\), retaining the top \(r\)
canonical directions per view. Averaging across all view pairs yields per-view
projectors \(P^{(v)} \in \mathbb{R}^{D \times r}\) defining the shared subspace.
For HSIC-based screening, we first prefilter coordinates using Pearson
correlation, then rank remaining dimensions by an unbiased HSIC estimate with
RBF kernels averaged over other views, and select the top \(r\) dimensions per
view. Alignment losses are applied only to these projected or masked
coordinates, so that dependence is enforced where cross-view signal is strongest
and private dimensions remain free to capture view-specific variation. Screening
can be performed once after warm-up or periodically refreshed during training.

For a subject-matched minibatch of size \(N\), the full training objective becomes

\begin{equation}
\mathcal{L}(\theta, \psi)
= \frac{1}{N} \sum_{n=1}^N \sum_{v=1}^V
\bigl[- \log p_{\theta}(x_n^{(v)})\bigr]
+ \lambda \, \mathcal{L}_{\text{align}}
\Bigl(\bigl\{\phi^{(v)}_{\psi}(z^{(v)}_{S,n})\bigr\}_{v,n}\Bigr),
\end{equation}

where \(\mathcal{L}_{\text{align}}\) is one of the available options: Barlow Twins, VICReg,
InfoNCE, Pearson correlation, or HSIC. \(\lambda\) controls the strength of
alignment.[^Kendall-Gal] 

[^Kendall-Gal]: 
In some early experiments we replaced the fixed weighting, \(\lambda\), by
learned task weights following the homoscedastic aleatoric uncertainty scheme of
Kendall and Gal [@kendall2018mtl]. In this formulation, each loss term \(L_i\)
is scaled as \(e^{-s_i} L_i + s_i\), where \(s_i = \log \sigma_i^2\) represents
task-dependent aleatoric (data) uncertainty, as opposed to epistemic (model)
uncertainty [@Hullermeier2021UncertaintyReview]. While this can automatically
balance losses with different units, in our multiview setting it tended to
inflate the alignment variance and drive the effective alignment weight
\(e^{-s_{\text{align}}}\) toward zero, effectively suppressing latent alignment.
Therefore, we report results using a fixed \(\lambda\) schedule in the main
experiments.


## View-specific Flow Architectures

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
(b) Generalized multiscale Glow architecture for imaging data (2D illustrated,
3D also supported).  An input image $x \in \mathbb{R}^{B \times C_1 \times H_1
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

### Tabular/IDP Views via RealNVP

For imaging-derived phenotypes (IDPs) and other tabular data, we use
single-scale flows based on RealNVP and masked autoregressive flows (MAF) with
affine couplings and masked multilayer perceptrons. Continuous variables are
preprocessed per view using dataset-owned normalization and imputation:
columns are coerced to numeric, NaNs are imputed (typically by the column mean),
and features are standardized to zero mean or rescaled to \([0,1]\) depending on
a user-selectable normalization mode. Very low-variance columns are stabilized
by floor-clamping the standard deviation. Positively skewed, non-negative
variables can optionally be $\log$- or $\log1p$-transformed before normalization to
reduce skewness.

We use two base distributions: a diagonal Gaussian and a Gaussian–PCA base that
performs an additional linear whitening of the flow latents (see Figure
\ref{fig:lamnr_diagrams}(a)). In the latter case, the flow acts as a learnable
multiview “whitener” that maps each tabular view to a standardized latent
\(\varepsilon\) with approximately independent components. Both the raw flow
latents \(z^{(v)}\) and the whitened coordinates \(\varepsilon^{(v)}\) can be
exported for downstream Gaussian modeling and diagnostics. This Gaussian–PCA
base distribution is particularly useful when tabular views have different
numbers of features. The per-view PCA yields an orthonormal, variance-ordered
latent in which we can select a common rank $r$ for alignment, producing
matched-dimension standardized coordinates \(\varepsilon^{(v)} \in
\mathbb{R}^r\) without altering the exact invertibility of the flow as truncation
is used only for the alignment head. 

### Image Views via Glow-based Multiscale Flows

For image views we adopt Glow-style discrete normalizing flows with \(L\) levels
and \(K\) coupling steps per level (see Figure \ref{fig:lamnr_diagrams}(b)) and
a diagonal Gaussian base distribution. Each coupling step comprises: (i) ActNorm
layers with data-dependent initialization, (ii) invertible \(1 \times 1 (\times
1)\) convolutions parameterized with LU factorization for efficient
log-determinant computation, and (iii) affine coupling layers whose scale and
shift fields are predicted by shallow convolutional subnetworks with a
configurable number of hidden channels. Squeeze and split operations provide a
multiscale representation in which shallower levels capture coarse structure
while deeper levels model fine texture. Our implementation follows the standard
Glow construction, instantiated via a model factory in ANTsTorch, with
configurable image size (both 2D and 3D), number of levels \(L\), steps per
level \(K\), and hidden channels.[^starflow] 

[^starflow]: Recent transformer autoregressive flows such as STARFlow achieve
strong high-resolution synthesis by operating as a normalizing flow in the
latent space of a pretrained autoencoder [@gu2025starflow]. This design does not
provide an exact, per-sample bijection from pixel space to the flow’s latents or
multiscale per-level latents for analysis, both of which we require for
per-level alignment and post-hoc Gaussian conditioning.  This motivates our
adoption of Glow-style multiscale flows that offer single-pass, exact
encoding/decoding in image space with explicit latent access [@kingma2018glow].

For image views we use a channel-wise diagonal Gaussian (“Glow base”) with one
mean and one log-scale per channel, broadcast across spatial locations. Let \(z
\in \mathbb{R}^{C\times N_1\times \dots \times N_S}\) with \(S\in\{2,3\}\)
spatial dimensions and \(d=C\prod_{i=1}^S N_i\). The log density is

\begin{equation}
\log p(z)
= -\tfrac12\, d\log(2\pi)
- \Big(\prod_{i=1}^S N_i\Big)\sum_{c=1}^C s_c
- \tfrac12 \sum_{c}\sum_{\mathbf{x}}\big[(z_{c,\mathbf{x}}-\mu_c)\,e^{-s_c}\big]^2,
\end{equation}

where \(\mu_c\) and \(s_c\) are per-channel parameters broadcast over all
spatial indices \(\mathbf{x}\). Compared to a conventional per-voxel diagonal
Gaussian, tying parameters within each channel reduces degrees of freedom,
matches Glow’s multiscale semantics, and avoids per-voxel scale collapse.
However, in the medical imaging context discussed here, only single channel
data is employed.

## High-dimensional Geometry and Latent Space Navigation

In high-dimensional standard normal latent spaces, such as those optimized by
LAMNr flows, the geometric properties of the data distribution become highly
counterintuitive due to the concentration of measure phenomenon
[@white2016sampling; @vershynin2018high; @blum2020foundations]. As
dimensionality increases, probability mass moves away from concentration at the
origin. Instead, the volume of the space grows exponentially with distance from
the center, causing the vast majority of the mass to concentrate within a narrow
spherical shell of radius $\approx \sqrt{d}$. This region is often referred to
as the typical set [@vershynin2018high; @blum2020foundations] or the "soap
bubble"[^blogpost] effect. Consequently, the latent origin $z=0$ is a highly
atypical point containing near-zero probability mass. The inverse mapping
$f^{-1}(0)$ must therefore be understood strictly as a barycentric geometric
anchor representing a central axis of symmetry for the learned bijection, rather
than a statistically representative anatomical mode.  

[^blogpost]: https://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/

Furthermore, while the normalizing flow successfully unfolds the global topology
of the anatomical data, the assumption that Euclidean operations in the latent
space seamlessly translate to valid anatomical transformations in the image
space is mathematically flawed. The latent space is not a flat Euclidean
manifold.  Rather, its intrinsic distances are governed by a stochastic
Riemannian metric induced by the generator's Jacobian, defined as $M_z =
J_z^\top J_z$ [@arvanitidis2018latent]. Because the network non-linearly expands
and compresses the data space to maximize likelihood, Euclidean straight lines
in the latent space do not correspond to the shortest paths (geodesics) on the
underlying image manifold. 

This geometric distortion has immediate, tangible consequences for cohort
alignment and interpolation. Linearly interpolating between two latent
points located on the typical set creates a trajectory that moves inward toward
the latent origin. In high dimensions, this effect forces the interpolation path
through unpopulated latent regions of extremely low probability, causing a
severe distribution mismatch [@agustsson2018optimaltransportmapsdistribution].
The resulting generated images exhibit blurriness, structural artifacts, and
anatomical inconsistencies. To better align deep generative models with the
principles of computational anatomy, Euclidean operations must be replaced with
distribution-preserving mechanisms.

To navigate this geometry appropriately, we replace standard linear
interpolation with spherical linear interpolation (SLERP) when traversing
the latent space between two generated samples $z_1$ and $z_2$
[@white2016sampling; @agustsson2018optimaltransportmapsdistribution]. For an
interpolation parameter $t \in [0, 1]$ and the angle $\theta =
\arccos\left(\frac{z_1 \cdot z_2}{\|z_1\|_2 \|z_2\|_2}\right)$ between the
vectors, the SLERP trajectory is defined as:

\begin{equation}\text{SLERP}(z_1,
z_2; t) = \frac{\sin((1-t)\theta)}{\sin(\theta)}z_1 +
\frac{\sin(t\theta)}{\sin(\theta)}z_2
\end{equation}

This formulation ensures that the interpolation path strictly follows the
high-probability manifold, preserving structural integrity. Similarly, we
adapted our distance metrics based on the evaluation context. When assessing the
semantic similarity between two individual images within the latent space, we
default to the geodesic (angular) distance rather than the Euclidean distance.
The geodesic distance effectively isolates the directional components of the
vectors:

\begin{equation}
d_{geo}(z_1, z_2) = \arccos\left( \frac{z_1 \cdot z_2}{|z_1|_2|z_2|_2} \right)
\end{equation}

This metric captures core semantic features while discarding magnitude
variations that primarily represent high-dimensional statistical noise. However,
for measuring a subject's deviation from the normative population, we utilize
the Mahalanobis distance relative to the Gaussian mean ($\mu = 0$):

\begin{equation} 
d_M(z) = \sqrt{z^\top\Sigma^{-1} z} 
\end{equation}

where $\Sigma$ represents the covariance matrix of the reference cohort. In this
scenario, the radial distance from the origin, captured by the Mahalanobis
metric, is precisely the signal required to quantify the statistical
unlikelihood of an abnormal sample.
