


<!-- ## Applications -->

## Conditional Gaussian Modeling Over Latents

Normalizing flows give us an explicit bijection between data space and a latent
space with a simple base density (e.g., Gaussian). Once the per-view flows have
been trained, every multiview observation \(x = \{x^{(v)}\}_{v=1}^V\) can be
mapped to a collection of latents \(z = \{z^{(v)}\}_{v=1}^V\) with an exactly
known Jacobian. Any joint distribution placed on these latents induces a valid
joint distribution on the original data via the change-of-variables formula. In
other words, specifying a model \(p_Z(z)\) in latent space is equivalent to
specifying a generative model \(p_X(x)\) in data space, but with the advantage
that inference and conditioning can be carried out where the geometry is
simpler.

LAMNr flows exploit this by choosing a multivariate Gaussian model on the
concatenated latents. This choice is deliberately simple as the flows absorb the
complex, non-Gaussian aspects of each view into the invertible mappings
\(f^{(v)}\), so that the residual cross-view structure can be captured by a
Gaussian dependence model in \(z\). Under this construction, the joint density
factorizes as

\begin{equation}
p_X(x) = p_Z(z)\,\prod_{v=1}^V \left|\det \frac{\partial f^{(v)}}{\partial x^{(v)}}\right|,
\end{equation}

with \(z = f(x)\). Because \(p_Z(z)\) is Gaussian, all conditionals
\(p_Z(z_U \mid z_O)\) are available in closed form, and exact conditional
inference in data space reduces to three steps: encode observed views to
latents, apply Gaussian conditioning in \(z\), and decode the resulting latents
back through the inverse flows. This yields closed-form posteriors, imputations,
and counterfactuals that are fully consistent with the learned flow model.

After training the per-view flows and projector alignment, we freeze the flow
parameters and collect latents for all subjects. For image views, we retain a
multiscale representation \(z^{(v)}_\ell\) at each level \(\ell \in
\{1,\dots,L\}\); for tabular views, we have a single level. Concatenating across
views and levels yields a joint latent vector

\begin{equation}
z = \bigl[z^{(1)}_1, \dots, z^{(1)}_L, \dots, z^{(V)}_1, \dots, z^{(V)}_L\bigr].
\end{equation}

We model this joint latent as Gaussian, \(z \sim \mathcal{N}(\mu, \Sigma)\).
However, directly computing and operating on the full covariance matrix
\(\Sigma\) poses a severe computational bottleneck due to the "curse of
dimensionality." While feasible for 2D images, a small 3D medical volume (e.g.,
\(64 \times 64 \times 64\)) yields a latent dimension \(D \approx 2.6 \times
10^5\). Storing the dense \(D \times D\) covariance matrix as 64-bit double
precision requires over 500 GB of memory, making direct Cholesky inversion
computationally intractable.

To resolve this in high-dimensional 3D settings, we employ a
low-rank-plus-diagonal parameterization via Singular Value Decomposition (SVD).
We approximate the covariance as \(\Sigma \approx U \Lambda U^T + \sigma^2 I\),
where \(U\) contains the top \(r\) eigenvectors (\(r \ll D\)), \(\Lambda\) is
the diagonal matrix of the top \(r\) eigenvalues, and \(\sigma^2 I\) captures
the isotropic residual variance. Given an observed subset of coordinates \(O\)
and an unobserved subset \(U\), the exact posterior \(p(z_U \mid z_O)\) is
Gaussian. The conditional mean and covariance are theoretically given by:

\begin{equation}
\mu_{U\mid O}
= \mu_U + \Sigma_{UO}\Sigma_{OO}^{-1}(z_O - \mu_O),
\end{equation}

\begin{equation}
\Sigma_{U\mid O}
= \Sigma_{UU} - \Sigma_{UO}\Sigma_{OO}^{-1}\Sigma_{OU}.
\end{equation}

To evaluate these equations without instantiating the massive $\Sigma_{OO}$
matrix, we utilize the Woodbury matrix identity [@henderson1981deriving].
Specifically, we apply the Push-Through identity to perform the inversion
strictly within the low-dimensional subspace $r$ [@rasmussen2006gaussian].
By reformulating the system to solve for a strictly positive-definite $r \times
r$ matrix ($K = \Lambda^{1/2} U_O^T U_O \Lambda^{1/2} + \sigma^2 I$), we bypass
problematic numerical cancellations [@higham2002accuracy] and reduce the
memory footprint to mere megabytes. 

Critically, the fidelity of these conditional results is intrinsically linked to
the inter-view latent alignment established during the initial training phase.
In the joint Gaussian model, the cross-covariance terms $\Sigma_{UO}$
encapsulate the statistical coupling between the observed and unobserved
manifolds. Effective alignment ensures that subject-specific anatomical features
are mapped to shared latent coordinates, enabling the conditional mean to
recover missing modalities with high individual specificity. In the absence of
such alignment (i.e., when $\Sigma_{UO} \approx 0$), the conditional mean
$\mu_{U|O}$ collapses to the population prior $\mu_U$. In this limit, the
imputed output reverts to a generic population template, losing the
subject-matched morphological details. 

Samples from this conditional Gaussian propagate uncertainty, while the
posterior mean provides a calibrated point estimate. Applying the inverse flows
to these posterior latents yields imputations, harmonized representations, and
latent edits in the original data space, with exact likelihoods available for
all configurations.

<!--
### Latent distance as a Riemannian biomarker

The multi-scale Glow architecture provides a bijective mapping $f_\theta: X \to
Z$ between the image space and a latent representation [@kingma2018glow], where
$Z$ is explicitly regularized to follow a standard multivariate Gaussian prior
$\mathcal{N}(0, I)$. We quantify the clinical relevance of an image $x$ via its
latent distance $\mathcal{D}(x)$, defined as the squared $L_2$ norm:

$$\mathcal{D}(x) = \|f_\theta(x)\|_2^2 = \sum_{i=1}^{d} z_i^2$$

In the context of the PPMI cohort, this metric functions as a measure of
"atypicality." We hypothesize that pathological alterations (e.g., dopaminergic
denervation or structural atrophy) map to the tails of the Gaussian
distribution, yielding higher latent distances than cognitively normal (CN)
controls. This is formally linked to the generative model's log-likelihood:

$$\log p(x) = -\frac{1}{2} \mathcal{D}(x) + \log \left| \det \frac{\partial
f_\theta(x)}{\partial x} \right| + C$$

where the Jacobian term captures local geometric deformations (e.g., ventricular
enlargement or cortical thinning). Crucially, by utilizing the CCA screening
mechanism, we define a shared latent distance $\mathcal{D}_{shared}(x)$
restricted to the subspace $\mathcal{Z}_{shared}$. This isolates biomarkers
consistently reflected across modalities (e.g., T1 and FA), enhancing
sensitivity to coordinated neurodegeneration while suppressing modality-specific
noise.

From a Riemannian perspective, $f_\theta$ induces an isometry between the curved
manifold of neuroanatomical images $\mathcal{X}$ and the flat Euclidean space
$Z$. Under the pull-back metric $g = J^T J$, the latent norm
$\sqrt{\mathcal{D}(x)}$ represents the geodesic distance between the subject $x$
and the learned population average (the latent mode at $z=0$). This formulation
draws a direct parallel to LDDMM [@Beg2005LDDMM] where the latent norm acts as a
surrogate for the geodesic distance on the manifold, but with the advantage of
being constrained to the biologically plausible image distribution learned by
the flow. Consequently, $\mathcal{D}(x)$ provides a robust, continuous scale for
quantifying disease severity as a displacement from the healthy anatomical
manifold.

### Shared-latent reconstructions, templates, and operational edits

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
-->