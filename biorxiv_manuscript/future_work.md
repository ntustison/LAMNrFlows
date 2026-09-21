\clearpage

# Future work

## Imputing Missing Data with LAMNr

We see two natural extensions of our framework to missing-data imputation, both
consistent with the way LAMNr models views as invertible mappings into a shared
latent structure. The first targets cross-view imputation, where one or more
views are absent for a subject but companion views are available. Because
training already aligns per-view latents on subject-matched batches and fits
per-level Gaussian statistics, inference becomes straightforward: we encode the
observed views, evaluate the closed-form conditional over the missing view’s
latent coordinates, and then invert that view’s flow to obtain an imputed sample
in input space. This pathway requires no new networks, preserves exact
likelihoods at each step, and yields imputations that are calibrated by
construction. In practice, we will report accuracy against held-out ground truth
together with conditional log-likelihoods, coverage of posterior intervals, and
sensitivity to the pattern and rate of missingness.

The second extension addresses intra-view gaps, where individual features are
missing within a single tabular block. Here we plan to start with a maximum a
posteriori approach that treats missing entries as optimization variables and
maximizes the joint log-density with respect to those coordinates. The
computation is simple—autodiff through the bijection—and benefits from good
initialization and early stopping, providing a competitive baseline without
retraining. Beyond that baseline, we will introduce mask-aware training that
randomly withholds features during learning and encourages the model to
reconstruct the masked coordinates from the observed ones. This amortizes the
conditional at test time and typically improves calibration. For views that use
a Gaussian–PCA base, we will also explore a lightweight refinement
loop—initialize the missing features, encode to latents, enforce Gaussian
consistency in the whitened coordinates, and invert back—repeating a small
number of times until convergence.

Across both regimes, our evaluation will mirror the rest of the paper: we will
select hyperparameters upstream using validation bits-per-dimension, then assess
imputation with error metrics (NRMSE, MAE, rank correlations), uncertainty
diagnostics (PIT, empirical coverage), and downstream utility (predictive
performance of simple models trained on complete data and tested on imputed
sets). The central advantage is conceptual continuity: imputation follows the
same exact, invertible modeling assumptions used elsewhere in LAMNr, allowing
ablations and comparisons to be scored on a common likelihood scale and
reproduced with identical train, validation, and test splits.


## Flow-based latent regions for ordered clinical categories

An important direction is to use normalizing flows to define and interrogate
clinically meaningful regions in feature space for ordered labels such as
“none”, “mild”, “moderate”, and “severe”. In a simple single-view setting, one
can first train an invertible flow on a large cohort of multi-dimensional
features \(x\) to obtain latent representations \(z = f_\theta(x)\) with a
class-conditional diagonal Gaussian base distribution, for example using a
`ClassCondDiagGaussian` base where each class \(y \in \{0,\dots,N-1\}\) has its
own mean and log scale in latent space. Given labelled subjects, the model then
learns \(p(z \mid y)\) as a separate Gaussian “bump” for each category, while
the flow \(f_\theta\) handles the nonlinear mapping between \(x\) and \(z\).
This setup naturally induces class-specific ellipsoidal regions in latent space
(level sets of \(p(z \mid y)\)) that can be mapped back through the inverse flow
\(f_\theta^{-1}\) to obtain corresponding domains in the original feature space,
providing a generative description of what “typical” or “boundary” cases look
like for each category.

This construction supports both prediction and interpretability. For
classification, a generative classifier can evaluate \(p(x \mid y = k)\) via the
flow and class-conditional base, combined with class priors, to assign new
subjects to the most likely category. For interpretation, one can sample from
the learned class-conditional Gaussians in latent space and invert the flow to
generate synthetic examples spanning the interior and boundary of each severity
grade, or examine where the class-conditional densities intersect to approximate
decision boundaries between “mild” and “moderate” or “moderate” and “severe”.
Although this is naturally formulated for categorical labels, the same framework
can be extended to ordered categories by encouraging class means to align along
a low-dimensional “severity axis” in latent space or by adding an ordinal loss
on a linear projection of \(z\). In future work, we plan to adapt this
class-conditional flow formulation to the LAMNr multiview setting, enabling
disease severity regions to be defined jointly across imaging-derived features
and non-imaging covariates.

## Mask-aware inpainting with conditional Gaussian modeling

A primary extension is mask-aware inpainting using the Conditional Gaussian
Modeling (CGM) machinery developed for our per-level latent statistics. Here we
treat a masked image as two “views” of the same subject: a context of
observed pixels and a hole of missing pixels. The key is to operate in
per-level latents (where we already model second-order structure) and to respect
the receptive field of each coupling network.

After \(\ell\) squeeze operations, each latent cell corresponds to a \(2^\ell
\times 2^\ell\) (2D) or \(2^\ell \times 2^\ell \times 2^\ell\) (3D) block in
image space. Let \(\Omega_{\mathrm{obs}}\subset\mathbb{Z}^d\) denote observed
pixels with indicator mask \(M\). We define a safe context band by morphological
erosion with a radius matched to the coupling network receptive field
\(r_\ell\):

\[
\Omega^{\text{safe}}_\ell \;=\; \big(\Omega_{\mathrm{obs}} \ominus B_{r_\ell}\big),
\]

and downsample \(\Omega^{\text{safe}}_\ell\) to latent indices via the squeeze
mapping. Latent positions whose entire receptive field lies in
\(\Omega^{\text{safe}}_\ell\) form the observed set \(X\); all remaining latent
positions form the missing set \(Y\). On projected per-level latents
\(\tilde{Z}_\ell\) we already estimate dataset moments
\((\mu_\ell,\Sigma_\ell)\), optionally after a CCA subspace of rank \(k\) with
shrinkage or jitter to maintain positive definiteness. Partitioning
\(\mu_\ell,\Sigma_\ell\) as \((Y,X)\), we can compute the Gaussian conditional

\[
\mu_{Y\mid X} \;=\; \mu_Y + \Sigma_{YX}\,\Sigma_{XX}^{-1}\big(x-\mu_X\big), 
\qquad 
\Sigma_{Y\mid X} \;=\; \Sigma_{YY} - \Sigma_{YX}\,\Sigma_{XX}^{-1}\Sigma_{XY}.
\]

We then fill \(\tilde Z_{\ell,Y} \leftarrow \mu_{Y\mid X}\) or sample
\(y\sim\mathcal{N}(\mu_{Y\mid X}, \tau^2\Sigma_{Y\mid X})\), invert the
projector to recover full \(Z_\ell\), merge per-level latents, and decode once
with the exact inverse \(f^{-1}\) to obtain an inpainted image \(\hat{x}\).
Per-voxel uncertainty maps follow from \(\operatorname{diag}\Sigma_{Y\mid X}\)
upsampled to image resolution. A natural schedule is coarse-to-fine
(\(\ell=L-1,\dots,0\)), so that global structure is imputed on coarser levels
and local texture is refined at finer levels.

## Energy-based latent posteriors for reconstruction

As an alternative to CGM, we can consider an energy-based posterior over latents
that enforces soft data consistency on the observed pixels without introducing
new architectural components. Starting from a standard Gaussian prior on \(z\),
we define

\[
\log p(z \mid x_{\text{obs}}) \;\propto\; -\tfrac{1}{2}\lVert z\rVert^2 
\;-\; \frac{1}{2\sigma^2}\,\big\lVert M\odot \big(f^{-1}(z)-x_{\text{obs}}\big)\big\rVert^2,
\]

where \(M\) is the observation mask and \(\sigma^2\) controls the strength of
the data-consistency term. We can approximate a MAP estimate by gradient descent
in \(z\) (or sample using Langevin dynamics or HMC), followed by a single decode
\(f^{-1}(z)\). This approach requires no extra networks or covariance
estimation, but it is iterative and may be more expensive than the closed-form
CGM for large numbers of masks or subjects. Comparing CGM and energy-based
inference in terms of reconstruction quality, speed, and stability would be a
natural follow-up study.

## Conditional Glow architectures for inpainting

A more architectural extension is conditional Glow for inpainting. Here we
introduce a context encoder \(g(x_{\mathrm{obs}},M)\) that processes the
observed pixels and mask, and condition the coupling networks on its features.
Training uses both a data-consistency term on the observed pixels and a
reconstruction term on the masked region. At test time, we can sample
\(z\sim\mathcal{N}(0,I)\) and decode conditioned on \(g(x_{\mathrm{obs}},M)\),
yielding diverse inpainted completions that respect both the global prior and
local context. This pushes the model towards fully conditional generation and
may improve fidelity or diversity at the cost of increased complexity and
training time. An open question is how much conditional capacity is actually
needed when a strong unconditional Glow backbone and CGM are already available.

## Scalable covariance modeling for CGM

To make CGM practical at scale and for arbitrary mask shapes, we must handle
covariance estimation and conditioning efficiently. Several strategies are
promising. First, local-window or block CGM would estimate
\((\mu_\ell,\Sigma_\ell)\) on sliding latent windows and compose conditionals
locally, blending overlapping predictions to approximate a global conditional at
reduced cost. Second, low-rank plus diagonal decompositions \(\Sigma \approx
UU^\top + \lambda I\) permit Woodbury-style updates for \(\Sigma_{XX}^{-1}\) and
fast Cholesky factorizations, reducing both memory and compute. Third, a
stationary Gaussian random field (GRF) / kriging view would estimate an
empirical kernel \(K_\ell(\Delta)\) over latent offsets and use kriging formulas
for \(Y\mid X\), so that storage scales with kernel parameters rather than full
covariance matrices. Systematic evaluation of these designs, including their
effect on inpainting quality and uncertainty, is an important line of work.

## Uncertainty calibration and evaluation

Because CGM naturally yields predictive variances for missing latents, it is
important to assess whether these uncertainties are well calibrated. Basic
diagnostics include empirical coverage—the frequency with which the true latent
or pixel values fall within \(\mu_{Y\mid X} \pm
q_\alpha\sqrt{\operatorname{diag}\Sigma_{Y\mid X}}\)—and the correlation between
squared error and predicted variance, computed inside the masked region. We can
also quantify boundary quality by reporting MSE or SSIM in a band of width \(b\)
around the mask edges, where seam artifacts are most likely to appear. These
tools would let us compare different CGM variants (e.g., with or without CCA,
different shrinkage strengths) not only on point estimates but also on the
reliability of their uncertainty maps.

## Computational efficiency and numerical stability

In practical settings, efficiency and numerics will be critical. One approach is
to group masks into a small number of batched patterns so that we can reuse
Cholesky factors of \(\Sigma_{XX}\) across subjects with similar missingness
patterns. For covariance operations, we can perform the forward pass with
automatic mixed precision while keeping Cholesky and linear solves in fp32,
adding automatic jitter (e.g., \(\epsilon I\)) when needed to stabilize
factorizations. Under low-rank covariance models, conditioning cost is dominated
by \(\mathcal{O}(rk^2)\) rather than \(\mathcal{O}(|X|^3)\), which favors
relatively small CCA ranks \(k\) and moderate low-rank dimensions \(r\). Careful
profiling and optimization of these components would be necessary to deploy
CGM-based inpainting on large 3-D volumes or cohort-scale datasets.

## Extensions beyond inpainting

The same machinery extends beyond single-image inpainting. CGM in latent space
can handle partial-view completion for slab or stack dropouts in 3-D MR, where
entire slices or chunks are missing. With multiple modalities, we can perform
cross-modal guided inpainting by including per-level latents from another
modality in the observed set \(X\), allowing, for example, T1 structure to guide
filling of T2 or FLAIR holes. More interactive applications are also possible:
user-drawn constraints (brush strokes, scribbles, or ROI averages) can be
treated as hard observations in \(X\), and CGM can propagate these constraints
through the latent space while preserving consistency with the learned prior.
Finally, training-time masking (e.g., CutOut-style random masks) can be used to
expose the model to inpainting-like regimes during training, potentially
tightening the Gaussian approximation and improving robustness of the
conditional statistics used at test time.

## Suggested ablations and experimental design

Several ablations follow directly from these proposals. For CGM, we can compare
inpainting quality and uncertainty with and without CCA, and across different
CCA ranks \(k\). Shrinkage parameters \(\lambda\) for covariance and jitter
schedules can be swept to understand stability and bias–variance trade-offs.
Multi-level conditioning schemes (coarse-to-fine vs single-shot conditioning at
a fixed level) can be evaluated to see how much global structure benefits from
hierarchical inference. The safe-band radius \(r_\ell\) can be tied to measured
receptive fields of the coupling networks, and we can study its effect on
boundary artifacts. Together, these experiments would clarify which components
of the CGM and inpainting pipeline contribute most to performance, and would
guide a practical design for clinical applications where missing data and
uncertainty-aware reconstruction are common.
