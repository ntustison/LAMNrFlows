

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
2D/3D images and RealNVP-style flows for IDPs and other tabular blocks. The
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
(histogram warping and bias field simulation [@Tustison:2021aa]), and additive
Gaussian noise treated as dequantization rather than biological variability. We
control the overall augmentation strength by a scalar schedule \(\alpha(t) \in
[0,1]\) as a function of normalized training time \(t\), and support linear,
cosine, and exponential decay: for example, a linear schedule reduces
augmentation proportionally to \(t\), a cosine schedule keeps stronger
perturbations early and then decays smoothly, and an exponential schedule
reduces aggressive warps and noise most rapidly at the beginning of training.
This allows us to start with heavier augmentations to regularize the flows and
discourage overfitting to discrete templates, then gradually emphasize fidelity
to the true data distribution as training progresses. For validation, we use the
same spatial transforms but disable additional noise and histogram warping. This
design preserves anatomical variability while preventing overfitting to
discrete, noise-free templates that would otherwise cause flows to collapse onto
spiky background modes.  

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
maintain a cross-microbatch queue to approximate large-batch behavior. 

The same training loop supports single and multiple views by instantiating one
flow per view, computing per-view log-likelihoods and latents for each
minibatch, flattening the multiscale latents, and applying the projector plus
alignment and screening logic described above. For IDP/tabular experiments, we
replace the image Glow architecture with RealNVP while keeping the multiview
alignment and conditional Gaussian stages unchanged, which allows LAMNr Flows to
operate uniformly across image contrasts and multiview IDP blocks within the
same methodological framework.

\begin{figure*}[!htb]
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

To ensure stable density estimation and prevent degenerate likelihoods due to
data quantization, we also employ uniform dequantization (jittering) during training,
following the variational framework established in Flow++ [@ho2019flowpp].
We apply lightweight, label-free augmentations during maximum-likelihood
training to improve robustness without changing the model’s exact likelihood
computation (augmentations act on inputs only)
[@Tustison:2024aa;@Tustison:2025aa] (cf Figure \ref{fig:aug-schedule}). For
image views (2D/3D), we use geometric transforms (linear and non-linear
transformations) shared across all views of a subject to preserve alignment
targets, and per-view intensity-based transforms (noise, simulated bias-field,
histogram warping).  Similar to the tabular case, the amplitude is controlled by
a scalar schedule \(\alpha(t)\) (linear, cosine, or exponential in training
time).

