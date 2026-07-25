
\clearpage

# Materials and methods

## Image acquisition

This retrospective analysis included 51 participants: young healthy volunteers
($n=4$), older healthy volunteers ($n=7$), and participants with cystic
fibrosis (CF; $n=14$), interstitial lung disease (ILD; $n=10$), or chronic
obstructive pulmonary disease (COPD; $n=10$). Hyperpolarized 129 Xe MRI was
performed under a protocol approved by the Institutional Review Board, with
written informed consent obtained from each participant. All imaging was
conducted under a physician-sponsored Investigational New Drug application
approved by the US Food and Drug Administration. MRI data were acquired on a
1.5-T whole-body scanner (Avanto; Siemens Medical Solutions, Malvern, PA, USA)
equipped with broadband capabilities and a flexible 129 Xe chest radiofrequency
coil (IGC Medical Advances, Milwaukee, WI, USA, or Clinical MR Solutions,
Brookfield, WI, USA). Participants inhaled approximately 1000 mL of
hyperpolarized 129 Xe mixed with nitrogen to a total volume equal to one-third
of their forced vital capacity (FVC). During a breath-hold of no more than 10 s,
15–17 contiguous coronal slices were acquired to cover the entire lungs.
Ventilation images were acquired using a gradient-recalled echo sequence with
spiral k-space sampling and 12 interleaves. Acquisition parameters were as
follows: repetition time/echo time, 7/1 ms; flip angle, 20$^\circ$; acquisition
matrix, $128 \times 128$; in-plane voxel size, $4 \times 4$ mm$^2$ ; slice
thickness, 15 mm; and interslice gap, 0 mm. All data were deidentified before
analysis. The deidentified data are available from the corresponding author upon
reasonable request and completion of an appropriate data-sharing agreement.

## Image preprocessing

Ventilation images were processed using ANTsX [@Tustison:2021aa]. Each image was
first resampled by linear interpolation to a voxel spacing of \(4 \times 4
\times 16\) mm\(^3\) and centrally padded or cropped to a matrix of \(128 \times
128 \times 16\) voxels. Through-plane resolution was then increased fourfold
using the pretrained three-dimensional MRI super-resolution model implemented in
ANTsPyNet [@avants2023superresolution], yielding a nominal voxel spacing of \(4
\times 4 \times 4\) mm\(^3\). To place all ventilation images in a common
spatial coordinate system, a rigid transformation was estimated between the lung
image and a designated lung template. Resampling into the template domain
produced the final common matrix of \(88 \times 72 \times 128\) voxels (voxel
resolution = $3.9 \times 3.9 \times 3.9$ mm$^3$).  No explicit lung mask was
applied to the ventilation images, thereby retaining both the pulmonary signal
and the surrounding background within the common image domain.

## Normalizing flows

Let $\mathbf{x}\in\mathbb{R}^{D}$ denote a vectorized, dequantized ventilation
volume and let $f_{\theta}$ denote an invertible transformation parameterized by
$\theta$. The model maps the image to a latent variable

$$
\mathbf{z}=f_{\theta}(\mathbf{x}),
$$

with inverse reconstruction

$$
\mathbf{x}=f_{\theta}^{-1}(\mathbf{z}).
$$

A standard multivariate Gaussian distribution was used as the base density,
$p_Z(\mathbf{z})=\mathcal{N}(\mathbf{0},\mathbf{I})$. Through the
change-of-variables formula, the image log likelihood was

$$
\log p_X(\mathbf{x}) = \log p_Z\!\left(f_{\theta}(\mathbf{x})\right)
+ \log\left|\det\frac{\partial f_{\theta}(\mathbf{x})}{\partial\mathbf{x}}\right|.
$$

The bijective construction permits direct likelihood evaluation and
reconstruction of each image from its latent representation [@dinh2016realnvp;
@papamakarios2021nfreview; @kobyzev2020nfsurvey].

### Multiscale 3D Glow network 

The model extended the Glow architecture to three spatial dimensions
[@kingma2018glow]. Each flow step comprised activation normalization, an
invertible $1\times1\times1$ convolution, and an affine coupling transformation.
Within the coupling layer, the input channels were partitioned into two
components. One component was passed unchanged, whereas a three-dimensional
convolutional subnetwork predicted the scale and translation applied to the
second component. This triangular construction provided an expressive nonlinear
transformation while retaining an efficiently computable Jacobian determinant
and analytic inverse. The model comprised four multiscale resolution levels,
with 32 invertible flow steps at each level. The convolutional subnetworks
within the affine coupling transformations contained 96 hidden channels.
Coupling layer scale outputs were transformed using a hyperbolic tangent mapping
and bounded by a scale cap of 1.5 to improve numerical stability. Channelwise
splitting was used for the multiscale factorization. A learned Glow base
distribution was employed, with its log-scale parameters constrained to the
interval $[-1,1]$. 

### Model training and optimization

Model parameters were estimated by unsupervised maximum-likelihood training.
The objective was the negative log likelihood, reported as bits per dimension,

$$
\operatorname{bpd}(\mathbf{x}) =
-\frac{\log p_X(\mathbf{x})}{D\log 2},
$$

where \(D\) denotes the number of voxels in the model input. Diagnostic group
labels were not used for data partitioning, augmentation, optimization, or
checkpoint selection.

Participants were randomly assigned at the subject level to training and
validation subsets using a validation fraction of 0.10 and random seed 0. The
split was not stratified by diagnostic group. All augmented samples derived
from a given participant remained within the same subset. The training dataset
generated 3,000 augmented samples per sampling cycle, whereas validation was
performed using eight nonaugmented samples.

Training-time augmentation comprised additive Gaussian intensity noise,
affine perturbations, spatial deformation, simulated intensity-bias fields,
and histogram warping. Augmentation strengths were progressively reduced over
the 150,000 training iterations. The standard deviation of the additive noise
decreased according to a cosine schedule from 0.02 to 0.001, and the affine
perturbation scale decreased from 0.05 to 0.01. The deformation parameter
decreased linearly from 12.0 to 10.0. Simulated bias field and histogram warping
parameters decreased according to cosine schedules from 0.20 to 0 and from
0.04 to 0, respectively. Spatial and intensity augmentation was disabled during
validation.

The model was trained for 150000 iterations in 32-bit floating-point precision
using the Adamax optimizer. The learning rate was linearly increased during the
first 5000 iterations to approximately $5 \times 10^{-5}$ and remained effectively
constant thereafter. No weight decay was applied. A minibatch size of seven
volumes and five-step gradient accumulation yielded an effective batch size of
35 volumes per parameter update. The global gradient norm was clipped at 0.2.
An exponential moving average of the model parameters was maintained with a
decay coefficient of 0.9997 and was used for validation and subsequent model
evaluation. Validation likelihood was evaluated every 1000 iterations using the
nonaugmented validation data. Training states containing both the directly
optimized and exponentially averaged model parameters were saved at each
evaluation. The final exponentially averaged model obtained after 150,000
iterations was used for latent encoding and subsequent statistical analysis.


## Multiscale latent geometry

### Empirical typical-set representation

For a high-dimensional Gaussian distribution, probability mass concentrates
within a relatively thin annular region rather than near the density mode
[@vershynin2018high; @blum2020foundations]. Consequently, the latent origin,
more generally the latent mean, does not represent a typical encoded image and
was not interpreted as a population-normality reference. Instead, the analysis
characterized directional relationships among subjects after projection onto an
empirically estimated typical-radius hypersphere.

For subject \(i\) and multiscale level \(l\), the flattened latent variable
\(\mathbf z_{il}\in\mathbb R^{d_l}\) was first centered using the empirical
level-specific latent mean,

$$
\mathbf a_{il}
=
\mathbf z_{il}-\boldsymbol{\mu}_l,
$$

where

$$
\boldsymbol{\mu}_l
=
\frac{1}{N}\sum_{i=1}^{N}\mathbf z_{il}.
$$

The representative radius at level \(l\) was defined as the median Euclidean
norm of the centered subject representations,

$$
r_l
=
\operatorname{median}_{i}
\left\lVert\mathbf a_{il}\right\rVert_2.
$$

Each centered representation was then projected onto the hypersphere of radius
\(r_l\):

$$
\widetilde{\mathbf a}_{il}
=
r_l
\frac{\mathbf a_{il}}
{\left\lVert\mathbf a_{il}\right\rVert_2}.
$$

The median radius provides a robust empirical estimate of the typical latent
scale at each resolution level. This projection retains the direction of each
centered subject representation while removing intersubject radial variation
from the geometric analysis.

### Pairwise distances and spherical interpolation

For subjects \(i\) and \(j\), the angular separation between their projected
representations at level \(l\) was

$$
\theta_{ijl}
=
\arccos\!\left[
\frac{
\widetilde{\mathbf a}_{il}^{\mathsf T}
\widetilde{\mathbf a}_{jl}
}{
\left\lVert\widetilde{\mathbf a}_{il}\right\rVert_2
\left\lVert\widetilde{\mathbf a}_{jl}\right\rVert_2
}
\right].
$$

The corresponding hyperspherical arc length was

$$
d_{ijl}=r_l\theta_{ijl}.
$$

Spherical linear interpolation between the projected representations was
defined for \(t\in[0,1]\) as

$$
\widetilde{\mathbf a}_{ijl}(t)
=
\frac{\sin[(1-t)\theta_{ijl}]}
{\sin(\theta_{ijl})}
\widetilde{\mathbf a}_{il}
+
\frac{\sin(t\theta_{ijl})}
{\sin(\theta_{ijl})}
\widetilde{\mathbf a}_{jl}.
$$

The interpolated representation was returned to the original latent coordinate
system by restoring the level-specific empirical mean,

$$
\mathbf z_{ijl}(t)
=
\boldsymbol{\mu}_l
+
\widetilde{\mathbf a}_{ijl}(t).
$$

For whole-image trajectories, interpolation was performed independently at all
four multiscale levels using the same value of \(t\). The resulting latent
variables were then jointly decoded through \(f_\theta^{-1}\) to visualize the
corresponding image-domain trajectory. Distances were evaluated separately at
levels \(L_0\)--\(L_3\). The combined multiscale distance was defined using the
product-space metric

$$
d_{ij}^{\mathrm{multi}}
=
\left[
\sum_{l=0}^{3}d_{ijl}^{\,2}
\right]^{1/2}.
$$

## Evaluation of clinical organization

Clinical labels were withheld during model training and used only afterward to
assess whether the unsupervised latent representation exhibited clinically
relevant organization. For each of the four resolution levels and the combined
multiscale representation, a complete $45 \times 45$ subject-distance matrix
was constructed. The analysis evaluated global differences in latent geometry
among the five clinical groups, followed by exploratory localization of these
differences to specific group pairs.

An omnibus permutational multivariate analysis of variance (PERMANOVA) was
performed separately for each of the five distance matrices. Statistical
significance was assessed by permuting the clinical labels among participants
while holding the distance matrix fixed. Because each participant contributed
to multiple pairwise distances, the participant—not an individual
distance—was treated as the exchangeable unit. For each analysis, the
pseudo-$F$ statistic and the proportion of distance variation attributable to
clinical group ($R^2$) were reported. Raw permutation $p$ values were adjusted
across the five omnibus tests using the Holm procedure.

Post-hoc analyses considered the ten unique pairs formed by the five clinical
groups at each of the four resolution levels and for the combined multiscale
distance, yielding 50 planned comparisons. For groups $A$ and $B$, the effect
size was defined as

$$
\Delta_{AB}
=
\overline{d}(A,B)
-
\frac{
\overline{d}(A,A)+\overline{d}(B,B)
}{2},
$$

where $\overline{d}(A,B)$ is the mean distance between participants from
different groups and $\overline{d}(A,A)$ and $\overline{d}(B,B)$ are the
corresponding mean within-group distances. Each unique within-group subject
pair was included once. Positive values of $\Delta_{AB}$ indicate that the
mean between-group distance exceeded the average of the two within-group
distances.

The null distribution of $\Delta_{AB}$ was obtained by permuting group labels
among the participants included in each comparison while preserving the
observed group sizes. When no more than 100,000 unique label assignments were
possible, all assignments were enumerated to obtain an exact permutation test;
otherwise, 100,000 Monte Carlo permutations were performed. One-sided $p$
values quantified evidence that the between-group distance exceeded the
average within-group distance. These values were adjusted jointly across the
50 post-hoc comparisons using the Holm procedure. The observed contrast,
within-group and between-group distance summaries, raw permutation $p$ value,
and Holm-adjusted $p$ value were reported for each comparison.

Because PERMANOVA can be sensitive to unequal within-group dispersion,
permutational analysis of multivariate dispersion (PERMDISP) was performed
separately for each of the five distance matrices. PERMDISP compares the
distances of individual participants from their respective group centroids,
thereby evaluating whether the clinical groups differ in their internal
dispersion. Statistical significance will be assessed by permuting clinical
labels at the participant level, and the resulting $p$ values will be adjusted
across the five tests using the Holm procedure. A significant PERMANOVA
accompanied by a nonsignificant PERMDISP is consistent with differences
in group location rather than detectable differences in within-group
dispersion. If both tests are significant, the PERMANOVA result cannot be
attributed unambiguously to group-location differences alone.


## Software and reproducibility

The three-dimensional flow model, preprocessing, latent encoding, image
reconstruction, and trajectory analysis were implemented within the
ANTsX/ANTsTorch software ecosystem. The complete framework was designed to use
publicly available, scriptable components to facilitate reproduction and
extension of the experiments [@ANTsWebsite].

**TODO:** Add the precise repository URL, release or commit identifier,
command-line invocation, configuration file, dependency versions, trained model
availability, and data-access statement. If the analysis code and model weights
will be released only upon publication, state that explicitly.
