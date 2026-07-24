
\clearpage

# Materials and methods

## Image acquisition

This retrospective analysis included 51 participants: young healthy volunteers
($n=10$), older healthy volunteers ($n=7$), and participants with cystic
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

## Three-dimensional multiscale normalizing flow

### Invertible transformation

Let $\mathbf{x}\in\mathbb{R}^{D}$ denote a vectorized, dequantized ventilation volume and let $f_{\theta}$ denote an invertible transformation parameterized by $\theta$. The model maps the image to a latent variable

$$
\mathbf{z}=f_{\theta}(\mathbf{x}),
$$

with inverse reconstruction

$$
\mathbf{x}=f_{\theta}^{-1}(\mathbf{z}).
$$

A standard multivariate Gaussian distribution was used as the base density, $p_Z(\mathbf{z})=\mathcal{N}(\mathbf{0},\mathbf{I})$. Through the change-of-variables formula, the image log likelihood was

$$
\log p_X(\mathbf{x}) = \log p_Z\!\left(f_{\theta}(\mathbf{x})\right)
+ \log\left|\det\frac{\partial f_{\theta}(\mathbf{x})}{\partial\mathbf{x}}\right|.
$$

The bijective construction permits direct likelihood evaluation and reconstruction of each image from its latent representation [@dinh2016realnvp; @papamakarios2021nfreview; @kobyzev2020nfsurvey].

### Flow blocks and multiscale factorization

The model extended the Glow architecture to three spatial dimensions [@kingma2018glow]. Each flow step comprised activation normalization, an invertible $1\times1\times1$ convolution, and an affine coupling transformation. Within the coupling layer, the input channels were partitioned into two components. One component was passed unchanged, whereas a three-dimensional convolutional subnetwork predicted the scale and translation applied to the second component. This triangular construction provided an expressive nonlinear transformation while retaining an efficiently computable Jacobian determinant and analytic inverse.

The architecture used four resolution levels, denoted $L_0$ through $L_3$. At each level, a three-dimensional squeeze operation exchanged spatial resolution for channels, followed by a sequence of invertible flow steps. A portion of the channels was factored from the transformation at successive levels, producing the multiscale latent representation

$$
\mathbf{z}=\left(\mathbf{z}_0,\mathbf{z}_1,\mathbf{z}_2,\mathbf{z}_3\right).
$$

This factorization allowed the complete image likelihood and reconstruction to be retained while enabling separate analysis of the latent variables associated with individual resolution levels. Because the current study used a single ventilation image per participant, no inter-view alignment loss or multimodal conditioning term was included.

**TODO:** Supply the number of flow steps per level, coupling-network hidden channels, scale-clamping rule, activation functions, parameter initialization, base-distribution parameterization, and total number of trainable parameters. Confirm whether all four levels used split operations and specify the exact dimensionality of each $\mathbf{z}_l$.

## Model optimization and selection

Parameters were optimized by minimizing the negative log likelihood, reported in bits per dimension:

$$
\operatorname{bpd}(\mathbf{x}) =
-\frac{\log p_X(\mathbf{x})}{D\log 2}.
$$

Training was entirely unsupervised with respect to the five clinical groups. Accordingly, neither the diagnostic labels nor any loss designed to increase between-group separation contributed to optimization. The selected checkpoint was determined from likelihood performance on data excluded from gradient-based training.

**TODO:** Specify the training, validation, and test allocation; whether splitting was stratified by group; random seed; optimizer; learning rate and schedule; warm-up duration; batch size and gradient accumulation; number of iterations or epochs; gradient clipping; exponential moving average; checkpoint-selection criterion; numerical precision; GPU model and number of GPUs; software versions; and approximate training time. Given the small cohort, also clarify whether a single split, repeated splits, or cross-validation was used.

## Multiscale latent geometry

### Typical-set representation

For a high-dimensional standard Gaussian distribution, typical samples concentrate within a relatively thin annular region rather than near the origin. The origin is therefore the mode of the density but is not representative of a typical encoded image. To avoid interpreting distance from $\mathbf{z}=\mathbf{0}$ as a population-normality score, the present analysis focused on relationships between subjects within the Gaussian typical set.

For subject $i$ and level $l$, the flattened latent variable $\mathbf{z}_{il}$ was projected onto a hypersphere of common radius $r_l$:

$$
\widetilde{\mathbf{z}}_{il}
= r_l\frac{\mathbf{z}_{il}}{\lVert\mathbf{z}_{il}\rVert_2}.
$$

**TODO:** State how $r_l$ was selected---for example, the theoretical Gaussian typical radius $\sqrt{d_l}$, the empirical mean radius, or another estimate---and whether centering or covariance normalization preceded radial projection.

### Pairwise distances and spherical interpolation

For subjects $i$ and $j$, the angular separation at level $l$ was

$$
\theta_{ijl}=\arccos\!\left(
\frac{\widetilde{\mathbf{z}}_{il}^{\mathsf{T}}
\widetilde{\mathbf{z}}_{jl}}
{\lVert\widetilde{\mathbf{z}}_{il}\rVert_2
\lVert\widetilde{\mathbf{z}}_{jl}\rVert_2}
\right).
$$

The corresponding hyperspherical arc length was $d_{ijl}=r_l\theta_{ijl}$. Spherical linear interpolation (Slerp) between the two latent representations was defined for $t\in[0,1]$ as

$$
\operatorname{Slerp}\!\left(
\widetilde{\mathbf{z}}_{il},
\widetilde{\mathbf{z}}_{jl};t
\right)
=
\frac{\sin((1-t)\theta_{ijl})}{\sin(\theta_{ijl})}
\widetilde{\mathbf{z}}_{il}
+
\frac{\sin(t\theta_{ijl})}{\sin(\theta_{ijl})}
\widetilde{\mathbf{z}}_{jl}.
$$

Intermediate latent variables could be decoded through $f_{\theta}^{-1}$ to visualize image-domain changes along the trajectory. Distances were calculated independently for $L_0$--$L_3$. A total multiscale distance summarized the level-specific distances for each subject pair.

**TODO:** Verify that the implemented distance is the hyperspherical arc length described above rather than the chord distance, an unscaled angular distance, or the numerical length of sampled Slerp points. Provide the exact formula used to combine $d_{ij0},\ldots,d_{ij3}$ into the total distance, including any normalization for the unequal latent dimensionalities.

## Evaluation of clinical organization

The clinical groups were used after training to evaluate whether the unsupervised latent representation contained clinically relevant organization. For each resolution level and for the combined representation, the complete $45\times45$ subject-distance matrix was constructed. Evaluation concerned two related properties: whether group membership was associated with systematic location differences in the latent geometry and whether the groups differed in within-group dispersion.

An omnibus permutational multivariate analysis of variance (PERMANOVA) was planned for each of the five distance matrices. Statistical significance was estimated by permuting the clinical labels at the subject level while leaving the distance matrix fixed. Because each subject contributes to multiple pairwise distances, the subject rather than the individual distance was the exchangeable unit. Permutational analysis of multivariate dispersion (PERMDISP) was used as a complementary analysis to determine whether a significant PERMANOVA result could be attributable to unequal group dispersion rather than differences in group location.

Following the omnibus analysis, pairwise comparisons were performed for the ten unique pairs formed by the five clinical groups. Applying these comparisons to the four resolution levels and the total distance yielded 50 planned pairwise tests. Family-wise error was controlled using Bonferroni correction, giving $\alpha_{\mathrm{adj}}=0.05/50=1.0\times10^{-3}$. For each comparison, an effect-size estimate and the groupwise within- and between-subject distance summaries were reported together with the permutation-derived $p$ value.

**TODO:** Implement or confirm the subject-level PERMANOVA and PERMDISP analyses; specify the software, pseudo-$F$ statistic, number of permutations, restricted-permutation scheme if applicable, effect-size definition, and confidence-interval procedure. The earlier Welch tests applied to individual pairwise distances should not be reported because pairwise distances sharing subjects are statistically dependent.

## Software and reproducibility

The three-dimensional flow model, preprocessing, latent encoding, image reconstruction, and trajectory analysis were implemented within the ANTsX/ANTsTorch software ecosystem. The complete framework was designed to use publicly available, scriptable components to facilitate reproduction and extension of the experiments [@ANTsWebsite].

**TODO:** Add the precise repository URL, release or commit identifier, command-line invocation, configuration file, dependency versions, trained model availability, and data-access statement. If the analysis code and model weights will be released only upon publication, state that explicitly.
