
\clearpage

# Results

## Cohort characteristics

The exploratory cohort comprised 45 participants distributed across five
clinical groups: young healthy volunteers, older healthy volunteers, cystic
fibrosis (CF), chronic obstructive pulmonary disease (COPD), and interstitial
lung disease (ILD). All participants contributed a three-dimensional
hyperpolarized xenon-129 ventilation image. Diagnostic labels were withheld from
model training and were introduced only after latent encoding for evaluation of
the learned representation.

**TODO:** Add a cohort table reporting the number of participants, age, sex,
pulmonary-function measurements, ventilation defect percentage (VDP), and
relevant clinical characteristics for each group. Report missing data explicitly
and provide the corresponding omnibus and pairwise demographic comparisons.

## Model optimization and reconstruction

The three-dimensional multiscale normalizing flow was trained by
exact-likelihood optimization after spatial resampling and probabilistic
dequantization of the ventilation volumes. Training remained numerically stable,
and all 45 images were successfully mapped to the four-level latent
representation and reconstructed through the analytic inverse. No diagnostic
information was used to construct the latent space.

**TODO:** Report the selected checkpoint, training and validation bits per
dimension, convergence behavior, and the criterion used for checkpoint
selection. Add quantitative reconstruction error, expected to be near numerical
precision for the dequantized model input, and distinguish this invertibility
check from reconstruction of the original pre-dequantization image. Include a
training-curve figure if it contributes information beyond the reported values.

Visual inspection indicated that reconstructed volumes retained the pulmonary
signal distribution and subject-specific ventilation abnormalities present in
their corresponding inputs. Because exact inversion alone does not establish
that samples or interpolations occupy well-supported regions of the learned
distribution, reconstruction fidelity was considered a verification of model
implementation rather than evidence of generative validity.

**TODO:** Add representative input/reconstruction pairs from each clinical group
using identical display windows and anatomical planes. If reconstruction
differences are imperceptible, report a difference image with an appropriately
amplified and explicitly labeled intensity scale.

## Organization of the multiscale latent representation

Encoding produced four latent components, $\mathbf{z}_0$ through $\mathbf{z}_3$,
corresponding to the successive levels of the multiscale architecture. The
complete representation retained a bijective correspondence with the input
image, whereas the level-specific components provided complementary descriptions
of variation within the cohort. After radial projection, encoded subjects
occupied the hyperspherical Gaussian typical set and could be compared without
treating the low-probability latent origin as a representative population image.

Pairwise distances varied across resolution levels, demonstrating that subject
relationships were not invariant to the level at which the representation was
examined. Fine and coarse latent components produced different relative
organizations of the cohort. The observed group relationships were consistent
with the multiscale representation capturing complementary aspects of
ventilation variation; however, architectural scale alone was insufficient to
assign a specific biological interpretation to an individual level.

**TODO:** Report, for each level, the latent dimensionality, empirical radius
before projection, radial dispersion, and distribution of pairwise distances.
Add a matrix or heat map showing the subject-by-subject distances ordered by
clinical group. Quantify the association between the level-specific distance
matrices using Mantel or rank correlations with subject-label permutations.
These results will establish whether the levels contain complementary
information rather than relying on visual interpretation.

## Image-domain interpretation of latent scale

Spherical interpolation provided continuous trajectories between subject
representations while remaining on a common-radius hypersphere. Decoding
intermediate points generated a corresponding sequence of three-dimensional
ventilation images. These trajectories demonstrated the operational advantage of
the invertible formulation: latent differences could be returned directly to the
image domain rather than interpreted solely through a two-dimensional embedding.

**TODO:** Select prespecified subject pairs representing (1) two participants
within the same group, (2) a healthy-to-disease comparison, and (3) two disease
groups with similar VDP. Show equally spaced Slerp points using fixed display
parameters. To support claims that $L_0/L_1$ represent localized abnormalities
and $L_3$ represents global organization, perform level-restricted latent
replacement or interpolation while holding the other levels fixed. Report
quantitative image changes at each level, such as spatial-frequency content,
connected-component characteristics of low-ventilation regions, or regional
displacement of ventilation signal.

## Clinical organization of latent distances

Although clinical labels were absent during optimization, the distance matrices
displayed group-associated structure after training. The strongest exploratory
relationships involved comparisons between young healthy volunteers and disease
groups and between the two obstructive disease groups, CF and COPD. Comparisons
involving CF versus ILD and CF versus older healthy volunteers showed greater
overlap. These observations suggest that the learned representation contains
clinically relevant information beyond an undifferentiated measure of overall
ventilation loss, while also revealing boundaries that remain ambiguous in the
small exploratory cohort.

The previously calculated pairwise Welch tests yielded very small nominal $p$
values for several comparisons. These values were not retained because distances
sharing a subject are statistically dependent and cannot be treated as
independent observations. Inferential results will instead be based on
permutation of labels at the subject level.

**TODO:** Replace this paragraph with the final statistical results. For each of
$L_0$--$L_3$ and the total distance, report:

- omnibus PERMANOVA pseudo-$F$, variance explained ($R^2$), and
  permutation-derived $p$ value;
- PERMDISP statistic and permutation-derived $p$ value;
- pairwise PERMANOVA effect sizes and multiplicity-adjusted $p$ values;
- within-group and between-group distance summaries with confidence intervals;
  and
- sensitivity analyses demonstrating that significant location effects are not
  explained solely by unequal group dispersion.

With five groups, ten pairwise group comparisons were possible at each of the
five distance definitions, yielding 50 planned tests and a Bonferroni
family-wise threshold of $1.0\times10^{-3}$. Exact permutation counts and
attainable $p$-value resolution should accompany the final results. A compact
table should report all effect sizes and corrected $p$ values, whereas the main
text should emphasize only the comparisons that address the principal
methodological hypotheses.

## Comparison with ventilation defect percentage

The normalizing-flow representation and VDP encode fundamentally different
quantities. VDP assigns each subject a single defect-burden value, whereas the
flow produces an invertible multiscale representation from which spatially
informed subject distances can be derived. The key empirical question is
therefore not whether the latent distance reproduces VDP, but whether it
distinguishes images that have comparable VDP yet different spatial
organizations of ventilation abnormality.

**TODO:** Add the subject-level VDP values and perform the following
prespecified analyses:

1. quantify associations between VDP differences and latent distances at each
   level using subject-label or matrix-based permutation testing;
2. identify subject pairs with similar VDP but large latent distance and show
   their ventilation images;
3. identify pairs with different VDP but relatively small latent distance to
   characterize potential failure modes; and
4. compare the ability of VDP and latent-distance features to recover clinical
   group structure using cross-validated or permutation-based metrics
   appropriate for the small sample.

These analyses are required to support the central claim that the flow retains
clinically relevant spatial information discarded by the scalar summary. Without
them, the comparison between VDP and the latent representation should remain
conceptual rather than be described as demonstrated superiority.

## Summary of findings

The experiments established the technical feasibility of exact-likelihood,
invertible modeling of three-dimensional hyperpolarized $^{129}$Xe ventilation
MRI in a sparse functional-imaging setting. The learned multiscale
representation supported reconstruction, level-specific subject comparison, and
image-domain interpolation without diagnostic supervision. Exploratory
organization of the latent distances was associated with clinical phenotype,
particularly for young healthy versus diseased ventilation and for CF versus
COPD, while other group boundaries remained less distinct. Final claims
concerning statistical group separation, scale-specific biological
interpretation, and added value relative to VDP await the subject-level
permutation and image-domain analyses specified above.
