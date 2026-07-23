
\clearpage

# Discussion and Conclusions

## Principal findings

Collectively, this framework extends quantitative functional lung imaging beyond
predefined voxel classes and one-dimensional defect summaries. It provides an
invertible connection between population geometry and complete three-dimensional
images, thereby supporting multiscale comparison and generative interrogation
within a single probabilistic model. The current study is intended as a
methodological proof of concept as it characterizes the representation, identifies
the assumptions required for its interpretation, and establishes an experimental
basis for subsequent validation in larger and more heterogeneous cohorts.

This study investigated three-dimensional normalizing flows as a quantitative
framework for hyperpolarized xenon-129 ($^{129}$Xe) pulmonary ventilation MRI.
The central methodological premise was that conventional scalar measurements and
an invertible generative representation answer different questions. Ventilation
defect percentage (VDP) estimates the total burden of signal classified as
defective, whereas the normalizing flow retains a bijective correspondence
between the complete ventilation image and a multiscale latent coordinate. The
latter therefore provides access to spatial relationships between subjects,
resolution-dependent variation, and image-domain trajectories that cannot be
recovered from VDP alone.

The experiments established the technical feasibility of applying
exact-likelihood, multiscale flow modeling to sparse three-dimensional
functional lung images. Despite extensive background and highly quantized
intensities, probabilistic dequantization enabled continuous-density estimation,
and each ventilation volume could be represented across four latent resolution
levels and reconstructed through the analytic inverse. The resulting
representation supported pairwise comparison on the Gaussian typical set and
spherical interpolation between subjects. Importantly, diagnostic labels were
excluded from training. Clinical organization observed after encoding therefore
reflects information present in the images and captured by the likelihood-based
model rather than direct optimization of a classification objective.

Exploratory analyses suggested that the learned geometry contains
group-associated information. The most apparent relationships involved young
healthy volunteers versus disease groups and CF versus COPD, whereas CF versus
ILD and CF versus older healthy volunteers showed greater overlap. These
observations are consistent with the possibility that diseases producing similar
overall ventilation impairment may differ in their spatial expression.
Nevertheless, the current results should be interpreted as evidence of
representational feasibility rather than definitive disease discrimination.
Valid inference requires subject-level permutation analysis, and estimates from
this small cohort will retain substantial uncertainty even after the dependence
among pairwise distances is handled correctly.

## Relationship to conventional ventilation quantification

VDP remains an important and clinically useful measure. It is readily
interpretable, can be calculated using several established segmentation
procedures, and has demonstrated reproducibility and potential utility in
multicenter and interventional settings [@Couch:2019aa; @Svenningsen:2020aa].
The present framework is therefore not motivated by the claim that VDP is
intrinsically erroneous. Rather, VDP is intentionally reductive: it maps a
three-dimensional image to a single proportion. That reduction is advantageous
when a simple measure of defect burden is desired, but it is insufficient when
the number, location, topology, or regional organization of abnormalities is
relevant.

This distinction extends our previous comparison of histogram- and image-based
quantification [@Tustison:2021aa]. That work showed that optimization in the
image domain uses spatial context discarded by histogram-based approaches and
can improve measurement precision in the presence of common MRI perturbations.
However, even an image-based segmentation is typically summarized by VDP at the
final stage. The current work moves the point of comparison from segmentation
algorithm to representation: instead of asking how best to assign every voxel to
a predefined class, it asks whether the complete image can be embedded in a
probabilistic coordinate system without an irreversible intermediate labeling.

The two approaches should consequently be regarded as complementary. VDP
provides a concise measure with direct clinical meaning, whereas latent
distances provide a higher-dimensional description whose interpretation must be
established empirically. A critical next analysis within the current cohort is
to identify subjects with comparable VDP but divergent latent representations
and determine whether their images exhibit meaningful spatial differences.
Conversely, cases with different VDP but small latent distance will reveal
whether the learned geometry discounts changes in total burden that remain
clinically important. Such discordant pairs will be more informative than a
simple correlation between VDP and latent distance because they directly
characterize the information preserved or deemphasized by each representation.

## Interpretation of the multiscale latent space

The multiscale construction is a principal advantage of the Glow-derived
architecture [@kingma2018glow]. Factoring latent variables across successive
spatial resolutions creates several related coordinate systems while preserving
the invertibility of their combination. The preliminary group relationships
differed across $L_0$--$L_3$, suggesting that the levels are not redundant.
However, it would be premature to equate individual levels directly with
specific biological scales. A finer architectural resolution does not
necessarily encode only small ventilation defects, nor must a deeper level
exclusively describe global pulmonary organization. Coupling transformations
preceding each split permit information to be redistributed across channels and
levels.

Biological interpretation therefore requires intervention in the latent
representation followed by decoding. Level-restricted interpolation,
replacement, or perturbation while holding the remaining components fixed can
reveal the image features controlled by each component. Spatial-frequency
analysis, regional summaries, and connected-component characteristics of
low-ventilation regions can then quantify whether the decoded effects are local,
lobar, or global. This image-domain validation is essential if the multiscale
factorization is to support more than an architectural description.

Similar caution applies to the latent distance. A normalizing flow supplies a
differentiable bijection and tractable probability model [@dinh2016realnvp;
@papamakarios2021nfreview; @kobyzev2020nfsurvey], but it does not guarantee that
Euclidean or hyperspherical proximity corresponds to biological similarity. The
Gaussian base distribution constrains the aggregate latent density, whereas the
learned transformation determines how image variation is arranged within that
density. Spherical interpolation avoids trajectories through the low-probability
origin and respects the radial concentration of high-dimensional Gaussian
samples, but the resulting arc length remains model dependent. Its clinical
meaning must be supported through external variables, decoded trajectories,
stability across training seeds, and validation in independent data
[@white2016sampling; @arvanitidis2018latent].

## Statistical considerations

Pairwise distances create a nonstandard inferential setting because the
$45\times45$ distance matrix contains far more entries than independent
subjects. Distances that share a subject are correlated; treating them as
independent observations inflates the apparent sample size and can yield
severely anti-conservative $p$ values. For this reason, the preliminary Welch
tests applied directly to distance entries should not be used to support group
separation.

The appropriate exchangeable unit is the subject. PERMANOVA with subject-label
permutations provides an omnibus test of group-associated location differences
in the distance geometry, while PERMDISP evaluates whether apparent separation
may instead reflect unequal within-group dispersion. Both are needed because
disease cohorts may be intrinsically more heterogeneous than healthy cohorts.
Pairwise permutation tests can subsequently localize effects, with multiplicity
correction across clinical comparisons and resolution levels. Effect sizes and
uncertainty intervals should be emphasized over significance alone, particularly
in this exploratory cohort.

The statistical question should also remain distinct from classification. A
significant difference in group geometry does not imply perfect separation,
diagnostic accuracy, or clinical utility. Demonstrating those properties would
require a prespecified prediction task, subject-level cross-validation,
comparison with appropriate baselines, and external validation. The present
study instead evaluates whether an unsupervised image representation contains
group-associated structure, which is a necessary but not sufficient condition
for subsequent biomarker development.

## Limitations and future work

Several limitations remain. First, the cohort of 45 subjects is small relative
to the dimensionality and capacity of the three-dimensional flow. The model may
encode cohort-specific acquisition characteristics or sampling variation in
addition to pulmonary biology. Group sizes and demographic distributions may
also be unbalanced, and age is partly confounded with clinical grouping when
young and older healthy volunteers are treated separately. Larger independent
cohorts will be required to estimate disease heterogeneity, adjust for
demographic and technical covariates, and assess generalization across
acquisition protocols.

Second, the current model is monocontrast and uses only ventilation MRI. This
design isolates the functional signal and avoids dependence on a paired
structural acquisition, but it limits anatomical contextualization. Pulmonary
boundaries, body habitus, lung volume, positioning, and registration may
contribute to latent distance. Image-domain experiments and sensitivity analyses
using masks, alternative spatial normalization procedures, and nuisance
regression will be required to determine how much of the geometry is
attributable to ventilation distribution rather than global shape or
preprocessing.

Third, dequantization is necessary for continuous likelihood modeling but
introduces a tunable stochastic perturbation. The selected scale of $0.15$
should be justified through sensitivity analysis. Too little dequantization may
leave discrete intensity structure that is poorly matched to the density model,
whereas excessive noise may obscure small or low-contrast ventilation
abnormalities. Comparisons across dequantization levels should therefore include
likelihood, sampling behavior, latent-distance stability, and image-domain
preservation of clinically relevant structure [@ho2019flowpp].

Fourth, exact-likelihood image flows impose substantial memory and computational
costs because invertibility and Jacobian evaluation require retaining or
recomputing intermediate transformations. These demands currently constrain
spatial resolution and network capacity for three-dimensional experiments.
Resampling to $48\times32\times48$ voxels makes the present analysis tractable
but may attenuate small peripheral defects. Future work should examine
memory-efficient invertible architectures, patch- or region-based strategies,
and higher-resolution models while ensuring that changes in architecture do not
compromise the comparability of latent distances.

Fifth, likelihood and perceptual or clinical fidelity are not equivalent. A
model can assign favorable likelihood while producing samples or trajectories
that fail to preserve subtle disease features. Evaluation must therefore combine
likelihood with reconstruction checks, decoded interpolation experiments,
spatial measurements, expert review, and task-specific validation. Stability
across checkpoints and random seeds is particularly important because the
orientation and local arrangement of a Gaussian latent space are not uniquely
identifiable.

Finally, the present work does not establish superiority to VDP, clinical
diagnostic performance, or generalization to a multicenter population. Those
claims require direct empirical comparisons and substantially larger data. The
future analysis of large multicenter collections should be treated as a separate
validation stage rather than as evidence supporting the current cohort. Such
studies can determine whether the framework captures reproducible
disease-related structure after accounting for site, scanner, acquisition
protocol, and demographic variation.

## Software availability and reproducibility

The methodological contribution is accompanied by a mature, publicly available
implementation within the ANTsX/ANTsTorch ecosystem [@ANTsWebsite]. Open
implementation is especially important for a framework whose behavior depends on
preprocessing, dequantization, multiscale architecture, checkpoint selection,
and distance construction. Release of the training configuration, model weights,
exact latent-distance code, and subject-level statistical workflow will permit
others to reproduce the reported geometry and test alternative modeling
assumptions. Reproducibility should include not only the final checkpoint but
also the parameters required to regenerate preprocessing and evaluation outputs.

## Conclusions

Multiscale three-dimensional normalizing flows provide an invertible
probabilistic framework for quantitative analysis of hyperpolarized $^{129}$Xe
pulmonary ventilation MRI. By retaining a direct correspondence between complete
images and a structured Gaussian latent representation, the approach extends
functional lung quantification beyond predefined voxel classes and scalar defect
burden. The framework supports reconstruction, resolution-specific analysis,
subject-to-subject distance measurement, and image-domain interrogation of
latent trajectories without using diagnostic supervision during training.

In the present exploratory cohort, the learned geometry exhibited clinically
associated organization, but its statistical strength, scale-specific biological
meaning, and added value relative to VDP require the planned subject-level
permutation and image-domain analyses. Accordingly, the principal contribution
of this study is not a finalized diagnostic biomarker but a technically grounded
and reproducible framework for studying spatial variation in pulmonary
ventilation. With rigorous validation in larger independent cohorts, this
representation may complement conventional defect summaries and support more
detailed phenotyping of functional lung disease.
