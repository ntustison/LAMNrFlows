\clearpage

# Results

## Study cohort

The exploratory cohort comprised 45 participants: 14 with cystic fibrosis
(CF), 10 with chronic obstructive pulmonary disease (COPD), 10 with
interstitial lung disease (ILD), seven older healthy participants, and four
young healthy participants. Each participant contributed one three-dimensional
hyperpolarized $^{129}$Xe ventilation image. Clinical labels were withheld
during model training and were used only for post-training evaluation of the
latent representation.

**TODO:** Add a cohort table reporting age, sex, pulmonary-function
measurements, ventilation defect percentage (VDP), and relevant clinical
characteristics by group. Report missing observations and the corresponding
demographic comparisons.

## Model fitting and invertible representation

The ventilation volumes were spatially resampled, probabilistically
dequantized, and used to train a three-dimensional multiscale normalizing flow
by exact-likelihood optimization. All 45 images were successfully encoded into
the four-level latent representation, comprising $\mathbf{z}_0$ through
$\mathbf{z}_3$, and contributed to the subsequent distance analyses. Because
the model is invertible by construction, the complete latent representation
retained a one-to-one correspondence with the dequantized model input. No
diagnostic or clinical information contributed to optimization.

**TODO:** Report the selected checkpoint, training and validation bits per
dimension, convergence behavior, and checkpoint-selection criterion. Quantify
the numerical inversion error for the dequantized input and distinguish this
implementation check from agreement with the original image before
dequantization. If representative reconstructions are shown, use identical
anatomical planes and display windows and include amplified difference images
when the errors are not visible at the native intensity scale.

## Multiscale latent geometry

For each resolution level, subject representations were radially projected to a
common-radius hypersphere within the Gaussian typical set. This construction
removed variation in latent radius and avoided using the low-probability
Gaussian origin as a reference representation. Pairwise geodesic distances were
then calculated separately at $L_0$--$L_3$. The combined multiscale distance was
defined by the $L_2$ norm across the four level-specific distances. Each
distance definition produced a complete $45 \times 45$ subject-distance
matrix.

The five matrices were Euclidean within numerical precision: principal
coordinate analysis yielded 44 positive axes and no negative axes for every
matrix. This permitted direct use of the distance matrices in the subsequent
PERMANOVA and PERMDISP analyses without correction for negative eigenvalues.
The presence of significant group-associated structure at every level, as
described below, indicated that the clinical signal was distributed across the
multiscale representation rather than restricted to one latent component.
However, these results alone do not establish a distinct biological
interpretation for any individual architectural level.

**TODO:** Report the dimensionality and chosen hyperspherical radius for each
level, the empirical latent-radius distribution before projection, and the
distribution of pairwise distances. Add subject-by-subject distance heat maps
ordered by clinical group. Quantify agreement among the level-specific matrices
using an appropriate matrix-correlation analysis before describing the levels
as complementary or nonredundant.

## Clinical-group organization

As a proof-of-concept analysis, we evaluated whether clinical group was
associated with the latent-distance structure observed in this small
exploratory cohort. Omnibus PERMANOVA identified group-associated organization
at each latent level and for the combined multiscale distance (Table 1).
Clinical group accounted for 10.7%--12.2% of the distance variation. The
largest proportion was observed at $L_0$ (pseudo-$F=1.390$, $R^2=0.122$,
permutation $p=0.0042$, FDR-adjusted $q=0.0064$). Significant effects were
also detected at $L_1$ (pseudo-$F=1.266$, $R^2=0.112$, $p=0.0020$,
$q=0.0064$), $L_2$ (pseudo-$F=1.194$, $R^2=0.107$, $p=0.0117$,
$q=0.0117$), and $L_3$ (pseudo-$F=1.249$, $R^2=0.111$, $p=0.0051$,
$q=0.0064$). The combined distance yielded a similar result
(pseudo-$F=1.244$, $R^2=0.111$, $p=0.0034$, $q=0.0064$). The consistency
of the omnibus effect across representations suggested that group-associated
organization was distributed throughout the multiscale geometry rather than
confined to a single resolution level.

**Table 1. Omnibus clinical-group analyses of the latent-distance matrices.**

| Distance | PERMANOVA pseudo-$F$ | $R^2$ | PERMANOVA $p$ | PERMANOVA FDR $q$ | PERMDISP $F$ | PERMDISP $p$ | PERMDISP FDR $q$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| $L_0$ | 1.390 | 0.122 | 0.0042 | 0.0064 | 1.025 | 0.4077 | 0.4077 |
| $L_1$ | 1.266 | 0.112 | 0.0020 | 0.0064 | 1.806 | 0.1389 | 0.3587 |
| $L_2$ | 1.194 | 0.107 | 0.0117 | 0.0117 | 1.624 | 0.1770 | 0.3587 |
| $L_3$ | 1.249 | 0.111 | 0.0051 | 0.0064 | 1.544 | 0.2152 | 0.3587 |
| Combined | 1.244 | 0.111 | 0.0034 | 0.0064 | 1.281 | 0.2943 | 0.3679 |

PERMDISP detected no significant difference in within-group dispersion for any
of the five distance matrices. Using bias-adjusted distances to group spatial
medians, the PERMDISP statistics ranged from $F=1.025$ to $F=1.806$; the raw
permutation $p$ values ranged from 0.1389 to 0.4077, and the FDR-adjusted
$q$ values ranged from 0.3587 to 0.4077. The ILD group showed comparatively
broad and asymmetric descriptive distance distributions at several levels, but
these differences did not produce a significant omnibus dispersion effect.
Thus, the PERMANOVA findings were not accompanied by a detectable difference
in within-group dispersion and were compatible with differences in group
location within the learned geometry. However, a nonsignificant PERMDISP does
not establish equality of dispersion. The small and unequal group sizes,
particularly the four young healthy participants, limited sensitivity to such
differences.

## Pairwise clinical-group comparisons

Post-hoc testing considered the ten unique clinical-group pairs at each of the
four latent levels and for the combined distance, yielding 50 planned
comparisons. The effect size $\Delta_{AB}$ compared the mean between-group
distance with the average of the two mean within-group distances. Clinical
labels were permuted at the participant level, thereby preserving the
dependence among distances that shared a participant. Exact enumeration was
used when no more than 100,000 unique assignments were possible; otherwise,
100,000 Monte Carlo permutations were performed. Given the proof-of-concept,
exploratory nature of this analysis, the 50 resulting $p$ values were adjusted
jointly using the Benjamini--Hochberg procedure to control the false discovery
rate. An FDR of 5% was retained as the primary criterion. A secondary,
hypothesis-generating analysis examined a less stringent FDR threshold of 10%.

At the primary 5% FDR threshold, none of the 50 individual comparisons remained
significant after multiplicity correction. At the secondary 10% threshold, 13
comparisons met the exploratory criterion, with $q$ values ranging from 0.0869
to 0.1000 (Table 2). These signals formed three descriptive patterns. First,
older healthy participants differed from one or more disease groups at
$L_0$--$L_2$, most consistently from COPD and ILD. Second, CF differed from ILD
at $L_0$ and $L_1$ and from COPD at $L_0$. Third, CF differed from young healthy
participants at $L_3$ and for the combined multiscale distance. The largest
observed contrasts were CF versus young healthy participants at $L_3$
($\Delta=23.743$, exact $p=0.00327$, $q=0.0869$) and for the combined distance
($\Delta=23.729$, exact $p=0.00621$, $q=0.0869$).

**Table 2. Pairwise contrasts meeting the secondary exploratory FDR threshold
of 10%.**

| Distance | Group A | Group B | $\Delta_{AB}$ | Permutation $p$ | FDR $q$ |
|---|---|---|---:|---:|---:|
| $L_0$ | CF | COPD | 2.335 | 0.01804 | 0.0869 |
| $L_0$ | CF | ILD | 2.726 | 0.01911 | 0.0869 |
| $L_0$ | CF | Older healthy | 2.976 | 0.02297 | 0.0957 |
| $L_0$ | COPD | Older healthy | 3.293 | 0.01702 | 0.0869 |
| $L_0$ | ILD | Older healthy | 4.101 | 0.01635 | 0.0869 |
| $L_1$ | CF | ILD | 3.596 | 0.00999 | 0.0869 |
| $L_1$ | CF | Older healthy | 3.088 | 0.02599 | 0.1000 |
| $L_1$ | COPD | Older healthy | 3.819 | 0.00967 | 0.0869 |
| $L_1$ | ILD | Older healthy | 5.091 | 0.00674 | 0.0869 |
| $L_2$ | COPD | Older healthy | 4.290 | 0.01424 | 0.0869 |
| $L_2$ | ILD | Older healthy | 5.571 | 0.01908 | 0.0869 |
| $L_3$ | CF | Young healthy | 23.743 | 0.00327 | 0.0869 |
| Combined | CF | Young healthy | 23.729 | 0.00621 | 0.0869 |

The 10% threshold was examined as a secondary analysis and was less stringent
than the primary 5% threshold. These pairwise findings were therefore treated
as hypothesis-generating rather than as evidence of reproducible separation
between specific clinical groups. The especially small young healthy subgroup
also made its two CF contrasts imprecise.

Taken together, the significant PERMANOVA and nonsignificant PERMDISP results
provided preliminary evidence that the unsupervised latent geometry contained
clinical-group organization that was not attributable to detectable
differences in within-group dispersion. The pairwise analysis generated
candidate group contrasts for future evaluation, but the available sample did
not provide confirmatory evidence of separation between any individual pair of
groups at the primary 5% FDR threshold.

## Image-domain interrogation of latent trajectories

Spherical interpolation generated continuous paths between radially projected
subject representations while maintaining the common latent radius. Decoding
points along these paths returned the latent trajectories to the
three-dimensional ventilation-image domain. This demonstrated that differences
represented in the latent space could be interrogated as spatially resolved
image changes rather than only through a low-dimensional visualization.

Because radial projection changes the original latent codes, the endpoints of
these trajectories correspond to decoded projected subject representations and
are not generally identical to the observed images. The interpolation results
should therefore be interpreted as trajectories between projected
representations rather than exact image-to-image transformations.

**TODO:** Show prespecified examples comprising a within-group pair, a
healthy-to-disease pair, and a between-disease pair with comparable VDP. Use
fixed display parameters and report the difference between each observed image
and its decoded projected representation. To determine what the individual
levels encode, perform level-restricted interpolation or replacement while
holding the remaining components fixed, and quantify the resulting spatial
changes.

## Relationship to ventilation defect percentage

VDP and the latent geometry summarize different properties of a ventilation
image. VDP measures the overall proportion of low-ventilation voxels, whereas
the flow retains an invertible multiscale representation from which spatially
informed subject relationships can be derived. The current analyses establish
clinical-group organization in the latent distances but do not yet determine
whether that organization contains information beyond VDP.

**TODO:** Quantify the association between absolute VDP differences and latent
distances at each level using a matrix-based permutation procedure. Identify
and visualize pairs with similar VDP but large latent distance, as well as
pairs with different VDP but small latent distance. Any comparison of group
information carried by VDP and latent features should use participant-level
cross-validation or permutation testing appropriate for this small cohort.
Until these analyses are complete, the added value of the latent representation
relative to VDP should remain an open empirical question.

## Summary

The three-dimensional normalizing flow produced an invertible, four-level
representation of hyperpolarized $^{129}$Xe ventilation MRI without diagnostic
supervision. Clinical group explained approximately 11% of variation in each
level-specific and combined subject-distance matrix, and all five omnibus
effects remained significant after FDR correction. No corresponding
difference in within-group dispersion was detected, although the small and
unequal group sizes limited the sensitivity of this analysis. None of the 50
pairwise contrasts met the primary 5% FDR threshold; 13 met a secondary 10%
exploratory threshold and were considered hypothesis-generating. The present
proof-of-concept results therefore support global clinical organization of the
learned representation and identify candidate pairwise patterns for subsequent
testing, but they do not establish definitive separation of specific diagnostic
groups or superiority to VDP. Confirmatory evaluation in the planned cohort of
approximately 1,200 participants will permit more precise effect estimation,
covariate adjustment, and held-out validation.
