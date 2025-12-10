

## Tabular LAMNr Flows

### Single-view tabular Gaussianization of UK Biobank IDPs

To assess whether normalizing flows are useful even for purely tabular imaging
phenotypes, we first analyzed a single-view setting using UK Biobank
imaging-derived phenotypes (IDPs) from three well-known packages:  ANTsX, 
FreeSurfer, and FSL [@Tustison:2024aa]. For each software package 
we treated its structural IDPs as one view and fitted a real-NVP–style flow
that Gaussianizes the joint feature distribution. Flows were trained in an
unsupervised fashion with maximum likelihood, using a standard Gaussian base and
$K$ coupling steps $(K \in {4, 8, 16, 32})$ but no dimensionality reduction. From each
trained model we extracted a "whitened representation", i.e., the latent
variables after flow inversion, linearly rescaled to zero mean and unit variance
so that the dimensionality matches the original IDPs. For each demographic or
lifestyle target (Age, BMI, Townsend deprivation index, and additional clinical
variables), we then fitted ridge regression models with 10-fold cross-validation
using either the raw IDPs, simple z-scores, or the whitened features.
Performance was summarized by cross-validated $R^2$, and we report uplift 
$\Delta R^2 = R^2(\mathrm{flow}) - R^2(\mathrm{raw})$ as a function of $K$ 
and package. 

\begin{figure}
  \centering
  \begin{tabular}{cc}
  \includegraphics[width=0.475\textwidth]{Figures/full_byK_Age_ridge.png} &
  \includegraphics[width=0.475\textwidth]{Figures/full_byK_BMI_ridge.png} \\
  (a) & (b)
  \end{tabular}
  \caption{Uplift in cross-validated \(R^2\) from flow-whitened features
  relative to raw imaging-derived phenotypes (IDPs) for two example targets in
  UK Biobank. Each panel shows \(\Delta R^2 = R^2_{\text{flow, full, ridge}} -
  R^2_{\text{raw}}\) as a function of the number of coupling steps \(K\) in the
  single-view real-NVP flow, for ANTsX, FreeSurfer, and FSL IDPs. (a) For
  Age, all curves lie below the dashed zero line, indicating that
  Gaussianization consistently worsens linear prediction, with degradation
  increasing at higher \(K\). (b) For BMI, flows substantially improve
  prediction, with positive uplift that grows with depth up to about \(K = 16\)
  for all three packages, illustrating that non-linear flow-based
  representations are most beneficial for targets whose relation to brain
  structure is not well captured by simple linear trends.
}
\end{figure}

Across packages, chronological age serves as a useful negative control. Age is
already known to exhibit strong, largely monotonic associations with global and
regional brain structure in large cohorts, including UK Biobank, where linear or
low-order non-linear models explain substantial variance in cortical thickness
and subcortical volumes [@tustison_antsx_2021;
@Bethlehem2022BrainChartsLifespan; @Dong2022UKBBStructuralCovariance;
@Tustison:2024aa]. Consistent with this, raw and z-scored IDPs achieved high
$R^2$ for Age, and flow-whitened features consistently *reduced* performance:
$\Delta R^2$ was negative for all $K$ and all three packages. Increasing $K$
further decreased $R^2$ in some cases. This suggests that, when the target is
already well approximated by a linear function of the original IDPs, an
unconstrained unsupervised flow tends to bend that linear manifold in ways that
are not aligned with the Age prediction task.  The downstream ridge decoder then
has to “undo” these warps and cannot fully recover the original signal.

In contrast, body-mass index (BMI) showed the largest and most robust positive
uplift. Prior work has reported complex, spatially heterogeneous associations
between adiposity and brain structure and function, including non-linear effects
and interactions with vascular and metabolic risk [@Dekkers2019ObesityBrain;
@Kim2015BMIThickness; @Bettcher2013BMIVascularWM; @Morys2021MidlifeObesity;
@Kullmann2012ObeseBrainRSFC].  In our experiments, $R^2$ for BMI from raw or
z-scored IDPs was modest, but the whitened representations yielded
substantial gains, with $\Delta R^2$ increasing steadily with $K$ and reaching
its maximum at the deepest flows. The pattern was consistent across ANTsX,
FreeSurfer, and FSL IDPs, though the absolute magnitude varied. This suggests
that Gaussianizing and whitening the joint IDP distribution helps straighten out
a curved, non-Gaussian manifold relating BMI to brain structure, making the
residual dependence more amenable to linear decoding.

The Townsend deprivation index, an area-level measure of socioeconomic
deprivation, produced intermediate behavior. Socioeconomic status is only
indirectly encoded in brain structure and is known to relate to cumulative
environmental exposures, health behaviors, and comorbidities
[@Tan2023TownsendCorticalThickness; @Brito2014SESBrianDevReview;
@Farah2017NeuroscienceSES; @Klee2023TownsendDeprivationHealth;
@Pampel2010SESHlthBehaviors]. For Townsend we observed small or near-zero uplift
at low $K$, with modest positive $\Delta R^2$ emerging only for deeper flows.
This is consistent with a weak but genuinely non-linear signal: the flows need
sufficient capacity before any advantage over simple z-scaling appears, and even
then the gains remain much smaller than for BMI.

Across the three exemplar targets, the uplift analysis paints a consistent
picture of when single-view tabular flows help and when they hurt. For Age,
flow-whitened features *always* reduced performance: mean uplift was negative
for every package and coupling depth, corresponding to drops of roughly 2–7
percentage points of cross-validated \(R^2\) relative to raw IDPs, and the
run-wise tests for \(\Delta R^2 > 0\) all yielded one-sided \(p\)-values close
to 1 (i.e., strong evidence of degradation). In contrast, BMI showed robust
gains: for FSL and FreeSurfer, mean uplift was positive at all \(K\), with
increases on the order of \(\sim 0.5\)–\(1.5\) percentage points of \(R^2\) and
one-sided \(p\)-values for \(\Delta R^2 > 0\) in the range \(\sim 7\times
10^{-4}\)–\(3\times 10^{-2}\); ANTsX exhibited a smaller but still significant
mean uplift at \(K = 8\), with shallow (\(K = 4\)) and very deep (\(K = 16,
32\)) flows either negative or indistinguishable from zero. The Townsend
deprivation index fell in between: for most package/\(K\) combinations the
mean uplift was near zero or slightly negative, but the deepest flows (\(K =
32\)) yielded small yet statistically detectable positive uplifts for ANTsX and
FreeSurfer (on the order of \(\sim 0.04\)–\(0.15\) percentage points of \(R^2\),
one-sided \(p \approx 0.01\)–\(0.02\)), while FSL remained effectively flat.
Taken together, these results support the idea that flow-based Gaussianization
of tabular IDPs is most beneficial for targets like BMI (and, weakly, Townsend)
whose relationship to brain structure is non-linear and non-Gaussian, and can
systematically harm prediction for targets like Age whose mapping is already
well captured by simple linear trends in the original feature space.

Most of the remaining demographic and clinical variables showed little benefit
from flow Gaussianization. For many targets, R² from raw IDPs was already very
low, indicating that the variable is only weakly expressed in the imaging
features; in that regime, an unsupervised representation step tends mainly to
add variance, leading to slightly negative $\Delta R^2$. For a few targets with clearer
linear structure, the behavior resembled Age, with mild degradation after
applying flows. Taken together, these single-view results indicate that tabular
normalizing flows are selectively useful: they provide the largest gains when
the target’s relationship to IDPs is strongly non-linear and non-Gaussian (BMI,
partially Townsend), but can modestly harm performance when the mapping is
already well captured by simple linear models (Age and similar measures). This
motivates our subsequent focus on multi-view, laminar settings where such
non-Gaussian structure is common and where flexible Gaussianization layers are
more likely to be beneficial.

