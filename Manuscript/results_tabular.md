

## Tabular LAMNr Flows

### Single-view tabular likelihood training on UK Biobank IDPs

To establish a stable single-view likelihood baseline for tabular LAMNr flows,
we evaluated UK Biobank structural imaging-derived phenotypes (IDPs) produced by
three standard processing packages: ANTsX, FreeSurfer, and FSL [@Tustison:2024aa].
In this setting, each package defines an independent view, and we trained
separate RealNVP-style normalizing flows *unsupervised* by maximum likelihood to
model the joint feature distribution of each view.

**Likelihood objective and evaluation metric.**  Models were selected using
validation bits-per-dimension (val_bpd), i.e. held-out negative log-likelihood
expressed per input dimension (lower is better). To isolate architectural
effects across sweeps, we fixed the base distribution to a GaussianPCA model
with latent dimension $\texttt{pca\_latent\_dimension}=31$, consistent with our
earlier SiMLR/NNHEmbed-aligned validation setting. For each run, we tracked the
full validation trajectory (val_history) and recorded the best checkpoint
(best_step) according to val_bpd.

Flow capacity is primarily controlled by (i) coupling depth ($K$), i.e. how many
affine coupling transforms are composed, and (ii) the hidden width
($\texttt{hidden\_channels}$) of the conditioner networks within each coupling
layer. Larger $K$ increases the number of invertible transformations applied to
the input, while larger $\texttt{hidden\_channels}$ increases the function class
used to predict scale/shift parameters, typically improving fit at higher
computational cost. Because these parameters trade off expressivity versus
stability and generalization, we swept them and selected settings using val_bpd
under equal-weighted aggregation across packages.

**Aggregation strategy for robust selection.**  Because feature composition
differs across packages, we ranked configurations using a two-stage aggregation:
(i) within each package, we averaged val_bpd across random seeds; (ii) across
packages, we computed an overall score with *equal package weighting* (mean of
the per-package means).

* **Stage 1 (coarse localization).**  We first localized a promising region in
($K$, hidden width) using a coarse grid: $K \in \{1, 4, 12, 32\}$ and
$\texttt{hidden\_channels} \in \{96, 192, 320\}$. Each configuration was
repeated across the three packages and two random seeds (72 runs total) with
$\texttt{max\_steps}=1500$ and $\texttt{val\_interval}=200$. Under
equal-weighted package aggregation, the best single-view setting was
$(K,\texttt{hidden\_channels})=(4,96)$. The overall differences among near-top
settings were small, indicating a relatively flat likelihood landscape around
the optimum.  Given the very small val_bpd differences among near-top
configurations, we adopt a parsimony bias i.e., when performance is statistically
indistinguishable, we prefer the smallest model (lower $K$ and narrower hidden
width). This choice is motivated by Occam/MDL-style arguments, improved
optimization stability across seeds/packages, and reduced computational cost,
enabling more replication at fixed budget.

* **Stage 2 (refinement and confirmation).**  Because the Stage 1 optimum lay near
the lower edge of the tested hidden widths, we expanded toward smaller models
and performed a two-phase refinement.

    * *Phase 1 (screen).*  We screened a denser local grid around the Stage 1 optimum:
$K \in \{2,3,4,5,6\}$ and
$\texttt{hidden\_channels} \in \{64,80,96,112,128\}$,
using one seed per package (75 runs) and increasing the training budget to
$\texttt{max\_steps}=3000$ (val_interval = 200). The best overall screen results
favored smaller widths (hidden = 64–80) with $K \approx 3$–4, while preserving
the same pattern of weak absolute differences among the top configurations.

    * *Phase 2 (confirm).*  We confirmed the top region using the compact grid
$K \in \{3,4\}$ and $\texttt{hidden\_channels} \in \{64,80\}$ with three seeds per
package (36 runs) and $\texttt{max\_steps}=6000$. After equal-weighted
aggregation, the most stable and best-performing configuration was
$(K,\texttt{hidden\_channels})=(4,80)$.

* **Final confirmation at extended training budget.**  Finally, we trained the
hyperparameter configuration 
$(K,\texttt{hidden\_channels})=(4,80)$ with $\texttt{max\_steps}=10000$
(3 packages × 3 seeds = 9 runs). The best checkpoints were no longer at the
maximum step (best_step typically occurred around 6.8k–7.6k), consistent with
approaching a plateau at this budget and validating the stability of the
selected setting. We therefore adopt $(K,\texttt{hidden\_channels})=(4,80)$ as
the default single-view likelihood configuration used in subsequent analyses.

### Single-view uplift analysis on clinical targets

\begin{table}[t]
\centering
\begin{tabular}{lrrr}
\hline
Target & ANTsX & FSL & FreeSurfer \\
\hline
Age & -8.297777 & -10.302203 & -11.539360 \\
Alcohol & -0.223586 & -0.325940 & -0.228319 \\
BMI & 1.403918 & 1.196208 & 2.123617 \\
FluidIntelligenceScore & -0.836901 & -0.605497 & -0.779686 \\
GeneticSex & -0.167546 & -0.219868 & -0.171760 \\
Hearing & -0.009873 & -0.010447 & -0.014611 \\
NeuroticismScore & -1.000343 & -1.030090 & -1.056362 \\
NumericMemory & -0.057217 & -0.095999 & -0.090725 \\
RiskTaking & -0.005101 & -0.005806 & -0.005586 \\
SameSexIntercourse & -0.000361 & -0.001562 & -0.001782 \\
Smoking & -0.015583 & -0.017823 & -0.008923 \\
TownsendDeprivationIndex & 0.062532 & 0.041367 & -0.053258 \\
\hline
\end{tabular}
\caption{Mean uplift ($\Delta R^2$) for OLS prediction of each clinical target,
computed as $R^2(\text{transformed}) - R^2(\text{raw})$ using the single-view
optimal configuration ($K{=}4$, \texttt{hidden\_channels}{=}80) for each package
(ANTsX, FSL, FreeSurfer).}
\label{tab:singleview_uplift_ols}
\end{table}

To quantify downstream utility of the single-view likelihood models, we
performed an “uplift” analysis that measures the change in predictive
performance on UK Biobank clinical/demographic targets when replacing raw
tabular inputs with the learned single-view representations. We focused on the
final single-view configuration selected by likelihood refinement,
$(K,\texttt{hidden\_channels})=(4,80)$, and evaluated each of the three IDP
packages (ANTsX, FSL, and FreeSurfer) separately.

For each package, we generated transformed single-view features from the trained
flow for three random seeds. These
transformed features are in a 31-dimensional PCA space. Since the PCA
projection used during flow training was not serialized, we constructed a
matched “raw” baseline by refitting a PCA ($k=31$) projection of the original
package-specific IDPs and using those PC scores as \texttt{view\_csv} in the
evaluation. This ensured that raw and transformed inputs had identical
dimensionality (31) for a fair comparison.

We evaluated ordinary least squares (OLS) models for each target using 10-fold
cross-validation, defining uplift as $\Delta R^2 = R^2(\text{transformed}) -
R^2(\text{raw})$. Despite the fact that ANTsX, FSL, and FreeSurfer compute
unique sets of structural brain measurements, the uplift patterns
were broadly consistent across packages (Table~\ref{tab:singleview_uplift_ols}).
This cross-package agreement provides an internal sanity check in that if the results
were dominated by noise or an implementation artifact, we would expect
substantially less concordance across independently derived feature sets.

Most targets exhibited negative uplift under this linear evaluator. Negative
uplift does not imply that the learned representation is universally “worse”;
rather, it indicates that under a linear probe and a strong PCA ($k=31$) baseline,
the likelihood-trained transform does not systematically increase supervised
signal for these endpoints. This is not unexpected for at least three reasons.
First, the likelihood-trained transformation is optimized to improve density
modeling (val\_bpd), not to preserve or amplify linear predictive signal for
arbitrary downstream phenotypes; in general, maximum-likelihood representation
learning does not guarantee improved $R^2$ for a supervised task. Second, our
baseline already uses PCA, which tends to concentrate the strongest
low-dimensional linear structure in the raw IDPs; relative to this strong
baseline, there is limited headroom for a purely unsupervised transform to
improve linear prediction, so small negative shifts can occur. Third, OLS is
sensitive to small changes in conditioning and signal-to-noise: if the learned
representation redistributes variance or deemphasizes weak but linearly
informative components, it can reduce $R^2$ even while improving likelihood.
Notably, BMI showed positive uplift across all three packages, suggesting that
for some phenotypes the learned representation can better capture brain–body
associations even within this unsupervised, likelihood-only setting.

### Nonlinear LAMNr Extension of the NNHEmbed Framework

To assess whether a nonlinear, latent-aligned normalizing-flow model offers any
practical advantage over the linear SiMLR/NNHEmbed framework used in previous
work [@Avants:2025aa], we performed a targeted comparison on the same UK Biobank
M3RI IDPs. We treated the T1, diffusion (DTI), and resting-state fMRI (rsfmri)
IDP blocks as three views for 8,361 subjects, retaining all features within each
block (51 T1, 77 DTI, 484 rsf). For each view, we applied the same preprocessing
pipeline used in the main NNH analyses (winsorization and z-scoring), and then
learned either (i) a linear Gaussian baseline equivalent to NNHEmbed/SiMLR
(per-view PCA to $k = 31$ components, where 31 is the minimum number of
principal components required to explain at least $95%$ of the variance in the
least variable modality) or (ii) a shallow LAMNr multiview normalizing flow with
the same 31-dimensional GaussianPCA base distribution per view. In the LAMNr
setting, each view’s IDPs are mapped to a shared isotropic Gaussian base via a
small number of RealNVP-style coupling layers (for example, $K = 1-4$ steps per
view with scale regularization to keep the transformations close to identity),
together with a cross-view alignment penalty that encourages corresponding
latent dimensions across T1, DTI, and rsf to share structure while preserving
exact invertibility and a fully specified joint density. All models were trained
on the same UKB training split and evaluated on held-out UKB test subjects using
identical demographic covariates derived from the accompanying table of
non-imaging variables (age at assessment, sex, and assessment centre/site) in
downstream linear models. We then compared (a) predictive performance for age
and selected physical measures such as grip strength and waist circumference
from the learned latent representations, and (b) the empirical Gaussianity and
cross-modal alignment of the resulting latents (marginal histograms, skewness
and kurtosis, and inter-view similarity scores such as the RV coefficient),
thereby directly testing whether a lightly nonlinear, generative LAMNr model
offers any measurable improvement over the linear SiMLR limit in this
near-Gaussian M3RI IDP regime.
