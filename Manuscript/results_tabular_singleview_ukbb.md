

\clearpage

## Tabular LAMNr Flows

### Single-view tabular likelihood training on UK Biobank IDPs

To establish a stable single-view likelihood baseline for tabular LAMNr flows,
we evaluated our approach using UK Biobank structural imaging-derived phenotypes
(IDPs) produced by three standard processing packages: ANTsX (302 IDPs),
FreeSurfer (301 IDPs), and FSL (166 IDPs) [@Tustison:2024aa]. In this setting,
each package defines an independent view of the same data, reflecting the
differences in parcellation, segmentation, parcellation, and other algorithmic
choices. We then trained separate RealNVP-style single-view normalizing flows,
unsupervised by maximum likelihood, to model the joint feature distribution for
each set of measurements. Importantly, these single-view experiments provide a
controlled setting for selecting core flow capacity hyperparameters (e.g.,
coupling depth and conditioner width) for subsequent experiments.  

In the RealNVP-style architecture, network capacity is mainly controlled by two
hyperparameters: 

(i) __the coupling depth $K$:__ the number of affine coupling transforms composed, and 
(ii) __the conditioner width $\texttt{hidden\_channels}$:__ the number of hidden
channels in the subnetworks predicting scale/shift.

Increasing $K$ increases the number of invertible transformations applied to the
input, whereas increasing $\texttt{hidden\_channels}$ enlarges the function
class used within each coupling layer. Both changes typically improve likelihood
at increased computational cost and with a greater risk of overfitting or
optimization instability. We therefore performed a dedicated single-view
sweep over $(K,\texttt{hidden\_channels})$ under replication, and fixed the
resulting capacity setting for all subsequent multiview experiments** so that
multiview differences could be attributed to alignment objectives rather than
model size.


We used bits-per-dimension on validation data (val_bpd) as the primary selection
criterion for single-view tabular flows, i.e., negative log-likelihood
normalized by input dimensionality. To make comparisons across sweeps
interpretable, we fixed the base distribution to a GaussianPCA model with
$\texttt{pca\_latent\_dimension}=31$, matching the latent rank used in our
SiMLR/NNHEmbed setting [@Avants:2025aa]. For each run, we tracked the full
validation trajectory (val_history) and recorded the best-performing checkpoint
(best_step) according to val_bpd.


**Aggregation strategy for robust selection.**  Because the three packages
define views with different feature compositions and dimensionalities, we ranked
configurations using a two-stage aggregation designed to emphasize robustness:
(i) within each package, we averaged val_bpd across random seeds; (ii) across
packages, we computed an overall score with *equal package weighting* (mean of
the per-package means). This prevents any single package from dominating the
selection and yields a capacity choice that generalizes across software-defined
views.

* **Stage 1 (coarse localization).**  We first localized a promising region in
$(K,\texttt{hidden\_channels})$ using a coarse grid: $K \in \{1,4,12,32\}$ and
$\texttt{hidden\_channels} \in \{96,192,320\}$. Each configuration was repeated
across the three packages and two random seeds (72 runs total), using
$\texttt{max\_steps}=1500$ and $\texttt{val\_interval}=200$. Under equal-weighted
aggregation, the best setting was $(K,\texttt{hidden\_channels})=(4,96)$. The
differences among the top configurations were small, indicating a relatively
flat likelihood surface in this regime. Given this near-tie behavior, we adopt a
parsimony preference: when val_bpd differences are negligible, we prefer smaller
models (lower $K$ and narrower hidden width) to improve stability across seeds
and reduce compute, enabling more replication at fixed budget.

* **Stage 2 (refinement and confirmation).**  Because the Stage 1 optimum lay
near the lower edge of the tested widths, we expanded toward smaller models and
performed a two-phase refinement.

    * *Phase 1 (screen).*  We screened a denser local grid around the Stage 1
      region: $K \in \{2,3,4,5,6\}$ and
      $\texttt{hidden\_channels} \in \{64,80,96,112,128\}$, using one seed per
      package (75 runs) and $\texttt{max\_steps}=3000$ ($\texttt{val\_interval}=200$).
      The best screen results favored smaller widths (64–80) with $K\approx3$–4,
      while preserving the same pattern of weak absolute differences among the
      top configurations.

    * *Phase 2 (confirm).*  We confirmed the top region using the compact grid
      $K \in \{3,4\}$ and $\texttt{hidden\_channels} \in \{64,80\}$ with three
      seeds per package (36 runs) and $\texttt{max\_steps}=6000$. After
      equal-weighted aggregation, the most stable and best-performing
      configuration was $(K,\texttt{hidden\_channels})=(4,80)$.

* **Final confirmation at extended training budget.**  Finally, we trained
$(K,\texttt{hidden\_channels})=(4,80)$ with an extended budget
$\texttt{max\_steps}=10000$ (3 packages $\times$ 3 seeds = 9 runs). The best
checkpoints were typically not at the final iteration (best_step commonly
occurred around 6.8k–7.6k), consistent with approaching a likelihood plateau at
this budget and supporting the stability of the selected setting. We therefore
adopt $(K,\texttt{hidden\_channels})=(4,80)$ as the **default single-view
capacity configuration** used in subsequent multiview tabular analyses.


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

