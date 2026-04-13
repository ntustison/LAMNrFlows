

## Nonlinear LAMNr Extension of the NNHEmbed Framework (UK Biobank M3RI IDPs)

To test whether a lightly nonlinear, generative multiview model provides a
practical advantage over the linear SiMLR/NNHEmbed regime used in our previous
work, we performed a focused comparison on the UK Biobank “M3RI” imaging-derived
phenotypes (IDPs). The goal of this experiment was deliberately narrow: hold the
cohort, view structure, preprocessing, and latent dimensionality fixed, and ask
whether replacing the linear per-view Gaussianization step with an invertible
normalizing-flow mapping yields measurable improvements in downstream utility.
To minimize confounding from architectural capacity, we fixed the flow depth and
conditioning-network width to **K=4** and **hidden\_channels=80**, respectively,
based on our prior single-view likelihood sweeps (val\_bpd) where this setting
emerged as the most stable and best-performing choice under equal-weighted
aggregation across processing packages and seeds.


### Data and baseline linear multiview embeddings (SiMLR/NNHEmbed)

We used **8,361 UKB subjects** and treated the modality-specific IDP blocks as
**three views**: **T1 structural IDPs (51 features)**, **DTI IDPs (77
features)**, and **resting-state fMRI IDPs (484 features)**. All features within
each block were retained, resulting in **612** total IDPs while preserving the
natural multiview partitioning. Each view was processed using the same tabular
pipeline adopted throughout the NNH analyses (winsorization followed by
z-scoring).

As a linear reference point, we used the same construction underlying the
SiMLR/NNHEmbed framework: per-view PCA embeddings with a shared dimensionality
**k = 31**, where 31 was chosen as the smallest number of principal components
required to explain approximately **95%** of the variance in the least-variable
modality. This produces three matrices of size **8361 × 31** (one per view), and
represents the “near-Gaussian linear limit” of the multiview embedding setting
we used previously.

### LAMNr multiview flows: invertible nonlinear Gaussianization with optional latent alignment

We then trained a multiview LAMNr model designed to match the dimensionality of
the linear baseline while allowing a controlled amount of nonlinearity. For each
view \(v\), LAMNr learns an **invertible** transformation \(f_v\) that maps IDPs
\(x^{(v)}\) to latents \(z^{(v)}\) under a shared-form Gaussian base
distribution:

\[
z^{(v)} = f_v(x^{(v)}), \qquad z^{(v)} \sim \mathcal{N}(0, I).
\]

Each \(f_v\) was implemented as a shallow RealNVP-style coupling-flow stack. We
intentionally constrained the expressivity to keep the comparison focused:
capacity was fixed to the single-view tuned setting (**K = 4 coupling steps per
view; hidden_channels = 80**) and we employed scale regularization (including
scale caps) to keep transformations close to identity. The base distribution was
the same across views, **GaussianPCA with latent dimension 31**, so that both
SiMLR and LAMNr ultimately produce 31-dimensional latent coordinates per
modality.

To encourage cross-view correspondence beyond what is implied by shared
dimensionality alone, we evaluated a family of cross-view regularizers applied
to the latents, weighted by \(\lambda\). We examined: **none** (pure likelihood
training), **VICReg**, **InfoNCE**, **Pearson correlation**, **Barlow Twins
(alignment)**, and **HSIC**, with \(\lambda \in \{0, 0.01, 0.03, 0.1\}\) when
applicable. Importantly, these penalties do not alter invertibility: they act as
auxiliary losses on the latent representations while the underlying per-view
flows remain exact bijections with tractable likelihood.

### Training, model selection, and downstream evaluation

All multiview flow variants were trained with identical data splits and
optimization settings. We monitored density modeling performance using
validation bits-per-dimension (**val_bpd**) and selected the best checkpoint per
run via **best_val_bpd**. For each configuration (penalty type and \(\lambda\)),
we trained **three seeds**, which allowed us to characterize seed-to-seed
variability at fixed capacity.

Our primary endpoint, however, was not likelihood alone. We evaluated whether
the learned latents provide improved utility for predicting selected non-imaging
outcomes from UKB. We focused on a compact target set with clear
interpretability and adequate prevalence: **BMI**, **Townsend deprivation
index**, **fluid intelligence**, **neuroticism**, and hearing measures derived
from the **speech reception threshold (SRT)**. For hearing, we included left and
right SRT and derived summaries (**mean SRT** and **left–right asymmetry**) to
capture both overall performance and lateralization. Smoking and alcohol status
were treated as categorical endpoints; we report both **accuracy** and **macro
one-vs-rest AUC** to reduce dependence on class imbalance.

To quantify comparative gains, we computed “uplift” in downstream performance
relative to (i) a nonlinear baseline flow without alignment (**LAMNr `none`,
\(\lambda=0\)**) and (ii) the linear SiMLR/PCA representations. Statistical
robustness was assessed using a **paired bootstrap over subjects** with
identical splits and resampling indices across compared methods. For each target
and metric, we report method A - method B with bootstrap **95%
confidence intervals**, two-sided **p-values**, and **Benjamini–Hochberg FDR**
q-values (computed separately for the “LAMNr vs baseline” and “LAMNr vs SiMLR”
families of tests).

### Likelihood results: alignment does not materially change density fit at fixed capacity

At the density modeling level, the top-performing multiview configurations were
essentially tied. In particular, **VICReg with \(\lambda=0.01\)** and **no
alignment** achieved nearly indistinguishable **best_val_bpd** (differences on
the order of \(10^{-5}\)–\(10^{-4}\) bpd). This indicates that, in this regime,
adding an alignment penalty does not substantially improve likelihood once the
latent dimension and flow capacity are fixed. In contrast, some penalties—most
notably **HSIC**—substantially degraded likelihood, suggesting that enforcing
strong dependence criteria can conflict with stable flow training in a
near-Gaussian tabular setting.

These likelihood trends already hint at an important point: model selection by
likelihood alone may not identify the configurations that maximize downstream
predictive utility, especially when alignment terms are introduced primarily to
shape representational structure rather than to improve density fit.

\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Multiview UK Biobank M3RI comparison (paired bootstrap, FDR-corrected).}
For each endpoint (target+metric), we report the best-performing LAMNr configuration (maximal $\Delta = A-B$) for (a) LAMNr vs SiMLR and (b) LAMNr vs LAMNr-none (penalty=none, $\lambda=0$).
CIs are paired-bootstrap 95\% intervals over subjects; $q$ values are Benjamini--Hochberg FDR within each comparison family.}
\label{tab:ukbb_m3ri_lamnr_stacked}

\begin{subtable}{\textwidth}
\centering
\caption{\textbf{LAMNr vs SiMLR}}
\begin{tabular}{ll l r l r r}
\toprule
\textbf{Outcome} & \textbf{Metric} & \textbf{Best LAMNr} & $\Delta$ & \textbf{95\% CI} & $p$ & $q$ \\
\midrule
BMI & R$^2$ & vicreg (0.01) & 0.0616 & [0.0495, 0.0744] & $<10^{-4}$ & $<10^{-4}$ \\
Townsend deprivation & R$^2$ & info\_nce (0.03) & 0.0122 & [0.0054, 0.0196] & 0.006 & 0.014 \\
Fluid intelligence & R$^2$ & vicreg (0.01) & 0.0162 & [0.0082, 0.0247] & $<10^{-4}$ & $<10^{-4}$ \\
Neuroticism & R$^2$ & hsic (0.01) & 0.0143 & [0.0080, 0.0205] & $<10^{-4}$ & $<10^{-4}$ \\
Hearing SRT (left) & R$^2$ & hsic (0.01) & 0.0151 & [0.0078, 0.0226] & $<10^{-4}$ & $<10^{-4}$ \\
Hearing SRT (right) & R$^2$ & pearson (0.01) & 0.0200 & [0.0132, 0.0270] & $<10^{-4}$ & $<10^{-4}$ \\
Hearing SRT (mean) & R$^2$ & barlow\_twins\_align (0.03) & 0.0239 & [0.0154, 0.0317] & $<10^{-4}$ & $<10^{-4}$ \\
Hearing SRT (L-R) & R$^2$ & hsic (0.1) & 0.0083 & [0.0035, 0.0132] & 0.002 & 0.005 \\
Smoking status & Acc & barlow\_twins\_align (0.1) & 0.0081 & [0.0023, 0.0140] & 0.006 & 0.014 \\
Smoking status & AUC$_{macro}$ (OVR) & vicreg (0.1) & -0.0132 & [-0.0517, 0.0211] & 0.492 & 0.525 \\
Alcohol status & Acc & barlow\_twins\_align (0.01) & 0.0005 & [0.0001, 0.0010] & 0.028 & 0.048 \\
Alcohol status & AUC$_{macro}$ (OVR) & hsic (0.01) & 0.1296 & [0.0480, 0.2085] & 0.01 & 0.021 \\
\bottomrule
\end{tabular}
\end{subtable}

\vspace{6pt}

\begin{subtable}{\textwidth}
\centering
\caption{\textbf{LAMNr vs LAMNr-none (penalty=none, $\lambda=0$)}}
\begin{tabular}{ll l r l r r}
\toprule
\textbf{Outcome} & \textbf{Metric} & \textbf{Best LAMNr} & $\Delta$ & \textbf{95\% CI} & $p$ & $q$ \\
\midrule
BMI & R$^2$ & vicreg (0.01) & 0.0171 & [0.0100, 0.0254] & $<10^{-4}$ & $<10^{-4}$ \\
Townsend deprivation & R$^2$ & info\_nce (0.03) & 0.0051 & [0.0007, 0.0099] & 0.02 & 0.175 \\
Fluid intelligence & R$^2$ & vicreg (0.01) & 0.0008 & [-0.0032, 0.0048] & 0.726 & 1 \\
Neuroticism & R$^2$ & hsic (0.01) & 0.0055 & [0.0005, 0.0105] & 0.022 & 0.184 \\
Hearing SRT (left) & R$^2$ & hsic (0.01) & 0.0037 & [-0.0015, 0.0085] & 0.142 & 0.568 \\
Hearing SRT (right) & R$^2$ & pearson (0.01) & 0.0027 & [-0.0008, 0.0066] & 0.122 & 0.532 \\
Hearing SRT (mean) & R$^2$ & pearson (0.1) & 0.0036 & [-0.0007, 0.0078] & 0.104 & 0.509 \\
Hearing SRT (L-R) & R$^2$ & hsic (0.1) & 0.0116 & [0.0078, 0.0155] & $<10^{-4}$ & $<10^{-4}$ \\
Smoking status & Acc & barlow\_twins\_align (0.1) & 0.0019 & [-0.0031, 0.0074] & 0.494 & 0.903 \\
Smoking status & AUC$_{macro}$ (OVR) & vicreg (0.1) & 0.0263 & [-0.0049, 0.0532] & 0.088 & 0.497 \\
Alcohol status & Acc & barlow\_twins\_align (0.01) & 0.0001 & [0.0000, 0.0004] & 0.786 & 1 \\
Alcohol status & AUC$_{macro}$ (OVR) & hsic (0.01) & 0.0582 & [0.0279, 0.0866] & $<10^{-4}$ & $<10^{-4}$ \\
\bottomrule
\end{tabular}
\end{subtable}

\end{table*}

### Downstream results: LAMNr consistently improves over SiMLR on continuous traits

Despite near-ties in best_val_bpd among the best-performing flow configurations,
downstream evaluation revealed a clearer separation between nonlinear and linear
representations. In paired bootstrap comparisons against **SiMLR**, the 
**LAMNr + VICReg (\(\lambda=0.01\))** model produced statistically significant
improvements (FDR-corrected) for several continuous endpoints. The strongest
gains were observed for **BMI** and the hearing-derived measures (SRT left/right
and mean SRT), along with smaller but significant improvements for **fluid
intelligence** and **neuroticism**. These results suggest that even a shallow
invertible nonlinearity—constrained to the same 31-dimensional latent space used
by the linear baseline—can yield latents that retain additional predictive
structure.

Categorical endpoints showed a more nuanced pattern. For smoking status,
improvements depended on the metric: LAMNr variants could provide modest gains
in accuracy, whereas **macro-AUC** was not consistently improved relative to
SiMLR (and in some cases SiMLR remained numerically better). For alcohol status,
several flow configurations improved accuracy, and some penalties provided
sizable gains in macro-AUC for specific parameterizations. Overall, these trends
are consistent with the idea that some categorical separations may be adequately
captured by linear multiview embeddings, while other endpoints benefit from
modest nonlinearity and/or alignment shaping.

### Incremental value of explicit alignment beyond “flow-only” training

When comparing aligned flows to the **LAMNr `none`** baseline (same architecture
and dimensionality), the incremental effect of alignment was present but more
target-specific and generally smaller than the overall LAMNr vs SiMLR gap. The
clearest and most robust incremental gain from alignment in our evaluation set
was for **BMI**, where **VICReg (\(\lambda=0.01\))** improved R² relative to the
unaligned flow baseline. Other targets showed mixed behavior: certain dependence
penalties improved specific endpoints while degrading others, and some penalties
that harmed likelihood could still improve selected downstream metrics. This
decoupling reinforces the view that alignment penalties function primarily as
representational regularizers, and that their value should be judged by
downstream goals rather than likelihood alone.

### Summary and implications

Taken together, these experiments support two practical conclusions. First, in a
near-Gaussian UKB IDP regime with matched latent dimensionality, a shallow
multiview normalizing flow can provide **measurable and statistically robust**
improvements over a linear PCA/SiMLR baseline for several clinically relevant
continuous outcomes. Second, explicit cross-view alignment offers
**incremental** gains for some endpoints but is not universally beneficial; its
impact is strongly metric- and target-dependent, and it can decouple density fit
from predictive utility. In practice, this suggests a two-stage strategy for
multiview generative modeling in tabular neuroimaging phenotypes: select stable
configurations by likelihood, but tune (or select) alignment penalties by
downstream utility when the scientific objective prioritizes prediction or
interpretable cross-view structure.


