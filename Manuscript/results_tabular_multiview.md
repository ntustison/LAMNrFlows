

### Multiview Comparison with the SiMLR NNHEmbed Framework

Building upon the baseline of individual views, we conducted a parallel
evaluation of three distinct classes of latent-alignment constraints to
characterize the likelihood-alignment trade-off across different mathematical
frameworks. Rather than relying on a single objective, we systematically
compared covariance-based (VICReg), kernel-based (HSIC), and contrastive
(InfoNCE) approaches to determine how varying geometric and statistical
pressures influence the resulting multiview representation. Simultaneously, we
performed a systematic sweep of the alignment weight ($\lambda$) for each
method. This allowed us to quantify the 'likelihood penalty' (i.e., the marginal
decrease in exact BPD) as a universal cost of prioritizing a unified, multiview
latent representation over view-specific details. Other related methods, such as
Barlow Twins or Pearson-based correlations, were omitted as they share
underlying functional principles with VICReg (specifically redundancy
reduction). 

Empirical results across both cohorts confirm that while all three
strategies successfully align the latent manifolds, they exhibit distinct
behaviors regarding density estimation and stability. HSIC emerged as the most
statistically robust objective for tabular data, providing a high-fidelity
alignment that effectively captures non-linear dependencies, albeit with a more
pronounced likelihood penalty in the smaller NNL cohort ($BPD = -2.871$ vs.
baseline $-4.239$). In contrast, VICReg and InfoNCE maintained likelihoods
closer to the baseline ($BPD \approx -4.16$ and $-4.00$ respectively for NNL),
suggesting a more conservative unfolding of the anatomical manifold. Notably, in
the larger PPMI dataset, the performance gap between methods narrowed
significantly, with VICReg yielding a BPD ($-8.142$) remarkably close to the
top-performaing HSIC($-8.198$). This consistency, combined with its superior
computational scalability, reinforces the selection of covariance-based
regularization (VICReg) as the primary alignment objective for the
high-dimensional imaging experiments using Glow-based LAMNr flows.

The robust latent alignment provides a biologically principled foundation for
statistical inference. This refined latent space directly translates into
enhanced sensitivity for clinical markers. As shown in Figure
\ref{fig:clinical_comparison}, LAMNr flows demonstrate significant performance
gains when predicting complex cognitive and functional phenotypes compared to
linear subspace projections. In the NNL cohort, the nonlinear mapping provides a
substantial "correlation uplift" ($\Delta r$) relative to SiMLR across multiple
domains, notably in *Recall Delayed* ($\Delta r = 0.190$, $q < 10^{-3}$) and
*Working Memory* ($\Delta r = 0.177$, $q = 0.024$). In contrast, for *Reading
Ability*, the nonlinear model shows a slight but non-significant decrease
compared to the linear baseline ($\Delta r = -0.004$, $q = 0.960$). However,
when compared to an unconstrained multiview model ($\lambda = 0$), LAMNr
alignment still retains a positive trend ($\Delta r = 0.069$), suggesting that
while linearity suffices for reading tasks, latent alignment remains beneficial
for overall model stability. Interestingly, the performance profiles differ
across populations. While the NNL cohort exhibits clear benefits from nonlinear
alignment, the linear SiMLR models remain highly competitive in the PPMI cohort.
This divergence likely reflects the different variance structures of the two
datasets.  Specifically, the NNL cohort captures a broad spectrum of healthy
variation where subtle nonlinear couplings are prevalent, whereas the PPMI
cohort is dominated by the strong, relatively linear pathological signal of
Parkinson’s disease progression.

\begin{figure*}[!htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{Figures/clinical_comparison_multipanel.png}
    \caption{The forest plot illustrates the correlation uplift ($\Delta r$)
    across two levels of comparison: (1) the gain from non-linear manifold
    mapping, represented by the difference between LAMNr flows (with VICReg) and
    the SiMLR baseline (i.e., red intervals), and (2) the gain from latent
    alignment, represented by the difference between the aligned LAMNr model and
    an unconstrained multi-view baseline ($\lambda = 0$, i.e., blue intervals).
    Error bars represent the 95\% confidence intervals derived from 1000
    bootstrap resamples. Top panel displays results for the NNL cohort, showing
    significant non-linear gains in memory and executive function. Bottom panel
    displays results for the PPMI cohort, where linear models remain highly
    competitive. Significant improvements ($q < 0.05$, FDR corrected) are
    indicated by intervals that do not cross the zero-reference line.}
    \label{fig:clinical_comparison}
\end{figure*}




