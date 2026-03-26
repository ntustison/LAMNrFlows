

### Multiview Comparison with the SiMLR NNHEmbed Framework

Having established a stable hyperparameter configuration, we next evaluate LAMNr
flows' capacity to align these latent spaces. We systematically compare VICReg,
HSIC, and InfoNCE.  Our empirical results (Table \ref{tab:alignment_results})
demonstrate that HSIC provides the most rigorous and stable latent alignment for
these tabular manifolds, effectively capturing non-linear dependencies that
simpler metrics might collapse. However, HSIC’s $O(N^2)$ complexity makes it
computationally prohibitive for high-resolution 3D Glow volumes.  Consequently, we
identify VICReg as the most viable candidate for scaling as it provides a superior
balance between numerical stability and computational efficiency, yielding
performance remarkably close to the kernel-based optimum while remaining
feasible for the subsequent Glow experiments.


\begin{table}[htbp]
\centering
\caption{Comparison of latent alignment constraints on the NNL and PPMI tabular cohorts. Values represent mean negative log-likelihood (BPD) $\pm$ standard deviation. While the HSIC criterion provides the most robust alignment for tabular data, the trade-off between statistical performance and computational scalability justifies the use of methods like VICReg for 3D volumes.}
\label{tab:alignment_results}
\begin{tabular}{@{}l l l r@{}}
\toprule
\textbf{Cohort} & \textbf{Method} & \textbf{Alignment Constraint} & {\textbf{Val. BPD ($\downarrow$)}} \\
\midrule
\textbf{NNL} & Baseline & None ($\lambda=0$) & $-4.239 \pm 0.001$ \\
             & HSIC & Kernel Independence & $-2.871$ \phantom{$\pm 0.000$}  \\
             & InfoNCE & Contrastive & $-3.999 \pm 0.114$ \\
             & VICReg & Var-Inv-Cov & $-4.159 \pm 0.124$ \\
\midrule
\textbf{PPMI} & Baseline & None ($\lambda=0$) & $-8.245 \pm 0.001$ \\
              & HSIC & Kernel Independence & $-8.198$ \phantom{$\pm 0.000$} \\
              & InfoNCE & Contrastive & $-7.380 \pm 0.634$ \\
              & VICReg & Var-Inv-Cov & $-8.142 \pm 0.228$  \\
\bottomrule
\end{tabular}
\end{table}

As shown in Figure \ref{fig:clinical_comparison}, LAMNr flows demonstrate
significant performance gains when predicting complex cognitive and functional
phenotypes compared to linear subspace projections. In the NNL cohort, the
nonlinear mapping provides a substantial "correlation uplift" ($\Delta r$)
relative to SiMLR across multiple domains, notably in *Recall Delayed* ($\Delta
r = 0.190$, $q < 10^{-3}$) and *Working Memory* ($\Delta r = 0.177$, $q =
0.024$). In contrast, for *Reading Ability*, the nonlinear model shows a slight
but non-significant decrease compared to the linear baseline ($\Delta r =
-0.004$, $q = 0.960$). However, when compared to an unconstrained multiview
model ($\lambda = 0$), LAMNr alignment still retains a positive trend ($\Delta r
= 0.069$), suggesting that while linearity suffices for reading tasks, latent
alignment remains beneficial for overall model stability. Interestingly, the
performance profiles differ across populations. While the NNL cohort exhibits
clear benefits from nonlinear alignment, the linear SiMLR models remain highly
competitive in the PPMI cohort. This divergence likely reflects the different
variance structures of the two datasets.  Specifically, the NNL cohort captures
a broad spectrum of healthy variation where subtle nonlinear couplings are
prevalent, whereas the PPMI cohort is dominated by the strong, relatively linear
pathological signal of Parkinson’s disease progression.

\begin{figure*}[!htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{Figures/clinical_comparison_multipanel.png}
    \caption{The forest plot illustrates the correlation uplift ($\Delta
    r$) across two levels of comparison: (1) the gain from non-linear manifold
    mapping, represented by the difference between LAMNr flows and the SiMLR
    baseline (i.e., red intervals), and (2) the gain from latent alignment,
    represented by the difference between the aligned LAMNr model and an
    unconstrained multi-view baseline ($\lambda = 0$, i.e., blue intervals).
    Error bars represent the 95\% confidence intervals derived from 1000
    bootstrap resamples. Top panel displays results for the NNL cohort, showing
    significant non-linear gains in memory and executive function. Bottom panel
    displays results for the PPMI cohort, where linear models remain highly
    competitive. Significant improvements ($q < 0.05$, FDR corrected) are
    indicated by intervals that do not cross the zero-reference line.}
    \label{fig:clinical_comparison}
\end{figure*}




