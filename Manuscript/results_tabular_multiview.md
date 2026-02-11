

## Nonlinear LAMNr comparison with the SiMLR NNHEmbed Framework

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{Figures/clinical_comparison_multipanel.png}
    \caption{\textbf{Clinical Predictive Power: LAMNr vs. Baselines.} 
    The forest plot illustrates the correlation uplift ($\Delta r$) of the LAMNr framework compared to the SiMLR linear baseline (blue) and the unconstrained ablation model ($\lambda = 0$, red). 
    Error bars represent the 95\% confidence intervals derived from 1000 bootstrap resamples. 
    \textbf{Panel A} displays results for the NNL cohort, highlighting significant non-linear gains in cognitive measures. 
    \textbf{Panel B} displays results for the PPMI cohort, where linear models remain highly competitive. 
    Significant improvements ($q < 0.05$, FDR corrected) are indicated by intervals that do not cross the zero-reference line.}
    \label{fig:clinical_comparison}
\end{figure}

<!-- \begin{table}[ht]
\centering
\caption{\textbf{Full Clinical Comparison: LAMNr vs SiMLR and Ablation (LAMNr-none)}}
\begin{tabular}{l rr l rr l}
\toprule
 & \multicolumn{3}{c}{\textbf{LAMNr vs SiMLR (Linear)}} & \multicolumn{3}{c}{\textbf{LAMNr vs Baseline ($\lambda > 0$)}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
\textbf{Outcome} & $\Delta r$ & \textbf{95\% CI} & $q_{Lin}$ & $\Delta r$ & \textbf{95\% CI} & $q_{None}$ \\
\midrule
\multicolumn{7}{l}{\textit{Panel: NNL}} \\
Recall Delayed & \textbf{0.190} & [0.099, 0.279] & \textbf{$<10^{-3}$} & 0.148 & [0.037, 0.263] & 0.064 \\
Working Memory & \textbf{0.177} & [0.051, 0.293] & \textbf{0.024} & \textbf{0.204} & [0.101, 0.319] & \textbf{0.032} \\
Reading Ability & 0.069 & [-0.018, 0.131] & 0.192 & -0.004 & [-0.076, 0.192] & 0.960 \\
Recall Total & 0.054 & [-0.054, 0.166] & 0.483 & -0.013 & [-0.143, 0.102] & 0.960 \\
Processing Speed & 0.017 & [-0.089, 0.299] & 0.714 & -0.002 & [-0.096, 0.248] & 0.960 \\
Executive Function & -0.024 & [-0.104, 0.052] & 0.483 & -0.004 & [-0.049, 0.127] & 0.960 \\
Focus And Control & -0.055 & [-0.186, 0.037] & 0.224 & -0.019 & [-0.152, 0.083] & 0.844 \\
Crystallized Intelligence & -0.078 & [-0.258, 0.115] & 0.505 & 0.055 & [-0.086, 0.195] & 0.844 \\
\midrule
\multicolumn{7}{l}{\textit{Panel: PPMI}} \\
ADAS-Q4 & -0.058 & [-0.125, 0.008] & 0.092 & -0.007 & [-0.064, 0.046] & 0.964 \\
CDR-SB & \textbf{-0.070} & [-0.133, -0.010] & \textbf{0.026} & 0.005 & [-0.043, 0.052] & 0.964 \\
FAQ & \textbf{-0.082} & [-0.153, -0.015] & \textbf{0.022} & -0.017 & [-0.074, 0.036] & 0.964 \\
mPACC (Digit) & \textbf{-0.082} & [-0.138, -0.026] & \textbf{0.004} & 0.004 & [-0.035, 0.046] & 0.964 \\
mPACC (Trails B) & \textbf{-0.083} & [-0.141, -0.029] & \textbf{$<10^{-3}$} & -0.002 & [-0.044, 0.040] & 0.964 \\
MMSE & \textbf{-0.084} & [-0.138, -0.028] & \textbf{$<10^{-3}$} & 0.004 & [-0.041, 0.044] & 0.964 \\
ADAS-13 & \textbf{-0.119} & [-0.180, -0.058] & \textbf{$<10^{-3}$} & 0.004 & [-0.063, 0.065] & 0.964 \\
\midrule
\bottomrule
\end{tabular}
\end{table} -->