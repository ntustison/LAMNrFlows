

\clearpage

## Glow-based LAMNr Flows

Prior to our registration-based evaluation of 3D LAMNr Flows, we first 
provide visualizations of the various possibilities of the proposed computional
anatomy framework restricted to 2D architectures due to modern hardware 
limitations[^comp]. 

[^comp]: 
    Unlike standard CNNs, Glow architectures require storing all
    intermediate activations to compute exact gradients, which quickly saturates
    Video RAM (VRAM) even when leveraging memory-efficient strategies such as
    gradient accumulation. The following table illustrates the exponential increase in voxel count compared to a baseline 2D slice:

    | Dimensionality | Resolution | Total Units (Pixels/Voxels) | Scaling Factor (vs. $256^2$) |
    | :--- | :--- | :--- | :--- |
    | **2D** | 256 $\times$ 256 | 65,536 | 1$\times$ |
    | **3D** | 48 $\times$ 48 $\times$ 48 | 110,592 | $\sim 1.7\times$ |
    | **3D** | 64 $\times$ 64 $\times$ 64 | 262,144 | 4$\times$ |
    | **3D** | 128 $\times$ 128 $\times$ 128 | 2,097,152 | 32$\times$ |
    | **3D** | 256 $\times$ 256 $\times$ 256 | 16,777,216 | 256$\times$ |

    Experimental procedures were executed using a single NVIDIA RTX A6000 GPU (48 GB
    VRAM). Empirically, processing a 3D volume of $48^3$ voxels (depth $L=3$,
    $K=32$, 64 hidden channels) consumes nearly all available memory with a
    micro-batch size of 14. Increasing the resolution to $64^3$ voxels imposes
    strict architectural compromises, forcing a reduction in hidden channels to 48
    ($L=3$, $K=32$) and batch size to 8 to prevent memory overflow. Projected memory
    requirements for $128^3$ or $256^3$ resolutions far exceed the 48 GB threshold,
    even with a batch size of 1. Future work concerns leveraging the existing 3D
    capabilities of our framework and the acquisition of high-capacity computational
    resources for image volumes $> 48^3$ voxels.

* Generative sampling 
* Fréchet mean approximation
* Latent distances 
* Pairwise image interpolation 
* Winsorize (tumor)


## Samples

\begin{figure}[htbp]
    \centering

    \begin{subfigure}{0.475\textwidth}
        \includegraphics[width=\linewidth]{Figures/samples_t1_temp_0.25.png}
        \caption{$\tau = 0.25$}
        \label{fig:t0.25}
    \end{subfigure}\hfill
    \begin{subfigure}{0.475\textwidth}
        \includegraphics[width=\linewidth]{Figures/samples_t1_temp_0.50.png}
        \caption{$\tau = 0.50$}
        \label{fig:t0.50}
    \end{subfigure}

    \vspace{0.2cm} 

    \begin{subfigure}{0.475\textwidth}
        \includegraphics[width=\linewidth]{Figures/samples_t1_temp_0.75.png}
        \caption{$\tau = 0.75$}
        \label{fig:t0.75}
    \end{subfigure}\hfill
    \begin{subfigure}{0.475\textwidth}
        \includegraphics[width=\linewidth]{Figures/samples_t1_temp_1.00.png}
        \caption{$\tau = 1.0$}
        \label{fig:t1.00}
    \end{subfigure}
    
    \caption{Samples drawn from the learned LAMNr flow prior for T1-weighted
    structural MRI at varying temperatures ($\tau$). Lower temperatures, such as
    $\tau = 0.25$ and $\tau = 0.50$, constrain sampling near the mode of the
    Gaussian prior, generating highly realistic, canonical anatomies with distinct
    tissue boundaries. As the temperature increases ($\tau = 0.75$ and $\tau =
    1.0$), sampling extends into the tails of the prior distribution. While this
    yields greater macroscopic morphological diversity, it also introduces
    high-frequency variance and slight structural artifacts, reflecting regions of
    the latent space with lower probability density.}

    \label{fig:t1_samples}
\end{figure}


\begin{figure}[htbp]
    \centering

    \begin{subfigure}{0.475\textwidth}
        \includegraphics[width=\linewidth]{Figures/samples_fa_temp_0.25.png}
        \caption{$\tau = 0.25$}
        \label{fig:t0.25}
    \end{subfigure}\hfill
    \begin{subfigure}{0.475\textwidth}
        \includegraphics[width=\linewidth]{Figures/samples_fa_temp_0.50.png}
        \caption{$\tau = 0.50$}
        \label{fig:t0.50}
    \end{subfigure}

    \vspace{0.2cm} 

    \begin{subfigure}{0.475\textwidth}
        \includegraphics[width=\linewidth]{Figures/samples_fa_temp_0.75.png}
        \caption{$\tau = 0.75$}
        \label{fig:t0.75}
    \end{subfigure}\hfill
    \begin{subfigure}{0.475\textwidth}
        \includegraphics[width=\linewidth]{Figures/samples_fa_temp_1.00.png}
        \caption{$\tau = 1.0$}
        \label{fig:t1.00}
    \end{subfigure}
    
    \caption{Samples drawn from the learned LAMNr flow prior for Fractional
    Anisotropy (FA) maps at varying temperatures ($\tau$). Consistent with the
    structural MRI samples, lower temperatures ($\tau \le 0.50$) generate coherent
    and anatomically representative white matter tracts. At higher temperatures
    (e.g., $\tau = 1.0$), the model samples extreme latent vectors from the prior
    tails. In highly sensitive quantitative modalities like FA, these extreme latent
    vectors occasionally map to focal, aberrant voxel intensities. When linearly
    normalized for visualization (min-max scaling), these isolated outlier values
    compress the visual contrast of the surrounding valid anatomical structures.}

    \label{fig:fa_samples}
\end{figure}


## Latent distances from $f^{-1}_{\theta}(0)$

\begin{figure}[htbp]
    \centering

    % --- Baseline Row ---
    \begin{subfigure}{0.32\textwidth}
        \includegraphics[width=\linewidth]{Figures/PPMI_template0_256x256x256_slice138.png}
        \caption{ANTsX Template}
        \label{fig:template_antsx}
    \end{subfigure}
    \hspace{0.05\textwidth} % Space to center the two images
    \begin{subfigure}{0.32\textwidth}
        \includegraphics[width=\linewidth]{Figures/template_T1_mu_sharpened_256x256.png}
        \caption{T1: $f^{-1}_{\theta}(0)$}
        \label{fig:template_flow}
    \end{subfigure}

    \vspace{0.5cm} 

    % --- Inliers (Min Distance) ---
    \begin{subfigure}{0.32\textwidth}
        \includegraphics[width=\linewidth]{Figures/sub-3859_ses-20120921_r0001_ppmixt1_min_total_t1_distance_to_gaussian_256x256_slice138.png}
        \caption{Min Distance (Total)}
        \label{fig:min_total}
    \end{subfigure}\hfill
    \begin{subfigure}{0.32\textwidth}
        \includegraphics[width=\linewidth]{Figures/sub-3327_ses-20170127_r0002_ppmixt1_min_distL0_t1_distance_to_gaussian_256x256_slice138.png}
        \caption{Min Distance (Level 0)}
        \label{fig:min_L0}
    \end{subfigure}\hfill
    \begin{subfigure}{0.32\textwidth}
        \includegraphics[width=\linewidth]{Figures/sub-3859_ses-20120921_r0001_ppmixt1_min_distL5_t1_distance_to_gaussian_256x256_slice138.png}
        \caption{Min Distance (Level 5)}
        \label{fig:min_L5}
    \end{subfigure}
    
    \vspace{0.5cm} 
    
    % --- Outliers (Max Distance) ---
    \begin{subfigure}{0.32\textwidth}
        \includegraphics[width=\linewidth]{Figures/sub-3150_ses-20101109_r0001_ppmixt1_max_total_t1_distance_to_gaussian_256x256_slice138.png}
        \caption{Max Distance (Total)}
        \label{fig:max_total}
    \end{subfigure}\hfill
    \begin{subfigure}{0.32\textwidth}
        \includegraphics[width=\linewidth]{Figures/sub-3318_ses-20120627_r0001_ppmixt1_max_distL0_t1_distance_to_gaussian_256x256_slice138.png}
        \caption{Max Distance (Level 0)}
        \label{fig:max_L0}
    \end{subfigure}\hfill
    \begin{subfigure}{0.32\textwidth}
        \includegraphics[width=\linewidth]{Figures/sub-3586_ses-20160810_r0001_ppmixt1_max_distL5_t1_distance_to_gaussian_256x256_slice138.png}
        \caption{Max Distance (Level 5)}
        \label{fig:max_L5}
    \end{subfigure}
    
    \caption{Visualization of the learned latent space across the cohort. The top
    row compares the standard population template with the mean image generated by
    the model. The subsequent rows illustrate the distance to the Gaussian prior
    distribution, identifying the most typical cases (minimum distance) and
    anomalies (maximum distance). Within the hierarchical Glow architecture,
    high-frequency details such as tissue texture and noise tend to be encoded in
    the lower levels whereas low-frequency morphological variations tend to be
    encoded in the higher levels. Consequently, evaluating distances specifically at
    $L=0$ isolates intensity and texture anomalies, while evaluating at $L=5$ highlights
    macroscopic structural deviations.}

    \label{fig:latent_space_distances}
\end{figure}


### Leveraging approximate template $\leftrightsquigarrow$ subject geodesic linearity for image registration via latent winsorization

To evaluate the utility of the learned latent representations for downstream
geometric tasks, we applied LAMNr Flows to the challenge of deformable image
registration in the presence of focal pathology (e.g., brain tumors).
Traditional registration metrics often struggle with "outlier" intensities
caused by tumors, leading to non-anatomical deformations as the algorithm
attempts to match pathological tissue to healthy templates. We hypothesize that
the bijective latent space of LAMNr exhibits approximate geodesic linearity,
where the path between subjects is largely governed by shared anatomical
features once view-specific or pathological "noise" is suppressed.

We implemented a latent winsorization schedule to guide the registration
process. By progressively relaxing the percentile bounds of the latent
variables—starting with a restrictive winsorization (e.g., 0.1) and
transitioning to the full latent signal (1.0)—we effectively regularize the
deformation field. In the early iterations, strong winsorization "flattens" the
latent outliers associated with the tumor, allowing the registration to focus on
the shared anatomical manifold. As the schedule progresses and the images become
globally aligned, the winsorization is lifted, permitting fine-grained local
adjustments.

Our results on the multimodal cohort demonstrate that this schedule prevents the
localized "warping" artifacts common in standard ANTs-based registration. By
leveraging the structured latent space of LAMNr, we achieve a registration that
is robust to focal pathology without requiring manual lesion masking,
effectively using the latent space as a prior for anatomical consistency.

