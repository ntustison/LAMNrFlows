

\clearpage

## Glow-based LAMNr Flows

Prior to our registration-based evaluation of 3D LAMNr Flows, we first 
provide visualizations of the various possibilities of the proposed computational
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

* __Generative sampling.__ Sampling in Normalizing/LAMNr Flows is achieved by drawing a
  latent vector $z$ from the isotropic Gaussian base distribution, 
  $z \sim \mathcal{N}(0, \tau^2 I)$, and passing it through the inverse of the learned bijective
  mapping, $x = f^{-1}_\theta(z)$, to reconstruct the high-dimensional image.
  Scaling the standard deviation of this prior with a temperature parameter
  ($\tau$) allows for explicit control over the fundamental trade-off between
  generating high-fidelity, canonical anatomies near the mean and exploring
  diverse, lower-probability structural variations in the tails.  Figures \ref{fig:t1_samples}
  and \ref{fig:fa_samples}.

* __Fréchet mean approximation.__ In the context of the learned data manifold,
  the Fréchet mean of the anatomical distribution can be efficiently
  approximated by decoding the origin of the latent space. By passing the
  zero-vector of the isotropic Gaussian prior through the inverse flow, $x =
  f^{-1}_\theta(0)$, we synthesize a canonical representation that captures the
  central morphometric tendency of the cohort without the computational overhead
  of iterative diffeomorphic averaging [@Avants:2010aa].  Figure \ref{fig:frechet_mean}.

* __Cohort template.__ Beyond simple point estimation, the generative nature of
  LAMNr flows allows for the construction of a high-fidelity cohort template
  that functions as an alternative to the Fréchet mean approximation. By
  calculating the arithmetic centroid of the subject-specific latent vectors and
  mapping this central point back to the physical domain, we generate a
  synthetic image that represents the cohort-specific anatomical
  characterization. Unlike the Fréchet mean approximation, this manifold-based
  synthesis preserves sharp morphological boundaries by operating within the
  linearized geometry of the "unfolded" anatomical distribution. Figure
  \ref{fig:cohort_template}.

* __Latent distances.__ The bijective nature of normalizing flows allows complex
  anatomical deviations to be quantified through a flexible suite of distance
  metrics in the learned latent space, depending on the analytical
  objective.Euclidean distance provides a straightforward measure of separation
  for basic similarity assessments. To account for the natural variance of each
  latent dimension, we implement a standardized Euclidean (diagonal Mahalanobis)
  distance, $d = \sqrt{ \sum_i \frac{(z_i - \mu_i)^2}{\sigma_i^2 + \epsilon} }$,
  which benchmarks a subject against the normative Gaussian mean ($\mu$) without
  artificially penalizing high-variance anatomical traits.  For point-to-point
  comparisons between the latents $z_j$ and $z_k$ of specific images, we utilize 
  geodesic distance derived from
  cosine similarity, $d = \arccos(\text{clamp}(\text{sim}(z_j, z_k)))$. By
  measuring the angular displacement on the hypersphere, this metric respects
  the spherical geometry of the isotropic Gaussian prior, ensuring that
  anatomical transitions are evaluated along the high-density manifold. These
  combined metrics yield a rigorous, variance-weighted framework for anomaly
  detection and longitudinal assessment. 
  Figures \ref{fig:latent_space_distances} and \ref{fig:twin_distances}.

* __Cross-modal imputation via Conditional Gaussian modeling.__ Missing
  modalities are synthesized by encoding the available observed images to the
  latent space, $z_O = f^{(O)}(X^{(O)})$, and computing the exact conditional
  expectation of the unobserved latent vectors, $\mu_{U|O}$, under the learned
  joint Gaussian prior. Projecting this optimal estimate through the target
  modalities' inverse flows, $\hat{X}^{(U)} = (f^{(U)})^{-1}(\mu_{U|O})$, yields
  a high-fidelity imputation that guarantees mathematical consistency with the
  population's cross-view dependencies. Crucially, because the joint prior
  models the full multi-view latent space simultaneously, this formulation is
  inherently flexible: it supports conditioning on any arbitrary subset of
  available data, enabling complex many-to-many translations (e.g., synthesizing
  a single FA map from combined T1 and T2 inputs, or simultaneously generating
  T2, FA, and MD from a single T1 scan).  Figure \ref{fig:imputation}.

* __Pairwise image interpolation.__ Smooth morphological transitions between two
  anatomical scans are generated by interpolating their representations in the
  learned latent space. To prevent the variance collapse and out-of-distribution
  artifacts characteristic of standard linear interpolation[^slerp], we employ a
  $\mu$-centered spherical linear interpolation (Slerp). By applying the
  spherical rotation relative to the population's empirical mean, $\mu$, the
  latent trajectory rigorously preserves the intrinsic data variance, ensuring
  all intermediate representations remain strictly on the high-probability
  anatomical manifold. Figure \ref{fig:interpolation}.

[^slerp]: Dans un espace latent de faible dimension (comme en 2D), on imagine
    souvent qu'une distribution Gaussienne concentre la majorité de ses points
    très près du centre (la moyenne $\mu$).
    Cependant, dans un espace latent de très grande dimension (comme celui d'une
    image IRM 3D), la géométrie change radicalement à cause du "fléau de la
    dimension" (curse of dimensionality). Le volume de l'espace croît de façon
    exponentielle en s'éloignant du centre. En conséquence, la quasi-totalité
    des échantillons réalistes (les cerveaux de votre cohorte) ne se trouve pas
    au centre $\mu$, mais forme une "coquille" ou une "bulle" sphérique à une
    distance fixe de ce centre. 
        
    * __L'Ensemble Typique (Typical Set)__ : Cet anneau sphérique est ce qu'on
    appelle la variété de haute probabilité (High-Probability Manifold). 
        
    * __L'équation $\|z - \mu\| \approx \text{const}$__ : Cette formule
    mathématique indique que la distance Euclidienne (la norme $\|\cdot\|$)
    entre un cerveau réaliste généré ($z$) et le cerveau moyen ($\mu$) est
    approximativement constante. C'est le rayon de notre sphère.  
    
    Si l'interpolation (Lerp) coupe à l'intérieur de cet arc, la distance $\|z -
    \mu\|$ chute considérablement. Le vecteur quitte la "coquille" des cerveaux
    réalistes et pénètre dans une zone morte, ce qui provoque l'artéfact en
    damier.

* __Temperature scaling.__ Weighting a latent representation by a scalar factor
  $\tau < 1.0$ contracts the vector toward the origin of the Gaussian
  prior. This variance reduction rigorously preserves the topology of the
  learned manifold, effectively suppressing out-of-distribution pathological
  anomalies and shifting the subject's anatomy toward the healthy population
  mean without introducing reconstruction artifacts. Figure
  \ref{fig:temperature_scaling}.


\begin{figure}[htbp]
    \centering

    \begin{subfigure}{0.475\textwidth}
        \includegraphics[width=\linewidth]{Figures/samples_t1_temp_0.25.png}
        \caption{$\tau = 0.25$}
    \end{subfigure}\hfill
    \begin{subfigure}{0.475\textwidth}
        \includegraphics[width=\linewidth]{Figures/samples_t1_temp_0.50.png}
        \caption{$\tau = 0.50$}
    \end{subfigure}

    \vspace{0.2cm} 

    \begin{subfigure}{0.475\textwidth}
        \includegraphics[width=\linewidth]{Figures/samples_t1_temp_0.75.png}
        \caption{$\tau = 0.75$}
    \end{subfigure}\hfill
    \begin{subfigure}{0.475\textwidth}
        \includegraphics[width=\linewidth]{Figures/samples_t1_temp_1.00.png}
        \caption{$\tau = 1.0$}
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
    \end{subfigure}\hfill
    \begin{subfigure}{0.475\textwidth}
        \includegraphics[width=\linewidth]{Figures/samples_fa_temp_0.50.png}
        \caption{$\tau = 0.50$}
    \end{subfigure}

    \vspace{0.2cm} 

    \begin{subfigure}{0.475\textwidth}
        \includegraphics[width=\linewidth]{Figures/samples_fa_temp_0.75.png}
        \caption{$\tau = 0.75$}
    \end{subfigure}\hfill
    \begin{subfigure}{0.475\textwidth}
        \includegraphics[width=\linewidth]{Figures/samples_fa_temp_1.00.png}
        \caption{$\tau = 1.0$}
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

\begin{figure}[htbp]
    \centering

    % --- Baseline Row ---
    \begin{subfigure}{0.30\textwidth}
        \includegraphics[width=\linewidth]{Figures/PPMI_template0_256x256x256_slice138.png}
        \caption{ANTsX Template}
        \label{fig:template_antsx}
    \end{subfigure}
    \hspace{0.01\textwidth} % Space to center the three images
    \begin{subfigure}{0.30\textwidth}
        \includegraphics[width=\linewidth]{Figures/template_T1_mu_sharpened_256x256.png}
        \caption{T1: $f^{-1}_{\theta}(0)$}
        \label{fig:template_flow}
    \end{subfigure}
    \hspace{0.01\textwidth} % Space to center the two images
    \begin{subfigure}{0.30\textwidth}
        \includegraphics[width=\linewidth]{Figures/template_T1_mu_sharpened_256x256.png}
        \caption{Need an image of the cohort-based template !}
        \label{fig:template_cohort_flow}
    \end{subfigure}

    \caption{Comparison of population Fréchet mean approximations. (a) The standard
    ANTsX template, constructed via traditional iterative diffeomorphic
    registration, representing a geometric spatial average that preserves
    high-frequency structural details. (b) The generative latent-mean,
    $f_\theta^{-1}(0)$, obtained in a single forward pass. The visually smoother
    appearance of the flow-generated template is a direct consequence of
    high-dimensional probabilistic modeling: as the exact mode of the latent
    distribution, it averages out idiosyncratic, high-frequency anatomical
    variations (such as specific cortical folding patterns) that do not strictly
    persist across the cohort. Instead of producing a single typical sample, it
    successfully isolates the macroscopic central morphological tendency and shared
    structural signal of the dataset.}

    \label{fig:frechet_mean}

\end{figure}

\begin{figure}[htbp]
    \centering

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


\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\linewidth]{Figures/comparaison_rangs_combinee} \\
    (a)\\
    \includegraphics[width=0.8\linewidth]{Figures/comparaison_rangs_combinee_3d} \\
    \caption{Latent geodesic distance rankings demonstrate structural biological
    relatedness across dimensionalities. Using a portion of the Queensland Twin
    Study ($N = 210$ subjects, mean age = $24.7 \pm 1.8$ years), we ranked the
    latent geodesic closeness of each subject relative to the rest of the cohort.
    The analysis compares two experimental configurations: (a) 2D LAMNr flows
    (resolution $256 \times 256$) optimized on mid-axial slices, with multiscale
    levels spanning $L_0$ to $L_5$; and (b) 3D LAMNr flows (resolution $48 \times 48
    \times 48$) with multiscale levels spanning $L_0$ to $L_2$. For both
    architectures, rankings were computed for the global latent space and individual
    resolution levels using models pre-trained on the PPMI dataset. The results
    illustrate that latent geodesic similarity strongly correlates with biological
    kinship in both settings ($p < 1\times10^{-10}$ for all comparisons). Notably, the
    detection of this genetic signature is significantly amplified following brain
    extraction (skull-stripping), despite the networks having been trained entirely
    on non-brain-extracted, significantly older PPMI data.}
    \label{fig:twin_distances}
\end{figure}




\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{Figures/imputation.pdf}
    \caption{Overview of the LAMNr cross-modal imputation framework. An observed
    source image (e.g., T1-weighted MRI, $X^{(1)}$) is strictly mapped to its
    latent representation $z_O$ via the learned bijective flow $f^{(1)}$.
    Leveraging population-level Gaussian priors (mean $\mu$ and low-rank
    covariance $\Sigma$), the missing latent vector $\tilde{z}_U$ is estimated
    through the conditional expectation $\mu_{U|O}$. To bypass the memory
    bottleneck of high-dimensional 3D data, the covariance inversion is
    efficiently computed in the reduced subspace using the Woodbury matrix
    identity. Finally, the target image (e.g., Fractional Anisotropy,
    $\hat{X}^{(2)}$) is synthesized by projecting the imputed latent vector back
    to the data space via the inverse flow $(f^{(2)})^{-1}$.}
    \label{fig:imputation}
\end{figure}



\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{Figures/interpolation.pdf}
    \caption{Within-cohort interpolation (PPMI). (Top) The generated
    morphological transition between a source image ($t=1.0$) and a
    target image ($t=0.0$). (Bottom) A geometric representation of
    the joint latent space. The empirical distribution of the training cohort is
    centered around $\mu$. Standard linear interpolation (Lerp, dotted red line)
    cuts through the interior of the latent hypersphere, causing a severe
    contraction of the vector's norm (variance collapse). This forces the
    decoding flow to evaluate out-of-distribution coordinates, generating
    checkerboard artifacts. Conversely, applying Slerp relative to the empirical
    mean $\mu$ (solid green arc) preserves the natural variance of the data,
    ensuring the trajectory remains strictly on the high-probability manifold.}
    \label{fig:interpolation}
\end{figure}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{Figures/interpolation_brats_24.pdf}
    \caption{Out-of-cohort interpolation (BraTS).   (Top) The generated
    morphological transition between a source image ($t=1.0$) and a
    target image ($t=0.0$). (Bottom) A geometric representation of
    the joint latent space. The empirical distribution of the training cohort is
    centered around $\mu$. Standard linear interpolation (Lerp, dotted red line)
    cuts through the interior of the latent hypersphere, causing a severe
    contraction of the vector's norm (variance collapse). This forces the
    decoding flow to evaluate out-of-distribution coordinates, generating
    checkerboard artifacts. Conversely, applying Slerp relative to the empirical
    mean $\mu$ (solid green arc) preserves the natural variance of the data,
    ensuring the trajectory remains strictly on the high-probability manifold.
    Note that the skull-stripped and extreme pathology interpolation is coherent
    even though the training data (PPMI) is non-skull-stripped and without the
    presence of tumors.  }
    \label{fig:interpolation_brats_024}
\end{figure}


\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{Figures/temperature_scaling.pdf}
    \caption{Effect of temperature scaling on anomaly expression. Weighting the
    latent representation by a temperature factor $\tau$ contracts the vector
    toward the origin. At a low value of $\tau = 0.01$, the reconstruction
    approaches the mean of the healthy population. Increasing the temperature
    scaling factor toward $\tau = 0.99$ increases the data variance. This
    progression gradually reveals the expression of the out-of-distribution
    anomaly, ultimately displaying the original subject with their complete
    pathology.}
    \label{fig:temperature_scaling}
\end{figure}





<!-- ### Leveraging approximate template $\leftrightsquigarrow$ subject geodesic linearity for image registration via latent winsorization

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
 -->
