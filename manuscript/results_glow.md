

\clearpage

## Glow-based LAMNr Flows

### Image Data and Preprocessing

Transitioning from tabular to image data within the LAMNr Flows context
introduces significant computational challenges.  The Glow architecture requires
storing all intermediate activations to compute exact gradients
[@kingma2018glow] which quickly saturates the VRAM (Video RAM), or memory, of
the graphics card. This memory bottleneck is particularly acute in 3D.  For
instance, while a 2D slice at a standard resolution of $256^2$ contains 65,536
pixels, a corresponding 3D volume exceeds 16 million voxels. On high-capacity
hardware (e.g., the NVIDIA RTX A6000 48 GB utilized here), processing volumes
even at reduced resolutions like $48^3$ or $64^3$ necessitates strict
architectural compromises, including reduced hidden channels and micro-batch
sizes, to prevent memory overflow. Consequently, we adopt a dual organizational
approach which utilizes high-resolution 2D slices as a practical sandbox for
visualizing the geometric properties of the framework, while demonstrating that
3D LAMNr flows remain robust and biologically informative even at lower
resolutions and structural applications. These constraints are primarily a
function of current hardware availability, as the software framework is
currently designed to scale with future computational resources.

We use five data cohorts in the experiments below: the Dallas Life Brain Study
[@ds004856:1.3.0;@Park:2025aa], the NIMH Healthy Research Volunteer Dataset
[@ds005752:2.1.0], the Queensland Twin IMaging dataset
[@ds004169:1.0.6;@Strike:2019aa], the Brain Tumor Sequence Registration
Challenge dataset [@baheti2024braintumorsequenceregistration], and the Open
Access Series of Imaging Studies 3 (OASIS-3) cohort [@lamontagne2019oasis3].
Whereas the first three datasets are openly available at
[OpenNeuro](https://openneuro.org/) (to facilitate reader reproducibility), the
BraTS-Reg dataset is available upon request from the challenge organizers, and
OASIS-3 is available upon request through the OASIS project. These datasets are
further summarized as follows:

* __Dallas Life Brain Study (DLBS).__ T1-weighted, FLAIR, diffusion-weighted
MRI. Three longitudinal "waves" are included ($N_1=463$, $N_2=298$, $N_3=191$).

* __NIMH Research Volunteer Dataset (NIMH).__ T1-weighted, T2-weighted, 
diffusion-weighted MRI for $N=234$ complete subjects.  

* __Queensland Twin IMaging (QTIM).__ T1-weighted MRI ($N=1202$)
including family identifiers.

* __Brain Tumor Sequence Registration Challenge (BraTS-Reg).__ T1-weighted,
T1-weighted contrast enhanced, T2-weighted, FLAIR from $N=140$ subjects,
featuring pre-operative and follow-up scans with expert-validated landmarks.

* __Open Access Series of Imaging Studies 3 (OASIS-3).__ Longitudinal multimodal 
neuroimaging (including T1-weighted MRI), clinical, and cognitive data (e.g., MMSE scores), 
utilizing standard FreeSurfer tabulated outputs to evaluate structural trajectories.

Common preprocessing steps for all data include rigid normalization to a common
reference space, specifically the Nathan Kline Institute (NKI) template
[@tustison_largescale_2014] (1 mm$^3$, $192 \times 256 \times 224$) using
brain-extracted T1-weighted images [@tustison_antsx_2021] and ANTs registration
[@Avants:2014aa].  Cropped volumetric left and right MTL sections for
modeling were derived from the NIMH and OASIS-3 images using DeepFlash
[@Tustison:2024aa], a deep-learning approach to parcellating specific structures
of the medial temporal lobe (MTL).  Similar cropped T1-w volumes from the DLBS
cohort were also generated for inference. Left and right MTLs for each subject
were rigidly registered independently using a label-based normalization approach
[@Roston2025.08.11.669599] to the DeepFlash template [@Tustison:2024aa],
oriented such that the long-axis of the hippocampus is perpendicular to the
coronal plane.  Fractional anisotropy (FA) images were derived from
diffusion-weighted imaging using Dipy [@dipy2014].

### Trained Models

\begin{figure}[!htbp]
    \centering
    \setlength{\tabcolsep}{0pt}
    \renewcommand{\arraystretch}{0} 

    % On utilise @{\hspace{1mm}} pour insérer exactement 1mm entre les colonnes
    \begin{tabular}{c@{\hspace{2mm}}c@{\hspace{1mm}}c@{\hspace{1mm}}c}
        {} & {\bf View 1: T1-w} & {\bf View 2: FLAIR} & {\bf View 3: FA} \\[2mm] % Espace vertical
        \rotatebox[origin=c]{90}{\textbf{25000 iterations}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.25\linewidth]{Figures/samples_view0_it025000_to01.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.25\linewidth]{Figures/samples_view1_it025000_to01.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.25\linewidth]{Figures/samples_view2_it025000_to01.png}} \\
        \vspace{2mm} \\
        \rotatebox[origin=c]{90}{\textbf{50000 iterations}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.25\linewidth]{Figures/samples_view0_it050000_to01.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.25\linewidth]{Figures/samples_view1_it050000_to01.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.25\linewidth]{Figures/samples_view2_it050000_to01.png}} \\
        \vspace{2mm} \\
        \rotatebox[origin=c]{90}{\textbf{75000 iterations}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.25\linewidth]{Figures/samples_view0_it075000_to01.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.25\linewidth]{Figures/samples_view1_it075000_to01.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.25\linewidth]{Figures/samples_view2_it075000_to01.png}} \\
        \vspace{2mm} \\
        \rotatebox[origin=c]{90}{\textbf{100000 iterations}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.25\linewidth]{Figures/samples_view0_it100000_to01.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.25\linewidth]{Figures/samples_view1_it100000_to01.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.25\linewidth]{Figures/samples_view2_it100000_to01.png}} \\
    \end{tabular}

    \caption{Generative samples drawn from the learned LAMNr flows model for the 3-view model 
    (T1, FLAIR, FA) over the course of optimization.  Samples were drawn at every
    1000 iterations to monitor the current training state of the model. Here we 
    show samples at 25000, 50000, 75000 and 100000 iterations. The sample temperature
    was $\tau = 0.8$ (i.e., samples were drawn from $\mathcal{N}(0, \tau^2 I)$.)}

    \label{fig:2d_dlbs_training}
\end{figure}


To demonstrate the versatility of the LAMNr flows framework across different
dimensionalities and anatomical scales, we trained three primary model
configurations. These models serve as the basis for the qualitative and
quantitative evaluations presented in the subsequent sections:

* __2D Multiview Model (T1-w, FLAIR, FA).__ This model was trained on mid-axial
  slices (index 115) from the DLBS wave 1 cohort. The architecture utilizes a
  $96 \times 128$ spatial resolution. Key hyperparameters include a multiscale
  depth of $L=5$ levels, $K=12$ coupling steps per level, and a hidden channel
  dimensionality of $HC=256$. Latent alignment was enforced using VICReg
  ($\lambda=0.005$) coupled with a CCA-based screening procedure (screening
  fraction = 0.5) to isolate shared anatomical features. Samples from the three
  modalities over the course of optimization is provided in Figure
  \ref{fig:2d_dlbs_training}.

* __3D Single-view Model (T1-w).__ This whole-head model was trained on DLBS
  wave 1 T1-weighted volumes downsampled to $48 \times 64 \times 56$ voxels. The
  architecture utilizes $L=3$ levels, $K=[16, 32, 64]$ steps, and $HC=96$. As this
  represents a single-view baseline, the alignment weight was set to
  $\lambda=0.0$, focusing purely on exact likelihood-based density estimation.

* __3D Multiview Model (T1-w, T2-w).__ This volumetric model of the left
  hippocampus used NIMH dataset inputs cropped to a size of $40 \times 40 \times
  64$ voxels. The network configuration consists of $L=3$ levels with $K=32$
  coupling steps and $HC=128$ hidden channels. Due to the high structural
  correlation between T1 and T2 modalities in the hippocampus, a stronger
  alignment weight was applied (VICReg $\lambda=1.0$) with CCA-based screening.

All models were optimized using the Adamax optimizer with a scheduled learning
rate ranging from $2.5 \times 10^{-5}$ to $5 \times 10^{-5}$. To ensure
numerical stability during the training of deep multiscale flows, we utilized
gradient norm clipping at $0.1$ or $0.2$ and employed mixed precision training
via a gradient scaler. To further stabilize the learned manifold and improve
generative quality, an Exponential Moving Average (EMA) of model parameters was
maintained throughout the optimization process.  Command line interfaces for these
workflows are provided through the Python-based CLIs ``train_lamnr_glow_2d.py`` and
``train_lamnr_glow_3d.py``, which natively handle dataset-level normalization,
coordinated data augmentation, and periodic logging of reconstructions for
visual quality assessment.

To support downstream inference and analysis, the framework includes the
``lamnr_glow_tool_2d.py`` and ``lamnr_glow_tool_3d.py`` CLI tools, which provide
extended functionality for sampling, reconstruction, and latent space
manipulation. These utilities allow for the fitting of conditional Gaussian
models to the learned latents via the ``gauss-fit`` sub-function, utilizing
low-rank (SVD) or diagonal covariance estimators to bypass memory errors when
processing high-dimensional image data. The ``gauss-impute`` sub-function
facilitates the synthesis of missing modalities (e.g., T1 to FA) by leveraging
the "Push-Through" Woodbury identity to ensure numerical stability in high
dimensions. Furthermore, the toolkit supports the construction of population
templates (``recon-template``), latent temperature modulation to suppress
anatomical anomalies (``recon-temperature``), and geodesic interpolation that
respects the high-probability manifold (``recon-interpolate``). Finally,
distances relative to the cohort mean is provided through the
calculation of Mahalanobis or Euclidean distances (``calc-distance``).





<!-- 
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
 -->


<!-- * __Temperature scaling.__ Weighting a latent representation by a scalar factor
  $\tau < 1.0$ contracts the vector toward the origin of the Gaussian
  prior. This variance reduction rigorously preserves the topology of the
  learned manifold, effectively suppressing out-of-distribution pathological
  anomalies and shifting the subject's anatomy toward the healthy population
  mean without introducing reconstruction artifacts. Figure
  \ref{fig:temperature_scaling}. 
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
-->





<!-- 
Scaling the LAMNr architecture to accommodate additional modalities (e.g., from
a bivariate T1/FA model to a trivariate T1/T2-FLAIR/FA configuration) introduces
a non-trivial geometric constraint on the shared latent subspace. Because the
inherent anatomical contrasts and noise profiles differ vastly across these
views—particularly the directional structural information in FA compared to the
scalar intensity profiles of T1 and T2-FLAIR—forcing a perfectly uniform latent
alignment becomes mathematically restrictive. In our bivariate models,
allocating 50% of the latent capacity to the shared space (SCREEN_FRAC=0.5,
PREFILTER_FRAC=0.5) provided ample bandwidth to encode complex cross-modal
structural dependencies. However, simply reducing this shared capacity (e.g., to
20%) to "protect" the network from aligning highly disparate signals creates a
severe information bottleneck. During conditional imputation, the network is
then forced to hallucinate missing high-frequency details from an overly
compressed shared subspace, degrading the fidelity of the generated target
modality.

To resolve this bottleneck in higher-dimensional multi-view settings, it is
necessary to maintain a wide shared latent bandwidth (e.g., SCREEN_FRAC=0.5,
PREFILTER_FRAC=0.5) while simultaneously relaxing the rigidity of the alignment
penalty. Rather than enforcing an exact L2 latent matching—which would force the
network to destroy unalignable fine-grained details—we utilize
Variance-Invariance-Covariance Regularization (VICReg). By lowering the global
alignment weight (e.g., ALIGN_WEIGHT=0.005) and softening the invariance penalty
(ALIGN_VICREG_INV=10.0), VICReg allows the network to learn correlated rather
than identical representations.
 -->


<!-- 
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
    resources for image volumes $> 64^3$ voxels.
-->


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

