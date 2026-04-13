

\clearpage

## Glow-based LAMNr Flows

### Image Data and Preprocessing

Transitioning from tabular to image data within the LAMNr Flows context
introduces significant computational challenges, particularly with the Glow
architecture which, unlike standard CNNs, requires storing all intermediate
activations to compute exact gradients [@Gomez2017RevNet;@kingma2018glow]. This
requirement quickly saturates the VRAM (Video RAM), or memory, of the graphics
card. This memory bottleneck is exacerbated when moving from 2D to 3D.  For
instance, while a 2D slice at a standard resolution of $256^2$ contains 65,536
pixels, a corresponding 3D volume exceeds 16 million voxels, constituting a
256-fold increase in unit count. On high-capacity hardware (e.g., the NVIDIA RTX
A6000 48 GB utilized here), processing volumes even at reduced resolutions like
$48^3$ or $64^3$ necessitates strict architectural compromises, including
reduced hidden channels and micro-batch sizes, to prevent memory overflow.
Consequently, we adopt a dual organizational approach which utilizes
high-resolution 2D slices as a practical sandbox for visualizing the geometric
properties of the framework, while demonstrating that 3D LAMNr flows remain
robust and biologically informative even at lower resolutions. These constraints
are primarily a function of current hardware availability, as the software
framework is currently designed to scale with future computational resources.

We use four data cohorts in the experiments below:  the Dallas Life Brain Study
[@ds004856:1.3.0;@Park:2025aa], the NIMH Healthy Research Volunteer Dataset
[@ds005752:2.1.0], the Queensland Twin IMaging dataset
[@ds004169:1.0.6;@Strike:2019aa;@Koenders:2016aa], and a T1-weighted structural
MRI study of cannabis users at baseline and 3 years follow-up, and the Brain
Tumor Sequence Registration Challenge dataset
[@baheti2024braintumorsequenceregistration].  Whereas the first four datasets
are openly available at [OpenNeuro](https://openneuro.org/) (to facilitate
reader reproducibility), the fourth dataset is available upon request from the
challenge organizers.  These datasets are further summarized as follows:

* __Dallas Life Brain Study (DLBS).__ T1-weighted, FLAIR, diffusion-weighted
MRI.  Three longitudinal "waves" are included ($N_1=463$, $N_2=298$, $N_3=191$).

* __NIMH Research Volunteer Dataset (NIMH).__ T1-weighted, T2-weighted, 
diffusion-weighted MRI for $N=234$ complete subjects.  

* __Queensland Twin IMaging (QTIM).__ T1-weighted MRI ($N=1202$)
including family identifiers.

* __Cannabis users: Baseline and Follow-up at 3 Years (CBF3).__ T1-weighted MRI
($N=42$) including demographics and characterization of cannabis use.

* __Brain Tumor Sequence Registration Challenge (BraTS-Reg).__ T1-weighted,
T1-weighted contrast enhanced, T2-weighted, FLAIR from $N=140$ subjects,
featuring pre-operative and follow-up scans with expert-validated landmarks.  

Common preprocessing steps for all data include rigid normalization to a common
reference space, specifically the Nathan Kline Institute (NKI) template
[@tustison_largescale_2014] (1 mm$^3$, $192 \times 256 \times 224$) using
brain-extracted T1-weighted images [@tustison_antsx_2021] and ANTs registration
[@Avants:2014aa].  Cropped volumetric left and right MTL sections for
modeling were derived from the NIMH and CBF3 T1-w images using DeepFlash
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
        \raisebox{-0.5\height}{\includegraphics[width=0.31\linewidth]{Figures/samples_view0_it025000_to01.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.31\linewidth]{Figures/samples_view1_it025000_to01.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.31\linewidth]{Figures/samples_view2_it025000_to01.png}} \\
        \vspace{2mm} \\
        \rotatebox[origin=c]{90}{\textbf{50000 iterations}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.31\linewidth]{Figures/samples_view0_it050000_to01.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.31\linewidth]{Figures/samples_view1_it050000_to01.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.31\linewidth]{Figures/samples_view2_it050000_to01.png}} \\
        \vspace{2mm} \\
        \rotatebox[origin=c]{90}{\textbf{75000 iterations}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.31\linewidth]{Figures/samples_view0_it075000_to01.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.31\linewidth]{Figures/samples_view1_it075000_to01.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.31\linewidth]{Figures/samples_view2_it075000_to01.png}} \\
        \vspace{2mm} \\
        \rotatebox[origin=c]{90}{\textbf{100000 iterations}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.31\linewidth]{Figures/samples_view0_it100000_to01.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.31\linewidth]{Figures/samples_view1_it100000_to01.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.31\linewidth]{Figures/samples_view2_it100000_to01.png}} \\
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

* __3D Multiview Model (T1-w, T2-w).__ This volumetric model of the left
  hippocampus used NIMH dataset inputs cropped to a size of $40 \times 40 \times
  64$ voxels. The network configuration consists of $L=3$ levels with $K=32$
  coupling steps and $HC=128$ hidden channels. Due to the high structural
  correlation between T1 and T2 modalities in the hippocampus, a stronger
  alignment weight was applied (VICReg $\lambda=1.0$) with CCA-based screening.

* __3D Single-view Model (T1-w).__ This whole-head model was trained on DLBS
  wave 1 T1-weighted volumes downsampled to $64 \times 80 \times 64$ voxels. The
  architecture utilizes $L=4$ levels, $K=32$ steps, and $HC=96$. As this
  represents a single-view baseline, the alignment weight was set to
  $\lambda=0.0$, focusing purely on exact likelihood-based density estimation.

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
processing high-dimensional 3D volumes. The ``gauss-impute`` sub-function
facilitates the synthesis of missing modalities (e.g., T1 to FA) by leveraging
the "Push-Through" Woodbury identity to ensure numerical stability in high
dimensions. Furthermore, the toolkit supports the construction of population
templates (``recon-template``), latent temperature modulation to suppress
anatomical anomalies (``recon-temperature``), and geodesic interpolation that
respects the high-probability manifold (``recon-interpolate``). Finally,
rigorous anomaly detection relative to the cohort mean is provided through the
calculation of Mahalanobis or Euclidean distances (``calc-distance``).

### Visualizing LAMNr Flows-based Deep Computational Anatomy

Using the 3-view model (2D) trained on the DLBS data, we visually demonstrate
multiple DCA-based applications of the LAMNr flows framework.  These include
the population template, generative sampling, latent distance calculations for
biological inference, cross-modal imputation, and latent-based image 
interpolation.

__LAMNr Flows-based population template.__ In the context of the learned data
manifold, the Fréchet mean of the anatomical distribution can be efficiently
approximated by decoding the origin ($z=0$) of the latent space. By passing the
zero-vector of the isotropic Gaussian prior through the inverse flow, $x =
f^{-1}_\theta(0)$, we synthesize a canonical representation that captures the
central morphometric tendency of the cohort without the computational overhead
of iterative diffeomorphic averaging [@Avants:2010aa].  See Figure
\ref{fig:frechet_mean}.

 
\begin{figure}[!htbp]
    \centering
    % Supprime les marges par défaut pour un contrôle précis
    \setlength{\tabcolsep}{0pt}
    \renewcommand{\arraystretch}{0} 

    % On utilise @{\hspace{1mm}} pour insérer exactement 1mm entre les colonnes
    \begin{tabular}{c@{\hspace{2mm}}c@{\hspace{1mm}}c@{\hspace{1mm}}c}
        {} & {\bf View 1: T1-w} & {\bf View 2: FLAIR} & {\bf View 3: FA} \\[2mm] % Espace vertical
        \rotatebox[origin=c]{90}{\textbf{ANTsX}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.24\linewidth]{Figures/T_templateT1_slice115.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.24\linewidth]{Figures/T_templateT2Flair_slice115.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.24\linewidth]{Figures/T_templateFA_slice115.png}} \\
        \vspace{2mm} \\
        \rotatebox[origin=c]{90}{\textbf{LAMNr Flows}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.24\linewidth]{Figures/L_templateT1.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.24\linewidth]{Figures/L_templateT2Flair.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.24\linewidth]{Figures/L_templateFA.png}}
    \end{tabular}

    \caption{Comparison of population Fréchet mean approximations. (Top) The standard
    multivariate ANTsX template, constructed via traditional iterative diffeomorphic
    registration, representing a geometric spatial average that preserves
    high-frequency structural details. (Bottom) The generative latent-means,
    $f_\theta^{-1}(0)$, obtained in a single forward pass. The visually smoother
    appearance of the flow-generated template is a direct consequence of
    high-dimensional probabilistic modeling. As the exact mode of the latent
    distribution, it averages out idiosyncratic, high-frequency anatomical
    variations (such as specific cortical folding patterns) that do not strictly
    persist across the cohort. Instead of producing a single typical sample from
    the typical set, it models the macroscopic central morphological tendency and shared
    structural signal of the dataset.}

    \label{fig:frechet_mean}

\end{figure}

__Generative sampling.__ Sampling in LAMNr flows is performed by drawing a
latent vector $z$ from the isotropic Gaussian base distribution, $z \sim
\mathcal{N}(0, \tau^2 I)$, and mapping it back to the image domain via the
inverse flow, $x = f^{-1}_\theta(z)$. In high-dimensional latent spaces,
however, the probability mass concentrates within a thin "typical set" located
on a spherical shell of radius $\sqrt{d}\tau$, rather than near the mode at the
origin. Adjusting the temperature parameter $\tau$ allows for explicit control
over this sampling radius: lower temperatures ($\tau < 1$) contract the sampling
toward the high-density (but low-volume) region near the mean to generate
high-fidelity, canonical anatomies, while $\tau \approx 1$ ensures that samples
are drawn from the typical set, capturing the diverse structural variations
characteristic of the true empirical distribution. See Figure \ref{fig:generative_samples}.

\begin{figure}[!htbp]
    \centering
    \setlength{\tabcolsep}{0pt}
    \renewcommand{\arraystretch}{0} 

    % On utilise @{\hspace{1mm}} pour insérer exactement 1mm entre les colonnes
    \begin{tabular}{c@{\hspace{2mm}}c@{\hspace{1mm}}c@{\hspace{1mm}}c@{\hspace{1mm}}c@{\hspace{1mm}}c}
        {} & {$\mathbf \tau = 0.1$} & {$\tau = 0.25$} & {$\tau = 0.5$} & {$\tau = 0.75$} & {$\tau = 1.0$} \\[2mm] % Espace vertical
        \rotatebox[origin=c]{90}{\textbf{View 1: T1-w}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/samples_t1_temp_0.10.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/samples_t1_temp_0.25.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/samples_t1_temp_0.50.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/samples_t1_temp_0.75.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/samples_t1_temp_1.00.png}} \\
        \vspace{2mm} \\
        \rotatebox[origin=c]{90}{\textbf{View 2: FLAIR}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/samples_t2flair_temp_0.10.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/samples_t2flair_temp_0.25.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/samples_t2flair_temp_0.50.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/samples_t2flair_temp_0.75.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/samples_t2flair_temp_1.00.png}} \\
        \vspace{2mm} \\
        \rotatebox[origin=c]{90}{\textbf{View 3: FA}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/samples_fa_temp_0.10.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/samples_fa_temp_0.25.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/samples_fa_temp_0.50.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/samples_fa_temp_0.75.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/samples_fa_temp_1.00.png}} \\
    \end{tabular}

    \caption{Generative samples drawn from the learned LAMNr flow prior for each
    view at varying temperatures ($\tau$). Lower temperatures ($\tau \le 0.50$) generate coherent
    and anatomically representations whereas at higher temperatures
    (e.g., $\tau = 1.0$), the model samples extreme latent vectors from the prior
    tails.}
    \label{fig:generative_samples}
\end{figure}

__Pairwise image interpolation.__ Smooth morphological transitions between
anatomical scans are generated by interpolating their representations in the
learned latent space. To prevent the variance collapse and out-of-distribution
artifacts characteristic of standard linear interpolation, we employ a
$\mu$-centered spherical linear interpolation (Slerp). By applying the spherical
rotation relative to the population's empirical mean, $\mu$, the latent
trajectory better preserves the intrinsic data variance, ensuring all
intermediate representations remain closer the high-probability anatomical
manifold. This is demonstrated in Figure \ref{fig:interpolation} for both
within-cohort data (i.e., DLBS Wave 2) and out-of-cohort data (i.e., BraTS-Reg),
in terms of the model training data (i.e., DLBS Wave 1).  Both the T1-w and
FLAIR images between the source ($t=0.0$) and target images ($t=1.0$).  In the
case of the DLBS Wave 2 cohort, we selected an older subject (age = 93) as the
source image and a younger subject (age 25) as the target image which
illustrates the morphological interpolation from larger to smaller ventricles
and from the presence to absence of white matter hyperintensities. We see
similar high quality interpolations in a BraTS-Reg subject (Subject 5, post- and
pre-resection scans).  It is noteworthy that training data did not include
skull-stripped images.  

\begin{figure}[!htbp]
\centering
\resizebox{\linewidth}{!}{%
\begin{tikzpicture}[>=latex, font=\sffamily]% --- PARTIE A : Séquence d'Images ---
  \begin{scope}[shift={(0, 9.5)}]

      % Étiquettes de rangées pivotées
      \node[rotate=90, font=\bfseries] at (-2., 2) {DLBS};
      \node[rotate=90, font=\bfseries] at (-2., -1) {BraTS-Reg};
      
      % --- RANGÉE DU HAUT : SLERP ---
      % Chaque nœud contient deux images de 1.6cm (Total 3.2cm)
      \node[inner sep=0, draw, thick, blue] (dlbs_img0) at (0,2) 
        {\includegraphics[width=1.6cm]{Figures/intra_dlbs_wave2_example_t1_t0.00.png}\includegraphics[width=1.6cm]{Figures/intra_dlbs_wave2_example_t2flair_t0.00.png}};
      \node[inner sep=0] (dlbs_img25) at (4,2) 
        {\includegraphics[width=1.6cm]{Figures/intra_dlbs_wave2_example_t1_t0.25.png}\includegraphics[width=1.6cm]{Figures/intra_dlbs_wave2_example_t2flair_t0.25.png}};
      \node[inner sep=0] (dlbs_img50) at (8,2) 
        {\includegraphics[width=1.6cm]{Figures/intra_dlbs_wave2_example_t1_t0.50.png}\includegraphics[width=1.6cm]{Figures/intra_dlbs_wave2_example_t2flair_t0.50.png}};
      \node[inner sep=0] (dlbs_img75) at (12,2) 
        {\includegraphics[width=1.6cm]{Figures/intra_dlbs_wave2_example_t1_t0.75.png}\includegraphics[width=1.6cm]{Figures/intra_dlbs_wave2_example_t2flair_t0.75.png}};
      \node[inner sep=0, draw, thick, orange] (dlbs_img100) at (16,2) 
        {\includegraphics[width=1.6cm]{Figures/intra_dlbs_wave2_example_t1_t1.00.png}\includegraphics[width=1.6cm]{Figures/intra_dlbs_wave2_example_t2flair_t1.00.png}};

      % --- Ligne de séparation ---
      \draw[thick, gray!30] (-3, 0.5) -- (18, 0.5);

      % --- RANGÉE DU BAS : Reg-BRATS ---
      \node[inner sep=0, draw, thick, blue] (img0) at (0,-1) 
        {\includegraphics[width=1.6cm]{Figures/inter_brats_example_t1_t0.00.png}\includegraphics[width=1.6cm]{Figures/inter_brats_example_t2flair_t0.00.png}};
      \node[inner sep=0] (img25) at (4,-1) 
        {\includegraphics[width=1.6cm]{Figures/inter_brats_example_t1_t0.25.png}\includegraphics[width=1.6cm]{Figures/inter_brats_example_t2flair_t0.25.png}};
      \node[inner sep=0] (img50) at (8,-1) 
        {\includegraphics[width=1.6cm]{Figures/inter_brats_example_t1_t0.50.png}\includegraphics[width=1.6cm]{Figures/inter_brats_example_t2flair_t0.50.png}};
      \node[inner sep=0] (img75) at (12,-1) 
        {\includegraphics[width=1.6cm]{Figures/inter_brats_example_t1_t0.75.png}\includegraphics[width=1.6cm]{Figures/inter_brats_example_t2flair_t0.75.png}};
      \node[inner sep=0, draw, thick, orange] (img100) at (16,-1) 
        {\includegraphics[width=1.6cm]{Figures/inter_brats_example_t1_t1.00.png}\includegraphics[width=1.6cm]{Figures/inter_brats_example_t2flair_t1.00.png}};

      % Flèches et étiquettes (inchangées)
      \draw[->, thick, gray] (dlbs_img0) -- (dlbs_img25);
      \draw[->, thick, gray] (dlbs_img25) -- (dlbs_img50);
      \draw[->, thick, gray] (dlbs_img50) -- (dlbs_img75);
      \draw[->, thick, gray] (dlbs_img75) -- (dlbs_img100);

      \draw[->, thick, gray] (img0) -- (img25);
      \draw[->, thick, gray] (img25) -- (img50);
      \draw[->, thick, gray] (img50) -- (img75);
      \draw[->, thick, gray] (img75) -- (img100);

      \node[below=0.2cm of img0, font=\bfseries, text=blue] {Source ($t=0.0$)};
      \node[below=0.2cm of img25] {$t=0.25$};
      \node[below=0.2cm of img50] {$t=0.50$};
      \node[below=0.2cm of img75] {$t=0.75$};
      \node[below=0.2cm of img100, font=\bfseries, text=orange] {Target ($t=1.0$)};
  \end{scope}

  % --- PARTIE B : Géométrie Latente (Slerp vs Lerp) ---

  \begin{scope}[shift={(8, 0)}]
      % Centre de l'espace (Moyenne Gaussienne)
      \coordinate (mu) at (0,0);
      \fill (mu) circle (2.5pt) node[below=0.1cm] {$\mu$ (Population Mean)};

      % Rayon de la sphère latente (Norme / Variance conservée)
      \def\R{4.5}
      % Angles pour les vecteurs Target et Source
      \def\angT{145} % Cible à gauche
      \def\angS{35}  % Source à droite

      % Arc représentant la variété de haute probabilité (Isocontour Gaussien)
      \draw[dashed, gray!80, thick] (mu) + (20:\R) arc (20:160:\R);
      \node[gray!80, align=center] at (0, 3.25) {High-Probability Manifold\\[-0.5ex] \small ($\|z - \mu\| \approx \text{const}$)};

      % Définition des coordonnées vectorielles
      \coordinate (ZT) at (\angT:\R);
      \coordinate (ZS) at (\angS:\R);
      
      % Vecteurs
      \draw[->, thick, blue] (mu) -- (ZT) node[midway, left=0.2cm] {$z_{source}$};
      \draw[->, thick, orange] (mu) -- (ZS) node[midway, right=0.2cm] {$z_{target}$};

      % Ligne d'interpolation Linéaire (Lerp)
      \draw[->, thick, red, dotted] (ZT) -- (ZS);
      \coordinate (L50) at (0, {4.5*sin(35)}); % Point central du Lerp
      \fill[red] (L50) circle (2pt);
      \node[red, align=center, below=0.1cm of L50] {Lerp\\ \small (Variance Collapse \\ \& Out-of-Distribution)};

      % Arc d'interpolation Sphérique (Slerp)
      \draw[->, ultra thick, green!60!black] (mu) + (\angT:\R) arc (\angT:\angS:\R);
      \node[green!60!black, above=0.3cm] at (0, \R) {\textbf{Slerp (Variance Conserved)}};

      % Calcul des points sur l'arc Slerp
      \coordinate (P25) at (117.5:\R); % 145 - (110*0.25)
      \coordinate (P50) at (90:\R);    % 145 - (110*0.5)
      \coordinate (P75) at (62.5:\R);  % 145 - (110*0.75)

      % Tracé des points
      \fill[blue] (ZT) circle (3pt);
      \fill[black] (P25) circle (3pt);
      \fill[black] (P50) circle (3pt);
      \fill[black] (P75) circle (3pt);
      \fill[orange] (ZS) circle (3pt);

      % Lignes de projection allongées pour s'adapter au nouvel espacement (y=6.8 au lieu de 4.3)
      \draw[dashed, blue!50, ->] (ZT) to[out=90, in=-90] (-8, 6.5);
      \draw[dashed, gray!50, ->] (P25) to[out=90, in=-90] (-4, 6.5);
      \draw[dashed, gray!50, ->] (P50) to[out=90, in=-90] (0, 6.5);
      \draw[dashed, gray!50, ->] (P75) to[out=90, in=-90] (4, 6.5);
      \draw[dashed, orange!50, ->] (ZS) to[out=90, in=-90] (8, 6.5);

  \end{scope}  
\end{tikzpicture}
}
\caption{Interpolation using the DLBS Wave 2 cohort (top row) and BraTS-Reg
cohort (second row).  Model training used only DLBS Wave 1 data (T1-w, FLAIR,
FA).  (Top) The generated morphological transition between a source image
($t=1.0$) and a target ($t=0.0$) multimodal images (T1-w, FLAIR).  Interpolation
DLBS data (Wave 2) included the source image (Subject 1225, age 93) and target
image (Subject 612, age 25). BraTS-Reg is demonstrated using pre and post-
resection T1-w and FLAIR images from Subject 5. (Bottom) A geometric
representation of the joint latent space. The empirical distribution of the
training cohort is centered around $\mu$. Standard linear interpolation (Lerp,
dotted red line) cuts through the interior of the latent hypersphere, causing a
severe contraction of the vector's norm (variance collapse). This forces the
decoding flow to evaluate out-of-distribution coordinates. Conversely, applying
Slerp relative to the empirical mean $\mu$ (solid green arc) better preserves
the natural variance of the data such the trajectory follows the
high-probability manifold.}
\label{fig:interpolation}
\end{figure}












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


* __Temperature scaling.__ Weighting a latent representation by a scalar factor
  $\tau < 1.0$ contracts the vector toward the origin of the Gaussian
  prior. This variance reduction rigorously preserves the topology of the
  learned manifold, effectively suppressing out-of-distribution pathological
  anomalies and shifting the subject's anatomy toward the healthy population
  mean without introducing reconstruction artifacts. Figure
  \ref{fig:temperature_scaling}.




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

