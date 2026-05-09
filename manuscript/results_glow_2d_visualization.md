
### Visualizing LAMNr Flows-based Deep Computational Anatomy

Using the 3-view model (2D) trained on the DLBS data, we demonstrate multiple
applications of the LAMNr flows framework within the context of Deep
Computational Anatomy (DCA). LAMNr flows provide a probabilistic,
likelihood-based alternative to traditional diffeomorphometry (e.g., LDDMM),
allowing for direct synthesis and manipulation of anatomical manifolds.

__LAMNr Flows-based population template.__ In the context of the learned data
manifold, the Fréchet mean of the anatomical distribution is efficiently
approximated by decoding the origin ($z=0$) of the latent space. By passing the
zero-vector of the isotropic Gaussian prior through the inverse flow, $x =
f^{-1}_\theta(0)$, we synthesize a canonical representation that captures the
central morphometric tendency of the cohort without the computational overhead
of iterative diffeomorphic averaging [@Avants:2010aa].  See Figure
\ref{fig:frechet_mean}.  Also, see ``lamnr_glow_tool_2/3d.py recon-template`` for
more details.
 
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
    multimodal ANTsX template, constructed via traditional iterative diffeomorphic
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
over this sampling radius.  Lower temperatures ($\tau<1$) contract the sampling
toward the high-density (but low-volume) region near the mean to generate
high-fidelity, canonical anatomies, while $\tau\approx1$ ensures that samples
are drawn from the typical set, capturing the diverse structural variations
characteristic of the true empirical distribution. See Figure
\ref{fig:generative_samples}. Also, see ``lamnr_glow_tool_2/3d.py sample`` for
more details.

\begin{figure}[!htbp]
    \centering
    \setlength{\tabcolsep}{0pt}
    \renewcommand{\arraystretch}{0} 

    % On utilise @{\hspace{1mm}} pour insérer exactement 1mm entre les colonnes
    \begin{tabular}{c@{\hspace{2mm}}c@{\hspace{1mm}}c@{\hspace{1mm}}c@{\hspace{1mm}}c@{\hspace{1mm}}c}
        {} & {$\mathbf \tau = 0.1$} & {$\tau = 0.25$} & {$\tau = 0.5$} & {$\tau = 0.75$} & {$\tau = 1.0$} \\[2mm] % Espace vertical
        \rotatebox[origin=c]{90}{\textbf{View 1: T1-w}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/Samples/samples_t1_temp_0.10.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/Samples/samples_t1_temp_0.25.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/Samples/samples_t1_temp_0.50.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/Samples/samples_t1_temp_0.75.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/Samples/samples_t1_temp_1.00.png}} \\
        \vspace{2mm} \\
        \rotatebox[origin=c]{90}{\textbf{View 2: FLAIR}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/Samples/samples_t2flair_temp_0.10.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/Samples/samples_t2flair_temp_0.25.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/Samples/samples_t2flair_temp_0.50.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/Samples/samples_t2flair_temp_0.75.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/Samples/samples_t2flair_temp_1.00.png}} \\
        \vspace{2mm} \\
        \rotatebox[origin=c]{90}{\textbf{View 3: FA}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/Samples/samples_fa_temp_0.10.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/Samples/samples_fa_temp_0.25.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/Samples/samples_fa_temp_0.50.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/Samples/samples_fa_temp_0.75.png}} &
        \raisebox{-0.5\height}{\includegraphics[width=0.19\linewidth]{Figures/Samples/samples_fa_temp_1.00.png}} \\
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
intermediate representations remain closer to the high-probability anatomical
manifold. This is demonstrated in Figure \ref{fig:interpolation} for both
within-cohort data (i.e., DLBS Wave 2) and out-of-cohort data (i.e., BraTS-Reg),
in terms of the model training data (i.e., DLBS Wave 1). We observe smooth
transitions for both the T1-w and FLAIR images between the source ($t=0.0$) and
target images ($t=1.0$).  In the case of the DLBS Wave 2 cohort, we selected two
subjects of different ages which illustrates the morphological interpolation
from larger to smaller ventricles and from the presence to absence of white
matter hyperintensities. We see similar high quality interpolations in a
BraTS-Reg subject (Subject 5, post- and pre-resection scans).  It is noteworthy
reiterating that training data did not include skull-stripped images.  See
``lamnr_glow_tool_2/3d.py recon-interpolate`` for specific implementation 
details.


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
        {\includegraphics[width=1.6cm]{Figures/interpolation/intra_dlbs_wave2_example_t1_t0.00.png}\includegraphics[width=1.6cm]{Figures/interpolation/intra_dlbs_wave2_example_t2flair_t0.00.png}};
      \node[inner sep=0] (dlbs_img25) at (4,2) 
        {\includegraphics[width=1.6cm]{Figures/interpolation/intra_dlbs_wave2_example_t1_t0.25.png}\includegraphics[width=1.6cm]{Figures/interpolation/intra_dlbs_wave2_example_t2flair_t0.25.png}};
      \node[inner sep=0] (dlbs_img50) at (8,2) 
        {\includegraphics[width=1.6cm]{Figures/interpolation/intra_dlbs_wave2_example_t1_t0.50.png}\includegraphics[width=1.6cm]{Figures/interpolation/intra_dlbs_wave2_example_t2flair_t0.50.png}};
      \node[inner sep=0] (dlbs_img75) at (12,2) 
        {\includegraphics[width=1.6cm]{Figures/interpolation/intra_dlbs_wave2_example_t1_t0.75.png}\includegraphics[width=1.6cm]{Figures/interpolation/intra_dlbs_wave2_example_t2flair_t0.75.png}};
      \node[inner sep=0, draw, thick, orange] (dlbs_img100) at (16,2) 
        {\includegraphics[width=1.6cm]{Figures/interpolation/intra_dlbs_wave2_example_t1_t1.00.png}\includegraphics[width=1.6cm]{Figures/interpolation/intra_dlbs_wave2_example_t2flair_t1.00.png}};

      % --- Ligne de séparation ---
      \draw[thick, gray!30] (-3, 0.5) -- (18, 0.5);

      % --- RANGÉE DU BAS : Reg-BRATS ---
      \node[inner sep=0, draw, thick, blue] (img0) at (0,-1) 
        {\includegraphics[width=1.6cm]{Figures/interpolation/inter_brats_example_t1_t0.00.png}\includegraphics[width=1.6cm]{Figures/interpolation/inter_brats_example_t2flair_t0.00.png}};
      \node[inner sep=0] (img25) at (4,-1) 
        {\includegraphics[width=1.6cm]{Figures/interpolation/inter_brats_example_t1_t0.25.png}\includegraphics[width=1.6cm]{Figures/interpolation/inter_brats_example_t2flair_t0.25.png}};
      \node[inner sep=0] (img50) at (8,-1) 
        {\includegraphics[width=1.6cm]{Figures/interpolation/inter_brats_example_t1_t0.50.png}\includegraphics[width=1.6cm]{Figures/interpolation/inter_brats_example_t2flair_t0.50.png}};
      \node[inner sep=0] (img75) at (12,-1) 
        {\includegraphics[width=1.6cm]{Figures/interpolation/inter_brats_example_t1_t0.75.png}\includegraphics[width=1.6cm]{Figures/interpolation/inter_brats_example_t2flair_t0.75.png}};
      \node[inner sep=0, draw, thick, orange] (img100) at (16,-1) 
        {\includegraphics[width=1.6cm]{Figures/interpolation/inter_brats_example_t1_t1.00.png}\includegraphics[width=1.6cm]{Figures/interpolation/inter_brats_example_t2flair_t1.00.png}};

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
      \draw[dashed, darkgray!80, thick] (mu) + (20:\R) arc (20:160:\R);
      \node[darkgray!80, align=center] at (0, 3.25) {High-Probability Manifold\\[-0.5ex] \small ($\|z - \mu\| \approx \text{const}$)};

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
      \node[red, align=center, below=0.1cm of L50] {Lerp\\ \small (Variance Collapse)};

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
      \draw[dashed, thick, blue!50, ->] (ZT) to[out=90, in=-90] (-8, 6.5);
      \draw[dashed, thick, darkgray!50, ->] (P25) to[out=90, in=-90] (-4, 6.5);
      \draw[dashed, thick, darkgray!50, ->] (P50) to[out=90, in=-90] (0, 6.5);
      \draw[dashed, thick, darkgray!50, ->] (P75) to[out=90, in=-90] (4, 6.5);
      \draw[dashed, thick, orange!50, ->] (ZS) to[out=90, in=-90] (8, 6.5);

  \end{scope}  
\end{tikzpicture}
}
\caption{Interpolation using the DLBS Wave 2 cohort (top row) and BraTS-Reg
cohort (second row).  Model training used only whole-head DLBS Wave 1 data
(T1-w, FLAIR, FA).  (Top) The generated morphological transition between a
source image ($t=1.0$) and a target ($t=0.0$) multimodal images (T1-w, FLAIR).
Interpolation DLBS data (Wave 2) included the source image (Subject 4488, Age
77) and target image (Subject 587, Age 53). BraTS-Reg is demonstrated using pre-
and post-resection T1-w and FLAIR images from Subject 5. (Bottom) A geometric
representation of the joint latent space. The empirical distribution of the
training cohort is centered around $\mu$. Standard linear interpolation (Lerp,
dotted red line) cuts through the interior of the latent hypersphere, causing a
severe contraction of the vector's norm (variance collapse). This forces the
decoding flow to evaluate out-of-distribution coordinates. Conversely, applying
Slerp relative to the empirical mean $\mu$ (solid green arc) better preserves
the natural variance of the data such that the trajectory follows the
high-probability manifold.}
\label{fig:interpolation}
\end{figure}

__Cross-modal imputation via Conditional Gaussian modeling.__ Missing modalities
are synthesized by encoding the available observed images to the latent space,
$z_O = f^{(O)}(\mathcal{X}^{(O)})$, and computing the exact conditional expectation of the
unobserved latent vectors, $\mu_{U|O}$, under the learned joint Gaussian prior.
Projecting this optimal estimate through the target modalities' inverse flows,
$\hat{\mathcal{X}}^{(U)} = (f^{(U)})^{-1}(\mu_{U|O})$, yields a high-fidelity imputation
that guarantees mathematical consistency with the population's cross-view
dependencies. Crucially, because the joint prior models the full multi-view
latent space simultaneously, this formulation is inherently flexible.  It
supports conditioning on any arbitrary subset of available data, enabling
complex many-to-many translations (e.g., synthesizing a single FA map from
combined T1-w and T2-w inputs, or simultaneously generating T2-w and FA from a
single T1-w scan).  See Figure \ref{fig:imputation}.  Also, see
``lamnr_glow_tool_2/3d.py impute`` for more details.

\begin{figure}[!htbp]
\centering
\resizebox{0.9\linewidth}{!}{%
\begin{tikzpicture}[>=latex, node distance=2cm]

  % --- CORE FLOW COMPONENTS ---
  
  % Input: T1
  \node (x1) at (0,0) {\textbf{Observed} ($\mathcal{X}$)};
  
  % Encoder
  \node[draw, fill=blue!10, minimum height=1.2cm, minimum width=1.5cm, align=center] (enc) at (2.5,0) {$f^{(O)}_{\theta}$};
  \node[text=blue, align=center, font=\scriptsize] at (2.5, -0.9) {Bijective\\Flow $\leftrightarrow$};

  % Observed Latent Vector
  \node[text=blue, font=\bfseries] (zO) at (4.5,0) {$z_O$};
  
  % Conditioning Box
  \node[draw, rounded corners, fill=gray!10, text width=5.cm, align=center, minimum height=2.5cm, inner sep=10pt] (cond) at (8,0) {
    \textbf{Latent Gaussian}\\
    \textbf{Conditioning}\\[0.3cm]
    $\mu_{U|O} = \mu_U + \Sigma_{UO}\Sigma_{OO}^{-1}(z_O - \mu_O)$\\[0.2cm]
    \scriptsize\textrm{(via Low-Rank Woodbury Identity)}
  };
  
  % Imputed Latent Vector
  \node[text=red, font=\bfseries] (zU) at (11.5,0) {$\tilde{z}_U$};
  
  % Decoder
  \node[draw, fill=red!10, minimum height=1.2cm, minimum width=1.5cm, align=center] (dec) at (13.5,0) {$(f^{(U)}_\theta)^{-1}$};
  \node[text=red, align=center, font=\scriptsize] at (13.5, -0.9) {Bijective\\Flow $\leftrightarrow$};

  % Output: FA
  \node (x2) at (16,0) {\textbf{Imputed} ($\hat{\mathcal{X}}$)};

  % --- IMAGE PLACEMENT (OBSERVED & IMPUTED) ---

  % --- PREMIERE RANGÉE ---

  % T1 Observed Images (Left side)
  \node[inner sep=0pt, below=2.5cm of x1, xshift=-0.8cm] (imgT1_00) 
    {\includegraphics[width=3cm]{Figures/dlbs_wave2_impute_FA_from_T1/000000_T1_input.png}};
  \node[inner sep=0pt, below=2.5cm of x1, xshift=2.4cm] (imgT1_01) 
    {\includegraphics[width=3cm]{Figures/dlbs_wave2_impute_FA_from_T1/000001_T1_input.png}};
  \node[inner sep=0pt, below=2.5cm of x1, xshift=5.6cm] (imgT1_02) 
    {\includegraphics[width=3cm]{Figures/dlbs_wave2_impute_FA_from_T1/000003_T1_input.png}};
  \node[below=0.1cm of imgT1_00, font=\normalsize] {Subject 00};
  \node[below=0.1cm of imgT1_01, font=\normalsize] {Subject 01};
  \node[below=0.1cm of imgT1_02, font=\normalsize] {Subject 02};
  \node[above=0.2cm of imgT1_01, font=\small\bfseries] {Observed T1};

  % FA Imputed Images (Right side)
  \node[inner sep=0pt, below=2.5cm of x2, xshift=-4.8cm] (imgFA_00) 
    {\includegraphics[width=3cm]{Figures/dlbs_wave2_impute_FA_from_T1/000000_FA.png}};
  \node[inner sep=0pt, below=2.5cm of x2, xshift=-1.6cm] (imgFA_01) 
    {\includegraphics[width=3cm]{Figures/dlbs_wave2_impute_FA_from_T1/000001_FA.png}};
  \node[inner sep=0pt, below=2.5cm of x2, xshift=1.6cm] (imgFA_02) 
    {\includegraphics[width=3cm]{Figures/dlbs_wave2_impute_FA_from_T1/000003_FA.png}};
  \node[below=0.1cm of imgFA_00, font=\normalsize] {Subject 00};
  \node[below=0.1cm of imgFA_01, font=\normalsize] {Subject 01};
  \node[below=0.1cm of imgFA_02, font=\normalsize] {Subject 02};
  \node[above=0.2cm of imgFA_01, font=\normalsize\bfseries] {Imputed FA};

  \draw[thick, gray!30] (-1.0, -8) -- (17.5, -8);

  % --- DEUXIÈME RANGÉE ---

  % FLAIR/FA Observed Images (Left side)
  \node[inner sep=0pt, below=2.5cm of imgT1_00, xshift=-0.75cm] (imgFlairFA_00) 
    {\includegraphics[width=1.5cm]{Figures/dlbs_wave2_impute_T1_from_T2FlairFA/000000_T2Flair_input.png}};
  \node[inner sep=0pt, below=2.5cm of imgT1_00, xshift=0.75cm] (imgFlairFA_10) 
    {\includegraphics[width=1.5cm]{Figures/dlbs_wave2_impute_T1_from_T2FlairFA/000000_FA_input.png}};

  \node[inner sep=0pt, below=2.5cm of imgT1_01, xshift=-0.75cm] (imgFlairFA_01) 
    {\includegraphics[width=1.5cm]{Figures/dlbs_wave2_impute_T1_from_T2FlairFA/000001_T2Flair_input.png}};
  \node[inner sep=0pt, below=2.5cm of imgT1_01, xshift=0.75cm] (imgFlairFA_11) 
    {\includegraphics[width=1.5cm]{Figures/dlbs_wave2_impute_T1_from_T2FlairFA/000001_FA_input.png}};
  
  \node[inner sep=0pt, below=2.5cm of imgT1_02, xshift=-0.75cm] (imgFlairFA_02) 
    {\includegraphics[width=1.5cm]{Figures/dlbs_wave2_impute_T1_from_T2FlairFA/000003_T2Flair_input.png}};
  \node[inner sep=0pt, below=2.5cm of imgT1_02, xshift=0.75cm] (imgFlairFA_12) 
    {\includegraphics[width=1.5cm]{Figures/dlbs_wave2_impute_T1_from_T2FlairFA/000003_FA_input.png}};
    
  \node[below=0.1cm of imgFlairFA_00, xshift=0.75cm, font=\normalsize] {Subject 00};
  \node[below=0.1cm of imgFlairFA_01, xshift=0.75cm, font=\normalsize] {Subject 01};
  \node[below=0.1cm of imgFlairFA_02, xshift=0.75cm, font=\normalsize] {Subject 02};
  \node[above=0.2cm of imgFlairFA_01, xshift=0.75cm, font=\small\bfseries] {Observed FLAIR/FA};

  % T1 Imputed Images (Right side)
  \node[inner sep=0pt, below=2.5cm of imgFA_00, xshift=-0.0cm] (imgT1w_00) 
    {\includegraphics[width=3cm]{Figures/dlbs_wave2_impute_T1_from_T2FlairFA/000000_T1.png}};
  \node[inner sep=0pt, below=2.5cm of imgFA_00, xshift=3.2cm] (imgT1w_01) 
    {\includegraphics[width=3cm]{Figures/dlbs_wave2_impute_T1_from_T2FlairFA/000001_T1.png}};
  \node[inner sep=0pt, below=2.5cm of imgFA_00, xshift=6.4cm] (imgT1w_02) 
    {\includegraphics[width=3cm]{Figures/dlbs_wave2_impute_T1_from_T2FlairFA/000003_T1.png}};
  \node[below=0.1cm of imgT1w_00, font=\normalsize] {Subject 00};
  \node[below=0.1cm of imgT1w_01, font=\normalsize] {Subject 01};
  \node[below=0.1cm of imgT1w_02, font=\normalsize] {Subject 02};
  \node[above=0.2cm of imgT1w_01, font=\normalsize\bfseries] {Imputed T1};

  % --- TROISIÈME RANGÉE ---

  \draw[thick, gray!30] (-1.0, -14.5) -- (17.5, -14.5);

  % T1 Observed Images (Left side)
  \node[inner sep=0pt, below=4.5cm of imgFlairFA_00, xshift=0.75cm] (imgT1x_00) 
    {\includegraphics[width=3cm]{Figures/dlbs_wave2_impute_T2FlairFA_from_T1/000000_T1_input.png}};
  \node[inner sep=0pt, below=4.5cm of imgFlairFA_00, xshift=3.95cm] (imgT1x_01) 
    {\includegraphics[width=3cm]{Figures/dlbs_wave2_impute_T2FlairFA_from_T1/000001_T1_input.png}};
  \node[inner sep=0pt, below=4.5cm of imgFlairFA_00, xshift=7.15cm] (imgT1x_02) 
    {\includegraphics[width=3cm]{Figures/dlbs_wave2_impute_T2FlairFA_from_T1/000003_T1_input.png}};
  \node[below=0.1cm of imgT1x_00, font=\normalsize] {Subject 00};
  \node[below=0.1cm of imgT1x_01, font=\normalsize] {Subject 01};
  \node[below=0.1cm of imgT1x_02, font=\normalsize] {Subject 02};
  \node[above=0.2cm of imgT1x_01, font=\small\bfseries] {Observed T1};

  % FLAIR/FA Observed Images (Right side)
  \node[inner sep=0pt, below=2.5cm of imgT1w_00, xshift=-0.75cm] (imgFlairFAx_00) 
    {\includegraphics[width=1.5cm]{Figures/dlbs_wave2_impute_T2FlairFA_from_T1/000000_T2Flair.png}};
  \node[inner sep=0pt, below=2.5cm of imgT1w_00, xshift=0.75cm] (imgFlairFAx_10) 
    {\includegraphics[width=1.5cm]{Figures/dlbs_wave2_impute_T2FlairFA_from_T1/000000_FA.png}};

  \node[inner sep=0pt, below=2.5cm of imgT1w_01, xshift=-0.75cm] (imgFlairFAx_01) 
    {\includegraphics[width=1.5cm]{Figures/dlbs_wave2_impute_T2FlairFA_from_T1/000001_T2Flair.png}};
  \node[inner sep=0pt, below=2.5cm of imgT1w_01, xshift=0.75cm] (imgFlairFAx_11) 
    {\includegraphics[width=1.5cm]{Figures/dlbs_wave2_impute_T2FlairFA_from_T1/000001_FA.png}};
  
  \node[inner sep=0pt, below=2.5cm of imgT1w_02, xshift=-0.75cm] (imgFlairFAx_02) 
    {\includegraphics[width=1.5cm]{Figures/dlbs_wave2_impute_T2FlairFA_from_T1/000003_T2Flair.png}};
  \node[inner sep=0pt, below=2.5cm of imgT1w_02, xshift=0.75cm] (imgFlairFAx_12) 
    {\includegraphics[width=1.5cm]{Figures/dlbs_wave2_impute_T2FlairFA_from_T1/000003_FA.png}};
    
  \node[below=0.1cm of imgFlairFAx_00, xshift=0.75cm, font=\normalsize] {Subject 00};
  \node[below=0.1cm of imgFlairFAx_01, xshift=0.75cm, font=\normalsize] {Subject 01};
  \node[below=0.1cm of imgFlairFAx_02, xshift=0.75cm, font=\normalsize] {Subject 02};
  \node[above=0.2cm of imgFlairFAx_01, xshift=0.75cm, font=\small\bfseries] {Imputed FLAIR/FA};

  % --- CONNECTIONS & DECORATIONS ---

  % Main flow arrows
  \draw[->, thick, blue] (x1) -- (enc);
  \draw[->, thick, blue] (enc) -- (zO);
  \draw[->, thick, blue] (zO) -- (cond);
  \draw[->, thick, orange] (cond) -- (zU);
  \draw[->, thick, orange] (zU) -- (dec);
  \draw[->, thick, orange] (dec) -- (x2);
  
  % Population Priors
  \node[align=center] (prior) at (8, 3.2) {Population Priors\\($\mu$, Low-Rank $\Sigma$)};
  \draw[->, dashed, thick, gray] (prior) -- (cond);

  % Connection lines from images to nodes
  % \draw[dashed, gray!50] (imgT1_00.north) -- (x1.south);
  % \draw[dashed, gray!50] (imgT1_01.north) -- (x1.south);
  % \draw[dashed, gray!50] (imgT1_02.north) -- (x1.south);
  % \draw[dashed, gray!50] (imgFA_00.north) -- (x2.south);
  % \draw[dashed, gray!50] (imgFA_01.north) -- (x2.south);
  % \draw[dashed, gray!50] (imgFA_02.north) -- (x2.south);

\end{tikzpicture}
}
\caption{(Top) Diagrammatic illustration of the Conditional Gaussian modeling
approach available through the LAMNr flows framework.  
Observed input features $\mathcal{X}$ are mapped to the latent representation
$z_O$ through the learned bijective flow $f_\theta$. Imputation of missing
modalities is performed via latent Gaussian conditioning modeling. The target
image $\hat{\mathcal{X}}$ is synthesized by projecting the imputed latent vector
$\tilde{z}_U$ back to the data space via the inverse flow $f_\theta^{-1}$.
(Bottom) Performance is demonstrated across three subjects under varying
observational constraints. (Row 1) Synthesis of Fractional Anisotropy (FA) maps
from observed T1-weighted inputs.  (Row 2) Joint reconstruction of T1-weighted
scans from observed FLAIR and FA modalities.  (Row
3) Simultaneous multi-modal imputation of FLAIR and FA from a single observed T1
input.}
\label{fig:imputation}
\end{figure}

__Latent distances.__ The bijective nature of normalizing flows allows complex
anatomical deviations to be quantified through a flexible suite of distance
metrics in the learned latent space, depending on the analytical objective.
Euclidean distance provides a straightforward measure of separation for basic
similarity assessments. To account for the natural variance of each latent
dimension, we implement a standardized Euclidean (diagonal Mahalanobis)
distance, $d = \sqrt{ \sum_i \frac{(z_i - \mu_i)^2}{\sigma_i^2 + \epsilon} }$,
which benchmarks a subject against the normative Gaussian mean ($\mu$) without
artificially penalizing high-variance anatomical traits.  For point-to-point
comparisons between the latents $z_j$ and $z_k$ of specific images, we utilize
geodesic distance derived from cosine similarity, $d =
\arccos(\text{clamp}(\text{sim}(z_j, z_k)))$. By measuring the angular
displacement on the hypersphere, this metric respects the spherical geometry of
the isotropic Gaussian prior, ensuring that anatomical transitions are evaluated
along the high-density manifold. These combined metrics yield a rigorous,
variance-weighted framework for anomaly detection and longitudinal assessment.
See Figure \ref{fig:latent_space_distances}.  Also, see ``lamnr_glow_tool_2/3d.py
calc-distance`` for more details.


\begin{figure*}[!htbp]
\centering
\begin{tikzpicture}[
    image_node/.style={inner sep=0pt, outer sep=0.5pt, anchor=north west},
    label_node/.style={font=\small\bfseries, anchor=base}
]

% 1. Paramètres de dimension
\def\imgw{2.1}   
\def\imgh{2.1}   
\def\hgap{0.08}  % Augmenté légèrement pour la clarté
\def\vgap{0.3}   
\def\groupgap{0.5} 

% 2. Header Labels (Calculs de centrage précis)
% Closest : milieu de l'image 2 (index 1) -> 1.5*imgw + hgap
\node[label_node] at ({(1.5*\imgw + 1*\hgap)*1cm - 0.25cm}, 0.3cm) {Closest};

% TEMPLATE : milieu de l'image 4 -> pos_template + 0.5*imgw
% pos_template = 3*(imgw + hgap) + groupgap
\node[label_node, color=blue!70!black] at ({(3.5*\imgw + 3*\hgap + \groupgap)*1cm - 0.25cm}, 0.3cm) {Template};

% Furthest : milieu de l'image 6 (index 1 du second bloc) -> pos_furthest_start + 1.5*imgw + hgap
% pos_furthest_start = 4*imgw + 3*hgap + 2*groupgap
\node[label_node] at ({(5.5*\imgw + 4*\hgap + 2*\groupgap)*1cm - 0.25cm}, 0.3cm) {Furthest};

% 3. Boucle principale
\foreach \display/\file [count=\r] in {
    Total/total, 
    Layer 1/layer_1, 
    Layer 2/layer_2, 
    Layer 3/layer_3, 
    Layer 4/layer_4, 
    Layer 5/layer_5%
} {
    \pgfmathsetmacro{\ypos}{-(\r-1) * (\imgh + \vgap)}
    
    % Titre de ligne
    \node[label_node, rotate=90, anchor=center] at (-0.3cm, {\ypos cm - (\imgh/2)*1cm}) {{\display}};

    % --- BLOC GAUCHE : 3 Closest ---
    \foreach \i in {1, 2, 3} {
        \pgfmathsetmacro{\xpos}{(\i-1) * (\imgw + \hgap)}
        \node[image_node] at (\xpos cm, \ypos cm) {
            \includegraphics[width=\imgw cm, height=\imgh cm, keepaspectratio]{Figures/min_max_distance_images/latent_distance_closest_\file_\i.png}
        };
    }
    
    % --- CENTRE : Le Template ---
    \pgfmathsetmacro{\xposT}{3 * (\imgw + \hgap) + \groupgap}
    \node[image_node, draw=blue!30, line width=1pt] at (\xposT cm, \ypos cm) {
        \includegraphics[width=\imgw cm, height=\imgh cm, keepaspectratio]{Figures/T_templateT1_slice115.png}
    };

    % --- BLOC DROIT : 3 Furthest ---
    \foreach \i in {1, 2, 3} {
        \pgfmathsetmacro{\xpos}{(\i-1) * (\imgw + \hgap) + (4*\imgw + 3*\hgap) + 2*\groupgap}
        \node[image_node] at (\xpos cm, \ypos cm) {
            \includegraphics[width=\imgw cm, height=\imgh cm, keepaspectratio]{Figures/min_max_distance_images/latent_distance_furthest_\file_\i.png}
        };
    }
}
\end{tikzpicture}
\caption{Visualization of latent space distance (cf Equation \ref{eq:geo_dist})
across the DLBS Wave 2 cohort with respect to the ANTsX T1-w template (cf Figure
\ref{fig:frechet_mean}).  We calculate the total latent space distance and the
distance for each hierarchical Glow layer and render the closest images (left) 
and the furthest images (right) centralizing the template as a visual reference
point.}
\label{fig:latent_space_distances}
\end{figure*}

