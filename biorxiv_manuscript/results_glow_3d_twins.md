### T1-w Volumetric LAMNr Flows Model

\begin{figure}[!htbp]
  \centering
  
  % --- Première rangée : Trois images ---
  \begin{subfigure}{0.2\linewidth}
    \includegraphics[width=\linewidth]{Figures/low_resolution_3d_axial.png}
    % \caption{Optionnel} 
    % \label{fig:img1}
  \end{subfigure}
  \begin{subfigure}{0.2\linewidth}
    \includegraphics[width=\linewidth]{Figures/low_resolution_3d_sagittal.png}
    % \caption{Optionnel}
    % \label{fig:img2}
  \end{subfigure}
  \begin{subfigure}{0.2\linewidth}
    \includegraphics[width=\linewidth]{Figures/low_resolution_3d_coronal.png}
    % \caption{Optionnel}
    % \label{fig:img3}
  \end{subfigure}
  
  \vspace{1.5em} % Espace vertical entre la rangée d'images et l'histogramme

  % --- Deuxième rangée : Histogramme ---
  \begin{subfigure}{\linewidth}
    \centering
    \includegraphics[width=0.9\linewidth]{Figures/histogramme_rangs_twins_with_labels.png}
  \end{subfigure}
  
  \caption{Top row:  Canonical views of the latent-defined template from
  the 3D, T1-w volumetric LAMNr flow model constructed from the DLBS wave 1 data.  Second row:
  Distribution of latent similarity ranks between twins (with and without brain
  extraction) from the QTIM dataset. The latent distance was calculated from
  each subject to every other subject for which a ranking was derived per
  subject. A rank closer to 1 indicates the highest possible similarity (i.e.,
  indicative of the twin counterpart). The overlaid distributions compare
  imaging data including the effects of brain extraction. The vertical dashed
  lines indicate the respective medians of the two groups. Skull-stripped images
  significantly lower the median similarity rank to 36.5, compared to a median
  rank of 83 for whole-head images ($p < 0.001$).}
  \label{fig:twins_histogramme}
\end{figure}

The utility of low resolution 3D single-view, T1-w LAMNr flows models was
demonstrated on the QTIM (Twins) dataset.  It was hypothesized that the latent
distance could serve as a similarity index for predicting twin pairs.  Although
it was trained on whole-head data (DLBS, wave 1), we also demonstrate that it
generalizes to skull-stripped data (cf. Figure \ref{fig:interpolation}) and, in
fact, this pre-processing step improves prediction performance. T1-weighted
imaging volumes from the QTIM cohort were analyzed under two preprocessing
conditions: original full-head volumes and skull-stripped volumes isolating the
brain parenchyma [@tustison_antsx_2021]. Images were projected into the LAMNr
flows latent space, and an $N \times N$ inter-subject distance matrix was
generated to evaluate anatomical affinity by converting these distances into
similarity ranks. A rank of 1 indicates that a subject's twin is their nearest
neighbor in the latent space. Statistical results demonstrate significantly
higher discriminative power when the model processes brain-only images, with the
median similarity rank for twin pairs improving from 83 (whole head) to 36.5
(brain only). A paired Wilcoxon test confirmed that this improvement is
significant ($p < 0.001$). 