

### Multiview Modeling of the Medial Temporal Lobe

\begin{figure}[!htbp]
    \centering
    \includegraphics[width=0.4\linewidth]{Figures/mtl_input_data_view0_cropped.png}
    \includegraphics[width=0.4\linewidth]{Figures/mtl_input_data_view1_cropped.png}
\caption{Representative 2D slices from the 3D T1-weighted (left) and T2-weighted
(right) medial temporal lobe (MTL) training pairs from the NIMH dataset. The
volumes have been cropped and rigidly normalized to the DeepFLASH template,
orienting the longitudinal axis of the hippocampus perpendicular to the coronal
plane.}
\label{fig:nimh_mtl_training_input}
\end{figure}

We evaluated the ability of the proposed framework to capture clinically
relevant changes of the medial temporal lobe (MTL). The LAMNr flows model is
volumetric ($40\times40\times64$ voxels) comprising two views/modalities (T1-w,
T2-w) common to imaging of the MTL [@Yushkevich:2015aa,@Wisse:2017aa]. T1-w/T2-w
imaging pairs ($N=249$) were taken from the NIMH dataset [@ds005752:2.1.0].
Training included both the left and right (flipped) MTLs. The cropped volume for
each image set was defined by applying DeepFLASH [@Tustison:2024aa], an MTL
segmentation application available in ANTsXNet, to linearly normalize each
hemisphere to the DeepFLASH template with the long axis of the hippocampus
oriented perpendicular to the coronal plane (see Figure
\ref{fig:nimh_mtl_training_input}).

Each view-specific normalizing flow architecture was scaled to maximize spatial
expressivity while adhering to hardware constraints, utilizing three multiscale
resolution levels ($L=3$), 32 coupling steps per level ($K=32$), and 128 hidden
channels. Training was stabilized using an effective batch size of 64
(``BATCH=8, GRAD_ACCUM=8``). The data augmentation schedule was: ``noise_std:
cos:0.05->0.02``, ``sd_deformation:linear: 6.0->0.2``,
``sd_simulated_bias_field: cos:0.20->0.01``, and ``sd_histogram_warping:
cos:0.04->0.002`` over 80000 iterations with a total of 100000 iterations. To
ensure anatomical synchronization between modalities without overriding their
respective characteristics, multimodal regularization was applied
to the latent space. Alignment was jointly driven by VICReg to maintain the
invariance of shared structural representations between T1-w and T2-w views, while
penalizing variance to prevent dimensional collapse of the latent
space. Specifically, the VICReg objective was configured with an overall
alignment weight of 1.0, utilizing penalty coefficients of 25.0 for invariance,
25.0 for variance, and 1.0 for covariance, alongside a variance hinge threshold
($\gamma$) of 1.0. Additionally, Canonical Correlation Analysis (CCA) screening
was dynamically deployed to isolate and project highly correlated
cross-modality features. The CCA screening was activated after a warmup period
of 1000 iterations and refreshed every 5000 iterations. To ensure robust
feature selection, the screening retained the top 50\% of the features
(``SCREEN_FRAC=0.5``, ``PREFILTER_FRAC=0.5``) and applied a ridge penalty of
$10^{-3}$ to stabilize the covariance matrix inversion.

To validate the clinical utility of the learned structural representations, we
evaluated the LAMNr latent space using longitudinal data from the OASIS-3 cohort
[@lamontagne2019oasis3] which includes standard Freesurfer output compiled in
tabular form. Intra-subject structural trajectories were quantified with our
LAMNr flows approach by calculating the spherical linear interpolation (Slerp)
geodesic distance in the latent space between a subject's baseline scan and
subsequent follow-up visits for both the left and right (flipped) MTLs. We
compared the resulting statistical model of these latent geometric deformations
against a FreeSurfer composite volumetric biomarker (hippocampus, entorhinal
cortex, and parahippocampal cortex) for modeling cognitive decline
[@schwarz2016large], measured via the Mini-Mental State Examination (MMSE).

A linear mixed-effects (LME) model incorporating the standardized geodesic distance, 
age at visit, and intracranial volume (ICV) as fixed effects, with a random intercept 
for each subject, was formulated to evaluate the longitudinal trajectories. Specifically, 
the LAMNr spatial deformation model is defined as:

\begin{equation}
MMSE_{ij} = \beta_0 + \beta_1 \Delta L_{ij} + \beta_2 t_{ij} + \beta_3 Age_{ij} + \beta_4 ICV_i + b_{0i} + \epsilon_{ij}
\end{equation}

where $MMSE_{ij}$ is the clinical cognitive score for subject $i$ at visit $j$,
$\Delta L_{ij}$ is the standardized Slerp geodesic distance from the subject's
baseline latent representation, $t_{ij}$ represents the longitudinal time
elapsed (in years) since the baseline scan, $Age_{ij}$ is the standardized age
at the time of the visit, and $ICV_i$ is the standardized intracranial volume.
The term $b_{0i} \sim \mathcal{N}(0, \sigma_b^2)$ represents the
subject-specific random intercept accounting for baseline cognitive variability,
and $\epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)$ is the residual error. 

Analysis of the cohort follow-up data revealed that longitudinal divergence
along LAMNr geodesic trajectories is significantly, and negatively, associated
with cognitive performance ($\beta = -0.142$, $p < 0.01$). Specifically, greater
geodesic distance from a subject's baseline representation corresponds to a
steeper decline in MMSE scores. Notably, when stratifying the cohort to evaluate
preclinical sensitivity (i.e., restricting the analysis exclusively to subjects
clinically diagnosed as cognitively normal) the latent distance remained a
robust predictor of subtle MMSE variations ($p < 0.001$). This indicates that
the proposed approach potentially captures early, sub-macroscopic morphological
shifts in the MTL that precede the gross volumetric tissue loss traditionally
isolated by macroscopic segmentation workflows.

For methodological comparison, the macroscopic volumetric model substitutes the
latent geodesic distance with the standardized FreeSurfer composite AD signature
volume ($V_{ij}$):

\begin{equation}
MMSE_{ij} = \beta_0 + \beta_1 V_{ij} + \beta_2 t_{ij} + \beta_3 Age_{ij} +
\beta_4 ICV_i + b_{0i} + \epsilon_{ij}.
\end{equation}

As anticipated for a targeted macroscopic biomarker, the FreeSurfer composite 
AD signature demonstrated a robust positive association with cognitive 
performance ($\beta = 0.830$, $p < 0.001$), confirming that gross volumetric 
atrophy in these regions strongly parallels clinical decline. To contextualize 
these findings against standard macroscopic techniques, we compared the global 
model fits. While the FreeSurfer composite AD formulation yielded a lower Akaike 
Information Criterion (AIC = 4290.3) than the LAMNr geodesic trajectory model 
(AIC = 4389.9) across the full follow-up cohort, this performance differential 
is methodologically consistent. FreeSurfer relies on explicit spatial priors 
and supervised atlas-based segmentations specifically engineered to isolate 
the macroscopic epicenters of Alzheimer's disease pathology. Conversely, the 
LAMNr framework is entirely unsupervised and data-driven, lacking explicit 
anatomical priors.