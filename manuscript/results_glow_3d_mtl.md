
### Multiview Modeling of the Medial Temporal Lobe

\begin{figure}[!htbp]
    \centering
    \includegraphics[width=0.4\linewidth]{Figures/mtl_input_data_view0_cropped.png}
    \includegraphics[width=0.4\linewidth]{Figures/mtl_input_data_view1_cropped.png}
\caption{Representative 2D slices from the 3D T1-weighted (left) and T2-weighted
(right) medial temporal lobe (MTL) training pairs from the NIMH dataset. The
volumes have been cropped and rigidly normalized to the DeepFLASH template,
orienting the longitudinal axis of the hippocampus perpendicular to the axial
plane.}
\label{fig:nimh_mtl_training_input}
\end{figure}


We evaluated the ability of the proposed framework to capture the complex
three-dimensional geometry and tissue intensities of the medial temporal lobe
(MTL). The LAMNr flows model is volumetric ($40\times40\times64$ voxels )
comprising two views/modalities (T1-w, T2-w) common to imaging of the MTL
[@Yushkevich:2015aa,@Wisse:2017aa].  T1-w/T2-w imaging pairs ($N=249$) were 
taken from the NIMH dataset. The cropped volume for each image set was defined
by applying DeepFLASH [@Tustison:2024aa], an MTL segmentation application 
available in ANTsXNet to rigidly normalize to the DeepFLASH template with 
the long axis of the hippocampus oriented perpendicular to the axial plane
(see Figure \ref{fig:nimh_mtl_training_input}).

The component normalizing flow architecture was scaled to maximize spatial
expressivity while adhering to hardware constraints, utilizing three multiscale
resolution levels ($L=3$), 32 coupling steps per level ($K=32$), and 128 hidden
channels. Training was stabilized using an effective batch size of 64 
(``BATCH=8, GRAD_ACCUM=8``).  The data augmentation schedule was 

* ``noise_std: cos:0.05->0.02``
* ``sd_affine:  cos:0.00->0.00``
* ``sd_deformation:linear: 6.0->0.2``
* ``sd_simulated_bias_field:  cos:0.20->0.01``
* ``sd_histogram_warping:cos:0.04->0.002``

over 80000 iterations with a total of 100000 iterations.  To ensure anatomical
synchronization between modalities without overriding their respective
radiological characteristics, multimodal regularization was applied to the
latent space. Alignment was jointly driven by VICReg to maintain the invariance
of shared structural representations between T1 and T2 views, while strongly
penalizing variance to prevent dimensional collapse of the latent space.
Additionally, CCA screening was dynamically deployed to isolate and project 
the most highly correlated cross-modality features.

