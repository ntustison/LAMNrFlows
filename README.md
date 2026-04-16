
## Deep Computational Anatomy via Latent-Aligned Multiview Normalizing Flows

<p align="center">
<img align="center" width="75%" src="manuscript/Figures/DiagramFromBrian.png">
</p>

<!-- <img align="center" width="50%" src="manuscript/Figures/lamnr_templates.png"><img align="center" width="45%" src="manuscript/Figures/lamnr_flows_illustration.png"> -->

----

Latent-aligned multiview normalizing (LAMNr) flows leverage exact-likelihood, bijective mappings to learn shared latent subspaces across heterogeneous, multimodal datasets. By applying formal latent-alignment constraints, the framework topologically unfolds anatomical manifolds into continuous vector spaces, enabling principled interpretations of computational anatomy concepts, such as population templates and geodesic interpolation. Evaluated on tabular IDPs and multimodal MRI, LAMNr flows improve calibrated likelihoods and downstream predictions compared to linear baselines. 


***

### RealNVP-based (tabulaire) LAMNr flows 

<details>
<summary>Network architecture</summary>


              RealNVP flow with alternative base distributions
              =================================================

                      +------------------------+
                      |       Input x          |
                      |        [B, D]          |
                      +------------------------+
                                   |
                                   v
                      +------------------------+
                      |  RealNVP block stack   |
                      |  K coupling steps      |
                      +------------------------+
                                   |
                                   v
                      +------------------------+
                      |      Latent z_K        |
                      |        [B, D]          |
                      +------------------------+
                          /               \
                         /                 \
                        v                   v

        +--------------------------------+      +-------------------------------------------+
        |     DiagGaussian base          |      |          GaussianPCA base                |
        |                                |      |                                           |
        |   z_K ~ N(0, I_D)              |      |   z_K ~ N(μ, W Wᵀ + σ² I_D)              |
        |                                |      |   u ~ N(0, I_M),  ε ~ N(0, I_D)          |
        |   (isotropic / diagonal        |      |   z_K = μ + W u + σ ε                    |
        |    Gaussian prior)             |      |   (low-rank + isotropic residual)        |
        +--------------------------------+      +-------------------------------------------+

              Same RealNVP encoder; only the base density p(z_K) differs.

</details>

<details>
<summary>Single view, uniform --> diagonal Gaussian (toy example)</summary>

<p align="center">
  <img src="examples/lamnr_tabular/Test_SimpleUniform/UniformSimulatedData/uniform_10000x4.png" alt="Input" width="75%"><br>
  Input<br>        
  <img src="examples/lamnr_tabular/Test_SimpleUniform/uniform_z_view0.png" alt="Output" width="75%"><br>
  Output
</p>

</details>

<details>
<summary>Multi-view NNHEmbed</summary>
  
Data from [*Joint representations from multi-view MRI-based learning support cognitive and functional performance domains*](https://www.medrxiv.org/content/10.1101/2025.09.27.25336706v2)

<p align="center">
  <img src="manuscript/Figures/clinical_comparison_multipanel2.png" alt="NNHEmbed" width="75%">
</p>

The forest plot illustrates the correlation uplift ($\Delta r$)
across two levels of comparison: (1) the gain from non-linear manifold
mapping, represented by the difference between LAMNr flows (with VICReg) and
the SiMLR baseline (i.e., red intervals), and (2) the gain from latent
alignment, represented by the difference between the aligned LAMNr model and
an unconstrained multi-view baseline ($\lambda = 0$, i.e., blue intervals).
Error bars represent the 95\% confidence intervals derived from 1000
bootstrap resamples. Top panel displays results for the NNL cohort, showing
significant non-linear gains in memory and executive function. Bottom panel
displays results for the PPMI cohort, where linear models remain highly
competitive. Significant improvements ($q < 0.05$, FDR corrected) are
indicated by intervals that do not cross the zero-reference line.

</details>

***

### Glow-based (image) LAMNr flows

<details>
<summary>Network architecture/configuration</summary>

__Single view normalizing flow__

```python
Input (image space)
-------------------
x : [B, 1, 128, 128]

          |
          | SQUEEZE (×4 channels, /2 spatial)
          v

Level 0 feature map
-------------------
h0: [B, 4, 64, 64]
    |
    | Glow blocks (K steps, invertible)
    v
    SPLIT (factor-out half the channels)
    +-----------------------------> z0: [B, 2, 64, 64]   (latent level 0)
    |
    +--> h1: [B, 2, 64, 64]  (remaining, goes deeper)

          |
          | SQUEEZE
          v

Level 1 feature map
-------------------
h1s: [B, 8, 32, 32]
     |
     | Glow blocks
     v
     SPLIT
     +----------------------------> z1: [B, 4, 32, 32]   (latent level 1)
     |
     +--> h2: [B, 4, 32, 32]

          |
          | SQUEEZE
          v

Level 2 feature map
-------------------
h2s: [B, 16, 16, 16]
     |
     | Glow blocks
     v
     SPLIT
     +----------------------------> z2: [B, 8, 16, 16]   (latent level 2)
     |
     +--> h3: [B, 8, 16, 16]

          |
          | SQUEEZE
          v

Level 3 feature map
-------------------
h3s: [B, 32, 8, 8]
     |
     | Glow blocks
     v
     SPLIT
     +----------------------------> z3: [B, 16, 8, 8]    (latent level 3)
     |
     +--> h4: [B, 16, 8, 8]

          |
          | SQUEEZE  (last time, because L=5)
          v

Level 4 (bottom level)
----------------------
h4s ≡ z4: [B, 64, 4, 4]          (latent level 4, NO split here)

All latents:
------------
z = { z0, z1, z2, z3, z4 }
```
__Latent-aligned multiview__

```python
                Latent-Aligned Multiview Normalizing Flows
                ==========================================

   x^(1) (T1)           x^(2) (T2)           x^(3) (FA)
 [B,1,128,128]        [B,1,128,128]        [B,1,128,128]
       |                     |                     |
       v                     v                     v
   +----------+          +----------+          +----------+
   |  Flow f1 |          |  Flow f2 |          |  Flow f3 |
   | (Glow,   |          | (Glow,   |          | (Glow,   |
   |  L = 5)  |          |  L = 5)  |          |  L = 5)  |
   +----------+          +----------+          +----------+
       |                     |                     |
       | z^(1) = {z_0..z_4}  | z^(2) = {z_0..z_4}  | z^(3) = {z_0..z_4}
       | (per-level latents) | (per-level latents) | (per-level latents)
       +----------+----------+----------+----------+
                  |                     |
                  v                     v

          +---------------------------------------------+
          |  Per-level alignment + Gaussian head        |
          |                                             |
          |  For ℓ = 0..4:                              |
          |    { z_ℓ^(v) }_(v=1..3)  ─→  projectors     |
          |                            ─→  alignment    |
          |    NLL from each flow     ─→  joint loss    |
          +---------------------------------------------+
```
</details>

<details>
<summary>Data augmentation:  (HCP-YA T1w template example)</summary>

```bash
iterations=100000
aug_params="noise_std:cos:0.05->0.015@${iterations},\
            sd_affine:cos:0.05->0.01@${iterations},\
            sd_deformation:linear:12.0->0.6@${iterations},\
            sd_simulated_bias_field:cos:0.20->0.03@${iterations},\
            sd_histogram_warping:cos:0.04->0.008@${iterations}"
```
Visualization cycle (~20 seconds)
<img src="manuscript/Figures/aug_dlbs.gif" alt="HCP YA T1 aug" width="100%">

</details>

<details>
Training evolution samples (100000 iterations)
<summary>Training LAMNr flows model (whole head, 2D, axial, 3 views)</summary>

* [OpenNeuro data](https://openneuro.org/datasets/ds004856) 
* [Command script](https://github.com/ntustison/LAMNrFlows/blob/main/examples/lamnr_glow/2D/command_train_lamnr_glow_2d_t1_t2flair_fa_whole_head.sh)
* [Trainer](https://github.com/ntustison/LAMNrFlows/blob/main/src/lamnrflows/train_lamnr_glow_2d.py)

<img src="manuscript/Figures/training_evolution_dlbs_2d_views.gif" alt="training evolution" width="100%">

</details>


<details>
<summary>Template comparison</summary>

<p align="center">
  <img src="manuscript/Figures/T_templateT1_slice115.png" alt="ANTsX T1" width="25%">
  <img src="manuscript/Figures/T_templateT2Flair_slice115.png" alt="ANTsX FLAIR" width="25%">
  <img src="manuscript/Figures/T_templateFA_slice115.png" alt="ANTsX FA" width="25%"><br>
  ANTsX multimodal template
  <br>
  <img src="manuscript/Figures/L_templateT1.png" alt="LAMNr flows T1" width="25%">
  <img src="manuscript/Figures/L_templateT2Flair.png" alt="LAMNr flows FLAIR" width="25%">
  <img src="manuscript/Figures/L_templateFA.png" alt="LAMNr flows FA" width="25%"><br>
  LAMNr flows multimodal template
</p>
Comparison of population Fréchet mean approximations. (Top) The standard
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
    structural signal of the dataset.

</details>


<details>
Training evolution samples (100000 iterations)
<summary>Training LAMNr flows model (MTL, 3D, 2 views)</summary>

<p align="center">
<img src="manuscript/Figures/training_evolution_hippo_views.gif" alt="training evolution" width="75%">
<\p>

</details>


***

### Funding support

We gratefully acknowledge the grant support of the Office of Naval Research (N0014-23-1-2317)
and the National Institute of Biomedical Imaging and Bioengineering (R01-EB031722).  
  

