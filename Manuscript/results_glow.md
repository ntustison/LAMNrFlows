

\clearpage

## Glow-based LAMNr Flows

### Justification for 2D Slice-Based Methodology

The decision to conduct the majority of experiments—specifically hyperparameter optimization and ablation studies—using 2D slice-based representations (**256 x 256**) instead of full 3D volumes was necessitated by the **cubic scaling laws** of memory consumption inherent to Normalizing Flows.

#### Computational Complexity and VRAM Constraints
Glow-based architectures require storing all intermediate activations for both forward and backward passes to facilitate exact gradient computation. While 2D convolutional neural networks (CNNs) scale quadratically ($N^2$) with spatial resolution, 3D architectures scale cubically ($N^3$). The disparity in voxel count across resolutions is detailed in the table below:

| Dimensionality | Resolution | Total Units (Pixels/Voxels) | Scaling Factor (vs. $256^2$) |
| :--- | :--- | :--- | :--- |
| **2D** | 256 x 256 | 65,536 | 1x |
| **3D** | 64 x 64 x 64 | 262,144 | 4x |
| **3D** | 128 x 128 x 128 | 2,097,152 | 32x |
| **3D** | 256 x 256 x 256 | 16,777,216 | 256x |

#### Hardware Limitations
Experimental procedures were executed using a single **NVIDIA RTX A6000 GPU (48 GB VRAM)**. Empirical measurements indicated that at a **64 x 64 x 64** resolution with an architecture depth of **L = 4** and width **K = 16**, VRAM consumption reached **29 GB** with a micro-batch size of **8**. Projections for higher resolutions indicate that:

* **128^3 Resolution**: Memory requirements for a batch size of **1** exceed the **48 GB** threshold using standard backpropagation.
* **256^3 Resolution**: Estimated memory requirements exceed **320 GB**, rendering training impossible on standard workstation hardware without distributed model parallelism or reversible networking implementations.

#### Strategic Implementation

A "2D-first" approach allowed for rapid iteration through the hyperparameter
space—including learning rates, coupling layer complexity, and alignment loss
weights—while maintaining an effective batch size of 128 to ensure
convergence stability and robust ActNorm statistics. Findings were subsequently
validated at $64 \times 64 \times 64$ resolution to ensure geometric and volumetric
consistency.

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

