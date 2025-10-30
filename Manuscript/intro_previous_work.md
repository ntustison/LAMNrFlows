
## Related work

### Cross-modal synthesis with flows

*DUAL-GLOW* (ICCV’19) proposes two **invertible, flow-based** networks—one for
MRI and one for PET—plus a latent relation network \(p(z_{\mathrm{PET}}\!\mid
z_{\mathrm{MRI}})\). The paper describes a RealNVP/Glow-style multiscale
architecture with affine coupling and reports **full 3-D** PET synthesis from
MRI after resampling to **\(64\times 96\times 64\)** at ~**1.5 mm** isotropic.
Conceptually this is close to our cross-modal mapping goal, but it does **not**
pursue per-level multi-view alignment or our CGM pipeline. [@sun2019dualglow]

Although the paper cites a public repository (**GitHub:** `haolsun/dual-glow`),
as of **2025-10-30** the released code provides **no inspectable evidence of a
normalizing-flow implementation**: there is no forward/inverse API with log-det
accumulation, no explicit affine-coupling/ActNorm modules or invertible
\(1{\times}1(\times1)\) convolutions, no multiscale squeeze/split operators, and
no round-trip/NLL tests. In short, the **released artifact does not substantiate
the claim of a flow-based model**; accordingly, we implement and report our own
**3-D Glow** baselines (matching depth/width where possible) and clearly label
them as ours.

### Invertible networks for multi-modal registration

**Invertible networks for multi-modal registration.** *INNReg* (ECCV’24)
converts multi-modal pairs into **mono-modal counterparts** via an **invertible
image-to-image (I2I) translation** network and then learns a **deformable
registration** on the translated images. The invertible module is described as
stacked affine‐coupling “InvBlocks” with **dynamic depthwise-convolution local
attention**; training couples pixel reconstruction, **INN cross-translation**
(enforcing \(T^{-1}(T(x))\approx x\)), a **barrier NMI** similarity term for
registration, and deformation **smoothness**. The registration head predicts a
**2-D displacement field** \(\phi=R(x,y)\). Conceptually adjacent in leveraging
invertibility with multiple contrasts, INNReg focuses on **geometric** alignment
after invertible translation; our work targets **latent** alignment inside a
likelihood-trained flow and **closed-form imputation** over aligned latents.
[@guo2024innreg;

**Reproducibility check (as of 2025-10-30).** The public repository (**GitHub:**
`MeggieGuo/INNReg`) is **present and populated** (e.g., `models/`, `util/`,
`options/`, `train.py`, `test.py`) and appears consistent with the paper’s
pipeline (**invertible I2I translation + deformable registration**). The paper
does **not** claim a normalizing-flow likelihood (no log-det/NLL), so the
absence of such code is **expected**. From the visible materials we **do not see
an external INN framework** (e.g., no explicit FrEIA dependency), suggesting the
invertible I2I network is **implemented in-repo**; however, the repo is
relatively sparse to inspect online, so we cannot definitively verify the exact
InvBlock internals without deeper code review. Experiments and deformation
prediction are **2-D**, which matches the paper’s scope. Overall: **the repo
broadly supports the paper’s claims** (unlike DUAL-GLOW); remaining uncertainty
concerns only low-level implementation details not easily inspectable from the
public pages. 

### Flow-based reconstruction

*Wen et al.* (ICML’23) introduce a **conditional normalizing flow (CNF)** for
**accelerated multi-coil MRI** that *samples from the posterior* rather than
producing a single point estimate. Technically, the method conditions an
invertible flow on features from a **UNet** applied to **zero-filled multi-coil
images**; the CNF **models the signal component in the measurement operator’s
nullspace**, then **combines** it with the measured component to enforce data
consistency. The implementation uses **FrEIA** (Framework for Easily Invertible
Architectures) with **conditional coupling blocks**,
**PyTorch/PyTorch-Lightning**, and **SigPy** for ESPIRiT coil maps; experiments
are **2-D slice-based** on **fastMRI (brain/knee)** with PSNR/SSIM and FID/cFID
metrics, and the authors emphasize **fast sampling** relative to
diffusion/Langevin alternatives. [@wen2023cnf]

**Reproducibility check (as of 2025-10-30).** The **official repository**
(**GitHub:** `jwen307/mri_cnf`) is present and populated (e.g., `models/`,
`train/`, `evals/`, configs), exposes **training/evaluation scripts** (e.g.,
`train_cnf.py`, `eval_cnf.py`), and documents the **FrEIA** dependency and
usage. Pretrained checkpoints are linked, and usage instructions target
**fastMRI** (multicoil) with PSNR/SSIM/FID evaluation. Overall, the codebase
**substantiates the paper’s claims** and is suitable as a baseline reference.
:contentReference[oaicite:1]{index=1}

**Scope and contrast to our work.** This CNF addresses a **single-modality
inverse problem** (MRI reconstruction) and does **not** tackle **multimodal
latent alignment** or **cross-modal imputation**. It operates on **2-D slices**
with conditional coupling via image-space features (no multiscale per-level
latent taps, no CCA-guided statistics). By contrast, our framework targets
**multimodal** inference with **per-level latent alignment** and **closed-form
Conditional Gaussian Modeling (CGM)** for missing-view imputation, followed by
exact flow decoding. [@wen2023cnf] 

### Harmonization with flows

*Beizaee et al.* propose **Harmonizing Flows**, an **unsupervised, source-free**
MRI harmonization framework that uses **normalizing flows** to model a *source*
scanner/site distribution and a lightweight **harmonizer** network to map
*target* images into that distribution. The pipeline is three-stage: (1) train a
**flow** on source images; (2) pre-train a **shallow harmonizer** to reconstruct
source images from their augmentations; (3) at **test time (source-free)**,
adapt the harmonizer so its outputs match the source distribution **under the
flow**, then deploy harmonized images for downstream tasks (e.g., adult/neonatal
segmentation, neonatal brain-age). This is a **distribution-alignment** method
across domains; it does not involve multi-view per-subject modeling.
[@beizaee2025harmonizingflows; @beizaee2024hf-arxiv]  
``Harmonizing Flows: Leveraging normalizing flows for unsupervised and
source-free MRI harmonization,'' **MedIA 2025**, with code released at
`farzad-bz/Harmonizing-Flows`. 

**Reproducibility check (as of 2025-10-30).** The public repository (**GitHub:**
`farzad-bz/Harmonizing-Flows`) mirrors the three-step design with top-level
folders `step1_Harmonizer_network/`, `step2_NF_model/`, and
`step3_Adapting_Harmonizer_using_NF/`, plus a README describing **2-D coronal
slice** preparation (ABIDE sites) and scripts to train the harmonizer and the
NF, then **adapt the harmonizer at test time** under the flow’s density.
Practically, this confirms a **slice-based (2-D)** implementation with a
**normalizing-flow density** and an adaptation loop; the code appears suitable
to reproduce the reported protocol.  Based on the structure and instructions
it’s reasonable to say home-grown PyTorch implementation, with no explicit
external NF library declared.

**Scope and contrast to our work.** Harmonizing Flows tackles
**cross-site/scanner distribution shift** using a learned **source density** and
**test-time harmonizer adaptation**, but it **does not** learn or use
**per-level, within-subject latent alignment** across modalities, nor any
**closed-form conditional Gaussian imputation** over flow latents. Our framework
targets **multimodal** (within-subject) inference: we align latents **per
level** with projector-guided objectives (Pearson/Barlow/VICReg/InfoNCE/HSIC)
and then perform **CGM** to impute missing views before exact inverse decoding
[@beizaee2025harmonizingflows].

### Latent-space imputation with flows

*EMFlow* casts imputation as **EM** over a Gaussian **latent prior** while
jointly learning an **invertible map** \(f\) between data and latents
(affine-coupling RealNVP-style). Training alternates (i) updating the flow
parameters by maximizing a masked likelihood/reconstruction on **observed**
entries and (ii) running **online EM** to update the latent Gaussian
\((\mu,\Sigma)\), which captures inter-feature dependence; missing entries are
then imputed by the latent conditional and mapped back through \(f^{-1}\). The
paper derives mini-batch EM with a Robbins–Monro stepsize and reports results on
tabular and image benchmarks. Experiments are on tabular and image datasets
(standard 2-D images), not 3-D volumes; there’s no mention of 3-D medical
imaging. We did **not** find a canonical public code release linked from the
paper/preprint at the time of writing even though the paper states that "they
provide code." [@ma2021emflow]

*CFMI* (Conditional Flow Matching for Imputation) trains **continuous
normalizing flows** (see comparison_df_vs_cnf.md) via **flow matching** to learn \(p(x_m \mid x_o, m)\)
directly. After training, imputation proceeds by solving the learned ODE for the
missing dimensions conditioned on observed ones. The authors benchmark across
many tabular datasets (and show zero-shot time-series imputation), emphasizing
computational efficiency relative to diffusion-style baselines. A research code
repository is available (GitHub: `vsimkus/cfmi`). All provided code targets
tabular and time-series data; there’s no 2-D/3-D imaging pipeline in the repo.
[@simkus2025cfmi]

**How this differs from our approach.** We also exploit **closed-form
conditioning** (EMFlow: closed-form Gaussian conditioning (single latent + EM).
CFMI: no closed-form—conditional ODE integration), but we do so over **multiscale, per-level latents** extracted
from an **exact-invertible image flow (Glow-style)** trained with **explicit
latent alignment** (Pearson/Barlow Twins/VICReg/InfoNCE/HSIC). Before
conditioning, we may project into a **CCA subspace** (with shrinkage/jitter for
SPD safety); after conditioning we **decode exactly** via the inverse flow,
enabling either posterior-mean or uncertainty-aware sampling. In contrast to
EMFlow’s single latent Gaussian and CFMI’s CNF-ODE sampling, our **Conditional
Gaussian Modeling (CGM)** provides **analytic** \(p(z_{\mathrm{mis}}\!\mid
z_{\mathrm{obs}})\) at each scale and is tailored to **multimodal medical
images** rather than generic tabular data. [@ma2021emflow; @simkus2025cfmi]

### Invertible fusion

*MMIF-INet* (Information Fusion, early access 2025) is representative of
**invertible image-fusion** methods that learn a **bijective map** between
multiple input modalities and a **single fused image**. Concretely, invertible
coupling blocks transform \((x^{(1)}, x^{(2)}, \ldots)\) into a shared latent in
which one branch is designated as the **fused output**, while the inverse map
recovers the individual modalities from the fused image. Training typically
combines **reconstruction/cycle-consistency** (exact invertibility),
**structure/edge preservation** (e.g., gradient or texture losses), and
**perceptual/SSIM** terms to balance contrast transfer and detail retention. In
practice, these systems are evaluated on **2-D** fusion benchmarks (e.g.,
infrared–visible or CT–MRI slice fusion) where a single fused image is desired.
[@he2025mmifinet]

**Scope and contrast to our work.** Invertible fusion aims to produce a **single
fused image** that mixes cues from all inputs and remains invertible for
modality recovery; it does **not** model a joint likelihood over modalities nor
expose **per-level latents** for statistics. By contrast, we keep
**modality-specific flows** trained under a **likelihood** objective, enforce
**explicit per-level latent alignment** (Pearson/Barlow/VICReg/InfoNCE/HSIC),
and perform **closed-form Conditional Gaussian Modeling (CGM)** to impute
**arbitrary missing patterns** before exact inverse decoding. Our target is
**multimodal inference** (alignment + imputation with calibrated NLL), not
fused-image generation.

**Reproducibility note** — MMIF-INet (invertible fusion)

- **Repository:** [`HeDan-11/MMIF-INet`](https://github.com/HeDan-11/MMIF-INet)
  — official implementation linked from the Information Fusion paper. As of
  **2025-10-30**, the repo shows **23 commits**, **16 stars**, and **4 forks**.
- **Key files / structure:** top-level Python sources include `MMIF_INet.py`
  (top network), `invblock.py` (invertible coupling blocks), `model.py`,
  `datasets_MSRS.py`, `test.py`, plus utility scripts. These align with an
  **invertible image-fusion** pipeline rather than a likelihood-trained flow.
- **2-D focus:** The README notes the method “**only suitable for the fusion of
  color images and grayscale images**,” and the provided dataset script targets
  standard **2-D** fusion benchmarks; there is **no advertised 3-D** support.
- **No NLL / log-det:** The codebase implements **invertible fusion blocks** but
  **does not expose** log-det/Jacobian accumulation, base densities, or NLL
  evaluation typical of **likelihood-trained normalizing flows**; training/eval
  rely on reconstruction/structure/perceptual losses, consistent with **fusion**
  rather than density modeling. 
- **Provenance:** The README explicitly credits prior invertible/fusion repos
  (e.g., **HiNet**) as references, indicating an **in-repo PyTorch**
  implementation (not FrEIA/nflows). 
- **Implication for our paper:** MMIF-INet is best cited as a representative
  **invertible fusion** baseline (bijective feature mixing with
  cycle/reconstruction losses). It is **not** a likelihood-trained flow and
  doesn’t target **multimodal latent alignment** or **closed-form imputation**,
  which are the focus of our approach.


**Summary of departures.** Compared with the above, we combine: (1)
**per-level** multi-view latent alignment (Pearson/Barlow/VICReg/InfoNCE/HSIC);
(2) a **CCA-guided safety clamp** to prevent collapse and stabilize statistics;
and (3) **conditional Gaussian modeling** to impute missing-view latents with
closed-form posteriors before exact decoding. Together these enable exact
likelihoods, cross-modal synthesis, and principled imputation in a single,
tested backbone.
