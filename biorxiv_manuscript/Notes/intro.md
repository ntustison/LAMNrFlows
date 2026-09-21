
\clearpage

# Introduction

Modern neuroimaging studies routinely acquire **co-registered, complementary
contrasts** (e.g., T1, T2, FA), and downstream analyses increasingly require
*both* high-fidelity synthesis across modalities and principled uncertainty
quantification. Among deep generative models, **normalizing flows** offer an
attractive backbone: they are *exactly invertible* and provide **tractable
log-likelihoods** and bits‑per‑dimension via the change‑of‑variables formula,
enabling calibrated comparisons and straightforward cross‑modal mappings
[@dinh2016realnvp; @kingma2018glow; @papamakarios2021nfreview]. In contrast,
**diffusion/score-based** models achieve state‑of‑the‑art perceptual quality but
typically require many function evaluations at sampling time and rely on
variational or ODE-based likelihood surrogates [@ho2020ddpm;
@nichol2021improved; @song2020score; @karras2022edm; @rombach2022ldm]. For
clinical or large‑scale cohort contexts, the ability to compute likelihoods
exactly and to invert mappings in a *single pass* is practically useful.

Recent work from Apple—TarFlow—demonstrates that flows can match diffusion-like
sample quality while setting new SOTA on likelihood for images, using a
Transformer-autoregressive flow over patch tokens that alternates causal
directions across layers [@zhai2024tarflow]. In short: normalizing flows are
again a front-line option for large-scale generative modeling, with active
follow-ups (e.g., STARFlow scaling in autoencoder latent space) underscoring
momentum [@gu2025starflow].

Trade-offs vs. Glow (our backbone): TarFlow leverages Transformers and
autoregression, which (i) introduces sequential dependencies during sampling
(slower than Glow’s fully parallel inverse), and (ii) incurs quadratic
self-attention cost in token count, raising memory/compute for high-res imagery.
By contrast, Glow uses convolutional coupling with invertible 1×1
convolutions, enabling single-pass parallel sampling and memory-efficient
training via reversibility at the cost of lower per-layer expressivity than a
full Transformer [@kingma2018glow; @papamakarios2021nfreview]. (Community
reports also note TarFlow’s sampling speed concerns in practice.)

Addtionally, to our knowledge, there is no publicly available 3-D/volumetric
TarFlow implementation; the paper and releases focus on 2-D images (and a video
variant), not voxel volumes. Practically, 3-D is currently prohibitive because
volumetric token counts grow cubic in resolution while Transformer
self-attention is quadratic in tokens (steep VRAM/time), and TarFlow’s
autoregressive sampling is sequential—so generation slows dramatically compared
with Glow’s fully parallel inverse. Moreover, training at medical-scale would
demand very large compute/datasets plus specialized memory-aware tricks
(windowed/axial attention, factorized autoregression) that aren’t yet standard
in flow toolchains.  In addition, 3-D Glow uses conv-coupling + invertible 1×1×1
convs, so compute scales roughly linearly in voxel count (per-voxel convs), not
$O(N)^2$ in tokens.  Autoregressive sampling is not used in 3-D Glow.
Generation is a single, fully parallel inverse pass, not a long sequential loop.

In practice, however, Glow implementations are brittle when implemented
naively: subtle mistakes in the **squeeze/unsqueeze** or **split/merge** order
can lead to channel mismatches at inversion time; unstable log‑det tracking and
insufficient shape asserts further complicate training at volume scale.
Moreover, simply coupling modalities in a shared encoder does not guarantee
**cross‑view consistency**: under‑alignment leads to ghosting or texture
leakage; over‑alignment blurs contrast‑specific detail. Finally, loss balancing
is non‑trivial—different modalities and levels exhibit **heteroscedastic
(aleatoric) variability**, making any single fixed weight sub‑optimal across the
training trajectory [@kendall2018mtl; @kendall2017uncertainties].



We address these issues in two layers. **First**, we provide a hardened 2D/3D
Glow implementation within *normflows* and integrate it with ANTsTorch data/IO.
The implementation corrects the multiscale reshape pipeline, adds 3D invertible
components (Invertible1×1×1Conv, ActNorm‑3D), stabilizes log‑det accumulation,
and exposes a reproducible CLI with mixed precision, EMA, resumable checkpoints,
and tests—mirroring ANTsX’s emphasis on *transparent, portable tooling*.
**Second**, we introduce **explicit latent alignment** applied **per multiscale
level** via lightweight projector heads. The framework unifies several
objectives—**Barlow Twins**, **VICReg**, **InfoNCE**, **HSIC**, and Pearson
correlation—under a single training interface [@zbontar2021barlow;
@bardes2021vicreg; @oord2018cpc; @gretton2005hsic]. To guard against collapse
along a few dominant axes and to stabilize downstream statistics, we employ a
**CCA‑guided** subspace and optional clamp (rank \(k\), strength \(\alpha\)) as
an alignment‑adjacent safety mechanism [@hotelling1936; @andrew2013dcca].

Building on aligned latents, we formulate **Conditional Gaussian Modeling
(CGM)** for **multimodal imputation**. For each level, we estimate dataset
means/covariances—optionally after projecting into the CCA subspace and applying
shrinkage/jitter for SPD safety—and compute the **closed‑form conditional**
\(p(Y\!\mid\!X)\) to impute missing‑view latents. We can decode either the
posterior mean (denoised) or samples (uncertainty‑aware) through the exact flow
inverse, with user‑visible controls for subspace rank \(k\), clamp \(\alpha\),
temperature \(\tau\), and jitter \(\varepsilon\). This approach preserves
**invertibility**, exposes **uncertainty** at the latent level, and accommodates
**arbitrary missingness** without retraining.

**Contributions.** In summary, this work (i) delivers a robust 2D/3D Glow
implementation with ANTsTorch integration suitable for volume data; (ii)
provides a **per‑level latent‑alignment** framework spanning Pearson/**Barlow
Twins**/**VICReg**/**InfoNCE**/**HSIC** with an optional **CCA‑guided**
subspace/clamp; and (iii) specifies a **CGM** pipeline for principled imputation
over aligned latents. Motivated by Kendall–Gal, we also outline an
**aleatoric‑aware** weighting of alignment terms as an optional extension to
reduce manual tuning. Together, these components produce a single, tested
backbone that supports exact likelihoods, cross‑modal synthesis, and
dataset‑level imputation—aligned with the ANTsX philosophy of modular,
reproducible scientific software.




<!--
*Organization.* Section 2 reviews related work in flows, alignment, and
uncertainty. Section 3 details library changes. Section 4 presents
latent‑aligned training. Section 5 describes CGM for imputation.
-->