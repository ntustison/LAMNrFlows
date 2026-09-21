
# Why the library matters (and why we built on `normflows`) — **strong internal note**

A dependable **normalizing flows** stack is the difference between “we have a
cool demo” and “we can **ship** multimodal, 3-D, likelihood-trained models with
tests, alignment, and imputation.” After surveying the ecosystem,
**`normflows`** was the only PyTorch library that let us build what we needed
**quickly and reliably**. It had clean abstractions, sane defaults, and enough
extensibility to support our **3-D Glow**, **per-level latent taps**, and
**CGM** pipeline. In our hands, it was simply **more robust and better
engineered** for image work than the other options we tried. (Stimper et al.,
JOSS 2023.)

By contrast, here is how the other commonly cited options landed for our
use-case:

- **FrEIA** — a solid invertible-net toolkit with a graph API. Powerful, but we
  found the **plumbing overhead** high for multiscale, image-centric Glow
  (especially **3-D**), with more ceremony to expose **per-level taps** and
  fewer “batteries-included” pieces for our workflow. The net effect for us:
  **slowed iteration**.

- **TensorFlow Probability (bijectors/RealNVP)** — conceptually rich, but the
  practical path to **conv-Glow** with **invertible 1×1(×1)** and multiscale
  squeeze/split required a lot of reshaping and custom scaffolding. We also hit
  **ergonomics/compatibility friction** (TF2/Keras variable scoping,
  conditioning templates). In our environment it was **brittle**.

- **nflows** — excellent for **tabular/VI** and general density estimation
  (MAF/NSF/IAF), but **not** a turnkey base for **image-centric, multiscale
  Glow**—let alone **3-D**. Great toolbox, wrong fit for this project.

- **FlowTorch / Zuko** — clean APIs and pleasant developer experience for
  general flows, but again **not aimed** at multiscale conv-Glow image
  pipelines. Good for research prototyping; we would still have had to build a
  lot ourselves.

Bottom line for our stack: **`normflows` was the only foundation that let us
move fast *and* stay correct**; from there, we invested heavily to turn it into
a **multimodal, 3-D-ready, likelihood-trained** platform with **explicit latent
alignment** and **closed-form imputation**.

**Our engineering adds on top of `normflows`**  
> • 3-D Glow backbone (Invertible **1×1×1** convs, ActNorm3D, multiscale
> plumbing with strict forward/inverse tests)  
> • Per-level projector taps + five alignment objectives (Pearson, Barlow Twins,
> VICReg, InfoNCE, HSIC)  
> • **CGM**: shrinkage covariances, CCA subspaces, Schur-complement
> conditioning, uncertainty maps, exact inverse decode  
> • AMP/EMA training, warmup, gradient clipping, resume semantics; PyTest suite
> (shape/Jacobian/SPD/round-trip)  
> • Data & eval tooling: multi-view loaders, missingness protocols, PSNR/SSIM,
> calibration diagnostics, retest checks

## Quick survey of normalizing-flow libraries (our pragmatic take)

Below is a concise survey you can keep in the notes. It’s deliberately
**engineering-centric** (what helps or slows us for image-centric, multiscale,
3-D work).

- **`normflows`** (PyTorch) — **Best practical base we found** for image flows.
  Good composability; easy to extend toward **Glow-like** conv stacks. We built
  our 3-D path, per-level taps, and CGM on top of it.

- **FrEIA** — Great **INN graph** framework; powerful for custom invertible
  networks. For our aims (multiscale conv-Glow, per-level statistics, 3-D), we
  hit **integration overhead** and slower iteration.

- **nflows** — Classic flow toolkit (MAF/NSF/etc.) with strong
  density-estimation pedigree. **Not** focused on conv-Glow or multiscale image
  pipelines; good for tabular/VI.

- **FlowTorch** — Clean PyTorch flows API; research-friendly.
  **General-purpose**, not tuned for conv-Glow image stacks.

- **Zuko** — Modern flows with nice docs and API. Similar story: **general
  flows**, not image-Glow turnkey.

- **Glasflow** — Convenience layers **on top of nflows**; helpful for
  experiments, but image-Glow still requires custom work.

- **Pyro transforms** — Flows inside a probabilistic programming ecosystem.
  Great for VI; not a plug-and-play image-Glow stack.

- **TensorFlow Probability (bijectors)** — Rich math catalog; **engineering
  friction** for conv-Glow and 3-D image work in our environment.

- **Flow-matching/CNF toolkits** (e.g., TorchCFM,
  facebookresearch/flow_matching) — Excellent for **CNFs with flow-matching**
  objectives. Different animal than **discrete** Glow; no exact NLL, ODE at
  inference.

---

## Feature snapshot

| Library             | Glow-style conv **image** flows | **3-D** conv path        | Multiscale squeeze/split | Built-in NLL/log-det utils | CNF / Flow-matching |
|---------------------|----------------------------------|---------------------------|--------------------------|----------------------------|---------------------|
| **normflows**       | \cmark{} good base, extensible   | \pmark{} via our extensions | \cmark{}                 | \cmark{}                   | \pmark{} limited    |
| **FrEIA**           | \pmark{} possible with effort    | \pmark{} possible         | \cmark{} (graph-based)   | \cmark{} (via modules)     | \xmark{}            |
| **nflows**          | \xmark{} (MLP-oriented)          | \xmark{}                  | \pmark{} partial         | \cmark{}                   | \xmark{}            |
| **FlowTorch**       | \pmark{} general flows           | \xmark{}                  | \pmark{} partial         | \cmark{}                   | \xmark{}            |
| **Zuko**            | \pmark{} general flows           | \xmark{}                  | \pmark{} partial         | \cmark{}                   | \xmark{}            |
| **Glasflow**        | \pmark{} nflows-based            | \xmark{}                  | \pmark{} partial         | \cmark{}                   | \xmark{}            |
| **Pyro transforms** | \pmark{} PPL-centric             | \xmark{}                  | \pmark{} partial         | \cmark{}                   | \xmark{}            |
| **TFP bijectors**   | \pmark{} vector-event oriented   | \xmark{}                  | \pmark{} manual          | \cmark{}                   | \xmark{}            |
| **TorchCFM / FM**   | \xmark{} (not discrete Glow)     | \xmark{}                  | \xmark{}                 | \xmark{}                   | \cmark{}            |
**Legend.**  
\cmark{} = fully supported / works out of the box  
\pmark{} = **partial / possible with non-trivial engineering** (limited or incomplete support)  
\xmark{} = not supported / out of scope

---

**Interpretation.** If your goal is **multiscale conv-Glow** with **3-D**
support, **per-level latents**, and **exact NLL**, you either (a) start from
**`normflows`** and extend, or (b) do significant engineering on top of a
general toolkit. That’s exactly what we did—and we’re packaging those extensions
so others don’t have to repeat the lift.

**Software infrastructure.** We build on the open-source **`normflows`** library
[Stimper et al., 2023], extending it for **3-D Glow**, multiscale **per-level**
latent taps, and a **conditional-Gaussian imputation** pipeline. We also
implemented five latent-alignment objectives and added a comprehensive test
suite. While other flow libraries offer valuable capabilities (e.g., FrEIA,
nflows), `normflows` provided the most direct path for our 
**image-centric, likelihood-trained** setting.


## Our engineering contributions (what we added, fixed, and hardened)

**Core flow architecture & 3-D support**
- **Glow3D**: Implemented a **3-D Glow** backbone (levels \(L\), steps \(K\)) with **`Invertible1x1x1Conv`** (LU-factorized) and **ActNorm3D**, mirroring the 2-D API.
- **Multiscale plumbing**: Audited and fixed **squeeze/unsqueeze** and **split/merge** ordering across levels to eliminate channel-mismatch bugs during inversion.
- **Round-trip stability**: Added strict **forward/inverse round-trip tests** on random tensors across shapes, dtypes (fp16/bf16/fp32), and devices.

**Per-level latent alignment (Section 2 of the paper)**
- **Projector taps** per level (shared or per-modality) with a uniform interface.
- Implemented and documented **Pearson**, **Barlow Twins**, **VICReg**, **InfoNCE**, and **HSIC (biased)** losses with numerically stable reductions and shape checks.
- Added **per-level weighting** schedules (heavier at coarse levels; tapered at fine levels), with logging of alignment statistics.

**Conditional Gaussian Modeling (CGM) module**
- **Dataset-level moment estimation** \((\mu_\ell,\Sigma_\ell)\) per level with **ridge** and **Ledoit–Wolf** shrinkage; **Cholesky** solves with auto-jitter.
- Optional **CCA subspace** projection (rank \(k\)) plus a **safety clamp** to moderate top canonical directions in low-\(n\) regimes.
- **Closed-form conditioning** \(p(z_{\text{mis}}\!\mid z_{\text{obs}})\) at test time; temperature-controlled sampling; exact inverse decode.
- Diagnostics: **Mahalanobis** residuals, **coverage** checks, and **uncertainty maps** from \(\operatorname{tr}\Sigma_{Y\mid X}\).

**Trainer & runtime**
- Mixed-precision (**AMP**) and **EMA** support; **warmup** schedulers; **gradient clipping**.
- **Resume/restart** semantics with deterministic seeding; **tqdm** progress; periodic **NLL/bpd** logging and inversion checks.
- Robust **checkpointing** (model/optimizer/scheduler/EMA) and **config export** for exact reproducibility.

**Data & evaluation utilities**
- Multi-view dataset loader with **family-wise** splitting, masking, and **missingness protocols** (MCAR, structured, block).
- Metrics for **imputation** (PSNR/SSIM), **calibration** (error–variance correlation), and **retest** reliability.
- Vectorized CGM block assembly for **batched volumes**; CPU-friendly conditioning with GPU decode.

**Testing & docs**
- **PyTest** suite: round-trip invariance, finite-logdet checks, per-level shape contracts, and CGM numerics (SPD, Cholesky success).
- **Documentation** for each alignment objective (equations, references, API examples) in your preferred docstring style, plus end-to-end examples.

**Net effect.** Starting from a strong base (`normflows`), we turned the stack
into a **multimodal, 3-D-ready, likelihood-trained** platform with
**statistically principled alignment** and **closed-form imputation**. Our hope
is that this raises the floor for researchers who want to revisit flows—not as
nostalgia, but as **probabilistic workhorses** with exact inverses, tractable
latents, and calibrated uncertainty.

