# Abstract {-}

We present a practical recipe for training **invertible** generative models that
jointly model multiple medical imaging modalities while exposing multiscale
latents that can be aligned, analyzed, and used for imputation. Concretely, we
harden 2D/3D **Glow** inside *normflows* with ANTsTorch integration—correcting
multiscale reshape orderings, adding 3D invertible layers, stabilizing
log-determinants, and providing a reproducible CLI with AMP/EMA, resumable
checkpoints, and tests [@kingma2018glow; @papamakarios2021nfreview]. On top of
the likelihood objective, we support **per-level latent alignment** via
projector heads with a family of objectives (Pearson, **Barlow Twins**,
**VICReg**, **InfoNCE**, **HSIC**) and an optional **CCA-guided** subspace/clamp
used to stabilize downstream estimation [@zbontar2021barlow; @bardes2021vicreg;
@oord2018cpc; @gretton2005hsic; @hotelling1936; @andrew2013dcca]. For missing
contrasts, we describe a **conditional Gaussian modeling** (CGM) pipeline:
estimate dataset means/covariances per level—optionally in a CCA subspace with
shrinkage/jitter—and compute closed‑form posteriors to impute latents before
exact decoding. Motivated by **aleatoric** variability across modalities, we
outline an uncertainty‑aware weighting of alignment terms à la **Kendall–Gal**
as an optional extension [@kendall2018mtl; @kendall2017uncertainties]. The
result is a single, tested backbone that supports exact likelihoods, cross‑modal
synthesis, and principled imputation, while remaining faithful to the ANTsX
emphasis on modularity and reproducibility.
