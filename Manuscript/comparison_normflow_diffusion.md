
---
title: "Gaussian Diffusion vs. Normalizing Flows (concise overview)"
bibliography: references.bib        # path to your .bib
csl: 
link-citations: true
citeproc: true
---


| Aspect               | Diffusion / Score-based                                                                                    | Normalizing Flows                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Core training target | Denoiser / score over noise levels \citep{ho2020ddpm,song2020score}                                        | Change-of-variables likelihood \citep{dinh2016realnvp,papamakarios2021nfreview} |
| Likelihood           | Typically intractable (bounds/ODE exceptions) \citep{nichol2021improved,karras2022edm}                     | **Exact** log-likelihood & bpd \citep{kingma2018glow}                           |
| Sampling cost        | Many steps; **~35** feasible with EDM \citep{karras2022edm}; faster in latent space \citep{rombach2022ldm} | **Single pass** (fast)                                                          |
| Invertibility        | No                                                                                                         | **Yes** (exact inverse)                                                         |
| Conditioning         | Very flexible (CFG, cross-attention) \citep{rombach2022ldm}                                                | Via conditioning in coupling/1×1 conv; integrates into likelihood               |
| Strengths            | SOTA perceptual quality; robust training                                                                   | Exact density; calibrated comparisons; natural for cross-modal transforms       |
| Typical pitfalls     | Slow sampling; compute-heavy; likelihood metrics awkward                                                   | Architecture care needed; coupling expressivity vs. Jacobian cost               |


* __Modeling objective__
    * Diffusion / score-based: learn time-indexed score/denoiser for a
      noise-perturbed data process; sampling integrates a reverse (S)DE or a
      discrete Markov chain over many steps \citep{ho2020ddpm,song2020score}.
    * Flows: learn a bijective map with the change-of-variables formula, giving
      exact log-likelihoods and exact latent inference in one pass
      \citep{dinh2016realnvp,kingma2018glow,papamakarios2021nfreview}.

* __Likelihood & calibration__
    * Diffusion: likelihood typically intractable (exceptions via variational
      bounds or specialized ODE setups); models are often tuned for perceptual
      quality \citep{ho2020ddpm,nichol2021improved,karras2022edm}.
    * Flows: tractable log-likelihood and bits-per-dim out-of-the-box; useful
      for model selection and uncertainty that relates to density
      \citep{papamakarios2021nfreview,kingma2018glow}.

* __Sampling cost__
    * Diffusion: tens to thousands of network evaluations per sample; modern designs
      bring this to ~35 evals while preserving quality \citep{karras2022edm}. Latent
      Diffusion further cuts cost by operating in autoencoder latent space
      \citep{rombach2022ldm}.
    * Flows: 1 forward pass from base noise to data (fast synthesis)
      \citep{kingma2018glow}.

* __Invertibility and cross-modal mapping__
    * Diffusion: not inherently invertible; inverse problems solved via guidance or
      conditional score estimation \citep{song2020score}.
    * Flows: exactly invertible by design, enabling clean cross-modal synthesis and
      level-wise latent taps (helpful for our per-level alignment and imputation).

* __Conditioning & control__
    * Diffusion: rich conditioning (classifier-free guidance, text/image/ROI
      prompts) and strong high-fidelity synthesis \citep{rombach2022ldm}.
    * Flows: conditioning via invertible affine/1x1 conv blocks or coupling
      networks; conditioning enters the likelihood cleanly
      \citep{kingma2018glow,papamakarios2021nfreview}.

* __Memory & training dynamics__
    * Diffusion: training is stable; losses are local denoising/score terms;
      large compute budgets often used \citep{ho2020ddpm,rombach2022ldm}.
    * Flows: training can be trickier (Jacobian/ActNorm/coupling design), but
      architectures like Glow offer reversible layers and stable multiscale
      training when implemented carefully
      \citep{kingma2018glow,papamakarios2021nfreview}.

* __Medical-imaging relevance__
    * Diffusion: excellent for inpainting/denoising/segmentation priors via
      guidance; strong perceptual realism.
    * Flows: exact likelihoods + invertibility make them natural for multimodal
      latent alignment and our Conditional Gaussian imputation pipeline.



## References
::: {#refs}
:::
