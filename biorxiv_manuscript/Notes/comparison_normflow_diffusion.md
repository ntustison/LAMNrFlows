
\clearpage

# Normalizing flows vs. Gaussian diffusion

\footnotesize

| Aspect               | Diffusion / Score-based                                                                                    | Normalizing Flows                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Core training target | Denoiser / score over noise levels [@ho2020ddpm,song2020score]                                        | Change-of-variables likelihood [@dinh2016realnvp,papamakarios2021nfreview] |
| Likelihood           | Typically intractable (bounds/ODE exceptions) [@nichol2021improved,karras2022edm]                     | **Exact** log-likelihood & bpd [@kingma2018glow]                           |
| Sampling cost        | Many steps; **~35** feasible with EDM [@karras2022edm]; faster in latent space [@rombach2022ldm] | **Single pass** (fast)                                                          |
| Invertibility        | No                                                                                                         | **Yes** (exact inverse)                                                         |
| Conditioning         | Very flexible (CFG, cross-attention) [@rombach2022ldm]                                                | Via conditioning in coupling/1×1 conv; integrates into likelihood               |
| Strengths            | SOTA perceptual quality; robust training                                                                   | Exact density; calibrated comparisons; natural for cross-modal transforms       |
| Typical pitfalls     | Slow sampling; compute-heavy; likelihood metrics awkward                                                   | Architecture care needed; coupling expressivity vs. Jacobian cost               |

\normalsize

##  A short history of flows vs. diffusion — and why diffusion “won” (for now)

If you rewind to the mid-2010s, **normalizing flows** were the cleanest
probabilistic story in deep generative modeling. Starting with **NICE** and
**RealNVP**, and crystallizing with **Glow** (2018), the promise was elegant:
learn an **exactly invertible** mapping between images and a base Gaussian; get
**tractable log-likelihoods**, calibrated density estimates, and **one-shot
sampling** for free. The field finally had a way to optimize the quantity it
claimed to care about (likelihood) without variational bounds or adversarial
games. But a truth lingered in the samples: flows were often **sharp but
plain**. The local, triangular Jacobian structure of coupling layers and the
reliance on convolutional inductive biases made flows **easier to optimize** but
**harder to scale** in expressivity at internet scale. They shined in density
estimation and anomaly detection; they did not (yet) ignite the public
imagination.

Then came the **score/diffusion** wave. The score-matching revival (Song et al.)
and **DDPM** (Ho et al.) reframed generation as **progressive denoising** along
a carefully designed noise schedule. Two things clicked. First, the **loss** was
simple and stable—just supervised regression to a noise target on top of a
**U-Net** backbone. Second, the **visual quality** scaled almost monotonically
with data and compute. In contrast to early GAN instability and flow
expressivity limits, diffusion models felt **predictable** to train and
**rewarding** to scale.

**2022** was the inflection. **Latent Diffusion** (a.k.a. **Stable Diffusion**)
put the denoising dance into a compressed latent space and married it to **text
conditioning via cross-attention** (think CLIP-like embeddings + classifier-free
guidance). Suddenly, anyone could type a prompt and get compelling images. The
model weights were released, the dataset (LAION-5B) was openly mined, and the
**tooling exploded**: web UIs, control modules (**ControlNet**), fine-tuning
recipes (**LoRA**), and a frenetic ecosystem of checkpoints. This wasn’t just a
research breakthrough—it was a **UX watershed**. Diffusion had a **killer
interface** (prompting) and a **killer distribution channel** (open weights +
easy fine-tuning). The community bootstrapped itself.

__Why did diffusion win adoption (at least in the short run)?__

- **Quality scaled first and fastest.** At large data/compute, diffusion
  **looked** better. For most creative and commercial uses, sample fidelity
  trumped exact likelihoods.
- **Training was boring—in the best way.** No adversarial min–max, no delicate
  Jacobian bookkeeping. If you could train a U-Net on images, you could train a
  diffusion model.
- **Conditioning was natural and powerful.** Cross-attention + classifier-free
  guidance made text→image, style control, and multi-modal conditioning
  straightforward.
- **The ecosystem flywheel.** Open weights, permissive licenses, Hugging Face
  hubs, and drag-and-drop GUIs invited millions of non-researchers. Flows rarely
  had that level of turnkey, high-quality, widely shared checkpoints.
- **Sampling got “fast enough.”** While diffusion takes multiple steps, samplers
  (DDIM/DPM-Solver/consistency) made **10–50 steps** viable—acceptable for many
  apps. The **one-shot** advantage of flows mattered less when diffusion gave
  better pictures and was still responsive.

Meanwhile, flows kept evolving—but in the background. Researchers pushed on
three fronts. First, **continuous normalizing flows** (neural ODEs) broadened
the design space beyond discrete stacks—though at the cost of ODE solves.
Second, **flow matching** reframed CNF training as **velocity field
regression**, sidestepping likelihood and score estimation and partially
shrinking the sampling gap. Third, the community experimented with
**Transformer-based flows** (e.g., autoregressive or latent-space variants),
showing that **capacity**, not the flow principle, was the bottleneck; recent
work attains excellent likelihoods and competitive fidelity. The message: flows
didn’t “fail”—they were **early**.

__So why do flows still matter—especially for medical imaging?__

Flows bring three things diffusion doesn’t natively:  
(1) **Exact likelihoods and invertibility,** which enable **calibrated density**
and principled diagnostics;  
(2) **Single-pass decoding** (no sampler loop), often **crucial in 3-D**; and  
(3) **Structured latents** that you can interrogate and manipulate with
**closed-form statistics**. That last point is where our work slots in: we train
a **multiscale Glow** and enforce **explicit latent alignment** across
modalities; then we perform **Conditional Gaussian Modeling** in those aligned
latents to impute missing views with **analytic conditionals** before exact
inversion. This turns the flow’s algebraic strengths into practical tools for
**multimodal inference**—a setting where calibrated likelihoods, invertibility,
and fast 3-D synthesis **do** matter.

If you zoom out, the recent picture looks less like “diffusion beat flows” and
more like **two complementary toolkits**: diffusion won **mindshare** by scaling
quality + UX + community; flows are resurfacing as **probabilistic workhorses**
in domains that value **tractable latents, exact inverses, and data-centric
inference**. As architectures cross-pollinate (Transformers in flows;
consistency/flow-matching bridging to diffusion), the boundary is blurring. For
now, if your goal is **creative text-to-image**, diffusion is the default. If
your goal is **multimodal medical inference with calibrated uncertainty and fast
3-D decoding**, a well-designed flow can be the quieter—but better—fit.


## Technical comparison

* __Modeling objective__
    * Diffusion / score-based: learn time-indexed score/denoiser for a
      noise-perturbed data process; sampling integrates a reverse (S)DE or a
      discrete Markov chain over many steps [@ho2020ddpm,song2020score].
    * Flows: learn a bijective map with the change-of-variables formula, giving
      exact log-likelihoods and exact latent inference in one pass
      [@dinh2016realnvp,kingma2018glow,papamakarios2021nfreview].

* __Likelihood & calibration__
    * Diffusion: likelihood typically intractable (exceptions via variational
      bounds or specialized ODE setups); models are often tuned for perceptual
      quality [@ho2020ddpm,nichol2021improved,karras2022edm].
    * Flows: tractable log-likelihood and bits-per-dim out-of-the-box; useful
      for model selection and uncertainty that relates to density
      [@papamakarios2021nfreview,kingma2018glow].

* __Sampling cost__
    * Diffusion: tens to thousands of network evaluations per sample; modern designs
      bring this to ~35 evals while preserving quality [@karras2022edm]. Latent
      Diffusion further cuts cost by operating in autoencoder latent space
      [@rombach2022ldm].
    * Flows: 1 forward pass from base noise to data (fast synthesis)
      [@kingma2018glow].

* __Invertibility and cross-modal mapping__
    * Diffusion: not inherently invertible; inverse problems solved via guidance or
      conditional score estimation [@song2020score].
    * Flows: exactly invertible by design, enabling clean cross-modal synthesis and
      level-wise latent taps (helpful for our per-level alignment and imputation).

* __Conditioning & control__
    * Diffusion: rich conditioning (classifier-free guidance, text/image/ROI
      prompts) and strong high-fidelity synthesis [@rombach2022ldm].
    * Flows: conditioning via invertible affine/1x1 conv blocks or coupling
      networks; conditioning enters the likelihood cleanly
      [@kingma2018glow,papamakarios2021nfreview].

* __Memory & training dynamics__
    * Diffusion: training is stable; losses are local denoising/score terms;
      large compute budgets often used [@ho2020ddpm,rombach2022ldm].
    * Flows: training can be trickier (Jacobian/ActNorm/coupling design), but
      architectures like Glow offer reversible layers and stable multiscale
      training when implemented carefully
      [@kingma2018glow,papamakarios2021nfreview].

* __Medical-imaging relevance__
    * Diffusion: excellent for inpainting/denoising/segmentation priors via
      guidance; strong perceptual realism.
    * Flows: exact likelihoods + invertibility make them natural for multimodal
      latent alignment and our Conditional Gaussian imputation pipeline.

