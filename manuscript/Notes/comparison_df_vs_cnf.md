
\clearpage

## Discrete flows vs. CNFs vs. CNFs trained with flow matching

**Discrete normalizing flows (Glow/RealNVP).**  
Stack **bijective layers** (coupling, ActNorm, invertible \(1{\times}1(\times1)\) convs) and train by **exact MLE**:
\[
\log p_X(x)=\log p_Z(f(x))+\log\left|\det J_f(x)\right|.
\]
**Sampling** is a **single parallel inverse** \(x=f^{-1}(z)\).  
**Pros:** exact likelihoods (calibrated NLL/bpd), fast generation, natural multiscale latents.  
**Cons:** architectural constraints (triangular Jacobians, local convs).  
*Refs:* [@kingma2018glow; @papamakarios2021nfreview]


**Continuous normalizing flows (CNFs / neural ODE flows).**  
Define a time-dependent vector field \(v_\theta\) and evolve
\[
\frac{d z_t}{d t} = v_\theta(z_t,t), \quad z_0 \sim p_0,\qquad
\frac{d}{dt}\log p_t(z_t) = -\nabla\!\cdot v_\theta(z_t,t).
\]
Train by **likelihood** (integrate ODE + divergence / Hutchinson trace); **sampling** also requires ODE solves (multiple function evaluations).  
**Pros:** flexible dynamics, no discrete layer design.  
**Cons:** no single-pass sampling; training/sampling can be slower than discrete flows.  
*Refs:* neural ODEs [@chen2018neuralode], FFJORD [@grathwohl2019ffjord], review [@papamakarios2021nfreview].

**CNFs trained with flow matching (FM / conditional FM).**  
Rather than MLE or score matching, **supervise the velocity field** along a designed probability path from base 
\(\to\) data:

\[
\min_{\theta}\; \mathbb{E}_{t,x_t}\,\|\,v_{\theta}(x_t,t)-u^{*}(x_t,t)\,\|^{2}
\]

where \(u^{*}\) is a **target velocity** (e.g., straight-line / rectified path). **Conditional FM** learns \(v_\theta(\cdot,t\mid x_{\text{obs}},m)\) for **imputation**.  
**Pros:** avoids log-det/score estimation; often **fewer steps** than diffusion at sampling.  
**Cons:** typically **no exact likelihood** (regression objective), still requires **ODE integration** at inference.  
*Refs:* flow matching [@lipman2022flowmatching]; conditional FM for imputation (CFMI) [@simkus2025cfmi].

**Takeaway for our setting.** We use **discrete Glow-style flows**: exact NLLs, **single-pass** decoding, and **multiscale latents** that we align per-level and then use for **closed-form CGM** \(p(z_{\mathrm{mis}}\!\mid z_{\mathrm{obs}})\) before exact inversion—an attractive fit for **3-D medical images** where speed, memory, and calibrated likelihoods matter. 
