

\clearpage

# Future work

## Mask-aware inpainting via CGM (primary)
We can repurpose our **Conditional Gaussian Modeling (CGM)** to perform **image inpainting** by treating the masked image as two “views” of the *same* subject: **context** (observed pixels) and **hole** (missing pixels). The key is to operate in **per-level latents** (where we already model second-order structure) and to respect coupling-network receptive fields.

**Per-level masking.** After \(\ell\) squeezes, each latent cell corresponds to a \(2^\ell\times 2^\ell\) (2-D) or \(2^\ell\times 2^\ell\times 2^\ell\) (3-D) block in image space. Let \(\Omega_{\mathrm{obs}}\subset\mathbb{Z}^d\) be observed pixels and \(M\) its indicator. Define a **safe context band** by morphological erosion with a radius that matches the coupling net’s receptive field \(r_\ell\):
\[
\Omega^{\text{safe}}_\ell \;=\; \big(\Omega_{\mathrm{obs}} \ominus B_{r_\ell}\big),
\]
and downsample \(\Omega^{\text{safe}}_\ell\) to latent indices (via squeeze mapping). Latent positions whose entire receptive field lies in \(\Omega^{\text{safe}}_\ell\) form the **observed set** \(X\); the remainder form the **missing set** \(Y\).

**Closed-form conditioning.** Using our dataset moments \((\mu_\ell,\Sigma_\ell)\) over projected per-level latents \(\tilde Z_\ell\) (optionally after a **CCA** subspace of rank \(k\) with shrinkage/jitter for SPD),
partition \(\mu_\ell,\Sigma_\ell\) as \((Y,X)\) and compute:
\[
\mu_{Y\mid X} \;=\; \mu_Y + \Sigma_{YX}\,\Sigma_{XX}^{-1}\big(x-\mu_X\big), 
\qquad 
\Sigma_{Y\mid X} \;=\; \Sigma_{YY} - \Sigma_{YX}\,\Sigma_{XX}^{-1}\Sigma_{XY}.
\]
Fill \(\tilde Z_{\ell,Y}\leftarrow \mu_{Y\mid X}\) (or sample \(y\sim\mathcal{N}(\mu_{Y\mid X},\tau^2\Sigma_{Y\mid X})\)), invert the projector to recover \(Z_\ell\), **merge** per-level latents, and decode once with the **exact inverse** \(f^{-1}\) to obtain the inpainted image \(\hat{x}\).
Uncertainty maps come from \(\operatorname{diag}\Sigma_{Y\mid X}\) upsampled to image resolution.

**Multi-scale schedule.** Run **coarse \(\to\) fine** (\(\ell=L-1,\dots,0\)) so global structure lands first and details are refined at lower levels.

---

## Energy-based posterior over \(z\) (no new modules)
As an alternative to CGM, define a **soft data-consistency** posterior in latent space
\[
\log p(z \mid x_{\text{obs}}) \;\propto\; -\tfrac{1}{2}\lVert z\rVert^2 \;-\; \frac{1}{2\sigma^2}\,\big\lVert M\odot \big(f^{-1}(z)-x_{\text{obs}}\big)\big\rVert^2,
\]
and recover a MAP estimate by gradient steps in \(z\) (or sample with Langevin/HMC), then decode \(f^{-1}(z)\). This requires **no extra networks**, but is iterative.

---

## Conditional Glow for inpainting (architectural fork)
Introduce a **context encoder** \(g(x_{\mathrm{obs}},M)\) and condition coupling nets on its features. Train with a data-consistency term on observed pixels and a reconstruction term on the hole. At test time, sample \(z\sim\mathcal{N}(0,I)\) and decode **conditioned** on \(g\). This is closer to conditional generation; higher complexity, but potentially stronger fidelity.

---

## Covariance modeling at scale
To keep CGM fast and memory-safe for arbitrary masks:
- **Local windows / block CGM:** estimate \((\mu_\ell,\Sigma_\ell)\) on sliding latent windows; compose conditionals locally and blend.
- **Low-rank + diagonal:** \(\Sigma \approx UU^\top + \lambda I\) permits Woodbury updates \((\Sigma_{XX}^{-1})\) and fast Cholesky.
- **Stationary GRF/Kriging view:** estimate an empirical kernel \(K_\ell(\Delta)\) over \(\tilde Z_\ell\) channels and use kriging for \(Y\mid X\); reduces storage to kernel params.

---

## Uncertainty calibration & evaluation
- **Coverage:** check empirical frequency of \(Y\) within \(\mu_{Y\mid X} \pm q_\alpha\sqrt{\operatorname{diag}\Sigma_{Y\mid X}}\).
- **Error–variance correlation:** correlate \((x-\hat{x})^2\) inside the hole with the decoded variance map.
- **Boundary quality:** report MSE/SSIM in a band of width \(b\) around the hole to quantify seam artifacts.

---

## Efficiency and numerics
- **Batched patterns:** pre-bucket masks by pattern to **reuse** Cholesky factors of \(\Sigma_{XX}\).
- **Precision:** AMP on the forward pass; perform **Cholesky** in fp32 with automatic jitter (e.g., add \(\epsilon I\) if needed).
- **Complexity:** conditioning cost dominated by \(\mathcal{O}(|X|^3)\) for Cholesky (or \(\mathcal{O}(rk^2)\) under low-rank). Favor small \(k\) in CCA.

---

## Beyond inpainting
- **Partial-view completion:** same CGM machinery for slab/stack drops in 3-D MR (e.g., corrupted slices).
- **Cross-modal guided inpainting:** when another modality is available, include its per-level latents in \(X\) to guide hole filling.
- **Interactive editing:** treat user-brushed constraints as hard observations in \(X\); CGM handles the rest.
- **Learning-time masking:** train with random masks (CutOut-style) to tighten the Gaussian approximation in inpainting regimes.

---

## Suggested ablations
- With/without CCA; varying rank \(k\).
- Shrinkage \(\lambda\) sweeps for \(\Sigma\); jitter schedules.
- Coarse\(\to\)fine vs. single-shot conditioning.
- Safe-band radius \(r_\ell\) tied to measured receptive fields.

*Impact.* Inpainting pushes the same probabilistic levers we already rely on—**closed-form Gaussian conditioning** and **exact invertibility**—and lets us target clinically relevant repairs (e.g., motion or dropout) with **uncertainty-aware** outputs, in both **2-D** and **3-D**.
