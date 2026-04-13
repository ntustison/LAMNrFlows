\clearpage

# From “Glow-lite” to **Canonical Glow** — what changed and why it mattered

This note documents the precise architectural and implementation changes that took our pipeline from a brittle, single-scale **Glow-lite** variant to a stable, multiscale **Canonical Glow** suitable for 2-D **and** 3-D medical images, with clean per-level latents for alignment and CGM.

---

## 1) What “Glow-lite” was (and the symptoms we saw)

**Architecture shortcuts**
- **Single-scale only** (no multiscale levels; no factor-out).
- **Additive coupling** only (no scale branch), limiting expressivity.
- **No ActNorm**, so early-epoch statistics drifted.
- **No invertible 1×1(×1) convolution**, so very weak channel mixing.
- **Squeeze/unsqueeze used inconsistently** or only at the input.
- **Per-level taps absent**, so there was no principled place to read latents for alignment/CGM.

**Inverse path bug**
- The inverse did **unsqueeze → merge** instead of **merge → unsqueeze**.  
  This produced the level-1 **channel mismatch** during inversion (e.g., “expected 8 channels, got 1”).

**Observable symptoms**
- Fragile training and worse NLL/bpd compared to canonical Glow baselines.
- Round-trip errors \(x \to z \to \hat{x}\) spiking on certain shapes.
- Alignment objectives acting on **noisy/non-stationary latents**.
- CGM statistics brittle (poorly conditioned \(\Sigma\), unreliable Cholesky).

---

## 2) Canonical Glow: the exact block and level structure we adopted

### 2.1 FlowStep (per step inside a level)
We restored the standard **Glow** step order with exact log-det bookkeeping:

1. **ActNorm\{2d/3d\}** (data-dependent init on first batch).  
   - Forward: \(y = s \odot x + b\)  
   - Log-det: \(\log|\det J| = \big(\sum\nolimits_{c} \log |s_c|\big)\cdot (H \times W\,[\times D])\)

2. **Invertible \(1\times 1\) (or \(1\times 1\times 1\)) convolution** with **LU** factorization.  
   - Forward: \(y = W\,x\)  
   - Log-det: \(\log|\det J| = (H \times W\,[\times D])\cdot \log|\det W|\), with \(\log|\det W| = \sum \log|\mathrm{diag}(U)|\)

3. **Affine coupling** (channels split: \(x = [x_a, x_b]\))  
   - Forward: \(y_a = x_a,\quad y_b = x_b \odot \exp(s(x_a)) + t(x_a)\)  
   - Log-det: \(\log|\det J| = \sum s(x_a)\) (sum over spatial dimensions)

**Why this order?** ActNorm stabilizes early statistics; the invertible \(1\times 1(\times 1)\) convolution mixes channels globally; affine coupling adds multiplicative capacity beyond additive-only variants.

### 2.2 Multiscale level layout
Each **level \(\ell\)** operates on a higher-channel, lower-spatial tensor via **squeeze**, runs **\(K\)** FlowSteps, then **factors out** part of the channels:

