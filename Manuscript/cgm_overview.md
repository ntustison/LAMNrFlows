
\clearpage

# Conditional Gaussian Modeling (CGM) for Multimodal Imputation

## Motivation and setting
Our goal is **imputation of missing modalities** using an *already trained*, **invertible** multimodal flow with **per-level latent taps**. Let $V$ denote the number of modalities, and let the flow factorize into $L$ multiscale levels. For subject $n$ and level $\ell$, we extract per-view latents $Z_{\ell}^{(v)}=f_{\ell}(x_n^{(v)})$ and pass them through a light projector $P_{\ell}$ to obtain features $\tilde Z_{\ell}^{(v)}=P_{\ell} Z_{\ell}^{(v)}$. We then **model the joint distribution across views at each level** as approximately Gaussian, estimated *across subjects*. This enables **closed-form conditioning** for imputation before decoding through the exact inverse of the flow. The approach leverages classical multivariate-Gaussian identities [@bishop2006prml; @murphy2012mlpp] while staying faithful to flow invertibility [@kingma2018glow; @papamakarios2021nfreview].

## Joint model and conditioning
Concatenate per-level features across views,
$$
\tilde Z_{\ell} = \big[\tilde Z_{\ell}^{(1)};\ldots;\tilde Z_{\ell}^{(V)}\big] \in \mathbb{R}^{D_{\ell}},
$$
and assume dataset-level moments $(\mu_{\ell}, \Sigma_{\ell})$ so that $\tilde Z_{\ell} \sim \mathcal N(\mu_{\ell}, \Sigma_{\ell})$. Partition into observed $X$ (views $S$) and missing $Y$ (views $M$):
$$
\begin{bmatrix} Y \\ X \end{bmatrix} \sim \mathcal N\!\left(
\begin{bmatrix} \mu_Y \\ \mu_X \end{bmatrix},
\begin{bmatrix} \Sigma_{YY} & \Sigma_{YX} \\ \Sigma_{XY} & \Sigma_{XX} \end{bmatrix}
\right).
$$
The **conditional Gaussian** used for imputation is
$$
\boxed{
\mu_{Y|X} = \mu_Y + \Sigma_{YX}\,\Sigma_{XX}^{-1}\,(x-\mu_X), \qquad
\Sigma_{Y|X} = \Sigma_{YY} - \Sigma_{YX}\,\Sigma_{XX}^{-1}\,\Sigma_{XY}
}
$$
[@bishop2006prml, ch. 2; @murphy2012mlpp, ch. 4].

**Decode.** Replace the missing-view latents at level $\ell$ with either the **posterior mean** or a **sample** $y_{\ell}\!\sim\!\mathcal N(\mu_{Y|X},\,\tau^2\Sigma_{Y|X})$ (temperature $\tau\in[0,1]$), aggregate level-wise latents, and apply the exact inverse $f^{-1}$ to obtain $\hat y$ (per missing view).

## Estimating $(\mu_{\ell}, \Sigma_{\ell})$ robustly
We compute per-level dataset statistics once (or per fold) over training/validation subjects:

- **Centering / scaling.** Mean-center features; optional per-dimension scaling for numerical conditioning.
- **Shrinkage covariance.** Use **ridge** ($\widehat\Sigma \leftarrow \widehat\Sigma + \varepsilon I$) and/or **Ledoit–Wolf** shrinkage $\widehat\Sigma_{\rho}=(1-\rho)\widehat\Sigma + \rho\,\mathrm{diag}(\widehat\Sigma)$ to improve SPD conditioning [@ledoit2004well; @schafer2005shrinkage].
- **CCA subspace (rank $k$).** Project $(X,Y)$ into a **canonical-correlation** subspace per level to emphasize shared structure and reduce dimensionality [@hotelling1936; @andrew2013dcca]. To avoid overconfident couplings, apply a **clamp** that scales top-$k$ canonical directions by a factor $\alpha\in(0,1]$ (empirically stabilizes $\Sigma_{XX}^{-1}$ in small/heterogeneous batches).
- **SPD numerics.** Prefer **Cholesky** solves on $\Sigma_{XX}$; use SVD fallback when needed; auto-jitter if the Cholesky test fails [@higham2002accuracy].

> **Why Gaussian at each level?** Final-flow latents are standard normal by construction; per-level projected latents are not guaranteed Gaussian but are often close after whitening/alignment. The Gaussian approximation gives a *closed-form* conditioning rule that is fast, stable, and empirically well-calibrated for cross-modal inference.

## Inference pipeline (per level)
**Inputs:** trained flow $f$, projectors $P_{\ell}$, per-level $(\mu_{\ell},\Sigma_{\ell})$, observed set $S$, missing set $M$.  
**Steps:**  
1. **Encode:** For $v\in S$, compute $Z_{\ell}^{(v)}=f_{\ell}(x^{(v)})$, then $\tilde Z_{\ell}^{(v)}=P_{\ell}Z_{\ell}^{(v)}$.  
2. **Subspace (optional):** Project $\tilde Z_{\ell}^{(v)}$ into the rank-$k$ CCA subspace; apply clamp $\alpha$.  
3. **Condition:** Build $(\mu_Y,\mu_X,\Sigma_{YY},\Sigma_{YX},\Sigma_{XX})$ blocks; compute $(\mu_{Y|X},\Sigma_{Y|X})$.  
4. **Sample/mean:** Draw $y_{\ell}\!\sim\!\mathcal N(\mu_{Y|X},\,\tau^2\Sigma_{Y|X})$ or use $\mu_{Y|X}$.  
5. **Decode:** Replace missing-view latents at level $\ell$ and invert $f^{-1}$ to reconstruct $\hat y$.  
6. **Aggregate:** Repeat over $\ell=0,\ldots,L-1$; concatenate level latents for final decode.

**Complexity.** Conditioning uses a single solve with $\Sigma_{XX}$ per level (cost dominated by a Cholesky $O(d_X^3)$ per level), where $d_X$ is the dimension of observed-view features in the chosen subspace. Since $(\mu_{\ell},\Sigma_{\ell})$ are **precomputed**, inference is CPU-friendly and trivially batchable across voxels/patches/subjects.

## Calibration, diagnostics, and controls
- **Uncertainty maps.** Use $\mathrm{tr}(\Sigma_{Y|X})$ to visualize aleatoric uncertainty per voxel.  
- **Coverage.** Check empirical coverage of $(1-\alpha)$-ellipsoids vs. samples; Mahalanobis residuals should be $\chi^2$-like [@bishop2006prml].  
- **Controls.** $\tau$ (noise–sharpness), $k$ (subspace rank), $\alpha$ (clamp strength), $\varepsilon$ (jitter).  
- **Ablations.** Compare posterior mean vs. sampling; with/without CCA; shrinkage choices; EMA vs. raw weights for encode/decode.

## Relationship to prior imputation work
General-purpose latent-imputation frameworks (e.g., EM with flows, flow matching for imputation) operate on a *single latent space* without per-level, per-view structure [@ma2021emflow]. Our pipeline (i) uses **multiscale, per-level** latents from an exact-invertible image model, (ii) introduces a **CCA-guided** subspace/clamp tailored to **cross-modality** coupling, and (iii) decodes through the **exact flow inverse**, yielding deterministic or uncertainty-aware reconstructions with a single parallel pass.

## Practical notes
- Precompute $(\mu_{\ell},\Sigma_{\ell})$ with held-out splits; cache factorizations of $\Sigma_{XX}$ when $S$ is fixed.  
- Use **vectorized** block assembly across spatial positions; pin memory for fast host device transfer.  
- For very high $D_{\ell}$, consider **per-level rank schedules** (e.g., larger $k$ for coarse levels, smaller for texture levels).

