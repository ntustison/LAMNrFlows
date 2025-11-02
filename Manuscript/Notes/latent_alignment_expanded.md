
\clearpage

# Latent alignment


\footnotesize


| Method                   | Objective (sketch)                                                                        | Encourages                           | Typical hyperparams                                              | Batch size need | Notes                                                                                                                |
| ------------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------ | ---------------------------------------------------------------- | --------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Pearson (multi)**      | Maximize mean pairwise corr: maximize $\mathrm{corr}(\tilde Z^{(i)},\tilde Z^{(j)})$      | Linear shared structure              | projector dim; feature normalization (pre-BN/L2)                 | **Low**         | Simple, fast, stable at small batch; second-order only—can over-smooth fine texture if over-weighted at high levels. |
| **Barlow Twins (multi)** | Cross-corr to identity: $\mathcal{L}=\sum_i (1-C_{ii})^2+\lambda!\sum_{i\neq j} C_{ij}^2$ | Invariance + decorrelation           | $\lambda$ (off-diag weight); covariance shrinkage; projector dim | **Med**         | No negatives; good default. Needs decent batch to estimate $C$; shrinkage helps stability.                           |
| **VICReg (multi)**       | $\alpha,\text{Inv}+\beta,\text{Var}+\gamma,\text{Cov}$                                    | Invariance while preserving variance | $\alpha,\beta,\gamma$; var margin; projector dim                 | **Med–Low**     | Collapse-resistant; tunable trade-offs. More knobs; match var margin to feature dim.                                 |
| **InfoNCE (multi)**      | Contrastive: $\mathcal{L}=-\log \frac{\exp(s/\tau)}{\sum \exp(s'/\tau)}$                  | Discriminative cross-view alignment  | temperature $\tau$; projector dim; (optional) aug strength       | **High**        | Strong signal with large batches; sensitive to batch/negatives; heavier compute.                                     |
| **HSIC (biased)**        | Maximize kernel dependence: $\mathrm{HSIC}(X,Y)$ (e.g., RBF)                              | Non-linear shared structure          | kernel type; bandwidth $\sigma$ (median heuristic); reg          | **Med**         | Captures beyond second-order; $O(B^2)$ cost; bandwidth selection matters.                                            |

\normalsize


We align per-level projected latents $\tilde Z_\ell^{(v)} = P_\ell Z_\ell^{(v)}$
across $V$ views to encourage a shared representation while preserving
modality-specific detail. Below, $B$ is batch size, $D$ feature dimension.

### Pearson multi-correlation (linear alignment)
**Idea.** Maximize average pairwise correlation between views’ features
(second-order, linear).  
**Sketch.** For mean-centered, variance-normalized features, 

$$
\mathcal{L}_{\text{Pearson}} = -\frac{2}{V(V-1)} \sum_{i<j}\,\frac{\langle
\tilde Z^{(i)}, \tilde Z^{(j)}\rangle}{\|\tilde Z^{(i)}\|\,\|\tilde Z^{(j)}\|}.
$$ 

**Pros.** Simple, stable at small $B$; negligible overhead.  
**Cons.** Only linear, second-order alignment; can over-smooth high-frequency
texture if over-weighted at deep levels.  
**Tips.** Pre-normalize features (BN/L2); moderate weights at low/coarse levels.

### Barlow Twins (decorrelated identity cross-corr) [@zbontar2021barlow]
**Idea.** Make cross-correlation between views’ features close to the identity;
penalize off-diagonals (reduces redundancy).  
**Sketch.** With cross-correlation matrix $C$, 

$$ 
\mathcal{L}_{\text{BT}} =
\sum_i (1 - C_{ii})^2 + \lambda \sum_{i\neq j} C_{ij}^2 . 
$$ 

**Pros.** No
negatives; robust and collapse-resistant; promotes invariance + decorrelation.  
**Cons.** Needs decent $B$ to estimate $C$; may suppress modality-specific axes
if over-strong.  
**Tips.** Use covariance shrinkage (moving average or ridge); tune $\lambda$;
pair with per-level weighting.

### VICReg (invariance–variance–covariance) [@bardes2021vicreg]
**Idea.** Combine an invariance term with a variance floor and covariance
regularizer to prevent collapse.  
**Sketch.** 

$$ 
\mathcal{L}_{\text{VICReg}} = \alpha\,\|\mu_1-\mu_2\|_2^2 +
\beta\,\sum_i \phi(\sigma_i) + \gamma\,\sum_{i\neq j}\! \mathrm{Cov}_{ij}^2, 
$$

with $\phi$ enforcing per-dimension std $\ge$ margin.  
**Pros.** Strong collapse avoidance; few negatives required.  
**Cons.** More knobs ($\alpha,\beta,\gamma$, margin); needs careful scaling by
feature dim.  
**Tips.** Start from published defaults; use per-level schedules (stronger at
coarse levels).

### InfoNCE (contrastive with in-batch negatives) [@oord2018cpc]
**Idea.** Increase similarity of positive pairs across views while pushing away
negatives (often other items in batch).  
**Sketch.** For similarity $s$ and temperature $\tau$, 
$$
\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B} \sum_{b=1}^B \log
\frac{\exp(s_{b,b}/\tau)}{\sum_{b'} \exp(s_{b,b'}/\tau)} . 
$$ 
**Pros.** Strong
discriminative signal; excellent alignment with sufficient negatives.  
**Cons.** **Batch-size sensitive**; heavier compute; temperature tuning
critical.  
**Tips.** If $B$ is limited, consider memory banks/queues or mix with
non-contrastive terms.

### HSIC (kernel dependence) [@gretton2005hsic]
**Idea.** Maximize a kernel-based dependence measure; captures non-linear shared
structure beyond second order.  
**Sketch.** With kernels $K, L \in \mathbb{R}^{B\times B}$, $$
\mathrm{HSIC}(X,Y) = \frac{1}{(B-1)^2}\,\mathrm{tr}(KHLH), \qquad
H=I-\frac{1}{B}\mathbf{1}\mathbf{1}^\top . $$ **Pros.** Flexible (RBF/poly
kernels), non-linear.  
**Cons.** $O(B^2)$ memory/time; bandwidth choice matters.  
**Tips.** Median heuristic for RBF bandwidth; mini-batch biased/unbiased
variants; combine with covariance shrinkage.

### CCA & “CCA-safe clamp” (stability aid) [@hotelling1936; @andrew2013dcca]
**Idea.** Use canonical correlation analysis per level to (i) select a shared
subspace of rank $k$ and (ii) **clamp** (scale) the top canonical directions by
factor $\alpha\in(0,1]$ to avoid runaway alignment along a few axes.  
**Use cases.**  
    - **Evaluation/CGM path** (current): project features into shared subspace
  before estimating Gaussian stats; clamp to stabilize $\Sigma_{XX}^{-1}$.  
    - **Training path** (optional): apply the same projection/clamp to features 
    **before** computing the alignment loss (not enabled by default).

### Choosing among objectives (practical guidance)
- **Small batches or limited compute:** Pearson or VICReg.  
- **Desire strong redundancy reduction without negatives:** Barlow Twins.  
- **Large batches / discriminative retrieval:** InfoNCE.  
- **Non-linear cross-modality relations:** HSIC.  
- **Always helpful:** per-level taps and optional CCA-based subspace/clamp for
  stability; moderate alignment on fine-detail levels to preserve
  modality-specific texture.

### Compact comparison (with pointers)

\footnotesize

| Method | What it optimizes | Captures | Negatives? | Batch need | Key knobs |
|---|---|---|---|---|---|
| Pearson | Mean pairwise corr | Linear, second-order | No | Low | Feature norm; weight |
| Barlow Twins [@zbontar2021barlow] | Cross-corr $\to I$ | Invariance + decorrelation | No | Med | Off-diag weight $\lambda$; shrinkage |
| VICReg [@bardes2021vicreg] | Inv/Var/Cov | Collapse-resistant invariance | No | Med–Low | $(\alpha,\beta,\gamma)$; var margin |
| InfoNCE [@oord2018cpc] | Contrastive MI lower bound | Discriminative alignment | Yes | **High** | Temp $\tau$; projector dim; augs |
| HSIC [@gretton2005hsic] | Kernel dependence | Non-linear relations | No | Med | Kernel/bandwidth; reg |
| CCA clamp [@hotelling1936; @andrew2013dcca] | Subspace + scaling | Stabilizes shared axes | — | Low | Rank $k$; clamp $\alpha$ |

\normalsize