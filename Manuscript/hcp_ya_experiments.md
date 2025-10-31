
\clearpage

# Results

```bash
# Minimize regularization
python eval_conditional_gaussian.py \
  --run-dir runs2/t1_t2_fa_128x128_vicreg \
  --use-ema \
  --gauss-samples 10000 --eval-samples 256 --batch 64 \
  --cov-mode perlevel \
  --cov-estimator diag --cov-lam 0.0 \
  --shrinkage 1e-6 \
  --cov-debug \
  --eval-tag gauss_minreg
```

\begin{figure}[!h]
  \centering

  \begin{subfigure}{0.48\linewidth}
    \centering
    \includegraphics[width=\linewidth]{Figures/vicreg_gt_fa_given_t1_t2.png}
    \caption{Ground truth:  FA $\leftarrow$ T1 + T2}
  \end{subfigure}\hfill
  \begin{subfigure}{0.48\linewidth}
    \centering
    \includegraphics[width=\linewidth]{Figures/vicreg_hat_fa_given_t1_t2.png}
    \caption{Predicted:  FA $\leftarrow$ T1 + T2}
  \end{subfigure}

  \vspace{0.6em}

  \begin{subfigure}{0.48\linewidth}
    \centering
    \includegraphics[width=\linewidth]{Figures/vicreg_gt_t1_fa_given_t2.png}
    \caption{Ground truth:  T1 + FA $\leftarrow$ T2}
  \end{subfigure}\hfill
  \begin{subfigure}{0.48\linewidth}
    \centering
    \includegraphics[width=\linewidth]{Figures/vicreg_hat_t1_fa_given_t2.png}
    \caption{Predicted:  T1 + FA $\leftarrow$ T2}
  \end{subfigure}

  \caption{Results with minimal CGM regularization.}
  \label{fig:two-by-two}
\end{figure}





## Dataset and modalities
We use the **Human Connectome Project – Young Adult (HCP-YA) S1200** cohort (ages 22–35) with high-quality 3 T multimodal MRI and standardized **minimal preprocessing pipelines** [@vanessen2013hcp; @glasser2013mpp]. From S1200 we draw **T1w/T2w** structural volumes (0.7 mm isotropic) and **diffusion MRI** (1.25 mm isotropic; three shells at **b = 1000/2000/3000 s/mm²** with ~**90** dir/shell; reverse phase-encoding runs), and compute **fractional anisotropy (FA)** from the preprocessed diffusion tensors to serve as our third view [@basser1996fa]. Where needed, we reference the **S1200 release manual** for exact acquisition and pipeline details [@hcp2017s1200].

## Preprocessing & dataset preparation
We consume the HCP **MNINonLinear** outputs (distortion-corrected, cross-modality registered volumes) [@glasser2013mpp]. For our experiments we
- resample/crop to **128³** or **160×192×160** (voxel spacing recorded),
- apply a robust **brain mask** (intersected across modalities),
- perform **per-modality z-scoring** within-mask per subject,
- and split train/val/test with **family-wise separation** (no twins/siblings across splits).

We reserve the **3 T retest** subset exclusively for stability checks (likelihood consistency and imputation repeatability) [@hcp_retest_manual].

## Model & training
We train a **3‑D Glow** model with **per-level latent taps** and **projector heads** for alignment. Default config: levels **L = 3–4**, steps per level **K = 2**, hidden channels 64–96, **AMP** + **EMA** enabled, warmup scheduler, gradient clipping. Alignment objective is selected from **Pearson**, **Barlow Twins**, **VICReg**, **InfoNCE**, or **HSIC**; we weight alignment **more at coarse levels** and reduce at fine‑detail levels. We log **NLL/bpd**, alignment statistics, and inversion round‑trip checks each epoch.

## Conditional Gaussian Modeling (CGM) stats
For each level, we estimate dataset moments \((\mu_\ell,\Sigma_\ell)\) over the **projected per‑level latents** (concatenated across views), optionally after projecting into a **CCA subspace** of rank \(k\). We use **ridge or Ledoit–Wolf shrinkage** and enforce **SPD** via Cholesky with auto‑jitter; we cache factorizations per missingness pattern. At inference, we compute the **closed‑form conditional** to obtain the posterior mean or samples for the missing‑view latents and decode via the **exact flow inverse**.

## Missingness protocols
We evaluate three regimes:
1) **MCAR single‑view** dropouts (randomly drop T2 or FA per subject),  
2) **Structured** drop‑T2 or drop‑FA (entire modality missing), and  
3) **Block** missingness (remove slabs/patches to mimic partial coverage).

## Metrics
- **Imputation fidelity:** **PSNR/SSIM** against held‑out ground truth; intensity bias within tissue masks.  
- **Calibration:** correlation between voxelwise squared error and **CGM posterior variance**; Mahalanobis residual diagnostics.  
- **Downstream stability:** consistency of simple parcellation/tissue metrics computed on imputed vs. real volumes.  
- **Retest reliability:** compare NLL and imputation metrics between test–retest pairs.

## Compliance
We follow HCP **Data Use Terms** and recommended **citation/acknowledgment** language; all experiments operate on the released, minimally preprocessed derivatives [@vanessen2013hcp; @hcp2017s1200].
