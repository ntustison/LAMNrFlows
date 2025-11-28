
## Latent-Aligned Multiview Normalizing Flows

We introduce Latent-Aligned Multiview Normalizing Flows, a general framework that learns a shared latent subspace across views, thereby treating the orthogonal complement as view-specific variation. Using subject-matched batches, we implement a library of latent-alignment constraints (Pearson, Barlow Twins, VICReg, InfoNCE, HSIC) and optionally use CCA (linear) or HSIC (kernel) to identify latent directions that are statistically shared across views, restricting alignment to those coordinates. After maximum-likelihood training, we model the joint latents as Gaussian, estimate per-level moments, and use the conditional Gaussian formulation to obtain closed-form posteriors for any subset. This enables principled cross-view imputation and, more generally, latent manipulations that preserve anatomy or identity while modulating modality- or view-specific factors; for images, replacing private components by their conditional means produces shared-latent images that act as contrast-robust surrogates. 


### Glow 2-D example

<details>

<summary>Command call</summary>

```bash
#!/usr/bin/env bash
set -eu pipefail

# total steps
iterations=120000          # phase 1
extra=40000               # phase 2
total=$((iterations + extra))   # horizon for phase-1 aug schedule

# 128×128 high-capacity arch
H=128; W=128; L=5; K=12; hidden=192
BATCH=64
align=vicreg
align_weight=0.01
OUTDIR="runs/hcp_t1_t2_fa_${H}x${W}_${align}_K${K}_H${hidden}_${align}_screen_phase1"

# Screening configuration
SCREEN_METHOD=cca           # options: none | cca | hsic
SCREEN_FRAC=0.5             # keep top 50% dims for alignment
SCREEN_WARMUP=1000          # start screening after N iters
SCREEN_REFRESH=0            # 0 = discover once; else refresh cadence
CCA_RIDGE=1e-3              # stability for CCA
PREFILTER_FRAC=0.5          # HSIC Pearson prefilter (ignored for CCA)

# ------------------------------
# Augmentation schedules
# ------------------------------

# Phase 1: original decreasing schedule (strong -> weak)
aug_params_phase1="noise_std:cos:0.05->0.004@${total},\
sd_affine:cos:0.05->0.00@$((total*3/5)),\
sd_deformation:linear:12.0->0.6@$((total*7/10)),\
sd_simulated_bias_field:cos:0.20->0.03@${total},\
sd_histogram_warping:cos:0.04->0.008@${total}"

# Phase 2: template-focused fine-tune (no shape; mild intensity jitter)
aug_params_phase2="noise_std:cos:0.004->0.004@${extra},\
sd_affine:cos:0.00->0.00@${extra},\
sd_deformation:linear:0.0->0.0@${extra},\
sd_simulated_bias_field:cos:0.00->0.00@${extra},\
sd_histogram_warping:cos:0.008->0.008@${extra}"

SLICE_IDX=116

# ---- Phase 1: strong->weak aug (as in earlier successful runs) ----
python train_cohort_screened.py \
  --view ~/Data/HCPTemplates/*/T_template0.nii.gz \
  --view ~/Data/HCPTemplates/*/T_template1.nii.gz \
  --view ~/Data/HCPTemplates/*/T_template2.nii.gz \
  --H ${H} --W ${W} --L ${L} --K ${K} --hidden ${hidden} \
  --batch ${BATCH} \
  --slice-idx ${SLICE_IDX} --val-frac 0.0 \
  --max-iter "${iterations}" \
  --devices cuda:1 --precision mixed --amp-dtype bf16 \
  --ema --ema-decay 0.9997 \
  --auto-resume \
  --aug-schedules "${aug_params_phase1}" \
  --lr 1e-4 --warmup-iters 1000 \
  --eval-interval 1000 --plot-interval 1000 \
  --grad-clip 1.0 \
  --train-samples 3000 --val-samples 128 \
  --smooth-alpha 0.05 \
  --sample-mode model \
  --weighting fixed \
  --align "${align}" \
  --align-weight "${align_weight}" \
  --screen "${SCREEN_METHOD}" \
  --screen-warmup "${SCREEN_WARMUP}" \
  --screen-refresh "${SCREEN_REFRESH}" \
  --screen-frac "${SCREEN_FRAC}" \
  --cca-ridge "${CCA_RIDGE}" \
  --prefilter-frac "${PREFILTER_FRAC}" \
  --out-dir "${OUTDIR}"

```

</details>

<details>

<summary>Input:  HCP templates with augmentation</summary>

<p align="center">
  <img src="Manuscript/Figures/input_data_view0.png" alt="alt1" width="30%">
  <img src="Manuscript/Figures/input_data_view1.png" alt="alt1" width="30%">
  <img src="Manuscript/Figures/input_data_view2.png" alt="alt1" width="30%">
</p>
  
</details>

### vicreg

__Ground truth:__ FA given T1 + T2
<img width="1042" height="522" alt="gt_FA_given_T1+T2" src="https://github.com/user-attachments/assets/f08a7ec4-9f62-4e33-8a0c-37ca110e0ee7" />

__Prediction:__ FA given T1 + T2
<img width="1042" height="522" alt="hat_FA_given_T1+T2" src="https://github.com/user-attachments/assets/ae591ab1-03a4-4416-afa1-13883e67107f" />

---

__Ground truth:__ T1 + FA given T2
<img width="1042" height="522" alt="gt_T1+FA_given_T2" src="https://github.com/user-attachments/assets/ccd9396c-9088-477c-a6d8-58277fc7872d" />

__Prediction:__ T1 + FA given T2
<img width="1042" height="522" alt="hat_T1+FA_given_T2" src="https://github.com/user-attachments/assets/039ae080-dc52-4f8e-a65a-bd28e363f880" />


```bash
# Optimal balance between data fidelity and regularization
python eval_conditional_gaussian.py \
    --run-dir runs2/t1_t2_fa_128x128_vicreg \
    --use-ema \
    --gauss-samples 10000 --eval-samples 256 --batch 64 \
    --cov-mode perlevel \
    --cov-estimator diag --cov-lam 0.10 \
    --shrinkage 1e-6 \
    --eval-tag diag_lam010_ridge1e-6
```

### vicreg

__Ground truth:__ FA given T1 + T2
<img width="1042" height="522" alt="gt_FA_given_T1+T2" src="https://github.com/user-attachments/assets/bc990285-4f91-4717-ac87-bab1899054a1" />

__Prediction:__ FA given T1 + T2
<img width="1042" height="522" alt="hat_FA_given_T1+T2" src="https://github.com/user-attachments/assets/3d014eb5-cba9-4cf2-aa33-45c9a85430db" />

__Ground truth:__ T1 + FA given T2
<img width="1042" height="522" alt="gt_T1+FA_given_T2" src="https://github.com/user-attachments/assets/87ddc84d-a5ba-44d5-8e16-62969588888f" />

__Prediction:__ T1 + FA given T2
<img width="1042" height="522" alt="hat_T1+FA_given_T2" src="https://github.com/user-attachments/assets/737637a2-5271-41a9-8862-af0caca50534" />

<details>

<summary>Funding support</summary>

We gratefully acknowledge the grant support of the Office of Naval Research (N0014-23-1-2317).  
  
</details>

