## Latent-Aligned Multimodal Normalizing Flows for Medical Images

```
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

__Ground truth:__ T1 + FA given T2
<img width="1042" height="522" alt="gt_T1+FA_given_T2" src="https://github.com/user-attachments/assets/87ddc84d-a5ba-44d5-8e16-62969588888f" />

__Prediction:__ T1 + FA given T2
<img width="1042" height="522" alt="hat_T1+FA_given_T2" src="https://github.com/user-attachments/assets/737637a2-5271-41a9-8862-af0caca50534" />


