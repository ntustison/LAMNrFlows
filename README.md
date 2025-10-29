## Latent-Aligned Multimodal Normalizing Flows for Medical Images

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

---

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

---

__Ground truth:__ T1 + FA given T2
<img width="1042" height="522" alt="gt_T1+FA_given_T2" src="https://github.com/user-attachments/assets/87ddc84d-a5ba-44d5-8e16-62969588888f" />

__Prediction:__ T1 + FA given T2
<img width="1042" height="522" alt="hat_T1+FA_given_T2" src="https://github.com/user-attachments/assets/737637a2-5271-41a9-8862-af0caca50534" />


