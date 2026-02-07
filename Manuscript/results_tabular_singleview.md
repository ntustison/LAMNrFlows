## Tabular LAMNr Flows

### Architecture and Hyperparameter Selection

To establish a robust latent representation for tabular data, we utilized the
pre-trained Similarity-driven Multiview Linear Reconstruction (SiMLR) framework
[@Tustison:2024aa]. Specifically, we projected the high-dimensional NNL and PPMI
input features into a shared, lower-dimensional basis ($k=31$) using the
established SiMLR projection matrices. These matrices encapsulate
population-level covariance structures derived from large-scale neuroimaging
initiatives, providing a stable initialization for our generative modeling.

Consistent with the complexity of this pre-trained SiMLR basis, we adopted a RealNVP-style normalizing flow architecture. The network capacity is controlled by two primary hyperparameters:
(i) **the coupling depth $K$** (number of transform layers), and
(ii) **the conditioner width `hidden_channels` ($HC$)** (neuronal width).

Based on the architectural specifications of the SiMLR framework and prior benchmarks on tabular neuroimaging data [@Tustison:2024aa], we initialized our search around a baseline configuration of **$K=4$** and **$HC=80$**.

### Targeted Validation on Clinical Cohorts

To verify that this architecture—originally calibrated for large-scale
population variance—was appropriate for the specific distributions of our
smaller clinical cohorts (NNL, $N \approx 360$; PPMI, $N \approx 400$), we
performed a targeted validation sweep. We evaluated a focused grid of
hyperparameters ($K \in \{3, 4, 5\}$, $HC \in \{64, 80, 96\}$) to assess model
stability and generalization.

**Results:**
The validation confirmed the robustness of the chosen architecture:
* **NNL Cohort:** The configuration $K=4$ was consistently identified as optimal across all modalities (T1, DTI, rsfMRI), minimizing the validation negative log-likelihood (bits per dimension).
* **PPMI Cohort:** While a slightly higher capacity configuration ($K=5$) yielded a negligible improvement in likelihood ($\Delta \text{BPD} < 0.001$), the baseline architecture remained highly competitive.

Prioritizing model parsimony and methodological consistency across datasets, we fixed the architecture at **$K=4$** and **$HC=80$** for all subsequent analyses. This approach ensures that the learned latent representations ($Z$) are comparable across different clinical populations while avoiding overfitting to dataset-specific noise.