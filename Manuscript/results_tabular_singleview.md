
## Tabular LAMNr Flows

Our tabular evaluation leverages the Normative Neurological Health Embedding
(NNHEmbed) [@Avants:2025aa], an optimized SiMLR-based multimodal modeling
framework engineered from IDPs from UK Biobank (UKBB) [@Miller2016aa]. These
IDPs, comprising T1-weighted MRI (T1-w), diffusion tensor imaging (DTI), and
resting-stage fMRI (rsfMRI), were generated using ANTsPyMM[^antspymm], an
ANTsX-based utility for generating tabular IDP data from neuroimaging cohorts.
For NNHEmbed, the resulting views comprised 51 T1-w IDPs, 77 DTI IDPs, and 484
rsfMRI IDPs. The UKBB-based projection matrices map the multimodal (i.e., three
view) input features of 1) the Normative Neurological Library (NNL)
[@Gage:2024aa] and 2) the Parkinson Progression Marker Initiative (PPMI)
[@PPMI:2011aa] cohorts into shared $k=31$ dimensional bases, which then serve
as the input for generating our LAMNr flows models.

[^antspymm]: https://github.com/ANTsX/ANTsPyMM

### Architecture and Hyperparameter Selection

Prior to the multiview SiMLR and LAMNr flows comparative evaluation, we used the
NNL and PPMI IDP data to determine optimal hyperparameter settings of the
RealNVP-style normalizing flow architecture across the single modalities in
terms of trained likelihoods, i.e., bits-per-dimension (BPD). The network
capacity is controlled by two primary hyperparameters: (i) the coupling depth
$K$ (number of transform layers), and (ii) the conditioner width or hidden
channels ($HC$). Rather than performing an exhaustive grid search over a broad
parameter space, we restricted our evaluation to a targeted window ($K \in \{3,
4, 5\}$, $HC \in \{64, 80, 96\}$). This focused selection is informed by a
caution against overfitting, given the broad range in cohort sizes. An
over-parameterized model risks capturing idiosyncratic noise rather than the
underlying manifold geometry. By selecting the smallest architecture capable of
minimizing the validation negative log-likelihood, we ensure that the model
remains a compact description of the distribution. 

Our hyperparameter sweep across the individual NNL and PPMI cohort views
confirmed the robustness of this architectural window. For the NNL cohort ($N =
346$), a depth of $K=4$ was consistently optimal across T1, DTI, and rsfMRI
modalities, effectively minimizing the validation (in terms of model training)
negative log-likelihood. In the PPMI cohort ($N = 1769$), while increasing
capacity to $K=5$ yielded slightly lower likelihoods, the improvement was
negligible ($\Delta \text{BPD} < 0.001$ for DTI) and did not justify the
additional model complexity. Prioritizing model parsimony and methodological
consistency between datasets, we fixed the configuration at $K=4$ and $HC=80$
for all subsequent multiview experiments. This stable parametric baseline
ensures that the learned latent representations are comparable across different
clinical populations while avoiding overfitting to dataset-specific noise.

