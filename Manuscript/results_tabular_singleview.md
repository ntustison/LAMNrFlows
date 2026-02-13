
\clearpage

## Tabular LAMNr Flows

Our tabular evaluation leverages the Normative Neurological Health Embedding
(NNHEmbed) framework [@Avants:2025aa]. To ensure methodological consistency with
population-level priors while adhering to data usage constraints, we utilize
optimized SiMLR-based projection matrices for multi-modal data derived from UK
Biobank (UKBB) [@Miller2016aa]. These IDPs, derived from T1-weighted MRI (T1-w),
diffusion tensor imaging (DTI), and resting-stage fMRI (rsfMRI), were generated
using ANTsPyMM[^antspymm], an ANTsX-based utility for generating tabular IDP
data from neuroimaging cohorts.  For NNHEmbed, the resulting views comprised 51
T1-w IDPs, 77 DTI IDPs, and 484 rsfMRI IDPs. The UKBB-based projection matrices
map the input features of these three views from the Normative Neurological
Library (NNL) [@Gage:2024aa] and the Parkinson Progression Marker Initiative
(PPMI) [@PPMI:2011aa] cohorts into a shared $k=31$ dimensional basis, which then
serves as the input for our LAMNr flows models.

[^antspymm]: https://github.com/ANTsX/ANTsPyMM

### Architecture and Hyperparameter Selection

Prior to the multiview SiMLR and LAMNr flows comparison, we used the original
NNL and PPMI IDP data to determine optimal hyperparameter configuration of the
RealNVP-style normalizing flow architecture across the single modalities in
terms of trained likelihoods, i.e., bits-per-dimension (BPD). The network
capacity is controlled by two primary hyperparameters: (i) the coupling depth
$K$ (number of transform layers), and (ii) the conditioner width or hidden
channels ($HC$).

Rather than performing an exhaustive 2D grid search over a broad parameter
space, we restricted our evaluation to a targeted window ($K \in \{3, 4, 5\}$,
$HC \in \{64, 80, 96\}$). This focused selection is informed by a caution
against overfitting, particularly with our cohort sizes, and the general Minimum
Description Length (MDL) model selection principle. An over-parameterized model
risks capturing idiosyncratic noise rather than the underlying manifold
geometry. By selecting the smallest architecture capable of minimizing the
validation negative log-likelihood, we ensure that the model remains a compact
description of the anatomical distribution. 

Our validation sweep across the NNL and PPMI cohorts confirmed the robustness of
this architectural window. For the NNL cohort ($N = 346$), a depth of
$K=4$ was consistently optimal across T1, DTI, and rsfMRI modalities,
effectively minimizing the validation negative log-likelihood. In the PPMI
cohort ($N = 1769$), while increasing capacity to $K=5$ yielded slightly
lower likelihoods, the improvement was negligible ($\Delta \text{BPD} < 0.001$
for DTI) and did not justify the additional model complexity. Prioritizing model
parsimony and methodological consistency between datasets, we fixed the
configuration at $K=4$ and $HC=80$ for all subsequent multiview experiments.
This stable parametric baseline ensures that the learned latent representations
are comparable across different clinical populations while avoiding overfitting
to dataset-specific noise.

