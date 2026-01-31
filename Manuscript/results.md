
\clearpage

# Results

We demonstrate our proposed framework by comparing established linear multiview
baselines with invertible flow models using both tabular and image data. Our
initial set of experiments uses UK Biobank imaging-derived phenotypes (IDPs) to
demonstrate statistical significance uplift with LAMNr flows over SiMLR-based
analysis.  The second set of experiments showcase the utility of LAMNr flows
built from Glow networks for parametrically characterizing multi-modal imaging
distributions.

Our tabular-based experiments is performed in two phases.  Both phases utilize
UKBB IDPs from previous research.  In prior work [@Tustison:2024aa], UKBB IDPs
generated from structural MRI data (i.e., T1-weighted) using three widely used
processing suites (FSL, FreeSurfer, and ANTsX).  More recently [@Avants:2025aa],
UKBB IDPs derived from T1-weighted, diffusion tensor imaging, and resting state
fMRI were used to perform a large-scale evaluation of the SiMLR framework.  We
use both sets of results to guide hyperparameter selection for a fair comparison
between LAMNr flows and SiMLR in predicting clinical variables of interest while
simultaneously exploring the effects of latent alignment.  

We also evaluate Glow-based LAMNr flow models using multi-modal MRI.  Using
publicly available multimodal MRI data from the Parkinson Progression Marker
Initiative (PPMI) [@PPMI:2011aa] and the Normative Neurological Library (NNL)
[@Gage:2024aa].  We demonstrate the utility of invertible flows by exploiting
simplified editing in latent space and conditional Gaussian modeling for
navigating between views (e.g., image imputation).  

