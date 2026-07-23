
# Abstract {-}

Quantification of hyperpolarized pulmonary ventilation MRI relies on
scalar summaries such as ventilation defect percentage, which omit consideration
of the spatial organization of functional abnormalities. We introduce a homeomorphic
multiscale latent geometry for spatially informed analysis of pulmonary
ventilation, constructed using trained three-dimensional normalizing flow (i.e.,
Glow) networks. Each ventilation volume is mapped through a continuous bijection
(with a continuous inverse) to a Gaussian latent distribution. The trained
network decomposes each image into multiscale resolution levels, facilitating
complementary characterization of local and global ventilation patterns without
loss of information. The resulting structured latent representation supports
comparative quantification through pairwise distances and spherical
interpolation trajectories within the Gaussian typical set. Moreover, the
homeomorphic construction preserves topological neighborhoods between the image
and latent spaces, such that local perturbations and continuous trajectories in
image space correspond to continuous changes in the latent space. Relationships
can therefore be decoded as complete ventilation volumes for direct visual and
quantitative interrogation. We evaluated the framework in an exploratory cohort
of 45 participants comprising cystic fibrosis, chronic obstructive pulmonary
disease, interstitial lung disease, and young and older healthy groups. The
learned geometry exhibited scale-dependent organization associated with clinical
phenotypes. Fine and coarse latent variables showed complementary group
relationships consistent with localized abnormalities and global ventilation
organization, respectively. These findings establish the feasibility of latent
modeling for functional lung MRI and provide a continuous, spatially informed
alternative to one-dimensional, defect-based summaries. The framework supports
unsupervised phenotyping, multiscale quantification, and generative
interrogation of pulmonary ventilation.

**Keywords:** functional lung imaging; hyperpolarized xenon-129 MRI; normalizing
flows; homeomorphism; latent geometry; unsupervised phenotyping
