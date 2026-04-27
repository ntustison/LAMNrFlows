
\clearpage 

# Discussion

LAMNr flows provide a flexible framework for data modeling which provides exact
likelihoods and bijective mappings between input spaces and their corresponding
latent spaces.  Among other possibilities, this permits a deep learning-based
perspective of traditional CA. Historically, traditional CA has been
fundamentally defined by image registration in which diffeomorphic
transformation groups are leveraged to model biological shape variability. DCA,
as described and implemented within the LAMNr flows framework, bypasses this
explicit image registration requirement by topologically unfolding non-linear
anatomical manifolds into a structured latent space via coordinated normalizing
flows instances. Instead of iterative registration workflows, DCA employs
single-pass bijective mappings per modality for inferring fundamental CA
concepts. 

As a salient example, traditional population template construction requires
image registration for establishing joint anatomical correspondence as a
prerequisite for computing the central intensity and morphological tendency of a
cohort. In contrast, the DCA-based population template is defined as the inverse
mapping of the latent origin which provides a barycentric anchor in latent space
for characterizing the cohort central tendency. Because this generative template
is the exact mode of the learned distribution, it naturally filters
idiosyncratic high-frequency noise, yielding a smooth representation of shared
structural signals. Furthermore, to navigate this space without the variance
collapse typical of high-dimensional Euclidean operations, we utilize spherical
linear interpolation. This ensures that interpolative trajectories
remain strictly on the typical set—the high-probability manifold where realistic
anatomical instances reside.  Similarly, the metric operations of DCA substitute
the image registration of CA with algebraic interpolations calculated directly
within the latent manifold which respect the underlying latent-space geometry
with efficient inverse single passes through the network(s). By leveraging the
exact-likelihood foundations of normalizing flows, this framework parallels the
geometric rigor of traditional CA while potentially providing a deep
learning-based approach for multimodal biological analysis.

Beyond CA modeling, the empirical results demonstrate the utility of LAMNr flows
for clinical and biological investigation. In the tabular experiments, LAMNr
flows achieved a significant "correlation uplift" over linear SiMLR baselines
when predicting cognitive outcomes such as working memory and delayed recall
within the NNL cohort. This suggests that the non-linear unfolding of the
anatomical manifold captures subtle biological couplings that are inaccessible
to linear subspace projections. However, the competitive performance of linear
models in the PPMI cohort indicates that pathological signals, such as those
associated with Parkinson's disease, may be dominated by stronger, more linear
variance structures. This divergence highlights the importance of selecting
alignment strategies, such as VICReg or HSIC, that balance density estimation
with the specific geometric attributes of the datasets of interest.

While traditional diffeomorphic image registration algorithms excel at alignment
for large deformation scenarios, significant topological disruptions, such as
tumor-induced changes, can limit accuracy.  One of our early hypotheses in the
development of this work was that DCA-based latent interpolation would be able
to  overcome such topological difficulties by providing an intermediate image
($t=0.5$) for more robust image registration.  The BraTS-Reg22 challenge
[@baheti2024braintumorsequenceregistration] provided the ideal opportunity to
test such an hypothesis as it involved image registration data pre- and 
post-resection with expert-annotated landmarks.  Although preliminary evaluations 
demonstrate competitive structural recovery (cf. Figure \ref{fig:interpolation})
for such data, the limited resolution of our 3D LAMNr models was insufficient
for the task and will be postponed for future work when hardware capabilities 
increase.

Finally, the detailed framework explores the current challenges associated with
scaling and training LAMNr flows and their constituent normalizing flows
architectures.  While Glow-style architectures are memory-intensive due to the
requirement of storing intermediate activations, we mitigate these constraints
through architectural refinements such as gradient microbatching and bounded
coupling scales. Additionally, the integration of a low-rank-plus-diagonal
Gaussian parameterization, solved via the Woodbury matrix identity, enables
exact conditional inference for cross-modal imputation in 3D. This allows LAMNr
flows to scale to volumetric data while maintaining a manageable memory
footprint. These advancements, integrated into the ANTsX ecosystem via
ANTsTorch, provide a robust and scalable foundation for future exploration of
deep computational anatomy.