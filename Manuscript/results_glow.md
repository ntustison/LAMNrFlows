

\clearpage

## Glow-based LAMNr Flows

### Leveraging approximate template $\leftrightsquigarrow$ subject geodesic linearity for image registration via latent winsorization

To evaluate the utility of the learned latent representations for downstream
geometric tasks, we applied LAMNr Flows to the challenge of deformable image
registration in the presence of focal pathology (e.g., brain tumors).
Traditional registration metrics often struggle with "outlier" intensities
caused by tumors, leading to non-anatomical deformations as the algorithm
attempts to match pathological tissue to healthy templates. We hypothesize that
the bijective latent space of LAMNr exhibits approximate geodesic linearity,
where the path between subjects is largely governed by shared anatomical
features once view-specific or pathological "noise" is suppressed.

We implemented a latent winsorization schedule to guide the registration
process. By progressively relaxing the percentile bounds of the latent
variables—starting with a restrictive winsorization (e.g., 0.1) and
transitioning to the full latent signal (1.0)—we effectively regularize the
deformation field. In the early iterations, strong winsorization "flattens" the
latent outliers associated with the tumor, allowing the registration to focus on
the shared anatomical manifold. As the schedule progresses and the images become
globally aligned, the winsorization is lifted, permitting fine-grained local
adjustments.

Our results on the multimodal cohort demonstrate that this schedule prevents the
localized "warping" artifacts common in standard ANTs-based registration. By
leveraging the structured latent space of LAMNr, we achieve a registration that
is robust to focal pathology without requiring manual lesion masking,
effectively using the latent space as a prior for anatomical consistency.

__Relationship to the population Fréchet mean__.  The generative capacity of
LAMNr Flows provides a direct link to classical anatomical template
construction. In the context of symmetric normalization (SyN) and diffeomorphic
mapping, a population template is formally defined as the Fréchet mean of the
group—the image that minimizes the sum of squared geodesic distances to all
subjects within a given population [@Avants:2010aa]. In the LAMNr framework, the
latent space is anchored by a centered Gaussian distribution where the origin
($z=0$) represents the statistical mode and mean.  Mapping this origin back to the
image domain yields a "latent-mean" reconstruction that captures the shared
anatomical features of the cohort, effectively serving as an approximation of
the Fréchet mean. Furthermore, due to the approximate geodesic linearity of the
learned latent space, the linear path from any subject's latent representation
$z_i$ toward the origin approximates the geodesic deformation toward the
population average. 