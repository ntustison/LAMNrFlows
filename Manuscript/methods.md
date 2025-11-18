# Methods

## Overview

LAM-Flow comprises two components. First, a set of per-view flows are trained by maximum likelihood with latent alignment on designated shared coordinates. Alignment can be applied across views directly or after an optional CCA or HSIC screen that selects directions with evidence of cross-view dependence. Second, a conditional Gaussian layer is fit over latents to enable closed-form posteriors and exact decoding for imputation, harmonization, counterfactual edits, and shared-latent reconstructions.

We denote a subject’s data by \( \{x^{(v)}\}_{v=1}^V \). Each view \(v\) has a flow \(f^{(v)}\) mapping \(x^{(v)}\) to latents \(z^{(v)}\) with density \(p(z^{(v)})\) and exact log-likelihood \(\log p(x^{(v)}) = \log p(z^{(v)}) + \log |\det \partial f^{(v)} / \partial x^{(v)}| \) [@papamakarios2021nfreview; @kingma2018glow]. We split latents into shared and private parts, \(z^{(v)} = [z^{(v)}_S, z^{(v)}_P]\), with selection defined by either a fixed allocation or by screening.

## Per-view flow architectures

### Image views

For images we adopt Glow-style discrete flows with L levels and K steps per level. Each step comprises ActNorm, invertible 1×1 convolutions, and affine coupling with convolutional predictors. Squeeze and split operations provide multiscale access to per-level latents. LU parameterization is used for stable log-determinant computation, and data-dependent initialization is used for ActNorm [@kingma2018glow]. We also considered variants with richer coupling and spline transforms [@ho2019flowpp; @durkan2019nsf]. Continuous-time flows are related but we do not use them here due to the cost of unbiased likelihood estimates [@grathwohl2019ffjord].

### IDP and CSV views

For tabular blocks we use RealNVP or MAF stacks with affine couplings and masked MLPs [@dinh2016realnvp; @papamakarios2017maf]. Continuous variables are standardized. Positive skewed variables can be log or log1p transformed. Bounded variables can be mapped to the real line with a logit transform. Categorical variables are one-hot encoded when needed. Each per-view flow yields a single-scale latent vector, to which the same projector, screening, and alignment steps are applied.

## Projector, alignment losses, and shared subspace screening

Let \(\phi^{(v)}(z^{(v)}) \in \mathbb{R}^D\) be the output of a small projector head that produces features with matched dimension across views. Given subject-matched batches \(\{\phi^{(v)}_n\}_{n=1}^N\), we apply an alignment loss on the coordinates designated as shared. We support Barlow Twins, VICReg, InfoNCE, Pearson correlation, and HSIC losses, each with standard hyperparameters [@zbontar2021barlow; @bardes2021vicreg; @oord2018cpc; @gretton2005hsic].

To identify shared coordinates automatically, we perform a short screening pass after an MLE warm-up. For CCA screening, we compute whitening matrices for two views and perform an SVD of the whitened cross-covariance \(X_a^\top X_b\). We retain the top r canonical directions per view and average pairwise subspaces across all view pairs to obtain per-view projectors \(P^{(v)} \in \mathbb{R}^{D \times r}\). For HSIC screening, we score coordinates by dependence using a two-stage procedure. First, a Pearson prefilter selects candidate dimensions. Second, an unbiased HSIC estimate with RBF kernels ranks dependence with the average of the other views. We then select the top r coordinates per view. The alignment loss is applied only to the projected or masked features. Screening can be performed once after warm-up or on a fixed refresh cadence [@Murphy2012ML; @bishop2006prml; @gretton2005hsic].

## Conditional Gaussian inference over latents

After training, we estimate per-level Gaussian statistics for latents. For images we collect \(z^{(v)}_\ell\) at each level \(\ell \in \{1,\dots,L\}\). For IDP blocks there is a single level. We model the joint latent vector across views as Gaussian with mean \(\mu\) and covariance \(\Sigma\). Given an observed subset \(O\) and an unobserved subset \(U\), the posterior \(p(z_U \mid z_O)\) is Gaussian with mean
\[
\mu_{U \mid O} = \mu_U + \Sigma_{UO} \Sigma_{OO}^{-1} (z_O - \mu_O)
\]
and covariance
\[
\Sigma_{U \mid O} = \Sigma_{UU} - \Sigma_{UO} \Sigma_{OO}^{-1} \Sigma_{OU}.
\]
We use shrinkage for numerical stability when needed and clip small eigenvalues to maintain positive definiteness [@schafer2005shrinkage; @ledoit2004well; @higham2002accuracy]. The posterior mean provides a calibrated point estimate. Samples from the posterior propagate uncertainty. Exact decoding through the inverse flows yields imputations, harmonized outputs, and latent edits with tractable likelihoods.

## Shared-latent reconstructions and operational edits

For image views we define shared-latent images by replacing private latents with their conditional means while holding shared latents fixed. This yields reconstructions that preserve geometry and suppress view-specific contrast. When a downstream task is sensitive to contrast or confounders, a transform can be estimated on these surrogates and then applied to the original images without further approximation. For IDP blocks, the same conditional layer provides calibrated imputation under missingness patterns, counterfactual queries with covariate edits, and harmonization across sites. We evaluate with likelihood calibration, imputation error, and predictive transfer in line with prior IDP analyses [@Tustison:2024aa].

## Relation to latent-space templates

Latent-space templates can be defined as means in latent space decoded back to image space, or as Monte Carlo expectations under the learned latent distribution. In the small-variance or locally linear regime the two constructions agree up to second order terms, which connects latent templates to Fréchet means in the induced metric. We present precise definitions, per-level composition, and conditions for agreement in the Supplementary Methods, and we use the constructs here as operational tools for surrogates and visualization.

## Practical training details

For images we use multiscale Glow with data-dependent ActNorm initialization, LU factorization for invertible convolutions, and a fixed number of steps per level. For tabular blocks we use 8 to 12 coupling layers with width between 256 and 512 units. We train by maximum likelihood with mixed precision and exponential moving averages of weights. Augmentations are used for images as described in the experimental section. Alignment losses are weighted relative to likelihood and turned on after a short warm-up. Screening uses CCA with ridge regularization or HSIC with a Pearson prefilter. We monitor exact NLL and simple latent calibration diagnostics for both images and IDPs [@kingma2018glow; @papamakarios2021nfreview; @zbontar2021barlow; @bardes2021vicreg].
