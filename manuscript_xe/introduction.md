
\clearpage

# Introduction

## Quantitative functional imaging of pulmonary ventilation

The lung presents a distinctive challenge for magnetic resonance imaging (MRI).
Its low proton density, rapid transverse signal decay, and numerous air--tissue
interfaces substantially limit conventional proton-based imaging of regional
pulmonary function. Hyperpolarized noble gases provide a means of circumventing
these limitations by directly visualizing the distribution of an inhaled
contrast agent within the airspaces of the lung. Early studies using
hyperpolarized helium-3 demonstrated the feasibility of depicting normal and
abnormal ventilation and established the sensitivity of regional signal
abnormalities to obstructive lung disease [@Bachert:1996aa; @Kauczor:1996aa;
@Kauczor:1997aa]. Subsequent development of hyperpolarized xenon-129
($^{129}$Xe) MRI extended this capability while offering practical advantages in
availability and the additional potential to characterize gas exchange
[@Mugler:2010aa; @Dregely:2011aa]. Ventilation imaging with hyperpolarized gas
has since been applied across a broad range of pulmonary conditions, including
asthma, cystic fibrosis (CF), chronic obstructive pulmonary disease (COPD), and
interstitial lung disease (ILD) [@Altes:2001aa; @Kirby:2012ab; @Santyr:2019aa;
@Mammarappallil:2019aa; @Myc:2020aa]. Recent reviews summarize the expanding set
of quantitative ventilation, gas-exchange, and microstructural measurements [@Stewart:2022aa],
including their emerging roles across cardiopulmonary disease [@Schmidt:2025aa].

The principal image feature used in many of these applications is the
ventilation defect. A ventilation defect is operationally defined as a lung
region exhibiting absent or abnormally low hyperpolarized gas signal, consistent
with nonventilated or underventilated airspaces.  Relatedly, ventilation defect
percentage (VDP) expresses the corresponding defect volume as a percentage of
total lung volume [@Niedbalski:2021aa]. Although such defects can be visually
conspicuous, their conversion into reproducible quantitative measurements
requires decisions concerning lung delineation, intensity normalization, and
assignment of image voxels to ventilation classes. A variety of algorithms have
therefore been proposed, including binary thresholding, linear binning,
hierarchical and adaptive clustering, fuzzy spatial clustering, and
probabilistic mixture modeling [@Tustison:2011aa; @Kirby:2012aa; @Zha:2016aa; @Hughes:2018aa;
@He:2016aa; @He:2020aa]. More recently, convolutional neural networks have
enabled direct image-domain segmentation while incorporating spatial context
unavailable to intensity-only approaches [@Tustison:2019ac; @Tustison:2021aa].

VDP is intuitive, clinically interpretable, and has demonstrated repeatability
and potential utility as an imaging biomarker in single- and multisite studies
[@Couch:2019aa; @Svenningsen:2020aa]. Nevertheless, the transformation from a
three-dimensional ventilation image to a single percentage is necessarily lossy.
Subjects with different numbers, sizes, locations, and spatial distributions of
defects can have the same VDP. The measurement consequently preserves overall
defect burden while discarding much of the spatial information that
distinguishes diffuse from focal abnormality, peripheral from central
involvement, and regionally organized from spatially dispersed dysfunction.
These distinctions are potentially important because pulmonary diseases may
produce overlapping global burdens while differing in their regional expression
and underlying pathophysiology.

Evidence that ventilation images contain clinically relevant information beyond
global defect burden predates recent deep learning approaches. In an early
analysis of hyperpolarized 3He ventilation MRI, approximately 1,600
image-derived features were evaluated in 55 participants with and without asthma
[@Tustison:2010ab]. The highest-ranked image features individually carried
substantially more information about diagnostic status than standard spirometric
measurements, while the imaging and spirometric features provided largely
complementary information. This study demonstrated the phenotypic value of
spatially resolved ventilation patterns, but it relied on a large set of
predefined, handcrafted descriptors. More recent studies have extended this
observation using spatial distribution indices, texture analysis, and learned
image representations. A three-dimensional defect distribution index
distinguished the spatial clustering of ventilation defects across obstructive
and restrictive disease groups, even when interpreted alongside VDP
[@Bdaiwi:2025aa]. Texture features derived from $^{129}$Xe ventilation MRI were also
associated with longitudinal quality-of-life improvement in long COVID
[@Kooner:2024aa]. Most recently, a supervised convolutional network classified
pulmonary diseases from ventilation images with performance exceeding that of
human observers, supporting the presence of disease-associated spatial patterns
not represented by VDP alone [@Matheson:2025aa].

This limitation is related to, but more fundamental than, the distinction
between histogram- and image-based segmentation. We previously showed that
histogram-based quantification discards contextual spatial information and can
exhibit reduced measurement precision under common MRI perturbations relative to
direct image-domain analysis [@Tustison:2021aa]. An image-based segmentation
model mitigates part of this loss by using spatial information to assign labels.
However, the final conversion of that segmentation to VDP again collapses the
result to a scalar. Thus, even an accurate segmentation does not by itself
provide a representation of how ventilation abnormalities are spatially
organized across subjects. A more general quantitative framework would retain
the complete image, support comparisons at multiple spatial scales, and provide
a principled geometry for measuring population variation without requiring
predefined ventilation classes.

## Invertible generative modeling as a quantitative framework

Deep generative models provide a possible route from scalar quantification
toward population-level representation of complete images. In contrast to
discriminative models optimized for a specific label or endpoint, generative
models attempt to characterize the distribution from which the observations
arise. This distinction is useful for functional lung imaging, where the desired
representation may need to support several downstream tasks, including
phenotyping, similarity analysis, interpolation, outlier detection, and
eventually conditional inference, without committing the training procedure to
a single clinical categorization.

Normalizing flows are particularly attractive for this purpose because they
define a bijective transformation between the image domain and a tractable
latent probability distribution [@dinh2016realnvp; @kingma2018glow;
@papamakarios2021nfreview; @kobyzev2020nfsurvey]. Unlike generative adversarial
networks, which learn an implicit distribution, or conventional variational
autoencoders, which generally trade exact reconstruction for a lower-dimensional
stochastic code, normalizing flows retain an explicit inverse and permits
exact likelihood evaluation under the specified model. Each observed image can
therefore be mapped to a unique latent representation and reconstructed from
that representation, subject to numerical precision. This correspondence
establishes a coordinate system in which the variability of the observed
population can be examined quantitatively [@Tustison:2026aa].

Flow-based modeling of pulmonary ventilation nevertheless presents several
technical challenges. Hyperpolarized gas ventilation MR images contain extensive
background regions, with informative signal largely confined to the lungs and
distributed heterogeneously within the pulmonary volume. To improve robustness
and reduce sensitivity to nearly uniform background intensities, we introduce
small random intensity and shape perturbations during training as a form of data
augmentation [@Tustison:2021aa;@Tustison:2024aa;@Tustison:2025aa]. The perturbation magnitude
provide sufficient regularization without obscuring clinically meaningful
regional ventilation patterns. In addition, full three-dimensional modeling
imposes substantial memory and computational requirements, making the balance
among spatial resolution, multiscale depth, and network capacity especially
consequential. We refer to this augmentation as ``dequantization noise'' for
consistency with flow-modeling terminology (i.e., [@ho2019flowpp]), although
here it is used primarily as a data smoothing operation rather than to
correct an assumed discrete data-generating process.

The multiscale construction introduced by Glow offers a natural organization for
medical and biological image data [@kingma2018glow]. Successive squeezing,
invertible transformation, and splitting operations factor the image
representation across resolution levels. Rather than interpreting the latent
variables as a single undifferentiated vector, the resulting hierarchy can be
interrogated by resolution level to determine how group relationships change
across spatial scales. Pretrained vision and vision-language foundation models
provide an alternative means of encoding medical images into compact
representations for similarity analysis, clustering, content-based retrieval,
and downstream prediction [@zhang2025biomedclip; @codella2024medimageinsight;
@denner2025foundation]. Such embeddings can capture transferable semantic
features, but they are generally optimized for invariance or discrimination
rather than information-preserving. By contrast, the flow-based mapping remains
invertible, allowing individual latent levels to be manipulated and decoded to
assess their contributions in the image domain. This combination of multiscale
organization and exact invertibility distinguishes the proposed analysis from
both post hoc dimensionality reduction and foundation-model embeddings, for
which correspondence with the complete image is generally noninvertible.

An additional consideration concerns the geometry of the Gaussian latent
distribution. In high dimensions, probability mass is concentrated within a
typical set located away from the origin. Consequently, the Gaussian mean is a
mathematically convenient reference but is not representative of a typical
encoded subject. Linear interpolation through the origin can similarly traverse
low-probability regions and produce trajectories that are inconsistent with the
radial organization of the latent distribution. Normalizing latent
representations to a common radius and using spherical linear interpolation
provides an alternative that follows the hyperspherical typical set. Pairwise
angular or arc-length distances along this set can then be evaluated separately
at each resolution level or combined across the complete latent representation.
Importantly, these distances describe geometry induced by the learned model and
their biological meaning must be established empirically rather than assumed
from the Gaussian prior alone [@white2016sampling; @arvanitidis2018latent].

## Motivation

\begin{figure}[!htb]
  \centering
  \resizebox{\textwidth}{!}{%
    \begin{tikzpicture}[x=1cm,y=1cm,>=Latex]
      % A slightly taller canvas gives the image captions and explanatory text
      % their own vertical bands instead of forcing them against the images.
      \useasboundingbox (0,0) rectangle (20,12.1);
      \fill[white] (0,0) rectangle (20,12.1);

      % ----- Panel A ----------------------------------------------------------
      \shade[left color=black!12,right color=black!3] (0,12.1) rectangle (20,11.1);
      \node[font=\bfseries\fontsize{12}{18}\selectfont] at (.5,11.6) {(a)};
      \node[font=\fontsize{14}{16}\selectfont] at (10,11.6)
        {\textbf{Ventilation Defect Percentage}};
      \node[font=\fontsize{15}{16}\selectfont] at (3.25,10.3) {Input};
      \node[font=\fontsize{15}{16}\selectfont] at (10.1,10.3) {Lossy Transformation};
      \node[font=\fontsize{15}{16}\selectfont] at (16.75,10.3) {Output};

      \node[inner sep=0pt] (youngA) at (1.80,8.72)
        {\includegraphics[width=2.55cm]{Figures/young_healthy_subject.png}};
      \node[inner sep=0pt] (cfA) at (4.85,8.72)
        {\includegraphics[width=2.55cm]{Figures/cf_subject.png}};
      \node[font=\fontsize{12}{13}\selectfont,below=1mm of youngA] {Young healthy};
      \node[font=\fontsize{12}{13}\selectfont,below=1mm of cfA] {Cystic fibrosis};

      \path[draw=black,line width=1.5pt,top color=softblue,bottom color=softblue!65!black]
        (7.5,9.88) .. controls (8,9.27) and (10.35,9.03) .. (11.72,9.00)
        -- (11.72,8.30) .. controls (10.35,8.27) and (8,8.05) .. (7.5,7.48) -- cycle;
      \node[font=\fontsize{15}{16}\selectfont] at (9.5,8.70) {Binarization};
      \path[draw=black,line width=1.5pt,fill=black!62]
        (11.72,8.78)--(12.5,8.78)--(12.5,9.10)--(13.12,8.62)--(12.5,8.14)--(12.5,8.47)--(11.72,8.47)--cycle;

      \draw[line width=1.8pt] (14.22,8.72)--(19.43,8.72);
      \foreach \x in {14.22,14.74,15.26,15.78,16.30,16.82,17.34,17.86,18.38,18.90,19.43}
        \draw[black!65] (\x,8.72)--(\x,8.53);
      \draw[line width=2pt] (14.22,8.90)--(14.22,8.54) (19.43,8.90)--(19.43,8.54);
      \draw[->, >=stealth,line width=2pt,color=deepred!90] (15.59,9.3) -- (15.59,8.8);
      % \draw[rounded corners=1pt,fill=black!18,draw=black!55,line width=1pt] (15.59,8.42) rectangle (15.71,9.04);
      \node[font=\fontsize{14}{15}\selectfont] at (14.22,8.17) {0};
      \node[font=\fontsize{14}{15}\selectfont] at (15.65,8.17) {0.25};
      \node[font=\fontsize{14}{15}\selectfont] at (16.85,8.17) {0.5};
      \node[font=\fontsize{14}{15}\selectfont] at (18.14,8.17) {0.75};
      \node[font=\fontsize{14}{15}\selectfont] at (19.43,8.17) {1};

      % ----- divider and panel B ---------------------------------------------
      % \draw[line width=1.5pt] (0,6.05)--(20,6.05);
      \shade[left color=black!8,right color=black!2] (0,6.38) rectangle (20,5.38);
      \node[font=\bfseries\fontsize{12}{18}\selectfont] at (.5,5.85) {(b)};
      \node[font=\fontsize{14}{15}\selectfont] at (10,5.85)
        {\textbf{Multiscale Latent Geometry}};
      \node[font=\fontsize{14}{16}\selectfont] at (3.20,4.58) {Input};
      \node[font=\fontsize{14}{16}\selectfont] at (10.05,4.58) {Homeomorphic Transformation};
      \node[font=\fontsize{14}{16}\selectfont] at (16.60,4.58) {Output};

      \node[inner sep=0pt] (youngB) at (1.75,3.0)
        {\includegraphics[width=2.60cm]{Figures/young_healthy_subject.png}};
      \node[inner sep=0pt] (cfB) at (4.78,3.0)
        {\includegraphics[width=2.60cm]{Figures/cf_subject.png}};
      \node[font=\fontsize{11}{12}\selectfont,below=1mm of youngB] {Young healthy};
      \node[font=\fontsize{11}{12}\selectfont,below=1mm of cfB] {Cystic fibrosis};
      \node[font=\fontsize{14}{15}\selectfont] at (3.2,.75) {Image space $\mathcal X$};

    \path[
      draw=black,
      line width=1.5pt,
      fill=softblue!75
    ]
      (6.90,3.7)
      .. controls (8.35,3.38) and (11.90,3.38) .. (13.25,3.7)
      -- (13.25,2.2)
      .. controls (11.90,2.5) and (8.35,2.5) .. (6.90,2.2)
      -- cycle;

    % Jet-stream curves
    \begin{scope}[
      every path/.style={
        <->,
        >=stealth,
        draw=deepred!80!black,
        line width=1.1pt
      }
    ]

      \draw
        (7.15,3.35)
        .. controls (8.70,3.17) and (11.45,3.17) ..
        (13.00,3.35);

      \draw
        (7.15,3.07)
        .. controls (8.75,2.98) and (11.40,2.98) ..
        (13.00,3.07);

      \draw
        (7.15,2.77)
        .. controls (8.75,2.85) and (11.40,2.85) ..
        (13.00,2.77);

      \draw
        (7.15,2.48)
        .. controls (8.70,2.66) and (11.45,2.66) ..
        (13.00,2.48);

    \end{scope}
        
      \node[font=\fontsize{14}{15}\selectfont,align=center] at (10.05,1.55) {Continuous bijection\\via normalizing flows };

      % Latent-space target
      \begin{scope}[shift={(16.58,2.65)}]
        \shade[shading=radial,inner color=deepred!60,middle color=softred,outer color=deepblue!60] (0,0) circle (1.48);
        \foreach \r in {.35,.66,.94,1.20,1.47} \draw[line width=1pt] (0,0) circle (\r);
        \draw[deepblue,line width=1.2pt,-{Latex[length=2.5mm]}] (1.50,-0.95)--(1.0,-.55);
        \draw[deepred,line width=1.2pt,-{Latex[length=2.5mm]}] (-1.36,-1.0)--(-.72,-.50);
        \node[font=\fontsize{9}{10}\selectfont] at (0,-.48) {$L_3$};
        \node[font=\fontsize{9}{10}\selectfont] at (0,-.78) {$L_2$};
        \node[font=\fontsize{9}{10}\selectfont] at (0,-1.08) {$L_1$};
        \node[font=\fontsize{9}{10}\selectfont] at (0,-1.37) {$L_0$};
      \end{scope}
      \node[deepblue,font=\fontsize{10}{13}\selectfont,align=center] at (19.0,1.8)
        {Finer-scale\\($L_0$, $L_1$)};
      \node[deepred,font=\fontsize{10}{13}\selectfont,align=center] at (14.18,1.5)
        {Coarser-scale\\($L_2$, $L_3$)};
      \node[font=\fontsize{10}{13}\selectfont,align=center] at (19.0,4.27) {Gaussian\\typical set};
      \draw[-{Latex[length=2.5mm]},line width=1pt] (18.12,3.95)--(16.85,3.34);

      \node[font=\fontsize{14}{15}\selectfont] at (16.60,.75) {Latent space $\mathcal Z$};

    \end{tikzpicture} 
 }
  \caption{
    Scalar reduction versus invertible multiscale representation of
    pulmonary ventilation. Representative $^{129}$Xe ventilation MR images from
    a young healthy participant and a participant with cystic fibrosis are shown
    to illustrate the two analysis pathways. (a) Conventional ventilation defect
    percentage (VDP) analysis requires binarization of the ventilation image
    into defect and nondefect classes followed by calculation of the fraction of
    lung assigned to the defect class. This lossy transformation preserves
    global defect burden but discards the number, size, location, and spatial
    arrangement of the defects. (b) The proposed normalizing flows framework
    learns a continuous bijection between image space $\mathcal{X}$ and latent
    space $\mathcal{Z}$. The bidirectional streamlines denote the forward and
    inverse transformations, through which each image is encoded and
    reconstructed. The multiscale architecture factors the latent representation
    across levels $L_0$--$L_3$, enabling finer- and coarser-scale contributions
    to population geometry to be examined separately or jointly. The concentric
    latent-space depiction schematically represents the Gaussian typical set and
    the multiscale organization.
  }
  \label{fig:vdp-latent-geometry}
\end{figure}

Figure \ref{fig:vdp-latent-geometry} summarizes the central methodological
contrast motivating this study. Conventional VDP quantification maps a complete
ventilation image, through binarization, to a single scalar value denoting the
global defect fraction. Our framework instead maps the image bijectively to a
multiscale latent representation, retaining the information required for
reconstruction while providing coordinates in which population variation can be
analyzed. Motivated by this contrast, we investigate normalizing flows as a
general quantitative framework for three-dimensional hyperpolarized $^{129}$Xe
ventilation MRI. Our objective is not to replace a specific VDP segmentation
algorithm or to optimize diagnostic classification. Instead, we ask whether an
invertible, multiscale density model can preserve pulmonary ventilation images
in a continuous latent coordinate system whose geometry contains clinically
relevant structure despite being learned without diagnostic supervision. This
formulation treats disease labels as external variables for evaluating the
learned representation rather than targets used to construct it.

We make four principal contributions. First, we adapt a multiscale normalizing
flow architecture to the low-resolution, spatially heterogeneous characteristics
of hyperpolarized gas ventilation MRI through shape- and intensity-based data
augmentation. Second, we define subject-level representations from the learned
flow latents and perform comparisons within the Gaussian typical set, enabling
pairwise distance analysis and spherical interpolation. Third, we exploit the
multiscale factorization of the flow to characterize complementary phenotypic
information at individual latent resolution levels and across their combination,
while retaining the ability to decode level-specific latent manipulations into
the image domain. Fourth, we evaluate the learned geometry in an exploratory
cohort comprising young and older healthy volunteers, together with participants
with CF, COPD, or ILD. Because diagnostic group labels are not used during
training, their correspondence with the latent organization provides an
independent, post hoc assessment of whether the unsupervised model captures
clinically meaningful variation in pulmonary ventilation.  Notably all functionality
is publicly available as mature open-source software via the ANTsX ecosystem.[^antsx]

[^antsx]: https://github.com/ANTsX

