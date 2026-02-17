#!/bin/bash
###############################################################################
# LAM-Flow Inference & Analysis Pipeline
# Project: PPMI T1/FA Latent Modeling
# Date: 2026-01-26
#
# This script manages the post-training workflow for 2D Glow models,
# including Gaussian distribution fitting, cross-modal imputation, 
# and population-level template reconstruction.
###############################################################################

# --- Global Configuration ---

# The base directory where your research project is housed.
base_dir="/Users/ntustison/Desktop/lamnr_glow_ppmi"

which_experiment="256x256"  # Options: "128x128", "256x256", etc. Adjusts image size and model config.

# Destination for all generated images, Gaussian models, and JSON summaries.
out_dir="${base_dir}/output${which_experiment}/"

# The training results directory. Contains 'training_state.pt' and logs.
if [[ ${which_experiment} == "128x128" ]]; then
  runs_dir="${base_dir}/runs/ppmi_t1_fa_${which_experiment}_K12_L5_HC192/"
elif [[ ${which_experiment} == "256x256" ]]; then
  runs_dir="${base_dir}/runs/ppmi_t1_fa_${which_experiment}_K12_L6_HC192/"
else
  echo "Error: Unrecognized experiment configuration '${which_experiment}'. Please set 'which_experiment' to a valid option."
  exit 1
fi

manifest_dir="${base_dir}/manifests/"

# The specific PyTorch checkpoint containing model weights (EMA/State Dict).
ckpt="${runs_dir}/training_state.pt"

# Manifest CSVs: These link subject IDs to their respective T1 and FA image paths.
manifest="${manifest_dir}/manifest_ppmi.csv"
manifest_short="${manifest_dir}/manifest_ppmi_short.csv"
manifest_lesions="${manifest_dir}/manifest_brats_short.csv"  

# Numerical parameters for slice extraction and computational batching.
SLICE_INDEX=138
WHICH_PYTHON="/Users/ntustison/anaconda3/bin/python3"
DEVICE="cpu"  # Options: 'cpu', 'cuda:0', etc.

# --- Derived Paths ---
# Serialized Gaussian model and its statistical summary for latent-space math.
gaussian_lr=${out_dir}/t1_fa_lowrank.npz
gaussian_lr_summary=${out_dir}/t1_fa_lowrank_summary.json

dist_csv=${out_dir}/t1_distance_to_gaussian.csv

###############################################################################
# PIPELINE EXECUTION START
###############################################################################


########################################
# Fit a Gaussian model (full or of low-rank covariance).
# Here we use the PPMI dataset which was used to train the LAMNr model.
# However, one could use any dataset of aligned multi-view images.
########################################

if [[ ! -f ${gaussian_lr} ]]; then
  echo "Gaussian model not found at ${gaussian_lr}. Proceeding to fit the model..."

  cov_mode="perlevel"  # Default to full covariance estimation.
  cov_estimator="full"  # Default to using the full dataset for covariance estimation.

  echo "Fitting Gaussian model to latent representations..."
  if [[ ${which_experiment} == "128x128" ]]; then
    echo "Using full covariance estimation for 128x128 experiment."
    cov_estimator="full"
  elif [[ ${which_experiment} == "256x256" ]]; then
    echo "Using low-rank covariance estimation for 256x256 experiment."
    cov_mode="perlevel"
    cov_estimator="lowrank"
  else
    echo "Error: Unrecognized experiment configuration '${which_experiment}'. Cannot determine covariance mode."
    exit 1
  fi

  ${WHICH_PYTHON} lamnr_glow_tool.py gauss-fit \
    --ckpt ${ckpt} \
    --manifest ${manifest} \
    --views T1,FA \
    --slice-axis 2 --slice-index ${SLICE_INDEX} \
    --batch 64 --devices ${DEVICE} \
    --cov-mode ${cov_mode} \
    --cov-estimator ${cov_estimator} \
    --rank 256 \
    --gauss-out ${gaussian_lr} \
    --gauss-summary ${gaussian_lr_summary}

else
  echo "Gaussian model already exists at ${gaussian_lr}. Skipping fitting step."
fi

########################################
# Utility to extract 2D slices from 3D NIfTI volumes.
# Prepares data for training or visual inspection at a specific axis and index.
# Supports NIfTI (.nii.gz) output to maintain floating-point precision.
########################################

# echo "Exporting 2D slices from 3D volumes for manifest input..."
# ${WHICH_PYTHON} lamnr_glow_tool.py export-slices \
#   --manifest ${manifest_short} \
#   --slice-axis 2 --slice-index ${SLICE_INDEX} \
#   --views T1,FA \
#   --image-size 128x128 \
#   --outdir ${out_dir}/manifest_input/ \
#   --output-format nii.gz

########################################
# Performs modality translation using the conditional Gaussian model.
# Predicts a target view (e.g., T1) from observed data (e.g., FA).
# The 'sample' strategy adds realistic texture variance to the prediction,
# while the 'mean' strategy provides the most likely anatomical structure.
########################################

# echo "Performing cross-modal imputation: Predicting T1 from FA..."
# ${WHICH_PYTHON} lamnr_glow_tool.py gauss-impute \
#   --ckpt ${ckpt} \
#   --gauss ${gaussian_lr} \
#   --manifest ${manifest_short} \
#   --views T1,FA \
#   --observed FA --target T1 \
#   --slice-axis 2 --slice-index ${SLICE_INDEX} \
#   --batch 2 --devices ${DEVICE} \
#   --strategy mean \
#   --temperature 0.1 \
#   --outdir ${out_dir}/impute_FA_from_T1/ \
#   --output-format nii.gz


########################################
# Reconstruction sanity panel.
# Visualizes the accuracy of the bijective mapping x <-> z.
# Produces a 3-column grid:
#   Column 1: Original input slice (x).
#   Column 2: Reconstruction (x_hat) after encoding and decoding.
#   Column 3: Absolute difference map |x - x_hat|.
########################################

# ${WHICH_PYTHON} lamnr_glow_tool.py recon \
#   --ckpt ${ckpt} \
#   --manifest ${manifest_short} --views T1,FA --view-index 0 \
#   --slice-axis 2 --slice-index ${SLICE_INDEX} --batch 6 --devices ${DEVICE} \
#   --out ${out_dir}/recon_t1_panel.png

########################################
# Generates a grid of synthetic brain images from the latent prior.
# The temperature parameter (tau) scales the variance of the Gaussian noise:
#   tau < 1.0: Samples are closer to the mean (structurally stable).
#   tau = 1.0: Samples follow the training distribution.
#   tau > 1.0: Higher diversity but increased risk of artifacts.
########################################

# for temp in 0.01 0.25 0.5 0.75 1.0 1.25 1.5;
#   do
#     echo "Sampling at temperature: ${temp}"
#     ${WHICH_PYTHON} lamnr_glow_tool.py sample \
#       --ckpt ${ckpt} \
#       --view-index 0 --sample-grid-size 6x6 \
#       --image-size 128x128 --temperature ${temp} \
#       --devices ${DEVICE} --sample-grid-out ${out_dir}/Samples/samples_t1_temp_${temp}.png \
#       --seed $RANDOM
#   done 

########################################
# Generates a population-level anatomical template.
# Decodes the mean latent vector (mu) derived from the Gaussian model.
# Using --mc-samples > 0 averages multiple stochastic samples in image space
# to produce a high-fidelity, denoised 'average' brain anatomy.
########################################

# echo "Generating population-level template from Gaussian mean latent vector..."
# ${WHICH_PYTHON} lamnr_glow_tool.py recon-template \
#   --ckpt ${ckpt} \
#   --gauss ${gaussian_lr} \
#   --views T1,FA \
#   --view-index 0 \
#   --mc-samples 0 \
#   --mc-temp 0.01 \
#   --devices ${DEVICE} \
#   --out ${out_dir}/template_T1_mu_sharpened_0.nii.gz \
#   --sharpen-image \
#   --seed ${RANDOM}

########################################
# Performs latent space interpolation between the subject and the population mean.
# Useful for visualizing patient-specific deviations (anomalies) vs. common anatomy,
# or for aggressive regularization/denoising by pulling latents toward the mean.
# Supports a global factor 't' (0.0=Mean, 1.0=Original) and granular per-level control
# (e.g., keeping original shape at L4 while normalizing texture at L0).
########################################
# echo "Performing latent space interpolation between subject and population mean..."
# for t_val in 0.60 0.90 ;
#   do
#     echo "Interpolating at t=${t_val}"
#     ${WHICH_PYTHON} lamnr_glow_tool.py recon-interpolate \
#       --ckpt ${ckpt} \
#       --gauss ${gaussian_lr} \
#       --manifest ${manifest_lesions} \
#       --views T1 \
#       --slice-axis 2 --slice-index ${SLICE_INDEX} \
#       --devices ${DEVICE} \
#       --t ${t_val} \
#       --out ${out_dir}/interpolation/interp_t${t_val}.png
#   done

${WHICH_PYTHON} lamnr_glow_tool.py calc-distance \
  --ckpt ${ckpt} \
  --gauss ${gaussian_lr} \
  --manifest ${manifest} \
  --devices ${DEVICE} \
  --views T1 \
  --slice-axis 2 --slice-index ${SLICE_INDEX} \
  --out ${dist_csv} \
  --save-levels  # ou --no-save-levels

########################################
# Performs 'Pseudo-Healthy Synthesis' via latent winsorization.
# Detects and clamps latent vectors that fall outside the learned distribution
# (e.g., tumors, lesions, or artifacts).
# Supports global thresholds (quantile or hard limit) and granular per-level
# control (e.g., aggressively clamping shape at L4 while preserving texture at L0).
########################################

# for quantile in 0.01 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95 0.99 0.999 1.00 ;
#   do
#     echo "Winsorizing at quantile: ${quantile}"
#     ${WHICH_PYTHON} lamnr_glow_tool.py recon-winsorize \
#       --ckpt ${ckpt} \
#       --manifest ${manifest_lesions} \
#       --views T1 \
#       --slice-axis 2 --slice-index ${SLICE_INDEX} \
#       --devices ${DEVICE} \
#       --out ${out_dir}/latent_winsorization/lesion_winsor_quantile_${quantile}.nii.gz \
#       --quantile ${quantile}

#   # for level in 0 1 2 3 4 ;
#   #   do
#   #     echo "Winsorizing at level: ${level}"
#   #     ${WHICH_PYTHON} lamnr_glow_tool.py recon-winsorize \
#   #       --ckpt ${ckpt} \
#   #       --manifest ${manifest_lesions} \
#   #       --views T1 \
#   #       --slice-axis 2 --slice-index ${SLICE_INDEX} \
#   #       --devices ${DEVICE} \
#   #       --out ${out_dir}/latent_winsorization/lesion_winsor_level_${level}_${quantile}.nii.gz \
#   #       --winsorize-level ${level},${quantile} 
#   #   done
#   done

# ${WHICH_PYTHON} lamnr_glow_tool.py recon \
#   --ckpt ${ckpt} \
#   --manifest ${manifest} --views T1,T2,FA --view-index 0 \
#   --slice-axis 2 --slice-index 64 --batch 6 --devices cuda:0 \
#   --gauss ${gaussian_lr} \
#   --edit-levels 0 \
#   --edit-what pc \
#   --edit-pc-index 0 \
#   --edit-pc-scale 2.0 \
#   --edit-pc-center sample \
#   --out ${out_dir}/recon_t1_pc0_k2_sample_0.png

# ${WHICH_PYTHON} lamnr_glow_tool.py recon \
#   --ckpt ${ckpt} \
#   --manifest ${manifest_lesions} --views T1 --view-index 0 \
#   --slice-axis 2 --slice-index 64 --batch 6 --devices cuda:0 \
#   --gauss ${gaussian_lr} \
#   --edit-levels 0 \
#   --edit-what pc \
#   --edit-pc-index 0 \
#   --edit-pc-scale 2.0 \
#   --edit-pc-center sample \
#   --out ${out_dir}/recon_t1_pc0_k2_sample_0.png



