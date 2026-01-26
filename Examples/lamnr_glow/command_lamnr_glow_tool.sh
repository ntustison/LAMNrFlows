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

# The training results directory. Contains 'training_state.pt' and logs.
runs_dir="${base_dir}/runs/ppmi_t1_fa_128x128_K12_L5_HC192_align-vicreg_screen-cca_2/"

# Destination for all generated images, Gaussian models, and JSON summaries.
out_dir="${base_dir}/output/"

# The specific PyTorch checkpoint containing model weights (EMA/State Dict).
ckpt="${runs_dir}/training_state.pt"

# Manifest CSVs: These link subject IDs to their respective T1 and FA image paths.
manifest="${base_dir}/manifest_ppmi.csv"
manifest_short="${base_dir}/manifest_ppmi_short.csv"

# Numerical parameters for slice extraction and computational batching.
SLICE_INDEX=138
WHICH_PYTHON="/Users/ntustison/anaconda3/bin/python3"
DEVICE="cpu"  # Options: 'cpu', 'cuda:0', etc.

# --- Derived Paths ---
# Serialized Gaussian model and its statistical summary for latent-space math.
gaussian_lr=${out_dir}/t1_fa_lowrank.npz
gaussian_lr_summary=${out_dir}/t1_fa_lowrank_summary.json

###############################################################################
# PIPELINE EXECUTION START
###############################################################################


########################################
# Fit a Gaussian model (full or of low-rank covariance).
# Here we use the PPMI dataset which was used to train the LAMNr model.
# However, one could use any dataset of aligned multi-view images.
########################################

# ${WHICH_PYTHON} lamnr_glow_tool.py gauss-fit \
#   --ckpt ${ckpt} \
#   --manifest ${manifest} \
#   --views T1,FA \
#   --slice-axis 2 --slice-index ${SLICE_INDEX} \
#   --batch 64 --devices ${DEVICE} \
#   --cov-mode perlevel \
#   --cov-estimator full \
#   --gauss-out ${gaussian_lr} \
#   --gauss-summary ${gaussian_lr_summary}

########################################
# Utility to extract 2D slices from 3D NIfTI volumes.
# Prepares data for training or visual inspection at a specific axis and index.
# Supports NIfTI (.nii.gz) output to maintain floating-point precision.
########################################

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
#   --manifest ${manifest_short} --views T1,FA --view-index 1 \
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

${WHICH_PYTHON} lamnr_glow_tool.py recon-template \
  --ckpt ${ckpt} \
  --gauss ${gaussian_lr} \
  --views T1,FA \
  --view-index 0 \
  --mc-samples 10 \
  --mc-temp 0.01 \
  --devices ${DEVICE} \
  --out ${out_dir}/template_T1_mu_sharpened.png \
  --sharpen-image True \
  --seed ${RANDOM}


# ${WHICH_PYTHON} lamnr_glow_tool.py recon-template \
# --ckpt runs/t1_fa_128x128_vicreg_K12_H192_vicreg/training_state.pt \
# --gauss /home/ntustison/Desktop/deep_simr_glow/output/t1_fa_lowrank.npz \
# --views T1,T2,FA \
# --view-index 0 \
# --devices ${DEVICE} \
# --mc-samples 32 \
# --mc-temp 0.25 \
# --seed 12345 \
# --out ${out_dir}/template_T1_mu_mc.png











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



