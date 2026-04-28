#!/bin/bash
###############################################################################
# Pipeline d'Inférence et d'Analyse LAM-Flow (2D)
# Projet : Modélisation Latente T1/FA (PPMI)
#
# Ce script gère le flux de travail post-entraînement pour les modèles Glow 2D,
# incluant l'ajustement de la distribution Gaussienne (prior), l'imputation 
# cross-modale, et la reconstruction de templates à l'échelle de la population.
###############################################################################

# --- Configuration Globale ---

# Le répertoire de base hébergeant votre projet de recherche.
base_dir="/Users/ntustison/Desktop/lamnr_glow_dlbs"
volume_size="48x64x56"  # Dimensions de l'image d'entrée (DxHxW)

data_dir="/Users/ntustison/Data/Public/OpenNeuro/ds004169"
manifest_dir="${data_dir}/manifests/"
manifest="${manifest_dir}/manifest_t1_short_brain.csv"
manifest_gaussian="/Users/ntustison/Desktop/lamnr_glow_dlbs/manifests/manifest_ds004856_wave1.csv"

out_dir="${data_dir}/output_brain_3d/"

# Le checkpoint PyTorch spécifique contenant les poids du modèle.
ckpt="${out_dir}/training_state.pt"
gaussian_lr="${out_dir}/t1_lowrank.npz"

WHICH_PYTHON="/Users/ntustison/anaconda3/bin/python3"
WHICH_LAMNR_TOOL=/Users/ntustison/Data/LAMNrFlows/src/lamnrflows/lamnr_glow_tool_3d.py
DEVICE="cpu"  # Options : 'cpu', 'cuda:0', 'mps', etc.

###############################################################################
# DÉBUT DE L'EXÉCUTION DU PIPELINE
###############################################################################

###############################################################################
# 1. Ajustement du Modèle Gaussien (Gauss-Fit)
###############################################################################
# Estime la moyenne (mu) et la covariance (Sigma) des représentations latentes.
# En 3D, l'estimateur 'lowrank' est forcé pour éviter la saturation mémoire (OOM).

if [[ ! -f ${gaussian_lr} ]]; then
  echo "Modèle Gaussien introuvable. Ajustement en cours..."

  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} gauss-fit \
    --ckpt ${ckpt} \
    --manifest ${manifest_gaussian} \
    --views T1 \
    --volume-size ${volume_size} \
    --batch 32 \
    --devices ${DEVICE} \
    --cov-estimator lowrank \
    --rank 256 \
    --gauss-out ${gaussian_lr}
else
  echo "[1] Modèle Gaussien déjà existant : ${gaussian_lr}. Étape ignorée."
fi


###############################################################################
# 8. Calcul des Distances Latentes (Détection d'Anomalies)
###############################################################################
# Calcule la distance Euclidienne entre les représentations latentes de 
# chaque sujet et l'image cible (ou la moyenne si non spécifiée).

dist_csv="${out_dir}/distances_pairwise_matrix_brain.csv"

if [[ ! -f "${dist_csv}" ]]; then
  echo "Calcul de la matrice complète des distances géodésiques..."
  
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} calc-distance \
    --ckpt ${ckpt} \
    --gauss ${gaussian_lr} \
    --manifest ${manifest} \
    --views T1 \
    --out "${dist_csv}" \
    --devices ${DEVICE} \
    --workers 8 \
    --batch 2 \
    --distance-metric geodesic \
    --variance-epsilon 1e-7 \
    --pairwise 
    
else
  echo "La matrice de distances ${dist_csv} existe déjà. Étape ignorée."
fi

