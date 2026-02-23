#!/bin/bash
###############################################################################
# LAM-Flow Inference & Analysis Pipeline
# Project: PPMI T1/FA Latent Modeling
# Date: 2026-01-26
#
# This script manages the post-training workflow for 3D Glow models,
# including Gaussian distribution fitting, cross-modal imputation, 
# and population-level template reconstruction.
###############################################################################

# --- Global Configuration ---

# The base directory where your research project is housed.
base_dir="/Users/ntustison/Desktop/lamnr_glow_ppmi"

which_experiment="48x48x48"  

# Destination for all generated images, Gaussian models, and JSON summaries.
out_dir="${base_dir}/output${which_experiment}/"

# The training results directory. Contains 'training_state.pt' and logs.
if [[ ${which_experiment} == "48x48x48" ]]; then
  runs_dir="${base_dir}/runs3d/hcp_t1_fa_${which_experiment}_K32_L3_H64/"
else
  echo "Error: Unrecognized experiment configuration '${which_experiment}'. Please set 'which_experiment' to a valid option."
  exit 1
fi

manifest_dir="${base_dir}/manifests/"

# The specific PyTorch checkpoint containing model weights (EMA/State Dict).
ckpt="${runs_dir}/training_state.pt"

# Manifest CSVs: These link subject IDs to their respective T1 and FA image paths.
manifest="${manifest_dir}/manifest_ppmi_100.csv"
manifest_short="${manifest_dir}/manifest_ppmi_short.csv"
manifest_lesions="${manifest_dir}/manifest_brats_short.csv"  

# Numerical parameters for slice extraction and computational batching.
WHICH_PYTHON="/Users/ntustison/miniconda3/bin/python3"

# MPS (Metal Performance Shaders) est l'API d'accélération matérielle développée 
# par Apple. Dans l'écosystème PyTorch, le backend mps est l'équivalent du backend 
# cuda de NVIDIA, mais il est conçu spécifiquement pour exploiter les cœurs 
# graphiques des puces Apple Silicon (séries M1, M2, M3, etc.).
DEVICE="mps"  # Options: 'cpu', 'cuda:0', etc.

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

  # Forçage de l'estimateur lowrank pour la gestion mémoire en 3D
  cov_estimator="lowrank"

  echo "Fitting Gaussian model to 3D latent representations using ${cov_estimator}..."

  ${WHICH_PYTHON} lamnr_glow_tool_3d.py gauss-fit \
    --ckpt ${ckpt} \
    --manifest ${manifest} \
    --views T1,FA \
    --volume-size ${which_experiment} \
    --batch 1 \
    --devices ${DEVICE} \
    --cov-estimator ${cov_estimator} \
    --rank 256 \
    --gauss-out ${gaussian_lr}

else
  echo "Gaussian model already exists at ${gaussian_lr}. Skipping fitting step."
fi

###############################################################################
# 6. Reconstruction 3D (Sanity Check : x -> z -> x_hat)
###############################################################################
# Vérifie la capacité du modèle à encoder et décoder fidèlement une image réelle.

recon_out_dir="${out_dir}/reconstructions/"

if [[ ! -d ${recon_out_dir} ]]; then
  echo "Exécution de la reconstruction 3D (Sanity Check)..."
  mkdir -p "${recon_out_dir}"

  # Extraction du premier chemin T1 valide depuis le manifeste court
  test_vol=$(tail -n +2 "${manifest_short}" | awk -F',' '{print $1}' | grep -v "^$" | head -n 1)

  if [[ -f "${test_vol}" ]]; then
    filename=$(basename "${test_vol}")
    out_file="${recon_out_dir}/recon_${filename}"

    echo "  Reconstruction de : ${filename}"

    ${WHICH_PYTHON} lamnr_glow_tool_3d.py recon \
      --ckpt ${ckpt} \
      --gauss ${gaussian_lr} \
      --manifest ${manifest_short} \
      --views T1 \
      --volume-size ${which_experiment} \
      --out "${out_file}" \
      --devices cpu
      
    echo "Reconstruction terminée. Fichier sauvegardé dans : ${out_file}"
  else
    echo "Erreur : Fichier d'entrée introuvable. Vérifiez le manifeste."
  fi
else
  echo "Le répertoire de reconstruction existe déjà : ${recon_out_dir}. Ignoré."
fi

###############################################################################
# 5. Échantillonnage (Génération de volumes 3D synthétiques)
###############################################################################
# Tire des échantillons aléatoires depuis la distribution normale (espace latent)
# et les décode pour créer de nouvelles images 3D anatomiques.

sample_out_dir="${out_dir}/generated_samples/"

echo "Génération de nouveaux échantillons 3D (Sampling)..."
mkdir -p "${sample_out_dir}"

# --view-index 0 correspond généralement à la modalité T1 selon votre ordre d'entraînement
# --temperature 0.8 réduit légèrement la variance pour des échantillons plus nets

if [[ -d ${sample_out_dir} && $(ls -A ${sample_out_dir}) ]]; then
  echo "Le répertoire d'échantillons existe déjà et n'est pas vide : ${sample_out_dir}. Ignoré."
else 
  ${WHICH_PYTHON} lamnr_glow_tool_3d.py sample \
    --ckpt ${ckpt} \
    --view-index 0 \
    --n-samples 5 \
    --volume-size ${which_experiment} \
    --temperature 0.25 \
    --out-dir "${sample_out_dir}" \
    --devices cpu
  echo "Échantillonnage terminé. Fichiers sauvegardés dans : ${sample_out_dir}"
fi

###############################################################################
# 2. Imputation Gaussienne (T1 -> FA)
###############################################################################
# Prédit la modalité FA manquante à partir de la modalité T1 observée.
# Utilise manifest_short pour accélérer les tests initiaux.

impute_out_dir="${out_dir}/imputed_FA/"

echo "Exécution de l'imputation conditionnelle (T1 -> FA)..."

if [[ -d ${impute_out_dir} && $(ls -A ${impute_out_dir}) ]]; then
  echo "Le répertoire d'imputation existe déjà et n'est pas vide : ${impute_out_dir}. Ignoré."
else
  ${WHICH_PYTHON} lamnr_glow_tool_3d.py gauss-impute \
    --ckpt ${ckpt} \
    --gauss ${gaussian_lr} \
    --manifest ${manifest_short} \
    --views T1,FA \
    --observed T1 \
    --target FA \
    --volume-size ${which_experiment} \
    --out-dir ${impute_out_dir} \
    --devices cpu
  echo "Imputation terminée. Fichiers sauvegardés dans : ${impute_out_dir}"
fi    

###############################################################################
# 1. Calcul de Distance Latente (Détection d'anomalies / OOD)
###############################################################################
# Évalue la distance entre les latents de chaque volume T1 et la moyenne de la population.

if [[ ! -f ${dist_csv} ]]; then
  echo "Calcul de la distance à la moyenne Gaussienne pour la vue T1..."  
  ${WHICH_PYTHON} lamnr_glow_tool_3d.py calc-distance \
    --ckpt ${ckpt} \
    --gauss ${gaussian_lr} \
    --manifest ${manifest} \
    --views T1 \
    --view-index 0 \
    --volume-size ${which_experiment} \
    --out-csv ${dist_csv} \
    --devices cpu
else
  echo "Le fichier CSV de distances existe déjà : ${dist_csv}. Ignoré."
fi

###############################################################################
# 3. Winsorisation (Suppression des Lésions/Artefacts)
###############################################################################
# Restreint les valeurs extrêmes dans l'espace latent (quantiles) pour 
# "guérir" ou atténuer les anomalies structurelles (ex: tumeurs de BraTS).

winsorize_out_dir="${out_dir}/winsorized_lesions/"

if [[ ! -d ${winsorize_out_dir} ]]; then
  echo "Exécution de la winsorisation sur le jeu de données des lésions..."
  mkdir -p "${winsorize_out_dir}"

  # Lecture du manifeste en ignorant l'en-tête. 
  # On suppose que la modalité T1 est la première colonne du CSV.
  tail -n +2 "${manifest_lesions}" | while IFS=',' read -r t1_path fa_path_or_other; do
    
    # Ignorer les lignes vides
    if [[ -z "${t1_path}" ]]; then continue; fi
    
    # Extraction du nom de fichier pour créer le chemin de sortie
    filename=$(basename "${t1_path}")
    out_path="${winsorize_out_dir}/winsorized_${filename}"

    echo "  Traitement de : ${filename}"

    ${WHICH_PYTHON} lamnr_glow_tool_3d.py recon-winsorize \
      --ckpt ${ckpt} \
      --manifest ${manifest_lesions} \
      --views T1 \
      --view-index 0 \
      --volume-size ${which_experiment} \
      --batch 2 \
      --hard-threshold 3.0 \
      --winsorize-level 0,0.95 \
      --out-dir ${winsorize_out_dir} \
      --devices cpu

  done
  
  echo "Winsorisation terminée. Résultats sauvegardés dans : ${winsorize_out_dir}"

else
  echo "Le répertoire de winsorisation existe déjà : ${winsorize_out_dir}. Ignoré."
fi

###############################################################################
# 4. Interpolation Latente 3D (Morphing / Transfert de Style)
###############################################################################
# Crée une transition fluide dans l'espace latent entre un volume source
# (sujet avec lésion) et un volume cible (sujet sain PPMI).

# interp_out_dir="${out_dir}/interpolation_brats_to_ppmi/"

# if [[ ! -d ${interp_out_dir} ]]; then
#   echo "Génération de l'interpolation latente 3D (Source -> Cible)..."
  
#   # Extraction du premier chemin T1 valide depuis les manifestes (en ignorant l'en-tête)
#   source_vol=$(tail -n +2 "${manifest_lesions}" | awk -F',' '{print $1}' | grep -v "^$" | head -n 1)
#   target_vol=$(tail -n +2 "${manifest}" | awk -F',' '{print $1}' | grep -v "^$" | head -n 1)

#   if [[ -f "${source_vol}" && -f "${target_vol}" ]]; then
#     echo "  Source : ${source_vol}"
#     echo "  Cible  : ${target_vol}"

#     ${WHICH_PYTHON} lamnr_glow_tool_3d.py recon-interpolate \
#       --ckpt ${ckpt} \
#       --source "${source_vol}" \
#       --target "${target_vol}" \
#       --out-dir ${interp_out_dir} \
#       --steps 5 \
#       --volume-size ${which_experiment} \
#       --devices ${DEVICE}
      
#     echo "Interpolation terminée. Fichiers NIfTI intermédiaires sauvegardés dans : ${interp_out_dir}"
#   else
#     echo "Erreur : Fichiers source ou cible introuvables. Vérifiez les chemins dans vos manifestes."
#   fi
# else
#   echo "Le répertoire d'interpolation existe déjà : ${interp_out_dir}. Ignoré."
# fi