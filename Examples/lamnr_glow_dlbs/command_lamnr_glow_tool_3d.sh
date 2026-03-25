#!/bin/bash
###############################################################################
# LAM-Flow Inference & Analysis Pipeline
# Project: PPMI T1/FA Latent Modeling
#
# Ce script gère le flux de travail post-entraînement pour les modèles Glow 3D,
# incluant l'ajustement de la distribution Gaussienne (prior), l'imputation 
# cross-modale, et la reconstruction de templates à l'échelle de la population.
###############################################################################

# --- Configuration Globale ---

# Répertoire de base du projet
base_dir="/Users/ntustison/Desktop/lamnr_glow_dlbs"
which_experiment="64x80x64"  
runs_dir="${base_dir}/runs3d/dlbs_t1_${which_experiment}_K32_L4_H64/"

# Destination pour toutes les images générées et modèles Gaussiens
out_dir="${base_dir}/output${which_experiment}/"

manifest_dir="${base_dir}/manifests/"
ckpt="${runs_dir}/training_state.pt"

# Manifestes CSV liant les identifiants sujets à leurs chemins d'images (T1, FA)
manifest="${manifest_dir}/manifest.csv"
manifest_short="${manifest_dir}/manifest_short.csv"
manifest_lesions="${manifest_dir}/manifest_brats_short.csv"  

ants_template="/Users/ntustison/Data/Public/OpenNeuro/ds004856/Template/nki_x.nii.gz"

WHICH_PYTHON="/Users/ntustison/anaconda3/bin/python3"

# MPS (Metal Performance Shaders) est l'API d'accélération matérielle d'Apple.
# Le backend 'mps' exploite les cœurs graphiques des puces Apple Silicon (M1/M2/M3).
# 'cpu' est utilisé ici pour garantir la compatibilité et éviter les limitations de VRAM.
DEVICE="cpu"

# --- Chemins Dérivés ---
# Modèle Gaussien sérialisé pour les mathématiques de l'espace latent
gaussian_lr="${out_dir}/t1_fa_lowrank.npz"
dist_csv="${out_dir}/t1_distance_to_gaussian.csv"

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

  ${WHICH_PYTHON} lamnr_glow_tool_3d.py gauss-fit \
    --ckpt ${ckpt} \
    --manifest ${manifest} \
    --views T1 \
    --volume-size ${which_experiment} \
    --batch 1 \
    --devices ${DEVICE} \
    --cov-estimator lowrank \
    --rank 256 \
    --gauss-out ${gaussian_lr}
else
  echo "[1] Modèle Gaussien déjà existant : ${gaussian_lr}. Étape ignorée."
fi

###############################################################################
# 2. Reconstruction 3D (Sanity Check : x -> z -> x_hat)
###############################################################################
# Vérifie la bijection du modèle : encodage vers l'espace latent puis décodage.

recon_out_dir="${out_dir}/reconstructions/"

if [[ ! -d ${recon_out_dir} ]]; then
  echo "Exécution de la reconstruction 3D (Sanity Check)..."
  mkdir -p "${recon_out_dir}"

  ${WHICH_PYTHON} lamnr_glow_tool_3d.py recon \
    --ckpt ${ckpt} \
    --gauss ${gaussian_lr} \
    --manifest ${manifest_short} \
    --views T1 \
    --view-index 0 \
    --volume-size ${which_experiment} \
    --batch 1 \
    --out-dir "${recon_out_dir}" \
    --devices ${DEVICE}
else
  echo "[2] Le répertoire de reconstruction existe déjà. Étape ignorée."
fi

###############################################################################
# 3. Échantillonnage (Génération de volumes 3D synthétiques)
###############################################################################
# Tire des échantillons stochastiques depuis la distribution Gaussienne (prior)
# et les décode via le flux inverse pour générer de nouvelles anatomies.

sample_out_dir="${out_dir}/generated_samples/"

if [[ -d ${sample_out_dir} && $(ls -A ${sample_out_dir}) ]]; then
  echo "[3] Le répertoire d'échantillons existe déjà et n'est pas vide. Étape ignorée."
else 
  echo "Génération de nouveaux échantillons 3D (Sampling)..."
  mkdir -p "${sample_out_dir}"

  ${WHICH_PYTHON} lamnr_glow_tool_3d.py sample \
    --ckpt ${ckpt} \
    --view-index 0 \
    --n-samples 5 \
    --volume-size ${which_experiment} \
    --temperature 0.10 \
    --out-dir "${sample_out_dir}" \
    --devices ${DEVICE}
fi

###############################################################################
# 4. Reconstruction de Templates (Moyenne de Population)
###############################################################################
# Génère le volume 3D correspondant au vecteur latent moyen (mu).
# Applique un lissage et un filtre Laplacien pour la netteté (--sharpen-image).

template_out_dir="${out_dir}/templates/"

if [[ ! -d ${template_out_dir} ]]; then
  echo "Génération du template de population 3D..."
  mkdir -p "${template_out_dir}"

  ${WHICH_PYTHON} lamnr_glow_tool_3d.py recon-template \
    --ckpt ${ckpt} \
    --gauss ${gaussian_lr} \
    --manifest ${manifest} \
    --views T1 \
    --view-index 0 \
    --volume-size ${which_experiment} \
    --mc-samples 10 \
    --mc-temp 0.1 \
    --sharpen-image \
    --out-dir ${template_out_dir} \
    --devices ${DEVICE}

  ${WHICH_PYTHON}  lamnr_glow_tool_3d.py recon-cohort-template \
    --ckpt ${ckpt} \
    --manifest ${manifest} \
    --views T1 \
    --volume-size ${which_experiment} \
    --view-index 0 \
    --out ${template_out_dir}/template_cohort.nii.gz \
    --sharpen-image

else
  echo "[4] Le répertoire de templates existe déjà. Étape ignorée."
fi




###############################################################################
# 5. Imputation Gaussienne (T1 -> FA)
###############################################################################
# Prédit le volume FA manquant conditionnellement au volume T1 observé
# en utilisant l'identité matricielle de Woodbury dans l'espace latent.

# impute_out_dir="${out_dir}/imputed_FA/"

# if [[ -d ${impute_out_dir} && $(ls -A ${impute_out_dir}) ]]; then
#   echo "[5] Le répertoire d'imputation existe déjà et n'est pas vide. Étape ignorée."
# else
#   echo "Exécution de l'imputation conditionnelle (T1 -> FA)..."
#   mkdir -p "${impute_out_dir}"
  
#   ${WHICH_PYTHON} lamnr_glow_tool_3d.py gauss-impute \
#     --ckpt ${ckpt} \
#     --gauss ${gaussian_lr} \
#     --manifest ${manifest_short} \
#     --views T1,FA \
#     --observed T1 \
#     --target FA \
#     --volume-size ${which_experiment} \
#     --out-dir ${impute_out_dir} \
#     --devices ${DEVICE}
# fi    

###############################################################################
# 6. Calcul de Distance Latente (Détection d'Anomalies / OOD)
###############################################################################
# Évalue la distance Euclidienne (L2) entre les latents de chaque volume T1 
# et la moyenne Gaussienne de la cohorte d'entraînement.

if [[ ! -f ${dist_csv} ]]; then
  echo "Calcul de la distance à la moyenne Gaussienne pour la vue T1..."  
  
  ${WHICH_PYTHON} lamnr_glow_tool_3d.py calc-distance \
    --ckpt ${ckpt} \
    --gauss ${gaussian_lr} \
    --manifest ${manifest_short} \
    --target-image ${ants_template} \
    --views T1 \
    --view-index 0 \
    --volume-size ${which_experiment} \
    --distance-metric geodesic \
    --variance-epsilon 0.000001 \
    --out-csv ${dist_csv} \
    --devices ${DEVICE}
else
  echo "[6] Le fichier CSV de distances existe déjà. Étape ignorée."
fi


###############################################################################
# 7. Mise à l'échelle par Température (Atténuation des Lésions/Artefacts)
###############################################################################
# Contracte l'espace latent vers la moyenne de la distribution (tau < 1.0) pour 
# projeter l'image pathologique vers une anatomie saine (pseudo-saine).

temperature_out_dir="${out_dir}/scaled_lesions/"

if [[ ! -d ${temperature_out_dir} ]]; then
  echo "Exécution du lissage par température sur le jeu de données pathologique..."
  mkdir -p "${temperature_out_dir}"

  ${WHICH_PYTHON} lamnr_glow_tool_3d.py recon-temperature \
    --ckpt ${ckpt} \
    --manifest ${manifest_lesions} \
    --views T1 \
    --view-index 0 \
    --volume-size ${which_experiment} \
    --tau 0.75 \
    --tau-level "0,0.95" \
    --out-dir ${temperature_out_dir} \
    --devices ${DEVICE}
else
  echo "[7] Le répertoire de mise à l'échelle existe déjà. Étape ignorée."
fi

###############################################################################
# 8. Interpolation Latente 3D (Transfert de Style / Morphing)
###############################################################################
# Crée une transition séquentielle de volumes dans l'espace latent entre une 
# distribution source (ex: cerveau avec lésion) et une cible (ex: cerveau sain).

interp_out_dir="${out_dir}/interpolation_brats"
interp_out=${interp_out_dir}/interp

if [[ ! -d ${interp_out_dir} ]]; then
  echo "Génération de l'interpolation latente 3D (Source -> Cible)..."
  mkdir -p "${interp_out_dir}"
  
  target_image=/Users/ntustison/Data/Public/BRATS/RegistrationCompetition2022/Data/BraTSReg_Training_Data_v2_in_DLBS_space/BraTSReg_003/BraTSReg_003_01_0029_t1.nii.gz
  source_image=/Users/ntustison/Data/Public/BRATS/RegistrationCompetition2022/Data/BraTSReg_Training_Data_v2_in_DLBS_space/BraTSReg_003/BraTSReg_003_00_0000_t1.nii.gz
  
  for t in 0.00 0.25 0.50 0.75 1.0; 
    do
      ${WHICH_PYTHON} lamnr_glow_tool_3d.py recon-interpolate \
        --ckpt ${ckpt} \
        --gauss ${gaussian_lr} \
        --source-image ${source_image} \
        --target-image ${target_image} \
        --views T1 \
        --view-index 0 \
        --volume-size ${which_experiment} \
        --batch 1 \
        --t $t \
        --interp-type nlerp \
        --out ${interp_out}_${t}.nii.gz \
        --devices ${DEVICE}
    done
else
  echo "[8] Le répertoire d'interpolation existe déjà. Étape ignorée."
fi