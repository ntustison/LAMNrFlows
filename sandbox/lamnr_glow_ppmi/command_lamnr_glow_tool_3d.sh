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
base_dir="/Users/ntustison/Desktop/lamnr_glow_ppmi"
which_experiment="48x48x48"  

# Destination pour toutes les images générées et modèles Gaussiens
out_dir="${base_dir}/output${which_experiment}/"

# Répertoire des résultats d'entraînement contenant 'training_state.pt'
if [[ ${which_experiment} == "48x48x48" ]]; then
  runs_dir="${base_dir}/runs3d/hcp_t1_fa_${which_experiment}_K32_L3_H64/"
else
  echo "Erreur : Configuration d'expérience '${which_experiment}' non reconnue."
  exit 1
fi

manifest_dir="${base_dir}/manifests/"
ckpt="${runs_dir}/training_state.pt"

# Manifestes CSV liant les identifiants sujets à leurs chemins d'images (T1, FA)
manifest="${manifest_dir}/manifest_ppmi.csv"
manifest_short="${manifest_dir}/manifest_ppmi_short.csv"
manifest_lesions="${manifest_dir}/manifest_brats_short.csv"  

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
    --views T1,FA \
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
    --temperature 0.25 \
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
    --sharpen-image \
    --out-dir ${template_out_dir} \
    --devices ${DEVICE}
else
  echo "[4] Le répertoire de templates existe déjà. Étape ignorée."
fi

###############################################################################
# 5. Imputation Gaussienne (T1 -> FA)
###############################################################################
# Prédit le volume FA manquant conditionnellement au volume T1 observé
# en utilisant l'identité matricielle de Woodbury dans l'espace latent.

impute_out_dir="${out_dir}/imputed_FA/"

if [[ -d ${impute_out_dir} && $(ls -A ${impute_out_dir}) ]]; then
  echo "[5] Le répertoire d'imputation existe déjà et n'est pas vide. Étape ignorée."
else
  echo "Exécution de l'imputation conditionnelle (T1 -> FA)..."
  mkdir -p "${impute_out_dir}"
  
  ${WHICH_PYTHON} lamnr_glow_tool_3d.py gauss-impute \
    --ckpt ${ckpt} \
    --gauss ${gaussian_lr} \
    --manifest ${manifest_short} \
    --views T1,FA \
    --observed T1 \
    --target FA \
    --volume-size ${which_experiment} \
    --out-dir ${impute_out_dir} \
    --devices ${DEVICE}
fi    

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
    --manifest ${manifest} \
    --views T1 \
    --view-index 0 \
    --volume-size ${which_experiment} \
    --out-csv ${dist_csv} \
    --devices ${DEVICE}
else
  echo "[6] Le fichier CSV de distances existe déjà. Étape ignorée."
fi

###############################################################################
# 7. Winsorisation (Atténuation des Lésions/Artefacts)
###############################################################################
# Restreint les valeurs extrêmes dans l'espace latent (hard-threshold) pour 
# projeter l'image pathologique vers une distribution anatomique saine.

winsorize_out_dir="${out_dir}/winsorized_lesions/"

if [[ ! -d ${winsorize_out_dir} ]]; then
  echo "Exécution de la winsorisation sur le jeu de données pathologique..."
  mkdir -p "${winsorize_out_dir}"

  ${WHICH_PYTHON} lamnr_glow_tool_3d.py recon-winsorize \
    --ckpt ${ckpt} \
    --manifest ${manifest_lesions} \
    --views T1 \
    --view-index 0 \
    --volume-size ${which_experiment} \
    --batch 5 \
    --hard-threshold 3.0 \
    --winsorize-level 0,0.95 \
    --out-dir ${winsorize_out_dir} \
    --devices ${DEVICE}
else
  echo "[7] Le répertoire de winsorisation existe déjà. Étape ignorée."
fi

###############################################################################
# 8. Interpolation Latente 3D (Transfert de Style / Morphing)
###############################################################################
# Crée une transition séquentielle de volumes dans l'espace latent entre une 
# distribution source (ex: cerveau avec lésion) et une cible (ex: cerveau sain).

interp_out_dir="${out_dir}/interpolation_brats_to_ppmi/"

if [[ ! -d ${interp_out_dir} ]]; then
  echo "Génération de l'interpolation latente 3D (Source -> Cible)..."
  mkdir -p "${interp_out_dir}"
  
  target_vol=$(tail -n +2 "${manifest}" | awk -F',' '{print $1}' | grep -v "^$" | head -n 1)

  if [[ -f "${target_vol}" ]]; then
    ${WHICH_PYTHON} lamnr_glow_tool_3d.py recon-interpolate \
      --ckpt ${ckpt} \
      --gauss ${gaussian_lr} \
      --manifest ${manifest_lesions} \
      --target-image ${target_vol} \
      --views T1 \
      --view-index 0 \
      --volume-size ${which_experiment} \
      --batch 1 \
      --steps 5 \
      --out-dir ${interp_out_dir} \
      --devices ${DEVICE}
  else
    echo "Erreur : Fichier cible introuvable. Vérifiez ${manifest}."
  fi
else
  echo "[8] Le répertoire d'interpolation existe déjà. Étape ignorée."
fi