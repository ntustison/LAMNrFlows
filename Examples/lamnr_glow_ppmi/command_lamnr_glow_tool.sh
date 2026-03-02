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
base_dir="/Users/ntustison/Desktop/lamnr_glow_ppmi"

# Image de référence cible (Template ANTs) utilisée pour l'interpolation et le calcul de distance.
ants_template="/Users/ntustison/Data/Public/PPMI/Template/PPMI_template0_256x256x256.nii.gz"
ppmi_example_image="/Users/ntustison/Data/Public/PPMI/PPMI_Dataset/sub-3781/ses-20140709/sub-3781_ses-20140709_r0001_ppmixt1.nii.gz"

# Options: "128x128", "256x256", etc. Ajuste la taille de l'image et la configuration du modèle.
which_experiment="256x256"  

# Destination pour toutes les images générées, les modèles Gaussiens et les résumés JSON.
out_dir="${base_dir}/output${which_experiment}/"

# Sélection dynamique du répertoire des résultats d'entraînement selon la résolution.
if [[ ${which_experiment} == "128x128" ]]; then
  runs_dir="${base_dir}/runs/ppmi_t1_fa_${which_experiment}_K12_L5_HC192/"
elif [[ ${which_experiment} == "256x256" ]]; then
  runs_dir="${base_dir}/runs/ppmi_t1_fa_${which_experiment}_K12_L6_HC192/"
else
  echo "Erreur : Configuration d'expérience '${which_experiment}' non reconnue."
  exit 1
fi

manifest_dir="${base_dir}/manifests/"

# Le checkpoint PyTorch spécifique contenant les poids du modèle.
ckpt="${runs_dir}/training_state.pt"

# Manifestes CSV : Ils lient les identifiants des sujets à leurs chemins d'images respectifs.
manifest="${manifest_dir}/manifest_ppmi.csv"
manifest_short="${manifest_dir}/manifest_ppmi_short.csv"
manifest_single="${manifest_dir}/manifest_ppmi_single.csv"
manifest_lesions="${manifest_dir}/manifest_brats_short.csv"  

# Paramètres numériques pour l'extraction des coupes 2D et le traitement.
SLICE_INDEX=138
WHICH_PYTHON="/Users/ntustison/anaconda3/bin/python3"
DEVICE="cpu"  # Options : 'cpu', 'cuda:0', 'mps', etc.

# --- Chemins Dérivés ---
# Modèle Gaussien sérialisé et son résumé statistique.
gaussian_lr="${out_dir}/t1_fa_lowrank.npz"
gaussian_lr_summary="${out_dir}/t1_fa_lowrank_summary.json"
dist_csv="${out_dir}/t1_distance_to_gaussian.csv"

###############################################################################
# DÉBUT DE L'EXÉCUTION DU PIPELINE
###############################################################################

###############################################################################
# 1. Ajustement du Modèle Gaussien (Gauss-Fit)
###############################################################################
# Estime la moyenne et la matrice de covariance de l'espace latent.
# Pour les images haute résolution (256x256), utilise obligatoirement
# l'estimateur "lowrank" (SVD) pour éviter de saturer la mémoire RAM.

if [[ ! -f ${gaussian_lr} ]]; then
  echo "Modèle Gaussien introuvable à ${gaussian_lr}. Ajustement en cours..."

  cov_mode="perlevel"
  cov_estimator="full" 

  if [[ ${which_experiment} == "128x128" ]]; then
    echo "Utilisation de l'estimateur de covariance complet (full) pour 128x128."
    cov_estimator="full"
  elif [[ ${which_experiment} == "256x256" ]]; then
    echo "Utilisation de l'estimateur de covariance de rang faible (lowrank) pour 256x256."
    cov_mode="perlevel"
    cov_estimator="lowrank"
  else
    echo "Erreur : Configuration '${which_experiment}' non reconnue pour le mode de covariance."
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
  echo "[1] Modèle Gaussien déjà existant. Étape ignorée."
fi

###############################################################################
# 2. Exportation des Coupes 2D (Slicing)
###############################################################################
# Utilitaire pour extraire des coupes 2D à partir des volumes NIfTI 3D.
# Prépare les données pour l'inspection visuelle en conservant la précision float.

manifest_input_dir="${out_dir}/manifest_input/"

if [[ ! -d ${manifest_input_dir} ]]; then
  echo "Exportation des coupes 2D depuis les volumes 3D..."
  mkdir -p "${manifest_input_dir}"

  ${WHICH_PYTHON} lamnr_glow_tool.py export-slices \
    --manifest ${manifest_short} \
    --slice-axis 2 --slice-index ${SLICE_INDEX} \
    --views T1,FA \
    --image-size ${which_experiment} \
    --outdir ${manifest_input_dir} \
    --output-format nii.gz
else
  echo "[2] Le répertoire d'exportation existe déjà. Étape ignorée."
fi 

###############################################################################
# 3a. Imputation Gaussienne (Translation de Modalité : FA -> T1)
###############################################################################
# Prédit une vue cible (ex: T1) à partir d'une donnée observée (ex: FA).
# Calcule la moyenne conditionnelle (MMSE) via l'identité de Woodbury.

impute_out_dir="${out_dir}/impute_T1_from_FA/"

if [[ -d ${impute_out_dir} && $(ls -A ${impute_out_dir}) ]]; then
  echo "[3a] Le répertoire d'imputation existe et n'est pas vide. Étape ignorée."
else 
  echo "Exécution de l'imputation cross-modale : Prédiction de T1 à partir de FA..."
  mkdir -p "${impute_out_dir}" 
  
  ${WHICH_PYTHON} lamnr_glow_tool.py gauss-impute \
    --ckpt ${ckpt} \
    --gauss ${gaussian_lr} \
    --manifest ${manifest_short} \
    --views T1,FA \
    --observed FA --target T1 \
    --slice-axis 2 --slice-index ${SLICE_INDEX} \
    --batch 2 --devices ${DEVICE} \
    --outdir "${impute_out_dir}" \
    --output-format nii.gz
fi

###############################################################################
# 3b. Imputation Gaussienne (Translation de Modalité : T1 -> FA)
###############################################################################
# Prédit une vue cible (ex: FA) à partir d'une donnée observée (ex: T1).
# Calcule la moyenne conditionnelle (MMSE) via l'identité de Woodbury.

impute_out_dir="${out_dir}/impute_FA_from_T1/"

if [[ -d ${impute_out_dir} && $(ls -A ${impute_out_dir}) ]]; then
  echo "[3b] Le répertoire d'imputation existe et n'est pas vide. Étape ignorée."
else 
  echo "Exécution de l'imputation cross-modale : Prédiction de T1 à partir de FA..."
  mkdir -p "${impute_out_dir}" 
  
  ${WHICH_PYTHON} lamnr_glow_tool.py gauss-impute \
    --ckpt ${ckpt} \
    --gauss ${gaussian_lr} \
    --manifest ${manifest_short} \
    --views T1,FA \
    --observed T1 --target FA \
    --slice-axis 2 --slice-index ${SLICE_INDEX} \
    --batch 2 --devices ${DEVICE} \
    --outdir "${impute_out_dir}" \
    --output-format nii.gz
fi


###############################################################################
# 4. Panneau de Vérification de Reconstruction (Sanity Check)
###############################################################################
# Visualise la précision de la bijection x <-> z.
# Génère une grille à 3 colonnes : [ Original (x) | Reconstruit (x_hat) | Différence Absolue ]

recon_panel_out="${out_dir}/recon_panel.png"

if [[ ! -f ${recon_panel_out} ]]; then
  echo "Génération du panneau de vérification de reconstruction..."
  ${WHICH_PYTHON} lamnr_glow_tool.py recon \
  --ckpt ${ckpt} \
  --manifest ${manifest_short} --views T1,FA --view-index 0 \
  --slice-axis 2 --slice-index ${SLICE_INDEX} --batch 6 --devices ${DEVICE} \
  --out ${recon_panel_out} 
else
  echo "[4] Le panneau de reconstruction existe déjà. Étape ignorée."
fi   

###############################################################################
# 5. Échantillonnage (Génération Stochastique)
###############################################################################
# Génère des images synthétiques à partir de la distribution latente.
# Le paramètre de température contrôle la diversité :
#   < 1.0 : Images proches de la moyenne (stables).
#   = 1.0 : Variance naturelle de l'ensemble d'entraînement.

which_modality="fa"  

# Correction de la syntaxe Bash pour les conditions
if [[ "${which_modality}" == "fa" ]]; then
  view_index=1
elif [[ "${which_modality}" == "t1" ]]; then
  view_index=0
else
  echo "Erreur : Modalité '${which_modality}' non reconnue. Utilisez 't1' ou 'fa'."
  exit 1
fi

grid_size="4x4" 

for temp in 0.01 0.25 0.50 0.75 1.00; do
  sample_output="${out_dir}/Samples/samples_${which_modality}_temp_${temp}.png" 
  
  if [[ -f ${sample_output} ]]; then
    echo "[5] Échantillons générés pour temp=${temp}. Ignoré."
    continue
  fi
  
  mkdir -p $(dirname "${sample_output}")
  echo "Échantillonnage à la température : ${temp}"
  
  ${WHICH_PYTHON} lamnr_glow_tool.py sample \
    --ckpt ${ckpt} \
    --view-index ${view_index} --sample-grid-size ${grid_size} \
    --image-size ${which_experiment} --temperature ${temp} \
    --devices ${DEVICE} --sample-grid-out "${sample_output}" \
    --seed $RANDOM
done 

###############################################################################
# 6. Reconstruction de Template (Atlas de Population)
###############################################################################
# Décode le vecteur latent moyen (mu) pour créer une anatomie "moyenne".
# L'utilisation de --mc-samples > 0 moyenne plusieurs échantillons pour créer
# une image haute-fidélité débruitée.

output_template="${out_dir}/template_T1_mu_sharpened.png"

if [[ ! -f ${output_template} ]]; then
  echo "Génération du template de population à partir de la moyenne Gaussienne..."
  ${WHICH_PYTHON} lamnr_glow_tool.py recon-template \
    --ckpt ${ckpt} \
    --gauss ${gaussian_lr} \
    --views T1 \
    --view-index 0 \
    --mc-samples 10 \
    --mc-temp 0.01 \
    --out "${output_template}" \
    --sharpen-image \
    --devices ${DEVICE} \
    --seed ${RANDOM}
else
  echo "[6] Le template de population existe déjà. Étape ignorée."
fi  

###############################################################################
# 7. Interpolation Latente (Morphing et Normalisation)
###############################################################################
# Crée une transition fluide dans l'espace latent.
# Utile pour visualiser l'écart d'un patient par rapport à la moyenne,
# ou pour forcer la régularisation d'une anatomie anormale.

echo "Interpolation dans l'espace latent (Sujet -> Moyenne de Population)..."
for t_val in 0.00 0.25 0.50 0.75 1.00; do
  output_interp="${out_dir}/interpolation/interp_mean_t${t_val}.nii.gz"
  
  if [[ -f ${output_interp} ]]; then
    echo "[7a] Image interpolée (moyenne) existante pour t=${t_val}. Ignoré."
    continue
  fi
  
  mkdir -p $(dirname "${output_interp}") 
  ${WHICH_PYTHON} lamnr_glow_tool.py recon-interpolate \
    --ckpt ${ckpt} \
    --gauss ${gaussian_lr} \
    --manifest ${manifest_single} \
    --views T1 \
    --slice-axis 2 --slice-index ${SLICE_INDEX} \
    --devices ${DEVICE} \
    --t ${t_val} \
    --interp-level 0,1.0 \
    --interp-level 1,1.0 \
    --out "${output_interp}"
done

echo "Interpolation dans l'espace latent (Sujet -> Image Cible Spécifique)..."
for t_val in 0.00 0.25 0.50 0.75 1.00; do
  # output_interp="${out_dir}/interpolation/interp_ants_template_t${t_val}.nii.gz"
  output_interp="${out_dir}/interpolation/interp_ppmi_example_t${t_val}.nii.gz"
  
  if [[ -f ${output_interp} ]]; then
    echo "[7b] Image interpolée (cible) existante pour t=${t_val}. Ignoré."
    continue
  fi
  
  mkdir -p $(dirname "${output_interp}")
  ${WHICH_PYTHON} lamnr_glow_tool.py recon-interpolate \
    --ckpt ${ckpt} \
    --gauss ${gaussian_lr} \
    --manifest ${manifest_single} \
    --views T1 \
    --slice-axis 2 --slice-index ${SLICE_INDEX} \
    --devices ${DEVICE} \
    --t ${t_val} \
    --out "${output_interp}" \
    --target-image "${ppmi_example_image}"
done

###############################################################################
# 8. Calcul des Distances Latentes (Détection d'Anomalies)
###############################################################################
# Calcule la distance Euclidienne entre les représentations latentes de 
# chaque sujet et l'image cible (ou la moyenne si non spécifiée).

if [[ ! -f ${dist_csv} ]]; then
  echo "Calcul des distances latentes par rapport à l'image cible..."
  ${WHICH_PYTHON} lamnr_glow_tool.py calc-distance \
    --ckpt ${ckpt} \
    --gauss ${gaussian_lr} \
    --manifest ${manifest_lesions} \
    --views T1 \
    --slice-axis 2 --slice-index ${SLICE_INDEX} \
    --out "${dist_csv}" \
    --target-image "${ants_template}" \
    --devices ${DEVICE} \
    --save-levels 
else
  echo "[8] Le fichier CSV de distances existe déjà. Étape ignorée."
fi

# ==============================================================================
# EXPÉRIENCE : MISE À L'ÉCHELLE PAR TEMPÉRATURE (TEMPERATURE SCALING)
# Réduit la variance des vecteurs latents en les multipliant par un scalaire tau.
# Contrairement au clipping (winsorisation), cela préserve la distribution Gaussienne.
# tau < 1.0 : Tire l'image vers la moyenne (lisse les anomalies).
# ==============================================================================

# 1. Test sur les Hautes Fréquences (Niveau 0 : Textures et bruit local)
# Un tau réduit ici lissera les micro-détails sans altérer la forme globale du cerveau.
for tau in 0.01 0.25 0.50 0.75 0.95 0.99; do
  echo "Processing Temperature Scaling L0 with tau=${tau}..."
  
  output_temp="${out_dir}/temperature/recon_temperature_L0_tau${tau}.nii.gz"
  
  if [[ ! -f ${output_temp} ]]; then
    mkdir -p $(dirname "${output_temp}")
    ${WHICH_PYTHON} lamnr_glow_tool.py recon-temperature \
      --ckpt ${ckpt} \
      --manifest ${manifest_lesions} \
      --views T1 \
      --slice-axis 2 --slice-index ${SLICE_INDEX} \
      --devices ${DEVICE} \
      --out "${output_temp}" \
      --tau-level 0,${tau}
  else 
     echo ${output_temp} déjà existante. Étape ignorée.
  fi 
  

done

# 2. Test sur les Basses Fréquences (Niveau 5 : Macro-structure et géométrie)
# Un tau réduit ici forcera la forme globale du cerveau à se rapprocher du 
# cerveau moyen de la population, tout en gardant les textures originales intactes.
for tau in 0.01 0.25 0.50 0.75 0.95 0.99; do
  echo "Processing Temperature Scaling L5 with tau=${tau}..."
  
  output_temp="${out_dir}/temperature/recon_temperature_L5_tau${tau}.nii.gz"
  if [[ ! -f ${output_temp} ]]; then
    mkdir -p $(dirname "${output_temp}")
    ${WHICH_PYTHON} lamnr_glow_tool.py recon-temperature \
      --ckpt ${ckpt} \
      --manifest ${manifest_lesions} \
      --views T1 \
      --slice-axis 2 --slice-index ${SLICE_INDEX} \
      --devices ${DEVICE} \
      --out "${output_temp}" \
      --tau-level 5,${tau}
   else
     echo ${output_temp} déjà existante. Étape ignorée.   
   fi  
done

# 3. Test Global (Tous les niveaux simultanément)
# Optionnel : Applique la réduction de variance à l'ensemble de l'espace latent.
for tau in 0.01 0.25 0.50 0.75 0.95 0.99; do
  echo "Processing Global Temperature Scaling with tau=${tau}..."
  
  output_temp="${out_dir}/temperature/recon_temperature_Global_tau${tau}.nii.gz"
  if [[ ! -f ${output_temp} ]]; then
    mkdir -p $(dirname "${output_temp}")
  
    ${WHICH_PYTHON} lamnr_glow_tool.py recon-temperature \
      --ckpt ${ckpt} \
      --manifest ${manifest_lesions} \
      --views T1 \
      --slice-axis 2 --slice-index ${SLICE_INDEX} \
      --devices ${DEVICE} \
      --out "${output_temp}" \
      --tau ${tau}
  else
    echo ${output_temp} déjà existante. Étape ignorée.
  fi

done