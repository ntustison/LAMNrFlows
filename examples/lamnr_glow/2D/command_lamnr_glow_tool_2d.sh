#!/bin/bash
###############################################################################
# PIPELINE D'INFÉRENCE ET D'ANALYSE LAM-FLOW (2D)
# Projet : Modélisation Latente Multimodale (T1 / T2Flair / FA)
#
# Description :
#   Ce script exécute le flux de travail post-entraînement complet pour les 
#   modèles Normalizing Flow 2D. Il gère l'estimation de la distribution, 
#   la traduction entre modalités (imputation), la synthèse de templates, 
#   et la manipulation avancée de l'espace latent (interpolation, scaling).
###############################################################################

# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

# Répertoire racine du projet
base_dir="/Users/ntustison/Desktop/lamnr_glow_dlbs"

# Hyperparamètres de l'architecture entraînée
which_experiment="96x128"  
runs_dir="${base_dir}/runs2d/dlbs_t1_t2flair_fa_${which_experiment}_K12_L5_HC256_Round2"
ckpt="${runs_dir}/training_state.pt"

# Répertoires de sortie et manifestes
out_dir="${base_dir}/output${which_experiment}_vicreg/"
manifest_dir="${base_dir}/manifests/"

manifest_dlbs_wave1="${manifest_dir}/manifest_ds004856_wave1.csv"
manifest_dlbs_wave2="${manifest_dir}/manifest_ds004856_wave2.csv"
manifest_dlbs_wave3="${manifest_dir}/manifest_ds004856_wave3.csv"
manifest_dlbs_wave1_short="${manifest_dir}/manifest_ds004856_wave1_short.csv"
manifest_dlbs_wave2_short="${manifest_dir}/manifest_ds004856_wave2_short.csv"
manifest_dlbs_wave1_notshort="${manifest_dir}/manifest_ds004856_wave1_notshort.csv"
manifest_dlbs_wave2_notshort="${manifest_dir}/manifest_ds004856_wave2_notshort.csv"

manifest_nimh="${manifest_dir}/manifest_ds005752.csv"
manifest_nimh_short="${manifest_dir}/manifest_ds005752_short.csv"
manifest_nimh_notshort="${manifest_dir}/manifest_ds005752_notshort.csv"

manifest_brats="${manifest_dir}/manifest_brats.csv"
manifest_brats_short="${manifest_dir}/manifest_brats_short.csv"
manifest_brats_notshort="${manifest_dir}/manifest_brats_notshort.csv"

# Image de référence cible (Template ANTs) pour l'alignement et l'interpolation
ants_template="/Users/ntustison/Data/Public/OpenNeuro/ds004856/ANTsTemplate/T_templateT1.nii.gz"

# Image d'un sujet spécifique pour les tests d'interpolation vers la cible
example_image="/Users/ntustison/Data/Public/OpenNeuro/ds004856/BIDSAlignedToTemplate/sub-1022/ses-wave1/anat/sub-1022_ses-wave1_acq-MPRAGE_run-1_T1w.nii.gz"

# Images Pre-/Post- d'un sujet spécifique dans l'ensemble BRATsReg
brats_dir="/Users/ntustison/Data/Public/BRATS/RegistrationCompetition2022/Data/BraTSReg_Training_Data_v2_in_DLBS_space/BraTSReg_003/"
brats_t1_pre="${brats_dir}/BraTSReg_003_00_0000_t1.nii.gz"
brats_t2flair_pre="${brats_dir}/BraTSReg_003_00_0000_flair.nii.gz"
brats_t1_post="${brats_dir}/BraTSReg_003_01_0029_t1.nii.gz"
brats_t2flair_post="${brats_dir}/BraTSReg_003_01_0029_flair.nii.gz"

# Images jeune/vieux - de deux sujets spécifiques dans l'ensemble DLBS Wave2
# Sujet jeune : sub-612 (âge 25 ans, f)
# Sujet vieux : sub-1225 (âge 93 ans, f)
dlbs_dir="/Users/ntustison/Data/Public/OpenNeuro/ds004856/BIDSAlignedToTemplate/"
dlbs_wave2_subj1_t1="${dlbs_dir}/sub-612/ses-wave2/anat/sub-612_ses-wave2_acq-MPRAGE_run-1_T1w.nii.gz"
dlbs_wave2_subj1_t2flair="${dlbs_dir}/sub-612/ses-wave2/anat/sub-612_ses-wave2_acq-FLAIR_run-1_T2w.nii.gz"
dlbs_wave2_subj2_t1="${dlbs_dir}/sub-1225/ses-wave2/anat/sub-1225_ses-wave2_acq-MPRAGE_run-1_T1w.nii.gz"
dlbs_wave2_subj2_t2flair="${dlbs_dir}/sub-1225/ses-wave2/anat/sub-1225_ses-wave2_acq-FLAIR_run-1_T2w.nii.gz" 

# Paramètres d'exécution
SLICE_INDEX=115
WHICH_PYTHON="/Users/ntustison/anaconda3/bin/python3"
WHICH_LAMNR_TOOL="${base_dir}/lamnr_glow_tool_2d.py"
DEVICE="cpu"

# Chemins des modèles dérivés
cov_mode="perlevel"
cov_estimator="full" 

gaussian_dlbs_wave1_lr="${out_dir}/t1_t2flair_fa_dlbs_wave1_${cov_estimator}.npz"
gaussian_dlbs_wave1_lr_summary="${out_dir}/t1_t2flair_fa_dlbs_wave1_${cov_estimator}_summary.json"

echo "========================================================"
echo " Démarrage du pipeline LAM-Flow 2D (${which_experiment})"
echo "========================================================"

###############################################################################
# 1. AJUSTEMENT DU MODÈLE GAUSSIEN (Gauss-Fit)
# Objectif : Estimer la moyenne multivariée et la matrice de covariance de 
# l'espace latent (prior). L'estimateur "lowrank" (SVD) est utilisé pour 
# approximer la covariance sans saturer la RAM. Indispensable pour l'imputation.
###############################################################################

if [[ ! -f ${gaussian_dlbs_wave1_lr} ]]; then
  echo "[1] Ajustement de la distribution Gaussienne..."

  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} gauss-fit \
    --ckpt ${ckpt} \
    --manifest ${manifest_dlbs_wave1_notshort} \
    --views T1,T2Flair,FA \
    --slice-axis 2 --slice-index ${SLICE_INDEX} \
    --batch 64 --devices ${DEVICE} \
    --cov-mode ${cov_mode} \
    --cov-estimator ${cov_estimator} \
    --rank 256 \
    --gauss-out ${gaussian_dlbs_wave1_lr} \
    --gauss-summary ${gaussian_dlbs_wave1_lr_summary}

    # --aug-params "sd_deformation:constant:0.1" \
    # --aug-epochs 10 \


else
  echo "[1] Modèle Gaussien déjà existant. Étape ignorée."
fi

###############################################################################
# 2. EXPORTATION DES COUPES 2D (Slicing)
# Objectif : Extraire la coupe axiale définie par SLICE_INDEX depuis les volumes 
# NIfTI 3D. Prépare les tenseurs exacts vus par le modèle pour inspection visuelle.
###############################################################################

# manifest_input_dir="${out_dir}/manifest_input/"

# if [[ ! -d ${manifest_input_dir} ]]; then
#   echo "[2] Exportation des coupes 2D depuis les volumes 3D..."
#   mkdir -p "${manifest_input_dir}"

#   ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} export-slices \
#     --manifest ${manifest_dlbs_wave1_short} \
#     --slice-axis 2 --slice-index ${SLICE_INDEX} \
#     --views T1,T2Flair,FA \
#     --image-size ${which_experiment} \
#     --outdir ${manifest_input_dir} \
#     --output-format nii.gz
# else
#   echo "[2] Le répertoire d'exportation existe déjà. Étape ignorée."
# fi 

###############################################################################
# 3. IMPUTATION GAUSSIENNE (Traduction Cross-Modale)
# Objectif : Prédire une modalité manquante (ex: T1) à partir de modalités 
# observées (ex: T2Flair, FA). Le modèle utilise l'identité de Woodbury pour calculer 
# la moyenne conditionnelle exacte (MMSE) dans l'espace latent.
###############################################################################

# 3A. T1 --> T2Flair,FA
impute_out_dir="${out_dir}/dlbs_wave2_impute_T2FlairFA_from_T1/"
if [[ -d ${impute_out_dir} && $(ls -A ${impute_out_dir}) ]]; then
  echo "[3a] Imputation T1 -> T2Flair, FA existante. Étape ignorée."
else 
  echo "[3a] Imputation cross-modale : Prédiction de T2Flair,FA à partir de T1..."
  mkdir -p "${impute_out_dir}" 
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} gauss-impute \
    --ckpt ${ckpt} --gauss ${gaussian_dlbs_wave1_lr} --manifest ${manifest_dlbs_wave2_short} \
    --views T1,T2Flair,FA --observed T1 --target T2Flair,FA \
    --slice-axis 2 --slice-index ${SLICE_INDEX} \
    --batch 2 --devices ${DEVICE} --outdir "${impute_out_dir}" --output-format png
  mogrify -rotate 90 ${impute_out_dir}/*.png  
fi

# 3B. T1 -> FA
impute_out_dir="${out_dir}/dlbs_wave2_impute_FA_from_T1/"
if [[ -d ${impute_out_dir} && $(ls -A ${impute_out_dir}) ]]; then
  echo "[3a] Imputation T1 -> FA existante. Étape ignorée."
else 
  echo "[3a] Imputation cross-modale : Prédiction de FA à partir de T1..."
  mkdir -p "${impute_out_dir}" 
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} gauss-impute \
    --ckpt ${ckpt} --gauss ${gaussian_dlbs_wave1_lr} --manifest ${manifest_dlbs_wave2_short} \
    --views T1,T2Flair,FA --observed T1 --target FA \
    --slice-axis 2 --slice-index ${SLICE_INDEX} \
    --batch 2 --devices ${DEVICE} --outdir "${impute_out_dir}" --output-format png
  mogrify -rotate 90 ${impute_out_dir}/*.png  
fi

# 3C. FA -> T1, T2Flair
impute_out_dir="${out_dir}/dlbs_wave2_impute_T1T2Flair_from_FA/"
if [[ -d ${impute_out_dir} && $(ls -A ${impute_out_dir}) ]]; then
  echo "[3b] Imputation FA -> T1,T2Flair existante. Étape ignorée."
else 
  echo "[3b] Imputation cross-modale : Prédiction de T1, T2Flair à partir de FA..."
  mkdir -p "${impute_out_dir}" 
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} gauss-impute \
    --ckpt ${ckpt} --gauss ${gaussian_dlbs_wave1_lr} --manifest ${manifest_dlbs_wave2_short} \
    --views T1,T2Flair,FA --observed FA --target T1,T2Flair \
    --slice-axis 2 --slice-index ${SLICE_INDEX} \
    --batch 2 --devices ${DEVICE} --outdir "${impute_out_dir}" --output-format png
  mogrify -rotate 90 ${impute_out_dir}/*.png  
fi

# 3D. T2Flair, FA -> T1
impute_out_dir="${out_dir}/dlbs_wave2_impute_T1_from_T2FlairFA/"
if [[ -d ${impute_out_dir} && $(ls -A ${impute_out_dir}) ]]; then
  echo "[3c] Imputation T2Flair,FA -> T1 existante. Étape ignorée."
else 
  echo "[3c] Imputation cross-modale : Prédiction de T1 à partir de T2Flair et FA..."
  mkdir -p "${impute_out_dir}" 
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} gauss-impute \
    --ckpt ${ckpt} --gauss ${gaussian_dlbs_wave1_lr} --manifest ${manifest_dlbs_wave2_short} \
    --views T1,T2Flair,FA --observed T2Flair,FA --target T1 \
    --slice-axis 2 --slice-index ${SLICE_INDEX} \
    --batch 2 --devices ${DEVICE} --outdir "${impute_out_dir}" --output-format png
  mogrify -rotate 90 ${impute_out_dir}/*.png    
fi

# 3D. T1 (BRATS) --> T2Flair. 

impute_out_dir="${out_dir}/brats_impute_T2Flair_from_T1/"
if [[ -d ${impute_out_dir} && $(ls -A ${impute_out_dir}) ]]; then
  echo "[3d] Imputation T2Flair -> T1 existante. Étape ignorée."
else 
  echo "[3d] Imputation cross-modale : Prédiction de T1 à partir de T2Flair..."
  mkdir -p "${impute_out_dir}" 
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} gauss-impute \
    --ckpt ${ckpt} --gauss ${gaussian_dlbs_wave1_lr} --manifest ${manifest_brats_short} \
    --views T1,T2Flair,FA --observed T1 --target T2Flair \
    --slice-axis 2 --slice-index ${SLICE_INDEX} \
    --batch 2 --devices ${DEVICE} --outdir "${impute_out_dir}" --output-format png
  mogrify -rotate 90 ${impute_out_dir}/*.png  
fi

# 3E. T1 (NIMH) --> FA.  

impute_out_dir="${out_dir}/nimh_impute_T2Flair_from_T1/"
if [[ -d ${impute_out_dir} && $(ls -A ${impute_out_dir}) ]]; then
  echo "[3e] Imputation T1-> FA existante. Étape ignorée."
else 
  echo "[3e] Imputation cross-modale : Prédiction de T2Flair à partir de FA..."
  mkdir -p "${impute_out_dir}" 
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} gauss-impute \
    --ckpt ${ckpt} --gauss ${gaussian_dlbs_wave1_lr} --manifest ${manifest_nimh_short} \
    --views T1,T2Flair --observed T1 --target T2Flair \
    --slice-axis 2 --slice-index ${SLICE_INDEX} \
    --batch 2 --devices ${DEVICE} --outdir "${impute_out_dir}" --output-format png
  mogrify -rotate 90 ${impute_out_dir}/*.png

fi

###############################################################################
# 4. VÉRIFICATION DE RECONSTRUCTION (Sanity Check)
# Objectif : Confirmer l'inversibilité parfaite du flux (x <-> z).
# Génère une grille comparative : [Original | Reconstruit | Erreur Absolue].
###############################################################################

recon_panel_out="${out_dir}/recon_panel.png"
if [[ ! -f ${recon_panel_out} ]]; then
  echo "[4] Génération du panneau de vérification de reconstruction..."
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} recon \
    --ckpt ${ckpt} --manifest ${manifest_dlbs_wave1_short} --views T1,T2Flair,FA --view-index 0 \
    --slice-axis 2 --slice-index ${SLICE_INDEX} --batch 6 --devices ${DEVICE} \
    --out ${recon_panel_out} 
else
  echo "[4] Le panneau de reconstruction existe déjà. Étape ignorée."
fi   

###############################################################################
# 5. ÉCHANTILLONNAGE STOCHASTIQUE (Génération)
# Objectif : Tirer des vecteurs aléatoires depuis la distribution normale et
# les décoder. La température (variance) module la netteté et la diversité.
###############################################################################

grid_size="5x4" 

for which_modality in fa t2flair t1; do
   if [[ "${which_modality}" == "fa" ]]; then view_index=2
   elif [[ "${which_modality}" == "t2flair" ]]; then view_index=1
   elif [[ "${which_modality}" == "t1" ]]; then view_index=0
   fi

   for temp in 0.10 0.25 0.50 0.75 1.00; do
     sample_output="${out_dir}/Samples/samples_${which_modality}_temp_${temp}.png" 
     
     if [[ -f ${sample_output} ]]; then
       continue
     fi
     
     mkdir -p $(dirname "${sample_output}")
     echo "[5] Échantillonnage (${which_modality}) à température = ${temp}..."
     
     ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} sample \
       --ckpt ${ckpt} --view-index ${view_index} --sample-grid-size ${grid_size} \
       --image-size ${which_experiment} --temperature ${temp} \
       --devices ${DEVICE} --sample-grid-out "${sample_output}" --seed $RANDOM

     convert ${sample_output} -rotate 90 ${sample_output}
     rm -f ${sample_output/png/json}
   done 
done

###############################################################################
# 6a. RECONSTRUCTION DE TEMPLATE (Atlas de Population)
# Objectif : Décoder le vecteur latent moyen (mu). L'échantillonnage de 
# Monte-Carlo (--mc-samples) ajoute une micro-variance moyennée pour obtenir 
# un atlas extrêmement net et dépourvu de bruit haute fréquence.
###############################################################################

output_template="${out_dir}/template_T1_mu_sharpened.nii.gz"
if [[ ! -f ${output_template} ]]; then
  echo "[6a] Génération du template de population (Moyenne Latente)..."
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} recon-template \
    --ckpt ${ckpt} --gauss ${gaussian_dlbs_wave1_lr} --views T1,T2Flair,FA --view-index 0 \
    --mc-samples 0 --mc-temp 0.01 --out "${output_template}" \
    --sharpen-image --devices ${DEVICE} --seed ${RANDOM}
  ConvertImagePixelType "${output_template}" "${out_dir}/L_templateT1.png" 3
else
  echo "[6a] Le template T1 de population existe déjà. Étape ignorée."
fi  

output_template="${out_dir}/template_T2Flair_mu_sharpened.nii.gz"
if [[ ! -f ${output_template} ]]; then
  echo "[6a] Génération du template de population (Moyenne Latente)..."
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} recon-template \
    --ckpt ${ckpt} --gauss ${gaussian_dlbs_wave1_lr} --views T1,T2Flair,FA --view-index 1 \
    --mc-samples 0 --mc-temp 0.01 --out "${output_template}" \
    --sharpen-image --devices ${DEVICE} --seed ${RANDOM}
  ConvertImagePixelType "${output_template}" "${out_dir}/L_templateT2Flair.png" 3
else
  echo "[6a] Le template T2Flair de population existe déjà. Étape ignorée."
fi  

output_template="${out_dir}/template_FA_mu_sharpened.nii.gz"
if [[ ! -f ${output_template} ]]; then
  echo "[6a] Génération du template de population (Moyenne Latente)..."
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} recon-template \
    --ckpt ${ckpt} --gauss ${gaussian_dlbs_wave1_lr} --views T1,T2Flair,FA --view-index 2 \
    --mc-samples 0 --mc-temp 0.01 --out "${output_template}" \
    --sharpen-image --devices ${DEVICE} --seed ${RANDOM}
  ConvertImagePixelType "${output_template}" "${out_dir}/L_templateFA.png" 3
else
  echo "[6a] Le template FA de population existe déjà. Étape ignorée."
fi  

###############################################################################
# 6b. RECONSTRUCTION DE TEMPLATE COHORTE (Atlas de Sous-Population)
# Objectif : Générer un template représentatif d'un sous-groupe clinique 
# spécifique (défini par le fichier manifest). Calcule la moyenne euclidienne 
# (barycentre) des vecteurs latents de cette cohorte exacte, puis la décode. 
# Ne nécessite pas de modèle Gaussien global.
###############################################################################

output_template_cohort="${out_dir}/template_T1_cohort.nii.gz"

if [[ ! -f ${output_template_cohort} ]]; then
  echo "[6b] Génération du template de cohorte (Moyenne Empirique)..."
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} recon-cohort-template \
    --ckpt ${ckpt} \
    --manifest ${manifest_dlbs_wave1_short} \
    --views T1,T2Flair,FA \
    --view-index 0 \
    --image-size ${which_experiment} \
    --slice-axis 2 \
    --slice-index ${SLICE_INDEX} \
    --out "${output_template_cohort}" \
    --sharpen-image \
    --devices ${DEVICE}
else
  echo "[6b] Le template de cohorte existe déjà. Étape ignorée."
fi

###############################################################################
# 7. INTERPOLATION LATENTE (Morphing Géodésique)
# Objectif : Calculer une trajectoire linéaire dans l'espace latent entre 
# deux cerveaux, produisant un morphing non-linéaire (anatomiquement continu) 
# dans l'espace de l'image.
###############################################################################

# # 7A. Trajectoire : Sujet -> Cerveau Moyen
# echo "[7A] Interpolation (Sujet -> Moyenne de Population)..."
# for t_val in 0.00 0.25 0.50 0.75 1.00; do
#   output_interp="${out_dir}/interpolation/interp_mean_t${t_val}.nii.gz"
#   if [[ -f ${output_interp} ]]; then continue; fi
  
#   mkdir -p $(dirname "${output_interp}") 
#   ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} recon-interpolate \
#     --ckpt ${ckpt} --gauss ${gaussian_dlbs_wave1_lr} --source-image ${example_image} \
#     --views T1 --slice-axis 2 --slice-index ${SLICE_INDEX} --devices ${DEVICE} \
#     --t ${t_val} --interp-level 0,1.0 --interp-level 1,1.0 --out "${output_interp}"
# done

# # 7B. Trajectoire : Sujet -> Template ANTs
# echo "[7B] Interpolation (Sujet -> Image Cible Spécifique)..."
# for t_val in 0.00 0.25 0.50 0.75 1.00; do
#   output_interp="${out_dir}/interpolation/inter_dlbs_example_t${t_val}.nii.gz"
#   if [[ -f ${output_interp} ]]; then continue; fi
  
#   mkdir -p $(dirname "${output_interp}")
#   ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} recon-interpolate \
#     --ckpt ${ckpt} --gauss ${gaussian_dlbs_wave1_lr} --source-image ${ants_template} \
#     --target-image ${example_image} --views T1 \
#     --slice-axis 2 --slice-index ${SLICE_INDEX} --devices ${DEVICE} \
#     --t ${t_val} --out "${output_interp}" 
# done

# 7C. Trajectoire : BRATs Pre -> BRATs Post
echo "[7C] T1 Interpolation (Sujet -> Image BRATs) ..."
for t_val in 0.00 0.25 0.50 0.75 1.00; do
  output_interp="${out_dir}/interpolation/inter_brats_example_t1_t${t_val}.nii.gz"
  if [[ -f ${output_interp/.nii.gz/png} ]]; then continue; fi
  
  mkdir -p $(dirname "${output_interp}")
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} recon-interpolate \
    --ckpt ${ckpt} --gauss ${gaussian_dlbs_wave1_lr} --source-image ${brats_t1_pre} \
    --target-image ${brats_t1_post} --views T1 \
    --slice-axis 2 --slice-index ${SLICE_INDEX} --devices ${DEVICE} \
    --t ${t_val} --out "${output_interp}" 
  ConvertImagePixelType "${output_interp}" ${output_interp/nii.gz/png} 3  
  rm -f "${output_interp}"
done

# 7D. Trajectoire : BRATs Pre -> BRATs Post
echo "[7D] T2Flair Interpolation (Sujet -> Image BRATs) ..."
for t_val in 0.00 0.25 0.50 0.75 1.00; do
  output_interp="${out_dir}/interpolation/inter_brats_example_t2flair_t${t_val}.nii.gz"
  if [[ -f ${output_interp/.nii.gz/png} ]]; then continue; fi
  
  mkdir -p $(dirname "${output_interp}")
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} recon-interpolate \
    --ckpt ${ckpt} --gauss ${gaussian_dlbs_wave1_lr} --source-image ${brats_t2flair_pre} \
    --target-image ${brats_t2flair_post} --views T2Flair \
    --slice-axis 2 --slice-index ${SLICE_INDEX} --devices ${DEVICE} \
    --t ${t_val} --out "${output_interp}" 
  ConvertImagePixelType "${output_interp}" ${output_interp/nii.gz/png} 3
  rm -f "${output_interp}"
done

# 7E. Trajectoire : DLBS Wave 2 jeune -> DLBS Wave 2 vieux (T1)
echo "[7E] T1 Interpolation (Jeune -> Vieux) ..."
for t_val in 0.00 0.25 0.50 0.75 1.00; do
  output_interp="${out_dir}/interpolation/intra_dlbs_wave2_example_t1_t${t_val}.nii.gz"
  if [[ -f ${output_interp/.nii.gz/png} ]]; then continue; fi
  
  mkdir -p $(dirname "${output_interp}")
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} recon-interpolate \
    --ckpt ${ckpt} --gauss ${gaussian_dlbs_wave1_lr} --source-image ${dlbs_wave2_subj1_t1} \
    --target-image ${dlbs_wave2_subj2_t1} --views T1 \
    --slice-axis 2 --slice-index ${SLICE_INDEX} --devices ${DEVICE} \
    --t ${t_val} --out "${output_interp}" 
  ConvertImagePixelType "${output_interp}" ${output_interp/nii.gz/png} 3  
  rm -f "${output_interp}"
done

# 7F. Trajectoire : DLBS Wave 2 jeune -> DLBS Wave 2 vieux (T2Flair)
echo "[7F] T2Flair Interpolation (Jeune -> Vieux) ..."
for t_val in 0.00 0.25 0.50 0.75 1.00; do
  output_interp="${out_dir}/interpolation/intra_dlbs_wave2_example_t2flair_t${t_val}.nii.gz"
  if [[ -f ${output_interp/.nii.gz/png} ]]; then continue; fi
  
  mkdir -p $(dirname "${output_interp}")
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} recon-interpolate \
    --ckpt ${ckpt} --gauss ${gaussian_dlbs_wave1_lr} --source-image ${dlbs_wave2_subj1_t2flair} \
    --target-image ${dlbs_wave2_subj2_t2flair} --views T2Flair \
    --slice-axis 2 --slice-index ${SLICE_INDEX} --devices ${DEVICE} \
    --t ${t_val} --out "${output_interp}" 
  ConvertImagePixelType "${output_interp}" ${output_interp/nii.gz/png} 3  
  rm -f "${output_interp}"
done

###############################################################################
# 8. DISTANCES LATENTES (Détection d'Anomalies)
# Objectif : Calculer la distance géodésique ou euclidienne de chaque sujet par 
# rapport au centre de la distribution. Isole les sujets atypiques (outliers).
###############################################################################

dist_csv="${out_dir}/distances_dlbs_wave2_to_antsx_template.csv"

if [[ ! -f ${dist_csv} ]]; then
  echo "[8] Calcul des distances latentes par rapport au modèle Gaussien..."
  ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} calc-distance \
    --ckpt ${ckpt} --gauss ${gaussian_dlbs_wave1_lr} --manifest ${manifest_dlbs_wave2} \
    --target-image ${ants_template} \
    --views T1 --slice-axis 2 --slice-index ${SLICE_INDEX} \
    --out "${dist_csv}" --distance-metric geodesic --devices ${DEVICE} --save-levels 
else
  echo "[8] Le fichier CSV de distances existe déjà. Étape ignorée."
fi

###############################################################################
# 9. MISE À L'ÉCHELLE PAR TEMPÉRATURE (Temperature Scaling des Lésions)
# Objectif : Contracter l'espace latent (tau < 1.0) pour forcer une image 
# pathologique à se rapprocher du manifold sain. L'application par niveau 
# permet de cibler des bandes de fréquences spécifiques (L0 = micro-structures, 
# L5 = macro-géométrie globale).
###############################################################################

# 9A. Scaling L0 (Micro-structures)
for tau in 0.01 0.25 0.50 0.75 0.95 0.99; do
  output_temp="${out_dir}/temperature/recon_temperature_L0_tau${tau}.nii.gz"
  if [[ ! -f ${output_temp} ]]; then
    echo "[9A] Temperature Scaling (Niveau 0) avec tau=${tau}..."
    mkdir -p $(dirname "${output_temp}")
    ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} recon-temperature \
      --ckpt ${ckpt} --manifest ${manifest_brats_short} --views T1 \
      --slice-axis 2 --slice-index ${SLICE_INDEX} --devices ${DEVICE} \
      --out "${output_temp}" --tau-level 0,${tau}
  fi 
done

# 9B. Scaling L5 (Macro-structure globale)
for tau in 0.01 0.25 0.50 0.75 0.95 0.99; do
  output_temp="${out_dir}/temperature/recon_temperature_L5_tau${tau}.nii.gz"
  if [[ ! -f ${output_temp} ]]; then
    echo "[9B] Temperature Scaling (Niveau 5) avec tau=${tau}..."
    mkdir -p $(dirname "${output_temp}")
    ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} recon-temperature \
      --ckpt ${ckpt} --manifest ${manifest_brats_short} --views T1 \
      --slice-axis 2 --slice-index ${SLICE_INDEX} --devices ${DEVICE} \
      --out "${output_temp}" --tau-level 5,${tau}
   fi  
done

# 9C. Scaling Global (Tous les niveaux)
for tau in 0.01 0.25 0.50 0.75 0.95 0.99; do
  output_temp="${out_dir}/temperature/recon_temperature_Global_tau${tau}.nii.gz"
  if [[ ! -f ${output_temp} ]]; then
    echo "[9C] Temperature Scaling Global avec tau=${tau}..."
    mkdir -p $(dirname "${output_temp}")
    ${WHICH_PYTHON} ${WHICH_LAMNR_TOOL} recon-temperature \
      --ckpt ${ckpt} --manifest ${manifest_dlbs_wave1_short} --views T1 \
      --slice-axis 2 --slice-index ${SLICE_INDEX} --devices ${DEVICE} \
      --out "${output_temp}" --tau ${tau}
  fi
done

echo "========================================================"
echo " Pipeline 2D terminé."
echo "========================================================"