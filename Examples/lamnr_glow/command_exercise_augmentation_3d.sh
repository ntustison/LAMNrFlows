python3 exercise_augmentation_3d.py \
  --view ~/Data/HCPTemplates/*/T_template0.nii.gz \
  --view ~/Data/HCPTemplates/*/T_template1.nii.gz \
  --view ~/Data/HCPTemplates/*/T_template2.nii.gz \
  --H 64 --W 64 --D 64 \
  --batch 4 \
  --n-per-view 32 \
  --out-dir debug_aug_3d
