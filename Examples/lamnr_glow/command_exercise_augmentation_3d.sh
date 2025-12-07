

# super-strong early augmentation
python exercise_augmentation_3d.py \
  --view ~/Data/HCPTemplates/*/T_template0.nii.gz \
  --view ~/Data/HCPTemplates/*/T_template1.nii.gz \
  --view ~/Data/HCPTemplates/*/T_template2.nii.gz \
  --H 64 --W 64 --D 64 \
  --batch 4 \
  --n-per-view 16 \
  --step 0 \
  --out-dir aug_step000000

# mid-training augmentation
python exercise_augmentation_3d.py \
  --view ~/Data/HCPTemplates/*/T_template0.nii.gz \
  --view ~/Data/HCPTemplates/*/T_template1.nii.gz \
  --view ~/Data/HCPTemplates/*/T_template2.nii.gz \
  --H 64 --W 64 --D 64 \
  --batch 4 \
  --n-per-view 16 \
  --step 60000 \
  --out-dir aug_step060000

# late / almost-off augmentation
python exercise_augmentation_3d.py \
  --view ~/Data/HCPTemplates/*/T_template0.nii.gz \
  --view ~/Data/HCPTemplates/*/T_template1.nii.gz \
  --view ~/Data/HCPTemplates/*/T_template2.nii.gz \
  --H 64 --W 64 --D 64 \
  --batch 4 \
  --n-per-view 16 \
  --step 120000 \
  --out-dir aug_step120000
