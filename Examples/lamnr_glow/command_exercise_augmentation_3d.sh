

/Users/ntustison/miniconda3/bin/python3 download_hcp_data.py

# super-strong early augmentation
/Users/ntustison/miniconda3/bin/python3 exercise_augmentation_3d.py \
  --view ~/.antstorch/hcp*T1Template.nii.gz \
  --view ~/.antstorch/hcp*T2Template.nii.gz \
  --view ~/.antstorch/hcp*FATemplate.nii.gz \
  --H 64 --W 64 --D 64 \
  --batch 4 \
  --n-per-view 16 \
  --step 0 \
  --out-dir aug_step000000

# mid-training augmentation
/Users/ntustison/miniconda3/bin/python3 exercise_augmentation_3d.py \
  --view ~/.antstorch/hcp*T1Template.nii.gz \
  --view ~/.antstorch/hcp*T2Template.nii.gz \
  --view ~/.antstorch/hcp*FATemplate.nii.gz \
  --H 64 --W 64 --D 64 \
  --batch 4 \
  --n-per-view 16 \
  --step 60000 \
  --out-dir aug_step060000

# late / almost-off augmentation
/Users/ntustison/miniconda3/bin/python3 exercise_augmentation_3d.py \
  --view ~/.antstorch/hcp*T1Template.nii.gz \
  --view ~/.antstorch/hcp*T2Template.nii.gz \
  --view ~/.antstorch/hcp*FATemplate.nii.gz \
  --H 64 --W 64 --D 64 \
  --batch 4 \
  --n-per-view 16 \
  --step 120000 \
  --out-dir aug_step120000
