python lam_flow_tool.py \
  --ckpt runs/t1_t2_fa_256x256_vicreg/training_state.pt \
  --view-index 0 \
  --grid-size 3x3 \
  --image-size 256x256 \
  --temperature 0.95 \
  --interp 0 \
  --out /home/ntustison/Desktop/samples_view0.png


python lam_flow_tool.py \
  --ckpt runs/t1_t2_fa_256x256_vicreg/training_state.pt \
  --view-index 0 \
  --recon 8 \
  --val-list ~/Data/NormalizingFlows/Nifti/*/T1.nii.gz \
  --recon-out recon_panel.png  