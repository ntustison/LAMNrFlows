

# Sample a specified view with a given temperature
# Render in a grid dictated by --grid-size
python lam_flow_tool.py \
  --ckpt runs/t1_t2_fa_128x128_vicreg_K12_H192_vicreg/training_state.pt \
  --view-index 0 \
  --grid-size 3x3 \
  --image-size 128x128 \
  --temperature 0.95 \
  --sample-grid-out /home/ntustison/Desktop/samples_view0.nii.gz

# Sample a set of files (n=args.recon), push them forward, and
# backward through the network and render as a nx3 mosaic where
# the first column is the original image, the second column is
# the reconstructed image and the third column is the difference.
python lam_flow_tool.py \
  --ckpt runs/t1_t2_fa_128x128_vicreg_K12_H192_vicreg/training_state.pt \
  --view-index 0 \
  --recon 4 \
  --slice-index 64 \
  --slice-axis 2 \
  --val-list '~/Data/NormalizingFlows/Nifti/*/T1.nii.gz' \
  --recon-out /home/ntustison/Desktop/recon_panel.png  