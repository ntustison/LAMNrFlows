
runs_dir="runs/t1_t2_fa_128x128_vicreg_K12_H192_vicreg/"
out_dir="/home/ntustison/Desktop/deep_simr_glow/output/"
ckpt="${runs_dir}/training_state.pt"
manifest=/home/ntustison/Desktop/deep_simr_glow/manifest_t1_t2_fa.csv

gaussian_lr=${out_dir}/t1_t2_fa_lowrank.npz
gaussian_lr_summary=${out_dir}/t1_t2_fa_lowrank_summary.json

# OAS, per-level
python lam_flow_tool.py gauss-fit \
  --ckpt ${ckpt} \
  --manifest ${manifest} \
  --views T1,T2,FA \
  --slice-axis 2 --slice-index 64 \
  --batch 64 --devices cuda:0 \
  --cov-mode perlevel \
  --cov-estimator lowrank --rank 256 --sigma2 auto --cov-lam 1e-3 \
  --gauss-out ${gaussian_lr} \
  --gauss-summary ${gaussian_lr_summary}

python lam_flow_tool.py gauss-impute \
  --ckpt ${ckpt} \
  --gauss ${gaussian_lr} \
  --manifest ${manifest} \
  --views T1,T2,FA \
  --observed T1,T2 --target FA \
  --slice-axis 2 --slice-index 64 \
  --batch 64 --devices cuda:0 \
  --strategy mean \
  --outdir ${out_dir}/impute_FAm_from_T1T2



# Sample grid (kept)
python lam_flow_tool.py sample \
  --ckpt ${ckpt} \
  --view-index 0 --sample-grid-size 6x8 \
  --image-size 128x128 --temperature 0.8 \
  --devices cuda:0 --sample-grid-out ${out_dir}/samples_t1.png

# Reconstruction sanity panel (new subcommand)
python lam_flow_tool.py recon \
  --ckpt ${ckpt} \
  --manifest ${manifest} --views T1,T2,FA --view-index 0 \
  --slice-axis 2 --slice-index 64 --batch 6 --devices cuda:0 \
  --out ${out_dir}/recon_t1_panel.png









# Low-rank + isotropic σ²
python lam_flow_tool_fixed.py gauss-fit \
  --ckpt runs/t1_t2_fa_128x128_vicreg/training_state.pt \
  --manifest /data/lam/manifest.csv \
  --views T1,T2,FA \
  --slice-axis 2 --slice-index 64 \
  --cov-mode perlevel \
  --cov-estimator lowrank --rank 64 --sigma2 auto --cov-lam 1e-6 \
  --gauss-out /data/lam/models/t1t2fa_gauss_perlevel_lr64.npz


# Sample from Σ_{U|O} with safety clamping
python lam_flow_tool_fixed.py gauss-impute \
  --ckpt runs/t1_t2_fa_128x128_vicreg/training_state.pt \
  --gauss /data/lam/models/t1t2fa_gauss_perlevel_lr64.npz \
  --manifest /data/lam/manifest.csv \
  --views T1,T2,FA \
  --observed T1 --target T2 \
  --slice-axis 2 --slice-index 64 \
  --strategy sample --samples 1 --temperature 1.0 \
  --safe-latent clamp --safe-k 2.0 \
  --seed 1234 \
  --outdir /data/lam/impute_T2_from_T1





# Impute with sampling + safety clamp
python lam_flow_tool_fixed.py gauss-impute \
  --ckpt runs/t1_t2_fa_128x128_vicreg \
  --gauss gaussian/t1_t2_fa_lowrank.pt \
  --manifest data/paths.csv --views T1,T2,FA \
  --observed T1,T2 --target FA \
  --slice-axis 2 --slice-index 64 --batch 64 \
  --strategy sample --temperature 1.0 \
  --safe-latent clamp --safe-k 2.0 \
  --outdir out_impute/fa_from_t1t2_sampled

# Sweep multiple pairs (observed,target[,subdir])
python lam_flow_tool_fixed.py gauss-impute \
  --ckpt runs/t1_t2_fa_128x128_vicreg \
  --gauss gaussian/t1_t2_fa_lowrank.pt \
  --manifest data/paths.csv --views T1,T2,FA \
  --slice-axis 2 --slice-index 64 \
  --pairs-csv pairs.csv \
  --outdir out_impute














# Sample a specified view with a given temperature
# Render in a grid dictated by --grid-size
python lam_flow_tool.py sample \
  --ckpt runs/t1_t2_fa_128x128_vicreg_K12_H192_vicreg/training_state.pt \
  --view-index 0 \
  --sample-grid-size 3x3 \
  --image-size 128x128 \
  --temperature 0.95 \
  --sample-grid-out /home/ntustison/Desktop/samples_view0.png

# Sample a set of files (n=args.recon), push them forward, and
# backward through the network and render as a nx3 mosaic where
# the first column is the original image, the second column is
# the reconstructed image and the third column is the difference.
python lam_flow_tool.py sample \
  --ckpt runs/t1_t2_fa_128x128_vicreg_K12_H192_vicreg/training_state.pt \
  --view-index 0 \
  --recon 4 \
  --slice-index 64 \
  --slice-axis 2 \
  --val-list '~/Data/NormalizingFlows/Nifti/*/T1.nii.gz' \
  --recon-out /home/ntustison/Desktop/recon_panel.png  


python lam_flow_tool.py gauss-fit \
--ckpt runs/t1_t2_fa_128x128_vicreg_K12_H192_vicreg/training_state.pt \
--manifest /home/ntustison/Desktop/deep_simr_glow/manifest_t1_t2_fa.csv \
--views T1,T2,FA \
--slice-axis 2 --slice-index 64 \
--batch 64 --devices cuda:0 \
--cov-mode perlevel --cov-estimator full --shrinkage 1e-6 --cov-lam 0.00 \
--gauss-out /home/ntustison/Desktop/t1t2fa_gauss_perlevel.pt \
--gauss-summary /home/ntustison/Desktop/t1t2fa_gauss_perlevel.json \
--save-fp 64

python lam_flow_tool.py gauss-impute \
  --ckpt runs/t1_t2_fa_128x128_vicreg_K12_H192_vicreg/training_state.pt \
  --gauss /home/ntustison/Desktop/t1t2fa_gauss_perlevel.pt \
  --manifest /home/ntustison/Desktop/deep_simr_glow/manifest_t1_t2_fa.csv \
  --views T1,T2,FA \
  --observed T1,T2 \
  --target FA \
  --slice-axis 2 --slice-index 64 \
  --devices cuda:0 --batch 64 \
  --strategy mean \
  --outdir /home/ntustison/Desktop/FA_from_T1T2/
