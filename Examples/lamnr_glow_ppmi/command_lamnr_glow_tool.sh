
base_dir="/Users/ntustison/Desktop/lamnr_glow_ppmi"
runs_dir="${base_dir}/runs/ppmi_t1_fa_128x128_K12_L5_HC192_align-vicreg_screen-cca_2/"
out_dir="${base_dir}/output/"
ckpt="${runs_dir}/training_state.pt"
manifest="${base_dir}/manifest_ppmi.csv"
manifest_short="${base_dir}/manifest_ppmi_short.csv"
manifest_lesions=/home/ntustison/Desktop/deep_simr_glow/manifest_lesions.csv

gaussian_lr=${out_dir}/t1_fa_lowrank.npz
gaussian_lr_summary=${out_dir}/t1_fa_lowrank_summary.json

# /Users/ntustison/anaconda3/bin/python3 lamnr_glow_tool.py gauss-fit \
#   --ckpt ${ckpt} \
#   --manifest ${manifest} \
#   --views T1,FA \
#   --slice-axis 2 --slice-index 128 \
#   --batch 64 --devices cpu \
#   --cov-mode perlevel \
#   --cov-estimator full \
#   --gauss-out ${gaussian_lr} \
#   --gauss-summary ${gaussian_lr_summary}

/Users/ntustison/anaconda3/bin/python3 lamnr_glow_tool.py gauss-impute \
  --ckpt ${ckpt} \
  --gauss ${gaussian_lr} \
  --manifest ${manifest_short} \
  --views T1,FA \
  --observed T1 --target FA \
  --slice-axis 2 --slice-index 128 \
  --batch 2 --devices cpu \
  --strategy sample \
  --temperature 0.0 \
  --outdir ${out_dir}/impute_FA_from_T1/ \
  --output-format nii.gz



# # Sample grid (kept)
# python lamnr_glow_tool.py sample \
#   --ckpt ${ckpt} \
#   --view-index 0 --sample-grid-size 6x8 \
#   --image-size 128x128 --temperature 0.8 \
#   --devices cuda:0 --sample-grid-out ${out_dir}/samples_t1.png

# # Reconstruction sanity panel (new subcommand)
# python lamnr_glow_tool.py recon \
#   --ckpt ${ckpt} \
#   --manifest ${manifest} --views T1,T2,FA --view-index 0 \
#   --slice-axis 2 --slice-index 64 --batch 6 --devices cuda:0 \
#   --out ${out_dir}/recon_t1_panel.png


# python lamnr_glow_tool.py recon \
#   --ckpt ${ckpt} \
#   --manifest ${manifest} --views T1,T2,FA --view-index 0 \
#   --slice-axis 2 --slice-index 64 --batch 6 --devices cuda:0 \
#   --gauss ${gaussian_lr} \
#   --edit-levels 0 \
#   --edit-what pc \
#   --edit-pc-index 0 \
#   --edit-pc-scale 2.0 \
#   --edit-pc-center sample \
#   --out ${out_dir}/recon_t1_pc0_k2_sample_0.png

# python lamnr_glow_tool.py recon \
#   --ckpt ${ckpt} \
#   --manifest ${manifest_lesions} --views T1 --view-index 0 \
#   --slice-axis 2 --slice-index 64 --batch 6 --devices cuda:0 \
#   --gauss ${gaussian_lr} \
#   --edit-levels 0 \
#   --edit-what pc \
#   --edit-pc-index 0 \
#   --edit-pc-scale 2.0 \
#   --edit-pc-center sample \
#   --out ${out_dir}/recon_t1_pc0_k2_sample_0.png



# python lamnr_glow_tool.py recon-template \
#   --ckpt runs/t1_fa_128x128_vicreg_K12_H192_vicreg/training_state.pt \
#   --gauss /home/ntustison/Desktop/deep_simr_glow/output/t1_fa_lowrank.npz \
#   --views T1,T2,FA \
#   --view-index 0 \
#   --devices cuda:0 \
#   --out ${out_dir}/template_T1_mu.png


# python lamnr_glow_tool.py recon-template \
# --ckpt runs/t1_fa_128x128_vicreg_K12_H192_vicreg/training_state.pt \
# --gauss /home/ntustison/Desktop/deep_simr_glow/output/t1_fa_lowrank.npz \
# --views T1,T2,FA \
# --view-index 0 \
# --devices cuda:0 \
# --mc-samples 32 \
# --mc-temp 0.25 \
# --seed 12345 \
# --out ${out_dir}/template_T1_mu_mc.png


