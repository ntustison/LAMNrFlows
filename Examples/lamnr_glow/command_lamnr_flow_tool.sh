
runs_dir="runs/t1_t2_fa_128x128_vicreg_K12_H192_vicreg/"
out_dir="/home/ntustison/Desktop/deep_simr_glow/output/"
ckpt="${runs_dir}/training_state.pt"
manifest=/home/ntustison/Desktop/deep_simr_glow/manifest_t1_t2_fa.csv

gaussian_lr=${out_dir}/t1_t2_fa_lowrank.npz
gaussian_lr_summary=${out_dir}/t1_t2_fa_lowrank_summary.json

python lamnr_flow_tool.py gauss-fit \
  --ckpt ${ckpt} \
  --manifest ${manifest} \
  --views T1,T2,FA \
  --slice-axis 2 --slice-index 64 \
  --batch 64 --devices cuda:0 \
  --cov-mode perlevel \
  --cov-estimator lowrank --rank 256 --sigma2 auto --cov-lam 1e-3 \
  --gauss-out ${gaussian_lr} \
  --gauss-summary ${gaussian_lr_summary}

python lamnr_flow_tool.py gauss-impute \
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
python lamnr_flow_tool.py sample \
  --ckpt ${ckpt} \
  --view-index 0 --sample-grid-size 6x8 \
  --image-size 128x128 --temperature 0.8 \
  --devices cuda:0 --sample-grid-out ${out_dir}/samples_t1.png

# Reconstruction sanity panel (new subcommand)
python lamnr_flow_tool.py recon \
  --ckpt ${ckpt} \
  --manifest ${manifest} --views T1,T2,FA --view-index 0 \
  --slice-axis 2 --slice-index 64 --batch 6 --devices cuda:0 \
  --out ${out_dir}/recon_t1_panel.png


python lamnr_flow_tool.py recon \
  --ckpt ${ckpt} \
  --manifest ${manifest} --views T1,T2,FA --view-index 0 \
  --slice-axis 2 --slice-index 64 --batch 6 --devices cuda:0 \
  --gauss ${gaussian_lr} \
  --edit-levels 0 \
  --edit-what pc \
  --edit-pc-index 0 \
  --edit-pc-scale 2.0 \
  --edit-pc-center sample \
  --out ${out_dir}/recon_t1_pc0_k2_sample_0.png


python lamnr_flow_tool.py recon-template \
  --ckpt runs/t1_t2_fa_128x128_vicreg_K12_H192_vicreg/training_state.pt \
  --gauss /home/ntustison/Desktop/deep_simr_glow/output/t1_t2_fa_lowrank.npz \
  --views T1,T2,FA \
  --view-index 0 \
  --devices cuda:0 \
  --out ${out_dir}/template_T1_mu.png


python lamnr_flow_tool.py recon-template \
--ckpt runs/t1_t2_fa_128x128_vicreg_K12_H192_vicreg/training_state.pt \
--gauss /home/ntustison/Desktop/deep_simr_glow/output/t1_t2_fa_lowrank.npz \
--views T1,T2,FA \
--view-index 0 \
--devices cuda:0 \
--mc-samples 32 \
--mc-temp 0.25 \
--seed 12345 \
--out ${out_dir}/template_T1_mu_mc.png


